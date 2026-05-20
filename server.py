#!/usr/bin/env python3
"""VoxCPM2 FastAPI UI server — proxies inference to a vllm-omni backend."""
from __future__ import annotations

import json
import os
import re
import struct
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Tuple

import httpx
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

_VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8001")
_MODEL_ID  = os.environ.get("MODEL_ID", "openbmb/VoxCPM2")
_TMP_DIR   = Path(tempfile.mkdtemp(prefix="voxcpm_uploads_"))

# Persistent HTTP client reused across requests to avoid per-request TCP
# connection setup when proxying to vllm-omni on localhost.
_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(timeout=None)
    yield
    await _http_client.aclose()
    _http_client = None


app = FastAPI(title="VoxCPM2 Streaming Server", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── WAV utilities ──────────────────────────────────────────────────────────────

def _parse_wav_header(data: bytes) -> Tuple[int, int, int, int]:
    """
    Scan a WAV buffer for the data chunk.
    Returns (data_offset, sample_rate, bits_per_sample, audio_format).
    audio_format: 1 = PCM int, 3 = IEEE float.
    Handles extra chunks (LIST, fact, …) before the data chunk.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a RIFF/WAVE file")
    if data[12:16] != b"fmt ":
        raise ValueError("Expected fmt chunk at offset 12")

    fmt_size       = struct.unpack_from("<I", data, 16)[0]
    audio_format   = struct.unpack_from("<H", data, 20)[0]
    sample_rate    = struct.unpack_from("<I", data, 24)[0]
    bits_per_sample = struct.unpack_from("<H", data, 34)[0]

    # Scan forward for the "data" sub-chunk
    pos = 12 + 8 + fmt_size          # past WAVE marker + fmt header + fmt body
    while pos + 8 <= len(data):
        chunk_id   = data[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
        if chunk_id == b"data":
            return pos + 8, sample_rate, bits_per_sample, audio_format
        pos += 8 + chunk_size + (chunk_size % 2)   # chunks are word-aligned

    raise ValueError("WAV data chunk not found")


def _pcm_to_float32(pcm: bytes, bits_per_sample: int, audio_format: int) -> np.ndarray:
    """Convert raw PCM bytes to float32 samples in [-1, 1]."""
    if audio_format == 3 or bits_per_sample == 32:          # IEEE float32
        return np.frombuffer(pcm, dtype=np.float32).copy()
    elif bits_per_sample == 16:
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    elif bits_per_sample == 24:
        # 24-bit is awkward — unpack manually
        arr = np.zeros(len(pcm) // 3, dtype=np.float32)
        for i in range(len(arr)):
            b = pcm[i*3 : i*3+3]
            val = struct.unpack("<i", b + (b"\xff" if b[2] & 0x80 else b"\x00"))[0] >> 8
            arr[i] = val / 8388608.0
        return arr
    else:
        raise ValueError(f"Unsupported bits_per_sample={bits_per_sample}")


# ── HTTP Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        r = await _http_client.get(f"{_VLLM_URL}/health", timeout=5.0)
        ready = r.status_code == 200
    except Exception:
        ready = False
    return {
        "status":       "ok" if ready else "loading",
        "model_loaded": ready,
        "model_id":     _MODEL_ID,
    }


@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    suffix = Path(file.filename or "audio").suffix or ".wav"
    fid    = str(uuid.uuid4())
    dest   = _TMP_DIR / f"{fid}{suffix}"
    dest.write_bytes(await file.read())
    return {"file_id": str(dest)}


# ── HTTP streaming ─────────────────────────────────────────────────────────────
# Frame format (length-prefixed binary):
#   [type: u8][length: u32 LE][payload: bytes]
#   type 0 → JSON  ({"type":"meta"|"done"|"error", ...})
#   type 1 → float32-LE PCM audio samples

def _json_frame(obj: dict) -> bytes:
    payload = json.dumps(obj).encode()
    return b"\x00" + struct.pack("<I", len(payload)) + payload

def _audio_frame(data: bytes) -> bytes:
    return b"\x01" + struct.pack("<I", len(data)) + data


@app.post("/api/stream")
async def api_stream(request: Request):
    try:
        params = await request.json()
    except Exception as exc:
        async def _err():
            yield _json_frame({"type": "error", "message": f"invalid request: {exc}"})
        return StreamingResponse(_err(), media_type="application/octet-stream")

    async def generate():
        text = (params.get("text") or "").strip()
        if not text:
            yield _json_frame({"type": "error", "message": "text is required"})
            return

        control = re.sub(r"[()（）]", "", (params.get("control") or "").strip()).strip()
        if control:
            text = f"({control}){text}"

        ref_path    = params.get("reference_wav_path") or ""
        prompt_path = params.get("prompt_wav_path") or ""
        prompt_text = (params.get("prompt_text") or "").strip()

        valid_ref    = ref_path    and Path(ref_path).exists()
        valid_prompt = prompt_path and Path(prompt_path).exists() and prompt_text

        # Build OpenAI-compatible /v1/audio/speech request for vllm-omni VoxCPM2.
        # vllm-omni uses `ref_audio` (file:// URI, data: URL, or http URL) and
        # `ref_text` — NOT the VoxCPM-native `reference_wav_path`/`prompt_wav_path`.
        # `max_new_tokens` maps to max audio token budget (≈ max_len in the old API).
        # `stream=True` is critical: without it vllm-omni awaits the entire audio
        # before sending any bytes (TTFT ≈ total latency).
        request_body: dict = {
            "model":           "voxcpm2",
            "input":           text,
            "response_format": "wav",
            "stream":          True,
            "max_new_tokens":  int(params.get("max_len", 4096)),
        }

        # Continuation (Ultimate Cloning): prompt audio + transcript
        if valid_prompt:
            request_body["ref_audio"] = Path(prompt_path).as_uri()
            request_body["ref_text"]  = prompt_text
        # Controllable Cloning / Preset: reference audio only
        elif valid_ref:
            request_body["ref_audio"] = Path(ref_path).as_uri()

        chunk_count      = 0
        header_done      = False
        buf              = bytearray()
        sample_rate      = 48000
        bits             = 16
        audio_fmt        = 1
        data_offset      = 44
        bytes_per_sample = 2

        try:
            async with _http_client.stream(
                "POST",
                f"{_VLLM_URL}/v1/audio/speech",
                json=request_body,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield _json_frame({
                        "type":    "error",
                        "message": f"vllm-omni error {resp.status_code}: "
                                   f"{body[:300].decode(errors='replace')}",
                    })
                    return

                async for raw in resp.aiter_bytes(chunk_size=4096):
                    buf.extend(raw)

                    # Parse WAV header once we have enough bytes (44 = minimal WAV header)
                    if not header_done and len(buf) >= 44:
                        try:
                            data_offset, sample_rate, bits, audio_fmt = _parse_wav_header(bytes(buf))
                        except ValueError as e:
                            yield _json_frame({"type": "error", "message": f"WAV parse error: {e}"})
                            return
                        bytes_per_sample = max(1, bits // 8)
                        buf = buf[data_offset:]
                        header_done = True
                        yield _json_frame({"type": "meta", "sample_rate": sample_rate})

                    if not header_done:
                        continue

                    # Flush complete samples from buffer (avoid splitting a sample)
                    n_bytes = (len(buf) // bytes_per_sample) * bytes_per_sample
                    if n_bytes >= bytes_per_sample * 256:
                        pcm_bytes = bytes(buf[:n_bytes])
                        buf = buf[n_bytes:]
                        float32 = _pcm_to_float32(pcm_bytes, bits, audio_fmt)
                        yield _audio_frame(float32.tobytes())
                        chunk_count += 1

                # Flush any remaining samples
                if header_done:
                    n_bytes = (len(buf) // bytes_per_sample) * bytes_per_sample
                    if n_bytes >= bytes_per_sample:
                        pcm_bytes = bytes(buf[:n_bytes])
                        float32 = _pcm_to_float32(pcm_bytes, bits, audio_fmt)
                        yield _audio_frame(float32.tobytes())
                        chunk_count += 1

        except Exception as exc:
            yield _json_frame({"type": "error", "message": str(exc)})
            return

        yield _json_frame({"type": "done", "chunks": chunk_count})

    return StreamingResponse(generate(), media_type="application/octet-stream")


# ── Voice presets API ──────────────────────────────────────────────────────────
_PRESETS_DIR = Path(__file__).parent / "voice_presets"

_LANG_NAMES = {
    "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish",
    "fr": "French", "hu": "Hungarian", "it": "Italian", "ja": "Japanese",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "tr": "Turkish",
    "zh": "Chinese",
}


@app.get("/api/presets")
async def get_presets():
    presets = []
    if not _PRESETS_DIR.exists():
        return {"presets": presets}
    for lang_dir in sorted(_PRESETS_DIR.iterdir()):
        if not lang_dir.is_dir() or lang_dir.name.startswith("."):
            continue
        lang = lang_dir.name
        for voice_dir in sorted(lang_dir.iterdir()):
            if not voice_dir.is_dir() or voice_dir.name.startswith("."):
                continue
            voice = voice_dir.name
            for wav_file in sorted(voice_dir.glob("*.wav")):
                emotion = wav_file.stem
                parts = emotion.split("_")
                if len(parts) > 1 and parts[0] == lang:
                    parts = parts[1:]
                if parts and parts[0] in ("child", "man", "woman"):
                    parts = parts[1:]
                emotion_label = "_".join(parts) if parts else emotion
                presets.append({
                    "lang":        lang,
                    "lang_name":   _LANG_NAMES.get(lang, lang.upper()),
                    "voice":       voice,
                    "emotion":     emotion_label,
                    "preview_url": f"/voice-presets/{lang}/{voice}/{wav_file.name}",
                    "server_path": str(wav_file.absolute()),
                })
    return {"presets": presets}


# ── Static files ───────────────────────────────────────────────────────────────
_STATIC = Path(__file__).parent / "static"
_STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

if _PRESETS_DIR.exists():
    app.mount("/voice-presets", StaticFiles(directory=str(_PRESETS_DIR)), name="voice-presets")


@app.get("/")
async def index():
    return FileResponse(str(_STATIC / "index.html"))


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import uvicorn

    p = argparse.ArgumentParser(description="VoxCPM2 UI server")
    p.add_argument("--host",     default="0.0.0.0")
    p.add_argument("--port",     type=int, default=8000)
    p.add_argument("--vllm-url", default="http://127.0.0.1:8001",
                   help="Base URL of the vllm-omni backend")
    args = p.parse_args()

    os.environ["VLLM_URL"] = args.vllm_url
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
