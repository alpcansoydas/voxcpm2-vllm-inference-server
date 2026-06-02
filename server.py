#!/usr/bin/env python3
"""VoxCPM2 FastAPI UI server — proxies inference to a vllm-omni backend."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import struct
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Tuple

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("voxcpm2")

# ── Configuration ─────────────────────────────────────────────────────────────

_VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8001")
_MODEL_ID = os.environ.get("MODEL_ID", "openbmb/VoxCPM2")

_UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", tempfile.mkdtemp(prefix="voxcpm_uploads_")))
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 50 * 1024 * 1024))  # 50 MB
_UPLOAD_TTL_SECONDS = int(os.environ.get("UPLOAD_TTL_SECONDS", 3600))  # 1 hour
_ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}

_MAX_CONCURRENT_STREAMS = int(os.environ.get("MAX_CONCURRENT_STREAMS", 8))
_VLLM_CONNECT_TIMEOUT = float(os.environ.get("VLLM_CONNECT_TIMEOUT", 30.0))
_VLLM_READ_TIMEOUT = float(os.environ.get("VLLM_READ_TIMEOUT", 300.0))

_PRESETS_DIR = Path(__file__).parent / "voice_presets"

# ── Runtime state ─────────────────────────────────────────────────────────────

_http_client: httpx.AsyncClient | None = None
_stream_semaphore: asyncio.Semaphore | None = None
_upload_registry: dict[str, Path] = {}  # opaque_id → absolute path
_preset_registry: dict[str, Path] = {}  # opaque_id → absolute path


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _http_client, _stream_semaphore
    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=_VLLM_CONNECT_TIMEOUT,
            read=_VLLM_READ_TIMEOUT,
            write=30.0,
            pool=10.0,
        )
    )
    _stream_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_STREAMS)
    _build_preset_registry()
    _schedule_upload_cleanup()
    yield
    await _http_client.aclose()
    _http_client = None


app = FastAPI(title="VoxCPM2 Streaming Server", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Upload cleanup ────────────────────────────────────────────────────────────

_cleanup_task: asyncio.Task | None = None


def _schedule_upload_cleanup():
    global _cleanup_task
    _cleanup_task = asyncio.create_task(_upload_cleanup_loop())


async def _upload_cleanup_loop():
    while True:
        await asyncio.sleep(300)  # every 5 minutes
        now = time.time()
        expired = []
        for fid, fpath in list(_upload_registry.items()):
            try:
                if fpath.exists() and (now - fpath.stat().st_mtime) > _UPLOAD_TTL_SECONDS:
                    fpath.unlink(missing_ok=True)
                    expired.append(fid)
            except OSError:
                expired.append(fid)
        for fid in expired:
            _upload_registry.pop(fid, None)
        if expired:
            logger.info("Cleaned %d expired uploads", len(expired))


# ── Preset registry ───────────────────────────────────────────────────────────

def _build_preset_registry():
    """Map each preset WAV to an opaque ID so clients never see server paths."""
    if not _PRESETS_DIR.exists():
        return
    for wav_file in _PRESETS_DIR.rglob("*.wav"):
        rel = wav_file.relative_to(_PRESETS_DIR)
        opaque_id = f"preset:{rel.as_posix()}"
        _preset_registry[opaque_id] = wav_file.resolve()


def _resolve_audio_id(audio_id: str) -> Path | None:
    """Resolve an opaque audio ID to a validated absolute path."""
    if not audio_id:
        return None
    if audio_id.startswith("preset:"):
        return _preset_registry.get(audio_id)
    return _upload_registry.get(audio_id)


# ── WAV utilities ─────────────────────────────────────────────────────────────

def _parse_wav_header(data: bytes) -> Tuple[int, int, int, int]:
    """
    Scan a WAV buffer for the data chunk.
    Returns (data_offset, sample_rate, bits_per_sample, audio_format).
    """
    if len(data) < 44:
        raise ValueError("WAV buffer too small")
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a RIFF/WAVE file")
    if data[12:16] != b"fmt ":
        raise ValueError("Expected fmt chunk at offset 12")

    fmt_size = struct.unpack_from("<I", data, 16)[0]
    audio_format = struct.unpack_from("<H", data, 20)[0]
    sample_rate = struct.unpack_from("<I", data, 24)[0]
    bits_per_sample = struct.unpack_from("<H", data, 34)[0]

    pos = 12 + 8 + fmt_size
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
        if chunk_id == b"data":
            return pos + 8, sample_rate, bits_per_sample, audio_format
        pos += 8 + chunk_size + (chunk_size % 2)

    raise ValueError("WAV data chunk not found")


def _pcm_to_float32(pcm: bytes, bits_per_sample: int, audio_format: int) -> np.ndarray:
    """Convert raw PCM bytes to float32 samples in [-1, 1]."""
    if audio_format == 3 or bits_per_sample == 32:
        return np.frombuffer(pcm, dtype=np.float32).copy()
    elif bits_per_sample == 16:
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    elif bits_per_sample == 24:
        n_samples = len(pcm) // 3
        raw = np.frombuffer(pcm, dtype=np.uint8).reshape(n_samples, 3)
        # Reconstruct 32-bit signed integers from 24-bit LE samples
        i32 = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        # Sign-extend from 24-bit
        i32[i32 >= 0x800000] -= 0x1000000
        return (i32 / 8388608.0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported bits_per_sample={bits_per_sample}")


# ── HTTP Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        r = await _http_client.get(f"{_VLLM_URL}/health", timeout=5.0)
        ready = r.status_code == 200
    except Exception:
        ready = False
    return {
        "status": "ok" if ready else "loading",
        "model_loaded": ready,
        "model_id": _MODEL_ID,
    }


@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    filename = file.filename or "audio.wav"
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(400, f"Unsupported audio format: {suffix}")

    fid = str(uuid.uuid4())
    dest = _UPLOAD_DIR / f"{fid}{suffix}"

    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(64 * 1024):
            size += len(chunk)
            if size > _MAX_UPLOAD_BYTES:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {_MAX_UPLOAD_BYTES // (1024*1024)} MB limit")
            f.write(chunk)

    _upload_registry[fid] = dest.resolve()
    return {"file_id": fid}


# ── HTTP streaming ────────────────────────────────────────────────────────────

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

        ref_id = (params.get("reference_wav_path") or "").strip()
        prompt_id = (params.get("prompt_wav_path") or "").strip()
        prompt_text = (params.get("prompt_text") or "").strip()

        ref_path = _resolve_audio_id(ref_id)
        prompt_path = _resolve_audio_id(prompt_id) if prompt_id else None

        valid_ref = ref_path is not None and ref_path.exists()
        valid_prompt = prompt_path is not None and prompt_path.exists() and prompt_text

        request_body: dict = {
            "model": "voxcpm2",
            "input": text,
            "response_format": "wav",
            "stream": True,
            "max_new_tokens": min(int(params.get("max_len", 4096)), 8192),
        }

        if valid_prompt:
            request_body["ref_audio"] = prompt_path.as_uri()
            request_body["ref_text"] = prompt_text
        elif valid_ref:
            request_body["ref_audio"] = ref_path.as_uri()

        chunk_count = 0
        header_done = False
        buf = bytearray()
        sample_rate = 48000
        bits = 16
        audio_fmt = 1
        data_offset = 44
        bytes_per_sample = 2

        acquired = _stream_semaphore.acquire() if _stream_semaphore else None
        if acquired:
            try:
                await asyncio.wait_for(acquired, timeout=30.0)
            except asyncio.TimeoutError:
                yield _json_frame({"type": "error", "message": "server busy, try again later"})
                return

        try:
            async with _http_client.stream(
                "POST",
                f"{_VLLM_URL}/v1/audio/speech",
                json=request_body,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield _json_frame({
                        "type": "error",
                        "message": f"vllm-omni error {resp.status_code}: "
                                   f"{body[:300].decode(errors='replace')}",
                    })
                    return

                async for raw in resp.aiter_bytes(chunk_size=4096):
                    buf.extend(raw)

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

                    n_bytes = (len(buf) // bytes_per_sample) * bytes_per_sample
                    if n_bytes >= bytes_per_sample * 256:
                        pcm_bytes = bytes(buf[:n_bytes])
                        buf = buf[n_bytes:]
                        float32 = await asyncio.to_thread(
                            _pcm_to_float32, pcm_bytes, bits, audio_fmt
                        )
                        yield _audio_frame(float32.tobytes())
                        chunk_count += 1

                if header_done:
                    n_bytes = (len(buf) // bytes_per_sample) * bytes_per_sample
                    if n_bytes >= bytes_per_sample:
                        pcm_bytes = bytes(buf[:n_bytes])
                        float32 = await asyncio.to_thread(
                            _pcm_to_float32, pcm_bytes, bits, audio_fmt
                        )
                        yield _audio_frame(float32.tobytes())
                        chunk_count += 1

        except httpx.ConnectError:
            yield _json_frame({"type": "error", "message": "backend unavailable"})
            return
        except httpx.ReadTimeout:
            yield _json_frame({"type": "error", "message": "backend timed out"})
            return
        except Exception as exc:
            logger.exception("Stream error")
            yield _json_frame({"type": "error", "message": str(exc)})
            return
        finally:
            if _stream_semaphore:
                _stream_semaphore.release()

        yield _json_frame({"type": "done", "chunks": chunk_count})

    return StreamingResponse(generate(), media_type="application/octet-stream")


# ── Voice presets API ─────────────────────────────────────────────────────────

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

                rel = wav_file.relative_to(_PRESETS_DIR)
                opaque_id = f"preset:{rel.as_posix()}"

                presets.append({
                    "lang": lang,
                    "lang_name": _LANG_NAMES.get(lang, lang.upper()),
                    "voice": voice,
                    "emotion": emotion_label,
                    "preview_url": f"/voice-presets/{lang}/{voice}/{wav_file.name}",
                    "id": opaque_id,
                })
    return {"presets": presets}


# ── Static files ──────────────────────────────────────────────────────────────
_STATIC = Path(__file__).parent / "static"
_STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

if _PRESETS_DIR.exists():
    app.mount("/voice-presets", StaticFiles(directory=str(_PRESETS_DIR)), name="voice-presets")


@app.get("/")
async def index():
    return FileResponse(str(_STATIC / "index.html"))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import uvicorn

    p = argparse.ArgumentParser(description="VoxCPM2 UI server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--vllm-url", default="http://127.0.0.1:8001",
                   help="Base URL of the vllm-omni backend")
    args = p.parse_args()

    os.environ["VLLM_URL"] = args.vllm_url
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
