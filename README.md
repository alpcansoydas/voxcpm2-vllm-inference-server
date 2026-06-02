# VoxCPM2 — Docker TTS Server with Web UI

Serves [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (2B tokenizer-free TTS, 48kHz) inside a single Docker container.

**Architecture inside the container:**

```
Browser → port 8000 → server.py (FastAPI UI + proxy)
                           │
                           └─→ vllm-omni (port 8001, internal)
                                    │
                                    └─→ VoxCPM2 model (GPU)
```

- **vllm-omni** runs the model on an internal port — GPU inference, not exposed externally.
- **server.py** serves the web UI on port 8000 and proxies generation requests to vllm-omni.

## Requirements

- Docker with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU with ~6 GB VRAM + headroom
- ~5 GB free disk for the Docker image; ~9 GB for model weights

## Build & Run

```bash
docker compose up --build
```

The first time this runs it downloads model weights from HuggingFace (~9 GB). Subsequent starts use the cached `hf_cache` volume.

Pass a HuggingFace token if the repo requires authentication:

```bash
HF_TOKEN=hf_... docker compose up --build
```

## Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `openbmb/VoxCPM2` | HF repo or local path |
| `HF_TOKEN` | — | HuggingFace access token |
| `MAX_UPLOAD_BYTES` | `52428800` (50 MB) | Max upload file size |
| `UPLOAD_TTL_SECONDS` | `3600` (1 hour) | Uploaded files auto-deleted after this |
| `MAX_CONCURRENT_STREAMS` | `8` | Max parallel TTS generation streams |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `VLLM_CONNECT_TIMEOUT` | `10.0` | Timeout connecting to vllm backend (s) |
| `VLLM_READ_TIMEOUT` | `300.0` | Timeout waiting for vllm response (s) |

## Web UI

Once you see `Application startup complete` in the logs, open:

```
http://localhost:8000/
```

The UI status dot will show **yellow** (loading) while vllm-omni loads the model weights (~60 s on first start), then turn **green** when ready.

Features:
- **Voice Design** — describe the voice you want in plain text
- **Preset Voice** — choose from bundled voice presets by language/emotion
- **Controllable Clone** — upload reference audio + style instruction
- **Ultimate Clone** — upload reference audio + its transcript for highest-fidelity cloning
- Real-time streaming audio playback
- Latency metrics (TTFT, RTF, total time)
- WAV download

## Using local weights

If you already have the model weights downloaded, mount them and point `MODEL_ID` at the path:

```yaml
# docker-compose.yml
volumes:
  - /path/to/local/VoxCPM2:/models/VoxCPM2
environment:
  MODEL_ID: /models/VoxCPM2
```

## API

The container exposes a single port (8000) with these endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/health` | GET | Model readiness (`model_loaded: true/false`) |
| `/upload-audio` | POST | Upload reference audio, returns opaque `file_id` |
| `/api/stream` | POST | Streaming TTS (binary frame protocol) |
| `/api/presets` | GET | List bundled voice presets |
| `/voice-presets/{lang}/{voice}/{file}` | GET | Serve preset WAV files |

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"model_id":"openbmb/VoxCPM2"}
```

### Upload audio

```bash
curl -F file=@reference.wav http://localhost:8000/upload-audio
# {"file_id": "a3f1b2c4-..."}
```

Returns an opaque `file_id` (not a filesystem path). Use this ID as `reference_wav_path` or `prompt_wav_path` in stream requests. Uploads expire after `UPLOAD_TTL_SECONDS`.

### Streaming protocol (`/api/stream`)

POST JSON params, receive a binary stream with length-prefixed frames:

**Frame format:** `[type: u8][length: u32 LE][payload: bytes]`
- type `0` → JSON control message
- type `1` → float32-LE PCM audio samples

**Sequence:**
1. `{"type": "meta", "sample_rate": 48000}` — audio format header
2. Binary audio frames — float32-LE PCM chunks (streamed in real time)
3. `{"type": "done", "chunks": N}` — end of stream
4. `{"type": "error", "message": "..."}` — on failure

**Params:**

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | required | Text to synthesize |
| `control` | string | `""` | Emotion/style instruction (Voice Design) |
| `reference_wav_path` | string | — | Opaque ID from `/upload-audio` or preset `id` |
| `prompt_wav_path` | string | — | Same as reference, for Ultimate Clone |
| `prompt_text` | string | — | Exact transcript of prompt audio |
| `max_len` | int | `4096` | Max output length (tokens, capped at 8192) |

### Presets

```bash
curl http://localhost:8000/api/presets
```

Returns presets with an opaque `id` field. Use this `id` as `reference_wav_path` in stream requests.

## Stopping

```bash
docker compose down
```

Remove cached weights too:

```bash
docker compose down -v
```
