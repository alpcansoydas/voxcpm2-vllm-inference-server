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
| `/upload-audio` | POST | Upload reference audio, returns `file_id` path |
| `/ws/stream` | WebSocket | Streaming TTS (see below) |
| `/api/presets` | GET | List bundled voice presets |
| `/voice-presets/{lang}/{voice}/{file}` | GET | Serve preset WAV files |

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"model_id":"openbmb/VoxCPM2"}
```

### WebSocket streaming protocol

Connect to `ws://localhost:8000/ws/stream`, send a JSON params message, receive:

1. `{"type": "meta", "sample_rate": 48000}` — audio format header
2. Binary frames — float32-LE PCM chunks (stream in real time)
3. `{"type": "done", "chunks": N}` — end of stream
4. `{"type": "error", "message": "..."}` — on failure

**Params:**

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | required | Text to synthesize |
| `control` | string | `""` | Emotion/style instruction (Voice Design) |
| `reference_wav_path` | string | — | File path returned by `/upload-audio` |
| `prompt_wav_path` | string | — | Same as reference, for Ultimate Clone |
| `prompt_text` | string | — | Exact transcript of prompt audio |
| `cfg_value` | float | `2.0` | Classifier-free guidance scale |
| `inference_timesteps` | int | `10` | Diffusion steps (speed vs quality) |
| `min_len` / `max_len` | int | `2` / `4096` | Output length bounds (tokens) |
| `normalize` | bool | `false` | Text normalization (numbers, dates) |
| `denoise` | bool | `false` | Denoise reference audio |
| `retry_badcase` | bool | `true` | Auto-retry poor quality outputs |
| `retry_badcase_max_times` | int | `3` | Max retries |
| `retry_badcase_ratio_threshold` | float | `6.0` | Retry quality threshold |

## Stopping

```bash
docker compose down
```

Remove cached weights too:

```bash
docker compose down -v
```
