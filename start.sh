#!/bin/sh
set -e

UPLOAD_DIR="${UPLOAD_DIR:-/tmp/voxcpm_uploads}"
mkdir -p "$UPLOAD_DIR"
export UPLOAD_DIR

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRESETS_DIR="${PRESETS_DIR:-$SCRIPT_DIR/voice_presets}"

# Start vllm-omni on internal port 8001.
# --allowed-local-media-path is restricted to uploads + presets only.
vllm-omni serve "${MODEL_ID}" \
  --omni \
  --host 127.0.0.1 \
  --port 8001 \
  --trust-remote-code \
  --served-model-name voxcpm2 \
  --allowed-local-media-path "$UPLOAD_DIR" \
  --allowed-local-media-path "$PRESETS_DIR" &

VLLM_PID=$!

cleanup() {
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  exit 0
}
trap cleanup TERM INT

exec python server.py --host 0.0.0.0 --port 8000 --vllm-url http://127.0.0.1:8001
