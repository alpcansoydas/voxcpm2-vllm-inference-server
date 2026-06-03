#!/bin/sh
set -e

MODEL_ID="${MODEL_ID:-openbmb/VoxCPM2}"
export MODEL_ID

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRESETS_DIR="${PRESETS_DIR:-$SCRIPT_DIR/voice_presets}"

# Uploaded reference audio. MUST live under MEDIA_ROOT (below) so vllm-omni
# accepts its file:// URIs.
UPLOAD_DIR="${UPLOAD_DIR:-$SCRIPT_DIR/uploads}"
mkdir -p "$UPLOAD_DIR"
export UPLOAD_DIR

# vllm-omni's --allowed-local-media-path takes a SINGLE directory — when passed
# more than once only the last value wins. It must therefore be a common parent
# of BOTH the uploads dir and the presets dir; otherwise uploads are rejected
# with "must be a subpath of --allowed-local-media-path".
MEDIA_ROOT="${MEDIA_ROOT:-$SCRIPT_DIR}"

# Start vllm-omni on internal port 8001.
vllm-omni serve "${MODEL_ID}" \
  --omni \
  --host 127.0.0.1 \
  --port 8001 \
  --trust-remote-code \
  --served-model-name voxcpm2 \
  --allowed-local-media-path "$MEDIA_ROOT" &

VLLM_PID=$!

cleanup() {
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  exit 0
}
trap cleanup TERM INT

exec python server.py --host 0.0.0.0 --port 8000 --vllm-url http://127.0.0.1:8001
