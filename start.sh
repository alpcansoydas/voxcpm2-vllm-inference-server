#!/bin/sh
# Start vllm-omni on internal port 8001, then the UI server on port 8000.
# The UI server proxies inference requests to vllm-omni.

vllm-omni serve "${MODEL_ID}" \
  --omni \
  --host 127.0.0.1 \
  --port 8001 \
  --trust-remote-code \
  --served-model-name voxcpm2 \
  --allowed-local-media-path / &

exec python server.py --host 0.0.0.0 --port 8000 --vllm-url http://127.0.0.1:8001
