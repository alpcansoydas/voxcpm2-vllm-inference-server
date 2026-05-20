FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System deps — no CUDA devel layer needed; torch bundles its own CUDA runtime.
# gcc is required by Triton, which JIT-compiles its CUDA driver utility at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv python3-dev \
      build-essential \
      git curl ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Use a venv to avoid PEP 668 restrictions on ubuntu:24.04
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --no-cache-dir uv

# vllm 0.20.0 — uv selects the right torch+CUDA wheel for the host GPU automatically
RUN uv pip install --no-cache "vllm==0.20.0" --torch-backend=auto

# vllm-omni at the matching v0.20.0 tag
RUN git clone --branch v0.20.0 --depth 1 \
      https://github.com/vllm-project/vllm-omni.git /opt/vllm-omni \
 && uv pip install --no-cache -e /opt/vllm-omni

WORKDIR /app

# UI server dependencies (FastAPI, uvicorn, httpx, …)
COPY requirements.txt ./
RUN uv pip install --no-cache -r requirements.txt

# UI and server files
COPY server.py ./
COPY static/ ./static/
COPY voice_presets/ ./voice_presets/

# Startup script
COPY start.sh ./
RUN chmod +x start.sh

# HuggingFace cache — mount a host volume here to avoid re-downloading weights
ENV HF_HOME=/root/.cache/huggingface
VOLUME /root/.cache/huggingface

# Model to serve: HF repo ID or a local path mounted at runtime
ENV MODEL_ID=openbmb/VoxCPM2

EXPOSE 8000

# Starts vllm-omni on internal :8001, then the FastAPI UI on :8000
CMD ["./start.sh"]
