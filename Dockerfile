FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# stage 2: the app
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    DEPTH_MODEL=depth-anything/Depth-Anything-V2-Large-hf

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch: the default wheel drags in ~2 GB of CUDA that never runs here
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0"

COPY requirements.txt .
RUN grep -viE '^torch' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt

# Backend/ = API + CLI, mathsandml/ = the science.
COPY backend/ ./backend/
COPY mathsandml/ ./mathsandml/
# The fine-tuned model (hybrid engine).
COPY models/ ./models/
COPY --from=web /web/dist ./frontend/dist

# Bake the checkpoint in, so the first run needs no network
RUN python -c "\
from transformers import pipeline; import os; \
pipeline('depth-estimation', model=os.environ['DEPTH_MODEL']); \
print('checkpoint cached into', os.environ['HF_HOME'])"

EXPOSE 8000
ENV HOST=0.0.0.0 PORT=8000
CMD ["python", "backend/server.py"]
