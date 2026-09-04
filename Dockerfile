# DepthWizard - one image, no internet at run time.
#
#   docker build -t depthwizard .
#   docker run --rm -p 8000:8000 depthwizard
#   open http://localhost:8000
#
# Two things are baked in so the container never reaches the network:
#   * the Depth Anything V2 checkpoint, pre-downloaded into the image
#   * the React bundle, built at image-build time and served by FastAPI
#
# ---------------------------------------------------------------------------
# stage 1: build the front end
# ---------------------------------------------------------------------------
FROM node:20-slim AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# stage 2: the app
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    DEPTH_MODEL=depth-anything/Depth-Anything-V2-Large-hf

WORKDIR /app

# rasterio and opencv ship manylinux wheels with GDAL bundled; these are the
# only system libraries OpenCV dlopens at import
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch: the default wheel drags in ~2 GB of CUDA that never runs here
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0"

COPY requirements.txt .
# torch is installed above; installing it again from PyPI would pull the CUDA build
RUN grep -viE '^torch' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt

COPY *.py ./
COPY --from=web /web/dist ./web/dist

# bake the checkpoint in, so the first run needs no network
RUN python -c "\
from transformers import pipeline; import os; \
pipeline('depth-estimation', model=os.environ['DEPTH_MODEL']); \
print('checkpoint cached into', os.environ['HF_HOME'])"

EXPOSE 8000
ENV HOST=0.0.0.0 PORT=8000
CMD ["python", "server.py"]
