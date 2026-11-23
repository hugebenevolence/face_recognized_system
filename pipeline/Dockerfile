# DeepStream 7.1 runtime: GStreamer + nvinfer/TensorRT/nvtracker/nvdsosd, plus the Python bindings.
# Needs an NVIDIA GPU (Docker Desktop + WSL2, or the NVIDIA Container Toolkit on Linux).
FROM nvcr.io/nvidia/deepstream:7.1-samples-multiarch

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 GIO_USE_PROXY_RESOLVER=dummy

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpython3.10 python3-gi python3-gst-1.0 unzip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt \
    && pip3 install https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_x86_64.whl

# Face models (insightface buffalo_l) converted for TensorRT. TensorRT engines are built on
# first start and cached in the /models volume.
COPY prepare_models.py .
RUN mkdir -p /tmp/pack /opt/models \
    && curl -fsSL -o /tmp/pack/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
    && unzip -q -j /tmp/pack/buffalo_l.zip det_10g.onnx w600k_r50.onnx -d /tmp/pack \
    && python3 prepare_models.py /tmp/pack /opt/models \
    && rm -rf /tmp/pack

COPY configs ./configs
COPY app ./app
COPY entrypoint.sh .
EXPOSE 8090
ENTRYPOINT ["./entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
