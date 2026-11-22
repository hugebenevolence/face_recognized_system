# CPU:  BASE_IMAGE=ubuntu:22.04                                 ORT_PACKAGE=onnxruntime
# GPU:  BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 ORT_PACKAGE=onnxruntime-gpu
# (docker-compose.gpu.yml sets the GPU values)
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ARG ORT_PACKAGE=onnxruntime
ARG MODEL_PACK=buffalo_l
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev build-essential libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# insightface compiles a Cython extension and needs numpy/cython visible at build time.
COPY constraints.txt .
ENV PIP_CONSTRAINT=/srv/constraints.txt
RUN pip3 install --upgrade pip setuptools wheel numpy==1.26.4 "cython<3.1"
COPY requirements.txt .
RUN pip3 install --no-build-isolation insightface==0.7.3 \
    && pip3 install ${ORT_PACKAGE}==1.19.2 \
    && pip3 install -r requirements.txt

# Bake the model weights into the image so first start needs no network.
RUN python3 -c "from insightface.app import FaceAnalysis; \
FaceAnalysis(name='${MODEL_PACK}', allowed_modules=['detection','recognition'], providers=['CPUExecutionProvider'])"

COPY app ./app
EXPOSE 8090
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
