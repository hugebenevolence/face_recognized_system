#!/bin/sh
# /models is a volume so the TensorRT engines survive container re-creation.
mkdir -p /models
cp -n /opt/models/*.onnx /models/
exec "$@"
