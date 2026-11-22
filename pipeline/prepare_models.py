"""Turns the insightface buffalo_l ONNX files into TensorRT-friendly models.

  det_10g.onnx   (SCRFD, dynamic input)  ->  scrfd_640.onnx    fixed 1x3x640x640, every output gets a
                                                               leading batch axis (nvinfer treats the
                                                               first axis as batch and would otherwise
                                                               misreport the tensor sizes)
  w600k_r50.onnx (ArcFace, batch 'None') ->  arcface_dyn.onnx  dynamic batch, 512-d output

Engines are built by nvinfer on first start and cached next to the ONNX files.
Usage: python3 prepare_models.py <dir containing the buffalo_l .onnx files> <output dir>
"""

import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def _set_dim(value_info, index: int, size) -> None:
    dim = value_info.type.tensor_type.shape.dim[index]
    dim.ClearField("dim_param")
    dim.ClearField("dim_value")
    if isinstance(size, int):
        dim.dim_value = size
    else:
        dim.dim_param = size


def _add_batch_axis(model: onnx.ModelProto) -> None:
    opset = max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    axes = helper.make_tensor("batch_axis", TensorProto.INT64, [1], [0])
    if opset >= 13:
        model.graph.initializer.append(axes)
    for out in list(model.graph.output):
        inner = out.name + "_nobatch"
        for node in model.graph.node:  # rename the producer's output
            for i, name in enumerate(node.output):
                if name == out.name:
                    node.output[i] = inner
        if opset >= 13:
            node = helper.make_node("Unsqueeze", [inner, "batch_axis"], [out.name])
        else:
            node = helper.make_node("Unsqueeze", [inner], [out.name], axes=[0])
        model.graph.node.append(node)
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        del out.type.tensor_type.shape.dim[:]
        for v in [1, *shape]:
            out.type.tensor_type.shape.dim.add().dim_value = v


def main(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)

    det = onnx.load(src / "det_10g.onnx")
    for i, size in enumerate((1, 3, 640, 640)):
        _set_dim(det.graph.input[0], i, size)
    _add_batch_axis(det)
    onnx.checker.check_model(det)
    onnx.save(det, dst / "scrfd_640.onnx")

    rec = onnx.load(src / "w600k_r50.onnx")
    _set_dim(rec.graph.input[0], 0, "batch")
    _set_dim(rec.graph.output[0], 0, "batch")
    onnx.save(rec, dst / "arcface_dyn.onnx")
    print("models written to", dst)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
