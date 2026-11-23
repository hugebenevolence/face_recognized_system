"""Small helpers for reading and writing DeepStream metadata from Python (pyds)."""

import ctypes
from typing import Optional

import numpy as np
import pyds

PGIE_ID, SGIE_ID = 1, 2
SKIP_CLASS = 1  # objects given this class are not sent to ArcFace
UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF  # pyds does not export this constant


def iter_list(head, cast):
    while head is not None:
        yield cast(head.data)
        head = head.next


def read_tensors(fm) -> dict[tuple[int, int], np.ndarray]:
    """Raw output tensors nvinfer attached to a frame, keyed by (rows, columns)."""
    out: dict[tuple[int, int], np.ndarray] = {}
    for um in iter_list(fm.frame_user_meta_list, pyds.NvDsUserMeta.cast):
        if um.base_meta.meta_type != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            continue
        tm = pyds.NvDsInferTensorMeta.cast(um.user_meta_data)
        for i in range(tm.num_output_layers):
            layer = pyds.get_nvds_LayerInfo(tm, i)
            total = layer.inferDims.numElements
            rows = layer.inferDims.d[0]
            ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
            arr = np.ctypeslib.as_array(ptr, shape=(total,)).reshape(rows, total // rows)
            out[(rows, total // rows)] = arr
    return out


def read_embedding(om) -> Optional[np.ndarray]:
    """L2-normalised 512-d ArcFace vector the secondary nvinfer attached to an object."""
    for um in iter_list(om.obj_user_meta_list, pyds.NvDsUserMeta.cast):
        if um.base_meta.meta_type != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            continue
        tm = pyds.NvDsInferTensorMeta.cast(um.user_meta_data)
        if tm.unique_id != SGIE_ID or tm.num_output_layers < 1:
            continue
        layer = pyds.get_nvds_LayerInfo(tm, 0)
        ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
        v = np.ctypeslib.as_array(ptr, shape=(layer.inferDims.numElements,)).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)
    return None


def set_rect(om, left: float, top: float, width: float, height: float) -> None:
    r = om.rect_params
    r.left, r.top, r.width, r.height = left, top, width, height


def add_object(bm, fm, box, score: float = 1.0, label: str = "face"):
    """Adds a detector object (as if PGIE had produced it) to a frame and returns it."""
    om = pyds.nvds_acquire_obj_meta_from_pool(bm)
    om.unique_component_id = PGIE_ID
    om.class_id = 0
    om.confidence = score
    om.object_id = UNTRACKED_OBJECT_ID
    om.obj_label = label
    set_rect(om, box[0], box[1], box[2] - box[0], box[3] - box[1])
    om.detector_bbox_info.org_bbox_coords.left = box[0]
    om.detector_bbox_info.org_bbox_coords.top = box[1]
    om.detector_bbox_info.org_bbox_coords.width = box[2] - box[0]
    om.detector_bbox_info.org_bbox_coords.height = box[3] - box[1]
    pyds.nvds_add_obj_meta_to_frame(fm, om, None)
    return om
