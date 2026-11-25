"""Small helpers for reading and writing DeepStream metadata from Python (pyds)."""

import ctypes
from typing import Optional

import numpy as np
import pyds

PGIE_ID, SGIE_ID = 1, 2
UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF  # pyds does not export this constant
# Marker in misc_obj_info[3]: fields 0..2 hold the face alignment matrix (see face_align.cpp)
ALIGN_MAGIC = 0x46414C4E


def set_alignment(om, matrix: np.ndarray) -> None:
    """Stores the 2x3 alignment matrix in the object's user-defined fields, where the alignment
    library of nvdspreprocess reads it: six float32 packed into three int64, then the marker."""
    packed = np.frombuffer(np.asarray(matrix, dtype=np.float32).reshape(6).tobytes(), dtype=np.int64)
    fields = _misc_obj_info(om)
    fields[0:3] = packed
    fields[3] = ALIGN_MAGIC


def _misc_obj_info(om) -> np.ndarray:
    """The object's misc_obj_info (gint64[4] in nvdsmeta.h) as a writable int64 array. pyds 1.2
    exposes it as an int32[4] view, which covers only half of the field, so the full field is
    addressed through the start of that view."""
    addr = om.misc_obj_info.ctypes.data
    return np.ctypeslib.as_array((ctypes.c_int64 * 4).from_address(addr))


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


def clip_box(box, width: int, height: int) -> tuple[float, float, float, float]:
    """A detector box limited to the frame. SCRFD can return boxes that run past the frame edge,
    and DeepStream's scaling of an object fails when its rectangle leaves the frame."""
    x1, y1 = min(max(box[0], 0.0), width - 1.0), min(max(box[1], 0.0), height - 1.0)
    x2, y2 = min(max(box[2], x1 + 1.0), width - 1.0), min(max(box[3], y1 + 1.0), height - 1.0)
    return x1, y1, x2, y2


def add_object(bm, fm, box, score: float = 1.0, label: str = "face", frame_size: tuple[int, int] = None):
    """Adds a detector object (as if PGIE had produced it) to a frame and returns it.
    With frame_size (width, height) the box is clipped to the frame first."""
    if frame_size is not None:
        box = clip_box(box, *frame_size)
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
