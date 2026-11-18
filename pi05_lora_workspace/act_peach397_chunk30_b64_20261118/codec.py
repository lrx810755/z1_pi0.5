"""Indexed per-frame zlib preserves the exact float32 deployed RGB pixels."""
import zlib
import numpy as np

SHAPE = (3, 224, 224)
FRAME_BYTES = 3 * 224 * 224 * 4


def pack_frame(pixels):
    array = np.asarray(pixels)
    if array.shape != SHAPE or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError('Require finite float32 RGB CHW224')
    return zlib.compress(np.ascontiguousarray(array).tobytes(), 1)


def unpack_frame(payload):
    data = zlib.decompress(payload)
    if len(data) != FRAME_BYTES:
        raise ValueError('Decoded frame has wrong length')
    return np.frombuffer(data, dtype=np.float32).reshape(SHAPE)
