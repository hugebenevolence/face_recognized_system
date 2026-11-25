#pragma once
#include <cuda_runtime.h>

/* Warps one face out of an RGBA frame with the inverse affine map (output pixel -> frame pixel)
 * and writes the normalised planar RGB tensor (value - offset) * scale. */
extern "C" void fa_launch(const unsigned char *src, int pitch, int src_w, int src_h, const float inv[6],
                          float *out, int out_w, int out_h, float scale, const float offset[3],
                          cudaStream_t stream);
