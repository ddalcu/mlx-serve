// Minimal C API over the C++ xatlas library (lib/xatlas/xatlas.h, MIT,
// jpcy/xatlas @ f700c7790aaa030e794b52ba7791a05c085faf0c) — the llama_shim
// discipline: a stable C surface so src/uvwrap.zig needs no C++ FFI.
//
// Reproduces the Python `xatlas.parametrize(positions, faces)` contract the
// Hunyuan3D-2.1 paint reference calls with ALL defaults (uvwrap_utils.py):
// ChartOptions + PackOptions default, positions + u32 index buffer only.
// UVs are returned normalized to [0,1] by atlas width/height (the Python
// binding's convention; raw xatlas UVs are in texel range).
#ifndef XATLAS_SHIM_H
#define XATLAS_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Unwrap `vertex_count` xyz positions (packed f32) with `index_count` u32
// indices (triangles). Returns an opaque result handle, or NULL on failure
// (empty input, AddMesh error, or no atlas produced). Free with uvw_free.
void *uvw_parametrize(const float *positions, uint32_t vertex_count,
                      const uint32_t *indices, uint32_t index_count);

uint32_t uvw_vertex_count(const void *r);   // output vertex count (>= input; seams duplicate)
uint32_t uvw_index_count(const void *r);    // == input index_count
uint32_t uvw_chart_count(const void *r);
uint32_t uvw_atlas_width(const void *r);
uint32_t uvw_atlas_height(const void *r);
const uint32_t *uvw_vmapping(const void *r); // [vertex_count] new vertex -> ORIGINAL vertex index
const uint32_t *uvw_indices(const void *r);  // [index_count] re-indexed triangles
const float *uvw_uvs(const void *r);         // [vertex_count*2] normalized [0,1]

void uvw_free(void *r);

#ifdef __cplusplus
}
#endif

#endif
