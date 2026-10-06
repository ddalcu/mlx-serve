// Minimal C API over Fast-Quadric-Mesh-Simplification (lib/fqms/Simplify.h,
// MIT, sp4cerat @ 65df07dc) — the xatlas_shim discipline: a stable C surface
// so src/mesh_simplify.zig needs no C++ FFI.
#ifndef FQMS_SHIM_H
#define FQMS_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Quadric-decimate a triangle mesh to at most `target_faces` triangles.
// Returns an opaque result handle, or NULL on failure. Free with fqms_free.
void *fqms_simplify(const float *positions, uint32_t vertex_count,
                    const uint32_t *indices, uint32_t index_count,
                    uint32_t target_faces);

uint32_t fqms_vertex_count(const void *r);
uint32_t fqms_index_count(const void *r);
const float *fqms_positions(const void *r);    // [vertex_count*3]
const uint32_t *fqms_indices(const void *r);   // [index_count]

void fqms_free(void *r);

#ifdef __cplusplus
}
#endif

#endif
