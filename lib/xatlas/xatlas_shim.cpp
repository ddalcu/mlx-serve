#include "xatlas_shim.h"
#include "xatlas.h"

#include <cstdlib>
#include <cstring>
#include <new>

namespace {
struct UvwResult {
    uint32_t vertex_count = 0;
    uint32_t index_count = 0;
    uint32_t chart_count = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t *vmapping = nullptr;
    uint32_t *indices = nullptr;
    float *uvs = nullptr;
    ~UvwResult() {
        std::free(vmapping);
        std::free(indices);
        std::free(uvs);
    }
};
} // namespace

extern "C" {

void *uvw_parametrize(const float *positions, uint32_t vertex_count,
                      const uint32_t *indices, uint32_t index_count) {
    if (!positions || !indices || vertex_count == 0 || index_count < 3)
        return nullptr;

    xatlas::Atlas *atlas = xatlas::Create();
    if (!atlas)
        return nullptr;

    xatlas::MeshDecl decl;
    decl.vertexPositionData = positions;
    decl.vertexPositionStride = sizeof(float) * 3;
    decl.vertexCount = vertex_count;
    decl.indexData = indices;
    decl.indexCount = index_count;
    decl.indexFormat = xatlas::IndexFormat::UInt32;

    if (xatlas::AddMesh(atlas, decl) != xatlas::AddMeshError::Success) {
        xatlas::Destroy(atlas);
        return nullptr;
    }
    // Reference call is xatlas.parametrize(...) with ALL defaults.
    xatlas::Generate(atlas);

    if (atlas->meshCount != 1 || atlas->width == 0 || atlas->height == 0) {
        xatlas::Destroy(atlas);
        return nullptr;
    }

    const xatlas::Mesh &m = atlas->meshes[0];
    UvwResult *r = new (std::nothrow) UvwResult();
    if (!r) {
        xatlas::Destroy(atlas);
        return nullptr;
    }
    r->vertex_count = m.vertexCount;
    r->index_count = m.indexCount;
    r->chart_count = m.chartCount;
    r->width = atlas->width;
    r->height = atlas->height;
    r->vmapping = static_cast<uint32_t *>(std::malloc(sizeof(uint32_t) * m.vertexCount));
    r->indices = static_cast<uint32_t *>(std::malloc(sizeof(uint32_t) * m.indexCount));
    r->uvs = static_cast<float *>(std::malloc(sizeof(float) * 2 * m.vertexCount));
    if (!r->vmapping || !r->indices || !r->uvs) {
        delete r;
        xatlas::Destroy(atlas);
        return nullptr;
    }
    const float inv_w = 1.0f / static_cast<float>(atlas->width);
    const float inv_h = 1.0f / static_cast<float>(atlas->height);
    for (uint32_t i = 0; i < m.vertexCount; i++) {
        r->vmapping[i] = m.vertexArray[i].xref;
        r->uvs[i * 2 + 0] = m.vertexArray[i].uv[0] * inv_w;
        r->uvs[i * 2 + 1] = m.vertexArray[i].uv[1] * inv_h;
    }
    std::memcpy(r->indices, m.indexArray, sizeof(uint32_t) * m.indexCount);
    xatlas::Destroy(atlas);
    return r;
}

uint32_t uvw_vertex_count(const void *r) { return static_cast<const UvwResult *>(r)->vertex_count; }
uint32_t uvw_index_count(const void *r) { return static_cast<const UvwResult *>(r)->index_count; }
uint32_t uvw_chart_count(const void *r) { return static_cast<const UvwResult *>(r)->chart_count; }
uint32_t uvw_atlas_width(const void *r) { return static_cast<const UvwResult *>(r)->width; }
uint32_t uvw_atlas_height(const void *r) { return static_cast<const UvwResult *>(r)->height; }
const uint32_t *uvw_vmapping(const void *r) { return static_cast<const UvwResult *>(r)->vmapping; }
const uint32_t *uvw_indices(const void *r) { return static_cast<const UvwResult *>(r)->indices; }
const float *uvw_uvs(const void *r) { return static_cast<const UvwResult *>(r)->uvs; }

void uvw_free(void *r) { delete static_cast<UvwResult *>(r); }

} // extern "C"
