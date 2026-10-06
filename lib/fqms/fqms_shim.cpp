#include "fqms_shim.h"
#include "Simplify.h"

#include <mutex>
#include <new>

namespace {
// Simplify.h keeps its mesh in namespace globals: one caller at a time.
std::mutex g_mu;

struct FqmsResult {
    std::vector<float> positions;
    std::vector<uint32_t> indices;
};
} // namespace

extern "C" {

void *fqms_simplify(const float *positions, uint32_t vertex_count,
                    const uint32_t *indices, uint32_t index_count,
                    uint32_t target_faces) {
    if (!positions || !indices || vertex_count == 0 || index_count < 3 || index_count % 3 != 0)
        return nullptr;
    for (uint32_t i = 0; i < index_count; ++i)
        if (indices[i] >= vertex_count) return nullptr;
    try {
        std::lock_guard<std::mutex> lock(g_mu);
        Simplify::vertices.clear();
        Simplify::triangles.clear();
        Simplify::refs.clear();
        Simplify::vertices.resize(vertex_count);
        for (uint32_t i = 0; i < vertex_count; ++i) {
            Simplify::Vertex &v = Simplify::vertices[i];
            v.p.x = positions[i * 3 + 0];
            v.p.y = positions[i * 3 + 1];
            v.p.z = positions[i * 3 + 2];
        }
        const uint32_t faces = index_count / 3;
        Simplify::triangles.resize(faces);
        for (uint32_t f = 0; f < faces; ++f) {
            Simplify::Triangle &t = Simplify::triangles[f];
            for (int k = 0; k < 3; ++k) t.v[k] = (int)indices[f * 3 + k];
            t.attr = 0;
            t.material = -1;
        }
        // Aggressiveness 7: the library's and fast_simplification's default.
        Simplify::simplify_mesh((int)target_faces, 7.0);

        FqmsResult *r = new FqmsResult();
        r->positions.resize(Simplify::vertices.size() * 3);
        for (size_t i = 0; i < Simplify::vertices.size(); ++i) {
            r->positions[i * 3 + 0] = (float)Simplify::vertices[i].p.x;
            r->positions[i * 3 + 1] = (float)Simplify::vertices[i].p.y;
            r->positions[i * 3 + 2] = (float)Simplify::vertices[i].p.z;
        }
        r->indices.resize(Simplify::triangles.size() * 3);
        for (size_t f = 0; f < Simplify::triangles.size(); ++f)
            for (int k = 0; k < 3; ++k) r->indices[f * 3 + k] = (uint32_t)Simplify::triangles[f].v[k];
        Simplify::vertices.clear();
        Simplify::triangles.clear();
        Simplify::refs.clear();
        return r;
    } catch (...) {
        return nullptr;
    }
}

uint32_t fqms_vertex_count(const void *r) { return (uint32_t)(static_cast<const FqmsResult *>(r)->positions.size() / 3); }
uint32_t fqms_index_count(const void *r) { return (uint32_t)static_cast<const FqmsResult *>(r)->indices.size(); }
const float *fqms_positions(const void *r) { return static_cast<const FqmsResult *>(r)->positions.data(); }
const uint32_t *fqms_indices(const void *r) { return static_cast<const FqmsResult *>(r)->indices.data(); }

void fqms_free(void *r) { delete static_cast<FqmsResult *>(r); }

}
