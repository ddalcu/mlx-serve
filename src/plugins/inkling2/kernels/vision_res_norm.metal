#pragma clang fp contract(off)
// Inkling 2 vision: fused projection bias + layer-scaled residual + optional next LayerNorm.
// X, Y [L, D] (T), B, SC, NW, NB [D] (T), eps (0-d f32) -> OUT [L, D] (T), NORM [L, D] or [1] (T).
// grid (TG, L, 1), threadgroup (TG, 1, 1) with TG <= 256 (multiple of 32).
const uint row = threadgroup_position_in_grid.y;
const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const size_t base = size_t(row) * uint(D);

threadgroup float sh[16];
constexpr uint GROUPS = uint(TG) / 32u;

if (D <= TG * 4) {
    float val[4];
    float local_sum = 0.0f;
    const uint c0 = tid * 4u;
#pragma clang loop unroll(full)
    for (uint e = 0; e < 4u; ++e) {
        const uint c = c0 + e;
        if (c < uint(D)) {
            const T yb = T(float(Y[base + c]) + float(B[c]));
            const T ys = T(float(yb) * float(SC[c]));
            const T m = T(float(X[base + c]) + float(ys));
            OUT[base + c] = m;
            val[e] = float(m);
            local_sum += val[e];
        }
    }
    if (!DO_NORM) return;

    float s = simd_sum(local_sum);
    if (lane == 0u) sh[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float t = simd_sum(lane < GROUPS ? sh[lane] : 0.0f);
        if (lane == 0u) sh[0] = t / float(D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = sh[0];

    float local_var = 0.0f;
#pragma clang loop unroll(full)
    for (uint e = 0; e < 4u; ++e) {
        if (c0 + e < uint(D)) {
            const float diff = val[e] - mean;
            local_var += diff * diff;
        }
    }
    float vs = simd_sum(local_var);
    if (lane == 0u) sh[8u + sg] = vs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float t = simd_sum(lane < GROUPS ? sh[8u + lane] : 0.0f);
        if (lane == 0u) sh[8] = metal::precise::rsqrt(t / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = sh[8];

#pragma clang loop unroll(full)
    for (uint e = 0; e < 4u; ++e) {
        const uint c = c0 + e;
        if (c < uint(D)) {
            NORM[base + c] = T((val[e] - mean) * inv * float(NW[c]) + float(NB[c]));
        }
    }
    return;
}

float local_sum = 0.0f;
for (uint c = tid; c < uint(D); c += uint(TG)) {
    const T yb = T(float(Y[base + c]) + float(B[c]));
    const T ys = T(float(yb) * float(SC[c]));
    const T m = T(float(X[base + c]) + float(ys));
    OUT[base + c] = m;
    local_sum += float(m);
}
if (!DO_NORM) return;

float s = simd_sum(local_sum);
if (lane == 0u) sh[sg] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0u) {
    float t = simd_sum(lane < GROUPS ? sh[lane] : 0.0f);
    if (lane == 0u) sh[0] = t / float(D);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float mean = sh[0];

float local_var = 0.0f;
for (uint c = tid; c < uint(D); c += uint(TG)) {
    const float diff = float(OUT[base + c]) - mean;
    local_var += diff * diff;
}
float vs = simd_sum(local_var);
if (lane == 0u) sh[8u + sg] = vs;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0u) {
    float t = simd_sum(lane < GROUPS ? sh[8u + lane] : 0.0f);
    if (lane == 0u) sh[8] = metal::precise::rsqrt(t / float(D) + eps);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float inv = sh[8];

for (uint c = tid; c < uint(D); c += uint(TG)) {
    NORM[base + c] = T((float(OUT[base + c]) - mean) * inv * float(NW[c]) + float(NB[c]));
}
