// One threadgroup of N / 4 threads per row: the RePo layer's LayerNorm in f32 (its affine weight and bias f32), out f32;
// SPLIT writes a row [hi | hi | lo] (hi exact in TF32: the low 13 mantissa bits cleared; lo = out - hi), the operand
// of a 3xTF32 product as one GEMM.
threadgroup float part[32];
const uint row = threadgroup_position_in_grid.x;
const uint lid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const uint N = X_shape[1], groups = N / 128;
const size_t at = size_t(row) * N + lid * 4;
const float4 v = float4(*(const device vec<T, 4>*)(X + at));
float s = simd_sum(v.x + v.y + v.z + v.w);
if (lane == 0) part[sg] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
float mean = 0;
for (uint k = 0; k < groups; k++) mean += part[k];
mean /= float(N);
const float4 d = v - mean;
s = simd_sum(d.x * d.x + d.y * d.y + d.z * d.z + d.w * d.w);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) part[sg] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
float var = 0;
for (uint k = 0; k < groups; k++) var += part[k];
const float r = metal::precise::rsqrt(var / float(N) + eps);
const uint c = lid * 4;
float4 o;
for (int e = 0; e < 4; e++) o[e] = d[e] * r * W[c + e] + BI[c + e];
if (SPLIT) {
    const float4 hi = as_type<float4>(as_type<uint4>(o) & 0xFFFFE000u);
    device float* r = O + size_t(row) * 3 * N + c;
    *(device float4*)(r) = hi;
    *(device float4*)(r + N) = hi;
    *(device float4*)(r + 2 * N) = o - hi;
} else {
    *(device float4*)(O + at) = o;
}
