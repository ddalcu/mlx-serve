// One threadgroup of N / 4 threads per row: LayerNorm (no affine) in f32, then x * (1 + scale) + shift rounded once
// (1 + scale rounds to T first, as the reference's add does).
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
vec<T, 4> o;
for (int e = 0; e < 4; e++) {
    const float s1 = float(T(float(SC[c + e]) + 1.0f));
    o[e] = T(d[e] * r * s1 + float(SH[c + e]));
}
*(device vec<T, 4>*)(O + at) = o;
