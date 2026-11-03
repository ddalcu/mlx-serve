// One threadgroup of 256 threads per row: the RePo LayerNorm folded into its gate | content projection. P [rows,
// PARTS * N2] is the row of H [rows, C] (f16 or bf16) times the stacked maps with the norm's weight folded in: their
// TF32-exact part, then the rest when PARTS is 2. Out GC [rows, N2] f32 = rstd * (P's parts summed - mean * CS) + BW,
// the statistics two-pass in f32 as the LayerNorm's.
threadgroup float part[8];
const uint row = threadgroup_position_in_grid.x;
const uint lid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const uint C = H_shape[1];
const uint N2 = CS_shape[0];
const device T* h = H + size_t(row) * C;
float s = 0;
for (uint i = lid; i < C; i += 256) s += float(h[i]);
s = simd_sum(s);
if (lane == 0) part[sg] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
float mean = 0;
for (uint k = 0; k < 8; k++) mean += part[k];
mean /= float(C);
float q = 0;
for (uint i = lid; i < C; i += 256) {
    const float d = float(h[i]) - mean;
    q += d * d;
}
q = simd_sum(q);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) part[sg] = q;
threadgroup_barrier(mem_flags::mem_threadgroup);
float var = 0;
for (uint k = 0; k < 8; k++) var += part[k];
const float r = metal::precise::rsqrt(var / float(C) + eps);
const device float* p = P + size_t(row) * PARTS * N2;
for (uint k = lid; k < N2; k += 256) {
    const float sum = PARTS == 2 ? p[k] + p[N2 + k] : p[k];
    GC[size_t(row) * N2 + k] = r * (sum - mean * CS[k]) + BW[k];
}
