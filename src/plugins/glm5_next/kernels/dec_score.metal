const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint tid = sg * 32 + lane;
const int q_pos = dims[0];
const int P = dims[1];
threadgroup float qT[DH * 32];
threadgroup float kp[SGS][DH];
for (uint i = tid; i < uint(32 * DH); i += SGS * 32) qT[(i % DH) * 32 + i / DH] = float(q[i]);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int pp = 0; pp < PPS; ++pp) {
const int p = (int(threadgroup_position_in_grid.y) * SGS + int(sg)) * PPS + pp;
if (p >= P) return;
// The pool's key: its completing row's gate half; lane l loads channels 4l..4l+3 (DH == 128).
const device vec<T, 4>* pooled = (const device vec<T, 4>*)(cache + (size_t(p) * KPOOL + KPOOL - 1) * (2 * DH) + DH);
((threadgroup float4*)kp[sg])[lane] = float4(pooled[lane]);
simdgroup_barrier(mem_flags::mem_threadgroup);
float dot = 0.0f;
for (int d = 0; d < DH; ++d) dot = fma(qT[d * 32 + lane], kp[sg][d], dot);
const float sc = simd_sum(w[lane] * metal::max(dot, 0.0f));
if (lane == 0) out[p] = (p * KPOOL + KPOOL - 1 <= q_pos) ? sc : -FLT_MAX;
simdgroup_barrier(mem_flags::mem_threadgroup);
}