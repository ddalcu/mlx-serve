// The exact verify's attention rows on MLX's two-pass vector kernels, pass 2: MLX 0.32.5's `sdpa_vector_2pass_2`
// (mlx/backend/metal/kernels/sdpa_vector.h, Copyright © 2024 Apple Inc., MIT), its statements in its order, over the
// partials pass 1 laid out for R rows (threadgroup (bh, j): row j of head bh's `blocks` partials, `blocks` read at run
// time as MLX passes it). Template constants T, V (sdpa_vector_2pass_2's D, the value dim); p = (R0, N0, blocks).
constexpr int BN = 32;
constexpr int BD = 32;
constexpr int elem_per_thread = V / BD;

const uint3 tid = threadgroup_position_in_grid;
const uint3 tpg = threadgroups_per_grid;
const uint simd_gid = simdgroup_index_in_threadgroup;
const uint simd_lid = thread_index_in_simdgroup;
const int blocks = p[2];

typedef float U;

thread U o[elem_per_thread] = {0};
threadgroup U outputs[BN * BD];

// Adjust positions
const int head_idx = tid.x;
const int q_seq_idx = tid.y;
const int q_offset = head_idx * tpg.y + q_seq_idx;
const device T* part = partials + q_offset * blocks * V + simd_gid * V + simd_lid * elem_per_thread;
const device float* sums_p = sums + q_offset * blocks;
const device float* maxs_p = maxs + q_offset * blocks;
device T* out = oout + q_offset * V + simd_gid * elem_per_thread;

// Set defaults
U sum_exp_score = 0.0;
U max_score = Limits<U>::finite_min;

// Reduce the max
for (int b = 0; b < blocks / BN; ++b) {
  max_score = max(max_score, maxs_p[simd_lid + BN * b]);
}
max_score = simd_max(max_score);

// Reduce the d
for (int b = 0; b < blocks / BN; ++b) {
  U factor = fast::exp(maxs_p[simd_lid + BN * b] - max_score);
  sum_exp_score += factor * sums_p[simd_lid + BN * b];
}
sum_exp_score = simd_sum(sum_exp_score);

// Reduce the sum exp and partials
for (int b = 0; b < blocks / BN; ++b) {
  U factor = fast::exp(maxs_p[simd_gid] - max_score);

  // Update the output accumulator
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] += factor * static_cast<U>(part[i]);
  }
  maxs_p += BN;
  sums_p += BN;
  part += BN * V;
}

// Use shared memory to transpose and reduce the final block
for (int i = 0; i < elem_per_thread; i++) {
  outputs[simd_lid * BD + simd_gid] = o[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
  o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// And write the output
if (simd_lid == 0) {
  for (int i = 0; i < elem_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}
