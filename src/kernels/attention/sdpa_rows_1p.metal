// The exact verify's attention rows on MLX's one-pass vector kernel: MLX 0.32.5's `sdpa_vector`
// (mlx/backend/metal/kernels/sdpa_vector.h, Copyright © 2024 Apple Inc., MIT) for several query rows in one launch
// (sdk/blocks/attention/sdpa_rows.zig). Threadgroup (bh, j) runs sdpa_vector's statements, in its order, for query row
// R0 + j of head bh over its own N = N0 + j keys: what MLX's one-row call over the first N keys runs (q_len 1, so no
// causal mask, every key below N), with no mask, sinks or window (their function constants false). Only the addressing
// of the rows is this file's: the query row r of qin [1, H, S, D] at its strides, the output row j of oout [1, H, R, V].
// Template constants T, D, V; p = (R0, N0).
constexpr int BN = 32;
constexpr int BD = 32;
constexpr int qk_per_thread = D / BD;
constexpr int v_per_thread = V / BD;

const uint3 tid = threadgroup_position_in_grid;
const uint3 tpg = threadgroups_per_grid;
const uint simd_gid = simdgroup_index_in_threadgroup;
const uint simd_lid = thread_index_in_simdgroup;

const int gqa_factor = qin_shape[1] / kin_shape[1];
const int N = p[1] + int(tid.y);
const size_t k_head_stride = kin_shape[1] == 1 ? size_t(kin_strides[0]) : size_t(kin_strides[1]);
const size_t k_seq_stride = size_t(kin_strides[2]);
const size_t v_head_stride = vin_shape[1] == 1 ? size_t(vin_strides[0]) : size_t(vin_strides[1]);
const size_t v_seq_stride = size_t(vin_strides[2]);
const float scale = sc[0];

int inner_k_stride = BN * int(k_seq_stride);
int inner_v_stride = BN * int(v_seq_stride);

typedef float U;

thread U q[qk_per_thread];
thread U k[qk_per_thread];
thread U o[v_per_thread];

threadgroup U outputs[BN * BD];
threadgroup U max_scores[BN];
threadgroup U sum_exp_scores[BN];

// Adjust positions
const int q_batch_head_idx = tid.x;
const int q_seq_idx = tid.y;
const int kv_head_idx = q_batch_head_idx / gqa_factor;
const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
const device T* queries = qin + q_batch_head_idx * qin_strides[1] + (p[0] + q_seq_idx) * qin_strides[2] +
    simd_lid * qk_per_thread;
const device T* keys = kin + kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
    simd_lid * qk_per_thread;
const device T* values = vin + kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
    simd_lid * v_per_thread;

device T* out = oout + o_offset * V + simd_gid * v_per_thread;

// Restrict the keys to the window
int lo = 0;
int hi = N;

// Read the query and 0 the output accumulator
for (int i = 0; i < qk_per_thread; i++) {
  q[i] = static_cast<U>(scale) * queries[i];
}
for (int i = 0; i < v_per_thread; i++) {
  o[i] = 0;
}

U max_score = Limits<U>::finite_min;
U sum_exp_score = 0;

// For each key in valid range
for (int i = lo + simd_gid; i < hi; i += BN) {
  bool use_key = true;
  if (use_key) {
    // Read the key
    for (int j = 0; j < qk_per_thread; j++) {
      k[j] = keys[j];
    }

    // Compute the i-th score
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      score += q[j] * k[j];
    }
    score = simd_sum(score);

    // Update the accumulators
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    // Update the output accumulator
    for (int j = 0; j < v_per_thread; j++) {
      o[j] = o[j] * factor + exp_score * values[j];
    }
  }

  // Move the pointers to the next kv
  keys += inner_k_stride;
  values += inner_v_stride;
}

// Each thread has a partial part of the output so we need to combine them.

// First let's communicate the max and sum_exp
if (simd_lid == 0) {
  max_scores[simd_gid] = max_score;
  sum_exp_scores[simd_gid] = sum_exp_score;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
max_score = max_scores[simd_lid];
U new_max = simd_max(max_score);
U factor = fast::exp(max_score - new_max);
sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

// Now we need to aggregate all the outputs
for (int i = 0; i < v_per_thread; i++) {
  outputs[simd_lid * BD + simd_gid] = o[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
  o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// And write the output
if (simd_lid == 0) {
  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}
