// PF_RMS_ROWS (deepseek_v41.zig `rmsRows`): MLX's rms_single_row (mlx/backend/metal/kernels/rms_norm.metal, MLX 0.32)
// over rows `x_strides[0]` apart, so a row slice of a wider projection (the query latent of q_kv's output) is normed
// where it lies instead of after MLX's contiguous copy. MLX's statements, its threadgroup (AX / 4 threads rounded up to
// whole simdgroups, one a row) and its four reads a thread: the same sums in the same order, the same words.
constexpr int N_READS = 4;
constexpr int SIMD_SIZE = 32;
const uint gid = threadgroup_position_in_grid.x;
const uint lid = thread_position_in_threadgroup.x;
const uint simd_lane_id = thread_index_in_simdgroup;
const uint simd_group_id = simdgroup_index_in_threadgroup;
const uint axis_size = AX;

threadgroup float local_inv_mean[1];
threadgroup float local_sums[SIMD_SIZE];

float acc = 0;
float thread_x[N_READS];
const device T* xr = x + gid * size_t(x_strides[0]) + lid * N_READS;
const device T* wr = w + lid * N_READS;
if (lid * N_READS + N_READS <= axis_size) {
  for (int i = 0; i < N_READS; i++) {
    thread_x[i] = xr[i];
    acc += thread_x[i] * thread_x[i];
  }
} else {
  for (int i = 0; i < N_READS; i++) {
    thread_x[i] = (lid * N_READS + i < axis_size) ? (float)xr[i] : 0;
    acc += thread_x[i] * thread_x[i];
  }
}
acc = simd_sum(acc);
if (simd_group_id == 0) {
  local_sums[simd_lane_id] = 0;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd_lane_id == 0) {
  local_sums[simd_group_id] = acc;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd_group_id == 0) {
  acc = simd_sum(local_sums[simd_lane_id]);
  if (simd_lane_id == 0) {
    local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps[0]);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
device T* o = out + gid * size_t(axis_size) + lid * N_READS;
if (lid * N_READS + N_READS <= axis_size) {
  for (int i = 0; i < N_READS; i++) {
    o[i] = wr[i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
  }
} else {
  for (int i = 0; i < N_READS; i++) {
    if ((lid * N_READS + i) < axis_size) {
      o[i] = wr[i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
    }
  }
}
