// One row reduction stage. `local_sums` is zeroed once at kernel start;
// the slots past the simdgroup count stay zero, so the 32-wide simd_sum
// below is MLX's tree. Two barriers a stage: the previous stage's reads
// of both planes complete before its second barrier, so a new stage's
// writes cannot race them.
METAL_FUNC float msv_row_inv_rms(float acc, float eps, int axis, threadgroup float* local_sums, threadgroup float* local_inv, uint simd_gid, uint simd_lid) {
  acc = simd_sum(acc);
  if (simd_lid == 0) local_sums[simd_gid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    acc = simd_sum(local_sums[simd_lid]);
    if (simd_lid == 0) local_inv[0] = metal::precise::rsqrt(acc / axis + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return local_inv[0];
}