// One threadgroup per row; preserve the bf16 division rounding before rint.
const uint row = threadgroup_position_in_grid.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint tid = sg * 32 + lane;
const int K = x_shape[1];
threadgroup float maxima[8];
float amax = 0.0f;
bool has_nan = false;
for (int k = int(tid); k < K; k += 256) {
  const float v = float(x[(long)row * K + k]);
  has_nan |= isnan(v);
  amax = max(amax, abs(v));
}
amax = simd_max(amax);
has_nan = simd_any(has_nan);
if (lane == 0) maxima[sg] = has_nan ? as_type<float>(0x7fc00000u) : amax;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float group_max = lane < 8 ? maxima[lane] : 0.0f;
has_nan = simd_any(isnan(group_max));
amax = simd_max(group_max);
// MLX's Max propagates NaNs; a nonfinite activation must not become finite here.
const float sc = has_nan ? as_type<float>(0x7fc00000u) : max(amax, 1e-8f) / 127.0f;
const float narrow_scale = float(bfloat16_t(sc));
if (tid == 0) scale[row] = narrow_scale;
for (int k = int(tid); k < K; k += 256) {
  const float norm = float(bfloat16_t(float(x[(long)row * K + k]) / narrow_scale));
  q[(long)row * K + k] = int8_t(clamp(rint(norm), -127.0f, 127.0f));
}
