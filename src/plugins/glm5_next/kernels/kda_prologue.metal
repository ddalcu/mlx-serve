// MLX_SERVE_KDA_PROMPT_PROLOGUE (kda_prologue.zig): a KDA layer's prompt prologue in one dispatch, kda.zig's op chain
// (conv1d.run's concat, depthwise conv and SiLU; the q / k / v slices; q's and k's rms_norm over the key head and their
// scales; the contiguous v the recurrence copies), each step the MLX kernel the chain runs, in its operand order and
// types:
//   conv   conv.metal `depthwise_conv_1d` over [prev rows | qkv rows]: acc = 0; acc += float(in) * w[i], taps in order;
//          the sum cast to InT
//   SiLU   unary_ops.h `Sigmoid` on that InT (kda_sigmoid), then binary_ops.h `Multiply` in InT: y * sigmoid(y)
//   rms    rms_norm.metal `rms_single_row` at axis DK: 32 lanes x 4 reads (lane l holds 4l .. 4l + 3), simd_sum, the
//          second-level simd_sum over lane 0's sum (the other lanes 0), precise::rsqrt(sum / axis + eps), the weight
//          (ones) times InT(x * inv)
//   scale  binary_ops.h `Multiply` in InT by the chain's own InT scalar (`q_scale`, `k_scale`)
// Grid (32, 3 * H, rows), threadgroup (32, 8, 1): one simdgroup a (row, section, head), x the lane.
const uint lane = thread_position_in_grid.x;
const uint sh = thread_position_in_grid.y;
const int t = int(thread_position_in_grid.z);
const uint sec = sh / H;
const uint h = sh - sec * H;
const uint cl = h * DK + lane * 4;
const uint c = sec * DIM + cl;
InT o[4];
for (int j = 0; j < 4; j++) {
  float acc = 0.0;
  const device InT* wj = w + size_t(c + j) * TAPS;
  for (int i = 0; i < TAPS; i++) {
    const int row = t + i;
    const InT xin = row < TAPS - 1 ? prev[size_t(row) * CDIM + c + j] : qkv[size_t(row - (TAPS - 1)) * CDIM + c + j];
    acc += static_cast<float>(xin) * wj[i];
  }
  const InT y = static_cast<InT>(acc);
  const InT sg = kda_sigmoid(y);
  o[j] = y * sg;
}
const size_t at = size_t(t) * DIM + cl;
if (sec == 2) {
  for (int j = 0; j < 4; j++) {
    v_out[at + j] = o[j];
  }
} else {
  float thread_x[4];
  float acc = 0;
  for (int j = 0; j < 4; j++) {
    thread_x[j] = o[j];
    acc += thread_x[j] * thread_x[j];
  }
  acc = simd_sum(acc);
  acc = simd_sum(lane == 0 ? acc : 0.0f);
  const uint axis_size = DK;
  const float inv = metal::precise::rsqrt(acc / axis_size + eps);
  const InT one = static_cast<InT>(1.0f);
  const InT sc = sec == 0 ? q_scale : k_scale;
  device InT* dst = sec == 0 ? q_out : k_out;
  for (int j = 0; j < 4; j++) {
    const InT r = one * static_cast<InT>(thread_x[j] * inv);
    dst[at + j] = r * sc;
  }
}
