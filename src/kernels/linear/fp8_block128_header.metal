// fp8res (mimo/fp8res, 2026-10-09): exact e4m3fn decoders for the release's FP8 residents, value = code x the f32
// scale of the code's 64-row x 128-column block. The bit trick is the DSV RCPROJ decoder's (src/kernels/exl3/
// header_rcproj.metal `rcp_dq4`, device-checked 2026-09-27), with an f32 block scale in place of its e8m0 group scale.
//
// Four codes c0..c3 of one little-endian word: lo = half2(c0, c2), hi = half2(c1, c3), each half the exact e4m3(c) / 256:
// (c & 127) << 7 puts the code's 4 exponent and 3 mantissa bits at the half's bits 7..13 (half bias 15 = e4m3 bias 7 + 8)
// and its sign at bit 15; subnormal codes land on half subnormals. The release holds no NaN code (S.1111.111: the
// converter refuses one), so every word decodes exactly; the half -> float widen is exact.
static inline void fp8r_halves(uint u, thread half2& lo, thread half2& hi) {
  lo = as_type<half2>(((u & 0x007F007Fu) << 7) | ((u << 8) & 0x80008000u));
  hi = as_type<half2>(((u >> 1) & 0x3F803F80u) | (u & 0x80008000u));
}

// The GEMV's weights: f32 code x scale, given s256 = scale x 256 (exact): (e4m3(c) / 256) x (256 x scale) is the
// exact real product code x scale, rounded once to f32: the converter's own f32 product, before any BF16 rounding.
static inline void fp8r_dq4(uint u, float s256, thread float* o) {
  half2 lo, hi;
  fp8r_halves(u, lo, hi);
  const float2 a = float2(lo) * s256;
  const float2 b = float2(hi) * s256;
  o[0] = a.x;
  o[1] = b.x;
  o[2] = a.y;
  o[3] = b.y;
}

// The GEMM's B operand: bfloat e4m3(c) / 256 (exact: at most 4 significant bits, inside bfloat's normal range, so the
// tensor unit sees no subnormal); the block's scale x 256 multiplies the f32 block sum instead.
static inline void fp8r_bf4(uint u, thread bfloat16_t* o) {
  half2 lo, hi;
  fp8r_halves(u, lo, hi);
  const float2 a = float2(lo);
  const float2 b = float2(hi);
  o[0] = static_cast<bfloat16_t>(a.x);
  o[1] = static_cast<bfloat16_t>(b.x);
  o[2] = static_cast<bfloat16_t>(a.y);
  o[3] = static_cast<bfloat16_t>(b.y);
}

// Two bf16 of one word -> floats (exact).
static inline float fp8r_bf_lo(uint u) {
  return as_type<float>(u << 16);
}
static inline float fp8r_bf_hi(uint u) {
  return as_type<float>(u & 0xFFFF0000u);
}
