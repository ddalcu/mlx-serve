
// q3_decode_rcproj (2026-09-27): exact fp8 decoders shared by the rcproj kernels.
// e8m0 scale byte -> float, fp8.h fp8_e8m0::operator float (0 -> the 0x00400000 subnormal 2^-127).
static inline float rcp_e8m0(uint e) {
  return as_type<float>(e == 0u ? 0x00400000u : (e << 23));
}
// Four e4m3 codes of one word -> their exact dequantized values code * scale, with s256 = scale * 256.
// half2 lo = codes (0, 2), hi = codes (1, 3): (c & 127) << 7 with the code's sign at bit 15 of its half is the exact
// half e4m3(c) / 256 for all 256 codes (fp8.h decodes through the same half); the widen and * s256 (a power of two)
// are exact.
static inline void rcp_dq4(uint u, float s256, thread float* o) {
  const uint lo = ((u & 0x007F007Fu) << 7) | ((u << 8) & 0x80008000u);
  const uint hi = ((u >> 1) & 0x3F803F80u) | (u & 0x80008000u);
  const float2 a = float2(as_type<half2>(lo)) * s256;
  const float2 b = float2(as_type<half2>(hi)) * s256;
  o[0] = a.x;
  o[1] = b.x;
  o[2] = a.y;
  o[3] = b.y;
}
// Two bf16 of one word -> floats (exact).
static inline float rcp_bf_lo(uint u) {
  return as_type<float>(u << 16);
}
static inline float rcp_bf_hi(uint u) {
  return as_type<float>(u & 0xFFFF0000u);
}
