// The decode kernels' mxfp8 reads, e4m3 at 2^-8 of its value (the half bias is 8 above e4m3's). By default MLX's
// (fp8.h, through fp4_scale.metal: put it first): S.1111.111 NaN, e8m0 0xFF NaN, each dot times 2^8 times its scale.
// FOLD (`MLX_SERVE_DSV41_FP8_FOLD`): the earlier form, the 2^8 on the scale (0xFF +Inf, codes 247 and up overflow) and
// S.1111.111 read as 480.

// e4m3 bytes 0 and 2 of `w` at 2^-8 of their value.
template <int FOLD>
inline float2 dsv41_e4m3x2(uint w) {
  if constexpr (FOLD != 0) {
    return float2(as_type<half2>(((w & 0x007F007Fu) << 7) | ((w & 0x00800080u) << 8)));
  } else {
    return float2(mlxserve_fp4_e4m3_x2m8(w), mlxserve_fp4_e4m3_x2m8(w >> 16));
  }
}

// The four e4m3 bytes of `w`, in order, at 2^-8 of their value.
template <int FOLD>
inline float4 dsv41_e4m3x4(uint w) {
  const float2 lo = dsv41_e4m3x2<FOLD>(w), hi = dsv41_e4m3x2<FOLD>(w >> 8);
  return float4(lo.x, hi.x, lo.y, hi.y);
}

// e8m0 code `e`'s scale and the factor on the dot it multiplies: MLX's read and 2^8, or (FOLD) 2^8 x the scale and 1.
template <int FOLD>
inline float2 dsv41_e8m0(uint e) {
  if constexpr (FOLD != 0) {
    return float2(as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 256.0f, 1.0f);
  } else {
    return float2(mlxserve_fp4_e8m0(e), 256.0f);
  }
}
