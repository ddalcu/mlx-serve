// The fp4 group scale, one decode for every fp4 kernel: MXFP4's e8m0 (a group of 32) and NVFP4's e4m3 (a group of
// 16), picked by the group size GS. Bit for bit MLX's `fp8_e8m0` / `fp8_e4m3` reads (fp8.h, `dequantize_scale`),
// one NaN rule: e4m3 0x7f and 0xff are NaN, e8m0 0xff is NaN, e8m0 0 is 2^-127.
// `_x2m8` is the value times 2^-8: a kernel whose e2m1 decode also leaves out a power of two folds both back once.

// e4m3 by bit placement into a half (MLX's `operator float16_t` before its x256): 0x7f / 0xff set the half's top
// exponent bit, a NaN.
inline float mlxserve_fp4_e4m3_x2m8(uint b) {
  const uint v = b & 0x7Fu;
  return float(as_type<half>(ushort((v << 7) | (((v + 1u) >> 7) << 14) | ((b & 0x80u) << 8))));
}

inline float mlxserve_fp4_e4m3(uint b) {
  return mlxserve_fp4_e4m3_x2m8(b) * 256.0f;
}

// e8m0 as a float exponent: 0 is 2^-127, 0xff NaN.
inline float mlxserve_fp4_e8m0(uint b) {
  return as_type<float>(b == 0u ? 0x400000u : (b == 0xFFu ? 0x7FC00000u : (b << 23)));
}

inline float mlxserve_fp4_e8m0_x2m8(uint b) {
  return mlxserve_fp4_e8m0(b) * 0.00390625f;
}

template <int GS>
inline float mlxserve_fp4_scale_x2m8(uint b) {
  static_assert(GS == 16 || GS == 32, "fp4 groups: e4m3 per 16 (NVFP4), e8m0 per 32 (MXFP4)");
  if constexpr (GS == 16) {
    return mlxserve_fp4_e4m3_x2m8(b);
  } else {
    return mlxserve_fp4_e8m0_x2m8(b);
  }
}

// The scale's value.
template <int GS>
inline float mlxserve_fp4_scale_of(uint b) {
  static_assert(GS == 16 || GS == 32, "fp4 groups: e4m3 per 16 (NVFP4), e8m0 per 32 (MXFP4)");
  if constexpr (GS == 16) {
    return mlxserve_fp4_e4m3(b);
  } else {
    return mlxserve_fp4_e8m0(b);
  }
}
