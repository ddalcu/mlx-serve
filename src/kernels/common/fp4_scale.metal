// The fp4 group scale, one decode for every fp4 kernel: MXFP4's e8m0 (a group of 32) and NVFP4's e4m3 (a group of
// 16), picked by the group size GS. MLX's own reads, copied from mlx/backend/metal/kernels/fp8.h (fp8_e8m0 and
// fp8_e4m3 `operator float`, `dequantize_scale` in fp_quantized.h): e8m0 0x00 is 2^-127, 0xff NaN, else a float
// exponent; e4m3 by bit placement into a half times 256, 0x7f / 0xff NaN, subnormals through the half. A product
// that MLX's kernels flush (a subnormal float), ours flush the same way.
// `mlxserve_fp4_e4m3_x2m8` is the e4m3 value times 2^-8 (always a normal half/float). There is no e8m0 one: folding
// 2^-8 into e8m0 codes under 9 makes a subnormal the GPU flushes where MLX's read does not.

inline float mlxserve_fp4_e4m3_x2m8(uint b) {
  const uint v = b & 0x7Fu;
  return float(as_type<half>(ushort((v << 7) | (((v + 1u) >> 7) << 14) | ((b & 0x80u) << 8))));
}

inline float mlxserve_fp4_e4m3(uint b) {
  return mlxserve_fp4_e4m3_x2m8(b) * 256.0f;
}

inline float mlxserve_fp4_e8m0(uint b) {
  return as_type<float>(b == 0u ? 0x400000u : (b == 0xFFu ? 0x7FC00000u : (b << 23)));
}

template <int GS>
inline float mlxserve_fp4_scale_of(uint b) {
  static_assert(GS == 16 || GS == 32, "fp4 groups: e4m3 per 16 (NVFP4), e8m0 per 32 (MXFP4)");
  if constexpr (GS == 16) {
    return mlxserve_fp4_e4m3(b);
  } else {
    return mlxserve_fp4_e8m0(b);
  }
}
