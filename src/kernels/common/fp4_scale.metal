// Ported from MLX (Copyright (c) 2023-2025 Apple Inc., MIT). See NOTICE.
// The fp4 group scale, one decode for every fp4 kernel: MXFP4's e8m0 (a group of 32) and NVFP4's e4m3 (a group of
// 16), picked by the group size GS. MLX's own reads, copied from mlx/backend/metal/kernels/fp8.h (fp8_e8m0 and
// fp8_e4m3 `operator float`, `dequantize_scale` in fp_quantized.h): e8m0 0x00 is 2^-127, 0xff NaN, else a float
// exponent; e4m3 by bit placement into a half times 256, 0x7f / 0xff NaN, subnormals through the half. A product
// that MLX's kernels flush (a subnormal float), ours flush the same way.
// `mlxserve_fp4_e4m3_x2m8` is the e4m3 value times 2^-8 (always a normal half/float). There is no e8m0 one: folding
// 2^-8 into e8m0 codes under 9 makes a subnormal the GPU flushes where MLX's read does not.

// A header joined after a line with no newline still starts here: the guard is past this comment.
#ifndef MLXSERVE_FP4_SCALE_METAL
#define MLXSERVE_FP4_SCALE_METAL

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

// fp4.h fp4_e2m1 operator float: an e2m1 code (low 4 bits) as its value, the magnitude placed in a half times 2^14,
// code 8 and up negative (-0 kept). The sign is a select on the finished value: placed in the half's sign bit, the NAX
// prefill gathers' per-weight dequantize ran slower.
inline float mlxserve_fp4_e2m1_value(uint c) {
  const float v = float(as_type<half>(ushort((c & 0x7u) << 9))) * 16384.0f;
  return (c & 0x8u) ? -v : v;
}

// Nibbles j and j+4 of `w` as their e2m1 values times 2^-14: one shift puts them in a half2's two halves, then the
// same bit placement serves both lanes, the sign bit with it.
inline float2 mlxserve_e2m1x2(uint w, int j) {
  const uint v = w >> (4 * j);
  return float2(as_type<half2>(((v & 0x70007u) << 9) | ((v & 0x80008u) << 12)));
}

// Two e4m3 bytes (bits 0-7 and 16-23) as MLX reads them, each times 2^-8, as a half2: the same bit placement,
// 0x7f / 0xff a NaN. Every e4m3 pair decode (mxfp8, fp8 block-128, the Sushi GEMV) reads through it.
inline half2 mlxserve_e4m3x2_h(uint w) {
  const uint v = w & 0x007F007Fu;
  return as_type<half2>((v << 7) | (((v + 0x00010001u) & 0x00800080u) << 7) | ((w & 0x00800080u) << 8));
}
#endif
