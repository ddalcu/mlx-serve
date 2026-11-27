// Ported from MLX (Copyright (c) 2023-2025 Apple Inc., MIT); its e4m3 encoder from PyTorch (BSD-3-Clause). See NOTICE.
// The fp8 / fp4 encoders of every kernel that quantizes inside itself: MLX's own, copied from
// mlx/backend/metal/kernels/fp8.h (the fp8_e4m3 and fp8_e8m0 constructors, mx_scale_round_up) and fp4.h (the
// fp4_e2m1 constructor and operator float). The e4m3 / e8m0 reads are fp4_scale.metal's: put it before this header.
// The block max and the clip that pick a scale keep a NaN as MLX's ops do (binary_ops.h Maximum / Minimum, clip =
// minimum(maximum(a, lo), hi); reduction/ops.h Max for a simdgroup): Metal's max / min / clamp / simd_max return the
// operand that is not NaN.

// fp8.h fp8_e4m3(float), from PyTorch's Float8_e4m3fn: round to nearest even, |x| >= 464 and +-Inf saturate to
// 0x7E (448), NaN is 0x7F, the sign kept.
inline uint mlxserve_fp8_e4m3_bits(float f) {
  const uint fp8_max = 543u << 21;
  const uint denorm_mask = 141u << 23;
  uint f_bits = as_type<uint>(f);
  const uint sign = f_bits & 0x80000000u;
  f_bits ^= sign;
  uchar bits;
  if (f_bits >= fp8_max) {
    bits = (f_bits > 0x7F800000u) ? 0x7F : 0x7E;
  } else {
    if (f_bits < (121u << 23)) {
      f_bits = as_type<uint>(as_type<float>(f_bits) + as_type<float>(denorm_mask));
      bits = uchar(f_bits - denorm_mask);
    } else {
      // resulting mantissa is odd
      const uchar mant_odd = (f_bits >> 20) & 1;
      f_bits += (uint(7 - 127) << 23) + 0x7FFFFu;
      f_bits += mant_odd;
      bits = uchar(f_bits >> 20);
    }
  }
  bits |= uchar(sign >> 24);
  return bits;
}

// `f` through e4m3 and back: fp8_e4m3(f) then its operator float.
inline float mlxserve_fp8_e4m3_round(float f) {
  return mlxserve_fp4_e4m3(mlxserve_fp8_e4m3_bits(f));
}

// fp8.h fp8_e8m0(float): non-finite 0xFF, negative 0x00, else round(log2(x)) clamped to [-127, 127], biased by 127.
inline uint mlxserve_fp8_e8m0_bits(float x) {
  if (!metal::isfinite(x)) {
    return 0xFF;
  }
  if (x < 0.0f) {
    return 0x00;
  }
  const float le = metal::log2(x);
  int n = int(metal::round(le));
  n = n < -127 ? -127 : n;
  n = n > 127 ? 127 : n;
  return uint(n + 127);
}

// fp8.h mx_scale_round_up: the smallest e8m0 at or above x (a NaN or infinite x reads back NaN), as a float. A
// positive normal x takes the bit form, its exponent one up when a mantissa bit is set, at most 2^127: MLX's round of
// log2 and its bump give that on every such float. Zero, subnormals, negatives and non-finite x run MLX's steps.
inline float mlxserve_mx_scale_round_up(float x) {
  const uint b = as_type<uint>(x);
  if (b - 0x00800000u < 0x7F000000u) {
    const uint e = (b >> 23) + ((b & 0x7FFFFFu) != 0u ? 1u : 0u);
    return as_type<float>(metal::min(e, 254u) << 23);
  }
  uint s = mlxserve_fp8_e8m0_bits(x);
  if (s < 0xFEu && mlxserve_fp4_e8m0(s) < x) {
    s += 1u;
  }
  return mlxserve_fp4_e8m0(s);
}

// fp4.h fp4_e2m1(float): NaN is 0x7 (6), else the sign bit and the nearest code, ties as the thresholds give them,
// past 5 saturating to 6.
inline uint mlxserve_fp4_e2m1_bits(float x) {
  if (metal::isnan(x)) {
    return 0x7;
  }
  const uchar sign_bit = (metal::signbit(x)) ? 0x8 : 0x0;
  x = metal::abs(x);
  uchar bits;
  if (x > 5.0f) {
    bits = 0x7;
  } else if (x >= 3.5f) {
    bits = 0x6;
  } else if (x > 2.5f) {
    bits = 0x5;
  } else if (x >= 1.75f) {
    bits = 0x4;
  } else if (x > 1.25f) {
    bits = 0x3;
  } else if (x >= 0.75f) {
    bits = 0x2;
  } else if (x > 0.25f) {
    bits = 0x1;
  } else {
    bits = 0x0;
  }
  return bits | sign_bit;
}

// `f` through e2m1 and back (the read is fp4_scale.metal's `mlxserve_fp4_e2m1_value`).
inline float mlxserve_fp4_e2m1_round(float f) {
  return mlxserve_fp4_e2m1_value(mlxserve_fp4_e2m1_bits(f));
}

// binary_ops.h Maximum / Minimum on floats: a NaN in either operand is the result.
inline float mlxserve_maximum(float x, float y) {
  if (metal::isnan(x)) {
    return x;
  }
  return x > y ? x : y;
}

inline float mlxserve_minimum(float x, float y) {
  if (metal::isnan(x)) {
    return x;
  }
  return x < y ? x : y;
}

// MLX's clip, minimum(maximum(a, lo), hi), for finite lo <= hi: a NaN a is the result, else Metal's clamp is the same.
inline float mlxserve_clip(float a, float lo, float hi) {
  return metal::isnan(a) ? a : metal::clamp(a, lo, hi);
}

// reduction/ops.h Max's simdgroup step on floats: NaN if any lane holds one.
inline float mlxserve_simd_max(float val) {
  if (metal::simd_any(val != val)) {
    return static_cast<float>(NAN);
  }
  return metal::simd_max(val);
}
