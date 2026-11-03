#pragma METAL fp contract(off)
// The NAX f32 GEMM's numerics on the HC combine (MLX_ENABLE_TF32): operands truncated to TF32, products below the
// f32 normal range flushed to a signed zero, the K = 4 terms summed in pairs.
inline float hcp_tf32(float v) { return as_type<float>(as_type<uint>(v) & 0xFFFFE000u); }
inline float hcp_ftz(float v) { return metal::fabs(v) < 0x1p-126f ? metal::copysign(0.0f, v) : v; }
