// Two e4m3 bytes (bits 0-7 and 16-23) as MLX's fp8_e4m3 reads them, each times 2^-8 (the shared fp4_scale header).
inline float2 mlxserve_e4m3x2(uint w) {
  return float2(mlxserve_e4m3x2_h(w));
}
