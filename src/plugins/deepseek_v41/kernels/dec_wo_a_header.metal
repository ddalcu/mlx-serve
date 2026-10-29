// The four e4m3 values of a weight word, each times 2^-8 (the half bit placement's exponent offset).
inline float4 dsv41_wo_a_e4m3x4(uint w) {
  const float2 lo = mlxserve_e4m3x2(w), hi = mlxserve_e4m3x2(w >> 8);
  return float4(lo.x, hi.x, lo.y, hi.y);
}
