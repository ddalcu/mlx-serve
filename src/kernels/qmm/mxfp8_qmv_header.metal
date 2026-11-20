// Two e4m3 bytes (bits 0-7 and 16-23) as MLX's fp8_e4m3 reads them (fp8.h), each times 2^-8: bit placement into a
// half, 0x7f / 0xff a NaN (the ((v + 1) >> 7) << 14 bit), subnormals through the half.
inline float2 mlxserve_e4m3x2(uint w) {
  const uint v = w & 0x007F007Fu;
  return float2(as_type<half2>((v << 7) | (((v + 0x00010001u) & 0x00800080u) << 7) | ((w & 0x00800080u) << 8)));
}
