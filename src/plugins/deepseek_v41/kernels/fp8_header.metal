inline float dsv41_fp8_roundtrip(float f) {
  uint fb = as_type<uint>(f);
  const uint sign = fb & 0x80000000u;
  fb ^= sign;
  uchar bits;
  if (fb >= (543u << 21)) bits = 0x7E;
  else if (fb < (121u << 23)) {
    const uint dm = 141u << 23;
    bits = uchar(as_type<uint>(as_type<float>(fb) + as_type<float>(dm)) - dm);
  } else {
    const uint odd = (fb >> 20) & 1u;
    fb += (uint(7 - 127) << 23) + 0x7FFFFu;
    fb += odd;
    bits = uchar(fb >> 20);
  }
  bits |= uchar(sign >> 24);
  const ushort v = bits & 127;
  const ushort u = (v << 7) | (((v + 1) >> 7) << 14) | (ushort((bits >> 7) & 1) << 15);
  return float(as_type<half>(u)) * 256.0f;
}
inline float dsv41_pow2ceil(float a) {
  const uint ab = as_type<uint>(a);
  return as_type<float>((((ab >> 23) & 0xFFu) + ((ab & 0x7FFFFFu) != 0u ? 1u : 0u)) << 23);
}
inline float dsv41_e2m1(float v) {
  const float m = metal::abs(v);
  const int k = int(m > 0.25f) + int(m > 1.25f) + int(m > 2.5f) + int(m > 5.0f) + int(m >= 0.75f) + int(m >= 1.75f) + int(m >= 3.5f);
  const float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return metal::sign(v) * lut[k];
}