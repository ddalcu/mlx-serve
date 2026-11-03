template <int BITS>
inline float rq_load(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  if (BITS == 8) {
    for (int i = 0; i < 8; i++) { sum += float(x[i]); xt[i] = float(x[i]); }
  } else {
    constexpr float d1 = BITS == 4 ? 16.0f : 64.0f, d2 = BITS == 4 ? 256.0f : 16.0f, d3 = BITS == 4 ? 4096.0f : 4.0f;
    for (int i = 0; i < (BITS == 4 ? 16 : 8); i += 4) {
      const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
      sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
      xt[i] = float(a); xt[i + 1] = float(b) / d1; xt[i + 2] = float(c) / d2; xt[i + 3] = float(d) / d3;
    }
  }
  return sum;
}
template <int BITS>
inline float rq_qdot(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  float accum = 0.0f;
  if (BITS == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < 4; i++)
      accum += (xt[4 * i] * (ws[i] & 0x000f) + xt[4 * i + 1] * (ws[i] & 0x00f0) +
                xt[4 * i + 2] * (ws[i] & 0x0f00) + xt[4 * i + 3] * (ws[i] & 0xf000));
  } else if (BITS == 6) {
    for (int i = 0; i < 2; i++) {
      const device uint8_t* b = w + 3 * i;
      const thread float* v = xt + 4 * i;
      accum += (b[0] & 0x3f) * v[0];
      accum += (b[0] & 0xc0) * v[1];
      accum += (b[1] & 0x0f) * (v[1] * 256.0f);
      accum += (b[1] & 0xf0) * v[2];
      accum += (b[2] & 0x03) * (v[2] * 256.0f);
      accum += (b[2] & 0xfc) * v[3];
    }
  } else {
    for (int i = 0; i < 8; i++) accum += xt[i] * w[i];
  }
  return scale * accum + sum * bias;
}
template <int K, int GS, int RPS, int BITS>
inline void rq_rowdot(const device uint8_t* w, const device bfloat* sc, const device bfloat* bi,
                      const device bfloat* x, uint lane, thread float* acc) {
  constexpr int VPL = BITS == 4 ? 16 : 8;
  constexpr int BPL = VPL * BITS / 8;
  constexpr int BLOCK = 32 * VPL;
  constexpr int KB = K * BITS / 8;
  constexpr int KG = K / GS;
  constexpr int FULL = K / BLOCK * BLOCK;
  w += lane * BPL;
  sc += lane / (GS / VPL);
  bi += lane / (GS / VPL);
  x += lane * VPL;
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < FULL; k0 += BLOCK) {
    float xt[VPL];
    const float sum = rq_load<BITS>(x, xt);
    for (int j = 0; j < RPS; j++) acc[j] += rq_qdot<BITS>(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += 32 * BPL; sc += BLOCK / GS; bi += BLOCK / GS; x += BLOCK;
  }
  if (FULL < K && int(lane) < (K - FULL) / VPL) {
    float xt[VPL];
    const float sum = rq_load<BITS>(x, xt);
    for (int j = 0; j < RPS; j++) acc[j] += rq_qdot<BITS>(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
  }
  for (int j = 0; j < RPS; j++) acc[j] = simd_sum(acc[j]);
}