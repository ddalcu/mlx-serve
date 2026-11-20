// The gather_qmv fp4 bodies' group scale (`fp4_scale.metal` decodes it) and where its power of two goes back, by
// the group size GS. A word's partial sums d carry the e2m1 decode's 2^-14.
// GS 16 (NVFP4): the e4m3 scale times 2^-8 (2^-17 .. 1.75, normal floats, exact for every code), and one 2^22 on
// the finished sum restores both. The folded sums are the true sums times 2^-22 bit for bit while every true partial
// is 0 or at least 2^-104 in magnitude; smaller ones lose bits to float subnormals.
// GS 32 (MXFP4): an e8m0 scale spans 2^-127 .. 2^127, past any fixed fold (2^127 x 2^-8 x 6 x 2^-14 stays finite,
// but 2^-127 x 2^-22 is under float's least subnormal). Each word's partial is restored (x 2^14, exact) and
// multiplied by the scale's exact float value, so the scale's product rounds where the unscaled sum's does.
template <int GS>
inline float mlxserve_gqmv_scale(uint b) {
  if constexpr (GS == 16) {
    return mlxserve_fp4_scale_x2m8<GS>(b);
  } else {
    return mlxserve_fp4_scale_of<GS>(b);
  }
}

template <int GS>
inline float mlxserve_gqmv_term(float d, float s) {
  if constexpr (GS == 16) {
    return d * s;
  } else {
    return (d * 16384.0f) * s;
  }
}

template <int GS>
constexpr float mlxserve_gqmv_fold() {
  return GS == 16 ? 4194304.0f : 1.0f;
}
