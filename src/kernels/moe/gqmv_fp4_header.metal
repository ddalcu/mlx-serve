// A gather_qmv fp4 word's term: its partial sums carry the e2m1 decode's 2^-14, restored here (x 2^14, exact), then
// times the group scale as MLX reads it (`mlxserve_fp4_scale_of<GS>`), so the product rounds where MLX's
// `scale * accum` does, for every e8m0 and e4m3 code.
// FOLD (`MLX_SERVE_FP4_FOLD22`, NVFP4 only): the pre-26.11.4 form, the e4m3 read times 2^-8 (exact) and one 2^22 on
// the finished sum; the same bytes wherever no partial sum leaves float32's normal range.
template <int FOLD>
inline float mlxserve_gqmv_term(float d, float s) {
  if constexpr (FOLD != 0) {
    return d * (s * 0.00390625f);
  } else {
    return (d * 16384.0f) * s;
  }
}
