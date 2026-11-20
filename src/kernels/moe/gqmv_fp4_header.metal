// A gather_qmv fp4 word's term: its partial sums carry the e2m1 decode's 2^-14, restored here (x 2^14, exact), then
// times the group scale as MLX reads it (`mlxserve_fp4_scale_of<GS>`), so the product rounds where MLX's
// `scale * accum` does, for every e8m0 and e4m3 code.
inline float mlxserve_gqmv_term(float d, float s) {
  return (d * 16384.0f) * s;
}
