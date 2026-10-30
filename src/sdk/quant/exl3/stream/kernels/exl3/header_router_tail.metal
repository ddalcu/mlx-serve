
// q3rc router tail: MLX's LogAddExp(x, y) / log1p(x) (float), statement for statement.
inline float q3rc_log1p(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == metal::numeric_limits<float>::infinity()) {
    return metal::numeric_limits<float>::infinity();
  }
  if (xp1 == 1.0f) {
    return x;
  }

  return x * (metal::log(xp1) / (xp1 - 1.0f));
}

inline float q3rc_logaddexp(float x, float y) {
  if (metal::isnan(x) || metal::isnan(y)) {
    return metal::numeric_limits<float>::quiet_NaN();
  }
  constexpr float inf = metal::numeric_limits<float>::infinity();
  float maxval = metal::max(x, y);
  float minval = metal::min(x, y);
  return (minval == -inf || maxval == inf)
      ? maxval
      : (maxval + q3rc_log1p(metal::exp(minval - maxval)));
}
