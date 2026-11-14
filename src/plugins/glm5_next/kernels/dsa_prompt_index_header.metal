// The order MLX's GPU sort (sort.h LessThan) puts the negated scores in, as an unsigned key that grows with the score:
// -0.0 equals 0.0, and a NaN, which sorts after every number among the negated scores, is below every score here.
inline uint glm5_dsa_score_key(float x) {
    if (isnan(x)) return 0u;
    const uint u = as_type<uint>(x == 0.0f ? 0.0f : x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
