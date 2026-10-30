inline uint msv_score_key(float x) {
    const uint u = as_type<uint>(x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}