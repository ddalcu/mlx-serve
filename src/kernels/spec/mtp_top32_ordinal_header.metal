inline uint msv_top32_ordinal(float v) {
    if (isnan(v))  { return 0xFFFFFFFFu; }
    if (v == 0.0f) { return 0x80000000u; }
    uint u = as_type<uint>(v);
    return (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
}