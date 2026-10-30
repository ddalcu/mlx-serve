
inline float unpack_e4m3(uint code) {
    ushort half_word = ushort((code & 127u) * 128u);
    half magnitude = as_type<half>(half_word) * half(256);
    return float((code & 128u) ? -magnitude : magnitude);
}
