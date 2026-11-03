static inline float4 msv_e4m3_decode4(uint w) {
    uint lo = ((w & 0x007F007Fu) << 7) | ((w & 0x00800080u) << 8);
    uint hs = w >> 8;
    uint hi = ((hs & 0x007F007Fu) << 7) | ((hs & 0x00800080u) << 8);
    half2 h02 = as_type<half2>(lo) * half2((half)256.0f);
    half2 h13 = as_type<half2>(hi) * half2((half)256.0f);
    return float4(float(h02.x), float(h13.x), float(h02.y), float(h13.y));
}