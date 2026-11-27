static inline float4 msv_e4m3_decode4(uint w) {
    // MLX's e4m3 read (the shared fp4_scale header): 0x7f / 0xff are NaN.
    half2 h02 = mlxserve_e4m3x2_h(w) * half2((half)256.0f);
    half2 h13 = mlxserve_e4m3x2_h(w >> 8) * half2((half)256.0f);
    return float4(float(h02.x), float(h13.x), float(h02.y), float(h13.y));
}