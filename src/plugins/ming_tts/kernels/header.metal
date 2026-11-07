inline float4 bf4(uint2 v) {
    return float4(as_type<float>(v.x << 16), as_type<float>(v.x & 0xffff0000u),
                  as_type<float>(v.y << 16), as_type<float>(v.y & 0xffff0000u));
}
inline float rb(float x) { return float(bfloat16_t(x)); }
#define LDW(p, i) bf4(*(const device uint2*)((p) + (i)))
inline float rsum_(threadgroup const float* red, int sgn, int no, int rows, int n, int r) {
    float a = 0.0f;
    for (int g = 0; g < sgn; g++) a += red[(g * no + n) * rows + r];
    return a;
}
inline float rinv_(threadgroup const float* ssr, int sgn, int rows, int r, int kd) {
    float a = 0.0f;
    for (int g = 0; g < sgn; g++) a += ssr[g * rows + r];
    return metal::precise::rsqrt(a / float(kd) + 1e-6f);
}
