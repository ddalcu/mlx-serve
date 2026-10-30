
// q3ht (DSV41_DECODE_HCTAPE) helpers
template <typename T>
inline float4 q3ht_ld4(const device T* p) {
    return float4(static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2]),
                  static_cast<float>(p[3]));
}
template <typename T>
inline void q3ht_st4(device T* p, float4 v) {
    p[0] = static_cast<T>(v.x);
    p[1] = static_cast<T>(v.y);
    p[2] = static_cast<T>(v.z);
    p[3] = static_cast<T>(v.w);
}
template <>
inline void q3ht_st4<float>(device float* p, float4 v) {
    *reinterpret_cast<device float4*>(p) = v;
}
template <typename T>
inline float4 q3ht_round4(float4 v) {
    return float4(static_cast<float>(static_cast<T>(v.x)), static_cast<float>(static_cast<T>(v.y)),
                  static_cast<float>(static_cast<T>(v.z)), static_cast<float>(static_cast<T>(v.w)));
}
inline float q3ht_sigmoid(float x) {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}
inline float q3ht_bfly(float v) {
    v = v + simd_shuffle_xor(v, ushort(16));
    v = v + simd_shuffle_xor(v, ushort(8));
    v = v + simd_shuffle_xor(v, ushort(4));
    v = v + simd_shuffle_xor(v, ushort(2));
    v = v + simd_shuffle_xor(v, ushort(1));
    return v;
}
