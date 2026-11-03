template <int HC, int ITERS, typename M, typename B, typename PO, typename CO>
inline void msv_hc_gates(M mix, B base, float pre_scale, float post_scale, float comb_scale,
                         float EPS, uint lane, threadgroup float* pre_shared,
                         PO post_out, CO comb_out) {
    constexpr int BASE_OFF = 2 * HC;
    const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
    const uint  llane  = metal::min(lane, (uint)(HC - 1));
    float pre_z  = mix[llane]      * pre_scale  + base[llane];
    float post_z = mix[HC + llane] * post_scale + base[HC + llane];
    float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
    float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
    if (lane < (uint)HC) {
        pre_shared[lane] = pre_v;
        post_out[lane]   = post_v;
    }
    const int o = BASE_OFF + int(llane) * HC;
    float4 v = (float4(mix[o], mix[o + 1], mix[o + 2], mix[o + 3]) * comb_scale
              + float4(base[o], base[o + 1], base[o + 2], base[o + 3])) * active;
    float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
    float4 e = metal::fast::exp(v - row_max) * active;
    float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS)) + EPS * active;
    float4 col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
    r *= col_inv;
    for (int iter = 1; iter < ITERS; ++iter) {
        r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
        col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
        r *= col_inv;
    }
    if (lane < (uint)HC) {
        comb_out[lane * HC] = r.x;
        comb_out[lane * HC + 1] = r.y;
        comb_out[lane * HC + 2] = r.z;
        comb_out[lane * HC + 3] = r.w;
    }
}

// msv_hc_gates' post and comb on one thread (HC == 4): no cross-lane sums per iteration.
template <int ITERS, typename M, typename B, typename PO, typename CO>
inline void msv_hc_gates1(M mix, B base, float post_scale, float comb_scale, float EPS,
                          PO post_out, CO comb_out) {
    for (int i = 0; i < 4; ++i)
        post_out[i] = 2.0f / (1.0f + metal::fast::exp(-(mix[4 + i] * post_scale + base[4 + i])));
    float4 r[4];
    for (int i = 0; i < 4; ++i) {
        const int o = 8 + 4 * i;
        const float4 v = float4(mix[o], mix[o + 1], mix[o + 2], mix[o + 3]) * comb_scale
                       + float4(base[o], base[o + 1], base[o + 2], base[o + 3]);
        const float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
        const float4 e = metal::fast::exp(v - row_max);
        r[i] = e * (metal::fast::divide(1.0f, e.x + e.y + e.z + e.w + EPS)) + EPS;
    }
    float4 ci = metal::fast::divide(1.0f, (r[0] + r[1]) + (r[2] + r[3]) + EPS);
    for (int i = 0; i < 4; ++i) r[i] *= ci;
    for (int iter = 1; iter < ITERS; ++iter) {
        for (int i = 0; i < 4; ++i) r[i] *= metal::fast::divide(1.0f, r[i].x + r[i].y + r[i].z + r[i].w + EPS);
        ci = metal::fast::divide(1.0f, (r[0] + r[1]) + (r[2] + r[3]) + EPS);
        for (int i = 0; i < 4; ++i) r[i] *= ci;
    }
    for (int i = 0; i < 4; ++i) {
        comb_out[4 * i] = r[i].x;
        comb_out[4 * i + 1] = r[i].y;
        comb_out[4 * i + 2] = r[i].z;
        comb_out[4 * i + 3] = r[i].w;
    }
}

template <typename T>
inline float4 msv_hc_load4(device T* h, int k) {
    device atomic_uint* w = (device atomic_uint*)(h + k);
    const uint a = atomic_load_explicit(w, memory_order_relaxed);
    const uint b = atomic_load_explicit(w + 1, memory_order_relaxed);
    return float4(float2(as_type<vec<T, 2>>(a)), float2(as_type<vec<T, 2>>(b)));
}

// Stream element k of `expand(y, h)`, rounded where `expand` rounds it.
template <typename T, int HC, int D, typename Y, typename P, typename C, typename H>
inline float msv_hc_hval(int k, Y y, P post, C comb, H h) {
    const int i = k / D;
    const int d = k - i * D;
    float acc = 0.0f;
    for (int j = 0; j < HC; ++j) acc = fma(comb[j * HC + i], float(h[j * D + d]), acc);
    return float(T(fma(post[i], float(y[d]), acc)));
}