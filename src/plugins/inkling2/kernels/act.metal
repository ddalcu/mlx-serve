// Inkling 2, one row: the routed experts' soft-capped SwiGLU, one thread per element (or 4 elements with VEC4).
// GU [KSEL, 1, 2 * I] (TI) holds each picked expert's gate (even) and up (odd) outputs; with SCALED each is times
// SC[IDX[k]] first (NVFP4's per-expert scale, f32 [E]). PR = { 1 / gate cap, gate cap, 1 / up cap, up cap }.
// ACT [KSEL, 1, I] (T) = cap_g tanh(g / cap_g) * cap_u tanh(u / cap_u) * sigmoid(g).
const uint e = thread_position_in_grid.x;
const uint I = GU_shape[2] / 2;
if constexpr (VEC4) {
    const uint I4 = I / 4;
    const uint k = e / I4;
    const uint i4 = (e % I4) * 4;
    const float sc = SCALED ? SC[IDX[k]] : 1.0f;
    const device vec<T, 4>* gp = (const device vec<T, 4>*)(GU + size_t(k) * 2 * I + 2 * i4);
    const vec<T, 4> a = gp[0];
    const vec<T, 4> b = gp[1];
    const float4 gs = float4(float(a[0]), float(a[2]), float(b[0]), float(b[2])) * sc;
    const float4 us = float4(float(a[1]), float(a[3]), float(b[1]), float(b[3])) * sc;
    const float p0 = PR[0], p1 = PR[1], p2 = PR[2], p3 = PR[3];
    vec<T, 4> out;
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) {
        const float capped = (precise::tanh(gs[j] * p0) * p1) * (precise::tanh(us[j] * p2) * p3);
        out[j] = T(capped / (1.0f + precise::exp(-gs[j])));
    }
    *((device vec<T, 4>*)(ACT + size_t(k) * I + i4)) = out;
} else {
    const uint k = e / I, i = e % I;
    const float sc = SCALED ? SC[IDX[k]] : 1.0f;
    const float g = float(GU[size_t(k) * 2 * I + 2 * i]) * sc;
    const float u = float(GU[size_t(k) * 2 * I + 2 * i + 1]) * sc;
    // The precise variants: the fast tanh is NaN at an infinity and a percent off near zero.
    const float capped = (precise::tanh(g * PR[0]) * PR[1]) * (precise::tanh(u * PR[2]) * PR[3]);
    ACT[e] = T(capped / (1.0f + precise::exp(-g)));
}
