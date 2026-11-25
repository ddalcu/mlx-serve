#pragma clang fp contract(off)
// Inkling 2 vision: fused optional projection bias + exact erf-GELU (0.5 * x * (1 + erf(x / sqrt(2)))).
// Y [L, M] (T), B [M] or [1] (T) -> O [L, M] (T).
// grid (ceil(M / 4), L, 1), threadgroup (min(ceil(M / 4), 256), 1, 1).
const uint M = Y_shape[1];
const uint L = Y_shape[0];
const uint row = thread_position_in_grid.y;
const uint c0 = thread_position_in_grid.x * 4u;
if (row >= L || c0 >= M) return;

const size_t base = size_t(row) * M;
const uint n = min(4u, M - c0);

// Cleanroom degree-11 Chebyshev approximation of erfcx(u) / t on t = 1 / (1 + 0.5 * u), u = |x| / sqrt(2).
auto gelu_erf = [](float x) -> float {
    if (isnan(x)) return x;
    if (isinf(x)) return x > 0.0f ? x : NAN;
    const float u = fabs(x) * 0.70710678118654752f;
    const float t = 1.0f / (1.0f + 0.5f * u);
    float p = -4.049579000e-3f;
    p = fma(p, t, 6.537067000e-2f);
    p = fma(p, t, -2.871196300e-1f);
    p = fma(p, t, 5.635457600e-1f);
    p = fma(p, t, -5.127186000e-1f);
    p = fma(p, t, 1.627825500e-1f);
    p = fma(p, t, -6.960271000e-2f);
    p = fma(p, t, 9.565474000e-2f);
    p = fma(p, t, 1.750473200e-1f);
    p = fma(p, t, 2.469013600e-1f);
    p = fma(p, t, 2.820933500e-1f);
    p = fma(p, t, 2.820948000e-1f);
    const float x2_hi = x * x;
    const float x2_lo = fma(x, x, -x2_hi);
    const float exp_val = metal::precise::exp(-0.5f * x2_hi) * fma(-0.5f, x2_lo, 1.0f);
    const float erfc_val = exp_val * t * p;
    const float half_x = 0.5f * x;
    return x >= 0.0f ? (x - half_x * erfc_val) : (half_x * erfc_val);
};

if (n == 4u && (M & 3u) == 0u) {
    const vec<T, 4> y = *(const device vec<T, 4>*)(Y + base + c0);
    const vec<T, 4> b = HAS_BIAS ? vec<T, 4>(B[c0], B[c0 + 1u], B[c0 + 2u], B[c0 + 3u]) : vec<T, 4>(0);
    vec<T, 4> out;
#pragma clang loop unroll(full)
    for (uint e = 0; e < 4u; ++e) {
        const float x = HAS_BIAS ? float(T(float(y[e]) + float(b[e]))) : float(y[e]);
        out[e] = T(gelu_erf(x));
    }
    *(device vec<T, 4>*)(O + base + c0) = out;
    return;
}

for (uint e = 0; e < n; ++e) {
    const uint c = c0 + e;
    const float x = HAS_BIAS ? float(T(float(Y[base + c]) + float(B[c]))) : float(Y[base + c]);
    O[base + c] = T(gelu_erf(x));
}
