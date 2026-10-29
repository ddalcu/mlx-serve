uint lane = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.y;
uint head = threadgroup_position_in_grid.z;
uint base = (row * uint(HV) + head) * uint(DV) + lane * 4;
float xs[4];
float sumsq = 0.0f;
for (uint i = 0; i < 4; ++i) {
    xs[i] = float(y[base + i]);
    sumsq += xs[i] * xs[i];
}
sumsq = simd_sum(sumsq);
float inv = metal::precise::rsqrt(sumsq / float(DV) + eps);
uint zbase = row * uint(ZSTRIDE) + uint(ZOFF) + head * uint(DV) + lane * 4;
for (uint i = 0; i < 4; ++i) {
    const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);
    const T zv = z[zbase + i];
    T sig;
    if constexpr (TAB) sig = sigtab[as_type<ushort>(zv)];
    else { T sy = T(1) / (T(1) + metal::exp(metal::abs(zv))); sig = zv < T(0) ? sy : T(1) - sy; }
    // swish gate: silu(z) * normed (qwen3.5); sigmoid gate: normed * sigmoid(z) (qwen4_exp, KDA)
    out[base + i] = SWISH ? (zv * sig) * normed : normed * sig;
}