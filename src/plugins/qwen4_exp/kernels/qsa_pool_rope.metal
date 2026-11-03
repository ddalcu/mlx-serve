uint lane = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.y;
uint blk = row % uint(nblk);
uint kbase = row * uint(R * 128) + lane * 4;
float xs[4];
float sumsq = 0.0f;
for (uint j = 0; j < 4; ++j) {
    float acc = 0.0f;
    for (uint r = 0; r < uint(R); ++r) acc = float(kb[kbase + r * 128 + j]) + acc;
    const T pooled = T(acc * inv_r);
    xs[j] = float(pooled);
    sumsq += xs[j] * xs[j];
}
sumsq = simd_sum(sumsq);
float inv = metal::precise::rsqrt(sumsq / 128.0f + eps);
for (uint j = 0; j < 4; ++j) {
    const T nv = norm_w[lane * 4 + j] * T(xs[j] * inv);
    const float nf = float(nv);
    const float up = simd_shuffle_down(nf, ushort(HALF / 4));
    const float dn = simd_shuffle_up(nf, ushort(HALF / 4));
    uint d = lane * 4 + j;
    T o = nv;
    if (d < uint(RD)) {
        const T rot = d < uint(HALF) ? T(-up) : T(dn);
        const T a = T(nf * float(cosv[blk * RD + d]));
        const T b = T(float(rot) * float(sinv[blk * RD + d]));
        o = T(float(a) + float(b));
    }
    out[row * 128 + d] = o;
}