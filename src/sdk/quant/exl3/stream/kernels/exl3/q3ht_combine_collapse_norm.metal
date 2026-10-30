
    // q3ht_combine_collapse_norm (ROUNDING-CLASS): q3ht_combine's fma chain per output, the OT round trip, then exactly
    // q3ht_collapse_norm's statements on the stored stream words.  One threadgroup per row m.

    constexpr int NC = D / (4 * T);                     // 4-column chunks per thread: t, t + T, ...
    constexpr int NSG = T / 32;
    static_assert(D == NC * 4 * T, "one threadgroup covers the row");
    static_assert(T % 32 == 0 && NSG <= 32, "whole simdgroups, one partial per lane");
    const int m = int(threadgroup_position_in_grid.y);
    const int t = int(thread_position_in_threadgroup.x);
    const int sg = t >> 5;
    const int lane = t & 31;
    threadgroup float red[2][32];
    const float pr0 = static_cast<float>(pre[m * 4 + 0]);
    const float pr1 = static_cast<float>(pre[m * 4 + 1]);
    const float pr2 = static_cast<float>(pre[m * 4 + 2]);
    const float pr3 = static_cast<float>(pre[m * 4 + 3]);
    float aq = 0.0f;
    float an = 0.0f;
    float4 cf[NC];

    float cb[16];
    for (int i = 0; i < 16; ++i) {
        cb[i] = static_cast<float>(comb[m * 16 + i]);
    }
    const float po0 = static_cast<float>(post[m * 4 + 0]);
    const float po1 = static_cast<float>(post[m * 4 + 1]);
    const float po2 = static_cast<float>(post[m * 4 + 2]);
    const float po3 = static_cast<float>(post[m * 4 + 3]);
    for (int n = 0; n < NC; ++n) {
        const int d0 = 4 * (t + n * T);
        const float4 xv = q3ht_ld4(x + int64_t(m) * D + d0);
        const float4 r0 = q3ht_ld4(r + (int64_t(m) * 4 + 0) * D + d0);
        const float4 r1 = q3ht_ld4(r + (int64_t(m) * 4 + 1) * D + d0);
        const float4 r2 = q3ht_ld4(r + (int64_t(m) * 4 + 2) * D + d0);
        const float4 r3 = q3ht_ld4(r + (int64_t(m) * 4 + 3) * D + d0);
        float4 v[4];
        for (int k = 0; k < 4; ++k) {
            const float pk = k == 0 ? po0 : (k == 1 ? po1 : (k == 2 ? po2 : po3));
            float4 mk = metal::fma(float4(cb[k]), r0, float4(-0.0f));
            mk = metal::fma(float4(cb[4 + k]), r1, mk);
            mk = metal::fma(float4(cb[8 + k]), r2, mk);
            mk = metal::fma(float4(cb[12 + k]), r3, mk);
            const float4 hk = metal::fma(float4(pk), xv, mk);
            q3ht_st4<OT>(h + (int64_t(m) * 4 + k) * D + d0, hk);
            v[k] = q3ht_round4<OT>(hk);
            q3ht_st4<float>(hf + (int64_t(m) * 4 + k) * D + d0, v[k]);
            aq = metal::fma(v[k].x, v[k].x, aq);
            aq = metal::fma(v[k].y, v[k].y, aq);
            aq = metal::fma(v[k].z, v[k].z, aq);
            aq = metal::fma(v[k].w, v[k].w, aq);
        }

        const float4 p0 = float4(pr0) * v[0];
        const float4 p1 = float4(pr1) * v[1];
        const float4 p2 = float4(pr2) * v[2];
        const float4 p3 = float4(pr3) * v[3];
        float4 c = p0 + float4(0.0f);
        c = c + p1;
        c = c + p2;
        c = c + p3;
        cf[n] = q3ht_round4<OT>(c);
        an = metal::fma(cf[n].x, cf[n].x, an);
        an = metal::fma(cf[n].y, cf[n].y, an);
        an = metal::fma(cf[n].z, cf[n].z, an);
        an = metal::fma(cf[n].w, cf[n].w, an);

    }

    aq = q3ht_bfly(aq);
    an = q3ht_bfly(an);
    if (lane == 0) {
        red[0][sg] = aq;
        red[1][sg] = an;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float tq = q3ht_bfly(lane < NSG ? red[0][lane] : 0.0f);
    const float ts = q3ht_bfly(lane < NSG ? red[1][lane] : 0.0f);
    if (t == 0) {
        ssq[m] = tq;
    }
    const float var = ts * 1.953125029e-04f;
    const float rs = metal::precise::rsqrt(var + 9.999999683e-21f);
    for (int n = 0; n < NC; ++n) {
        const int d0 = 4 * (t + n * T);
        const float4 wv = q3ht_ld4(w + d0);
        const float4 xr = cf[n] * float4(rs);
        const float4 o = wv * xr;
        q3ht_st4<OT>(y + int64_t(m) * D + d0, o);
    }
