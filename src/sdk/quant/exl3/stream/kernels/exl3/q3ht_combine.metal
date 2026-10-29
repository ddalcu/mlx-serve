
    // q3ht_combine (ROUNDING-CLASS): h[m, k, d] = OT(post[m, k] * x[m, d] + sum_j comb[m, j, k] * r[m, j, d]) as ONE f32
    // fma chain per output (the prefill HCFUSE q3_hf_combine arithmetic): mk = comb[m,0,k] * r0 rounded once,
    // mk = fma(comb[m,j,k], rj, mk) for j = 1, 2, 3, h = fma(post[m,k], x, mk).  Thread = (4 consecutive d, row m): a
    // row reads only its own x / r / post / comb, M is the grid's row count only.
    const int d0 = int(thread_position_in_grid.x) * 4;
    const int m = int(thread_position_in_grid.y);
    const float4 xv = q3ht_ld4(x + int64_t(m) * D + d0);
    const float4 r0 = q3ht_ld4(r + (int64_t(m) * 4 + 0) * D + d0);
    const float4 r1 = q3ht_ld4(r + (int64_t(m) * 4 + 1) * D + d0);
    const float4 r2 = q3ht_ld4(r + (int64_t(m) * 4 + 2) * D + d0);
    const float4 r3 = q3ht_ld4(r + (int64_t(m) * 4 + 3) * D + d0);
    for (int k = 0; k < 4; ++k) {
        const float c0 = static_cast<float>(comb[m * 16 + k]);
        const float c1 = static_cast<float>(comb[m * 16 + 4 + k]);
        const float c2 = static_cast<float>(comb[m * 16 + 8 + k]);
        const float c3 = static_cast<float>(comb[m * 16 + 12 + k]);
        const float pk = static_cast<float>(post[m * 4 + k]);
        float4 mk = metal::fma(float4(c0), r0, float4(-0.0f));
        mk = metal::fma(float4(c1), r1, mk);
        mk = metal::fma(float4(c2), r2, mk);
        mk = metal::fma(float4(c3), r3, mk);
        q3ht_st4<OT>(h + (int64_t(m) * 4 + k) * D + d0, metal::fma(float4(pk), xv, mk));
    }
