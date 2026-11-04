
    constexpr uint C = 4;
    constexpr uint DC = (uint(D) + C - 1) / C;
    const uint g = thread_position_in_grid.x;
    const uint row = g / DC;
    const uint col = (g - row * DC) * C;
    const ulong lbase = ulong(row) * ulong(K);
    const ulong obase = ulong(row) * ulong(D) + ulong(col);
    float t[C];
    {
        const float w = static_cast<float>(weights[ulong(row) * ulong(K)]);
        const device bfloat16_t* p0 = q3jl_pick<bfloat16_t>(loc[2 * lbase], s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23);
        const ulong o0 = ulong(loc[2 * lbase + 1]) * ulong(D) + ulong(col);
        for (uint i = 0; i < C; ++i) {
            if (col + i < uint(D)) {
                const float p = static_cast<float>(p0[o0 + i]) * w;     // Multiply(routed, w)
                t[i] = p + 0.0f;                                                // lid.y 0: op(v_0, Sum::init)
            }
        }
    }
    for (uint k = 1; k < uint(K); ++k) {
        const float w = static_cast<float>(weights[ulong(row) * ulong(K) + k]);
        const device bfloat16_t* pk = q3jl_pick<bfloat16_t>(loc[2 * (lbase + k)], s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23);
        const ulong ok = ulong(loc[2 * (lbase + k) + 1]) * ulong(D) + ulong(col);
        for (uint i = 0; i < C; ++i) {
            if (col + i < uint(D)) {
                const float p = static_cast<float>(pk[ok + i]) * w;
                t[i] = (p + 0.0f) + t[i];                                       // op(shared_vals[k], totals)
            }
        }
    }
    for (uint i = 0; i < C; ++i) {
        if (col + i < uint(D)) {
            out[obase + i] = t[i] + static_cast<float>(shared[obase + i]);     // Add(sum, shared)
        }
    }
