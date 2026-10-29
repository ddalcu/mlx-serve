
    constexpr uint C = 4;
    constexpr uint DC = (uint(D) + C - 1) / C;
    const uint g = thread_position_in_grid.x;
    const uint row = g / DC;
    const uint col = (g - row * DC) * C;
    const ulong rbase = ulong(row) * ulong(K) * ulong(D) + ulong(col);
    const ulong obase = ulong(row) * ulong(D) + ulong(col);
    float t[C];
    {
        const float w = static_cast<float>(weights[ulong(row) * ulong(K)]);
        for (uint i = 0; i < C; ++i) {
            if (col + i < uint(D)) {
                const float p = static_cast<float>(routed[rbase + i]) * w;     // Multiply(routed, w)
                t[i] = p + 0.0f;                                                // lid.y 0: op(v_0, Sum::init)
            }
        }
    }
    for (uint k = 1; k < uint(K); ++k) {
        const float w = static_cast<float>(weights[ulong(row) * ulong(K) + k]);
        for (uint i = 0; i < C; ++i) {
            if (col + i < uint(D)) {
                const float p = static_cast<float>(routed[rbase + ulong(k) * ulong(D) + i]) * w;
                t[i] = (p + 0.0f) + t[i];                                       // op(shared_vals[k], totals)
            }
        }
    }
    for (uint i = 0; i < C; ++i) {
        if (col + i < uint(D)) {
            out[obase + i] = t[i] + static_cast<float>(shared[obase + i]);     // Add(sum, shared)
        }
    }
