
        using namespace metal;
        constexpr uint ITERS = 20;
        constexpr float EPS = 1.000000000e-06f;

        const uint gid = thread_position_in_grid.x;
        const uint lane = thread_index_in_simdgroup;
        const uint e = gid & 15u;
        const bool live = (gid >> 4) < uint(nmat);
        const ushort rb = ushort((lane & 16u) + (e & 12u));   // lane of c[i][0] (this element's row i)
        const ushort cb = ushort((lane & 16u) + (e & 3u));    // lane of c[0][k] (this element's column k)

        float c = live ? comb[gid] : 0.0f;

        // row-softmax over k, then + EPS  (stock: m = c[i][0]; m = max(m, c[i][k]) k = 1..3;
        // e = exp(c - m); s += e in k order; c = e / s + EPS)
        float m = simd_shuffle(c, rb);
        m = metal::max(m, simd_shuffle(c, ushort(rb + 1)));
        m = metal::max(m, simd_shuffle(c, ushort(rb + 2)));
        m = metal::max(m, simd_shuffle(c, ushort(rb + 3)));
        const float ex = metal::exp(c - m);
        float s = 0.0f;
        s += simd_shuffle(ex, rb);
        s += simd_shuffle(ex, ushort(rb + 1));
        s += simd_shuffle(ex, ushort(rb + 2));
        s += simd_shuffle(ex, ushort(rb + 3));
        c = ex / s + EPS;

        // column normalise (sum over rows j in order, / (sum + EPS))
        {
            float cs = 0.0f;
            cs += simd_shuffle(c, cb);
            cs += simd_shuffle(c, ushort(cb + 4));
            cs += simd_shuffle(c, ushort(cb + 8));
            cs += simd_shuffle(c, ushort(cb + 12));
            const float den = cs + EPS;
            c = c / den;
        }

        // ITERS-1 alternating passes: row normalise, then column normalise
        for (uint it = 0; it < (ITERS - 1); ++it) {
            float rs = 0.0f;
            rs += simd_shuffle(c, rb);
            rs += simd_shuffle(c, ushort(rb + 1));
            rs += simd_shuffle(c, ushort(rb + 2));
            rs += simd_shuffle(c, ushort(rb + 3));
            const float rden = rs + EPS;
            c = c / rden;
            float cs = 0.0f;
            cs += simd_shuffle(c, cb);
            cs += simd_shuffle(c, ushort(cb + 4));
            cs += simd_shuffle(c, ushort(cb + 8));
            cs += simd_shuffle(c, ushort(cb + 12));
            const float cden = cs + EPS;
            c = c / cden;
        }

        if (live) { out[gid] = c; }
    