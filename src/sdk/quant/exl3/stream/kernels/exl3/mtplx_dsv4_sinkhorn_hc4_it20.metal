
        using namespace metal;
        constexpr uint HC = 4;
        constexpr uint N = 16;
        constexpr uint ITERS = 20;
        constexpr float EPS = 1.000000000e-06f;

        uint gid = thread_position_in_grid.x;
        if (gid >= nmat) { return; }
        const uint off = gid * N;

        float c[N];
        for (uint i = 0; i < N; ++i) { c[i] = comb[off + i]; }

        // row-softmax over the last axis k (row i = c[i*HC + k]), then + EPS
        for (uint i = 0; i < HC; ++i) {
            float m = c[i * HC];
            for (uint k = 1; k < HC; ++k) { m = metal::max(m, c[i * HC + k]); }
            float s = 0.0f;
            for (uint k = 0; k < HC; ++k) {
                float e = metal::exp(c[i * HC + k] - m);
                c[i * HC + k] = e;
                s += e;
            }
            for (uint k = 0; k < HC; ++k) { c[i * HC + k] = c[i * HC + k] / s + EPS; }
        }

        // column normalise: sum over rows j (axis=-2), divide by (sum + EPS)
        for (uint k = 0; k < HC; ++k) {
            float cs = 0.0f;
            for (uint j = 0; j < HC; ++j) { cs += c[j * HC + k]; }
            float den = cs + EPS;
            for (uint j = 0; j < HC; ++j) { c[j * HC + k] = c[j * HC + k] / den; }
        }

        // iters-1 alternating passes: row normalise then column normalise
        for (uint it = 0; it < (ITERS - 1); ++it) {
            for (uint i = 0; i < HC; ++i) {      // row normalise (axis=-1)
                float rs = 0.0f;
                for (uint k = 0; k < HC; ++k) { rs += c[i * HC + k]; }
                float den = rs + EPS;
                for (uint k = 0; k < HC; ++k) { c[i * HC + k] = c[i * HC + k] / den; }
            }
            for (uint k = 0; k < HC; ++k) {     // column normalise (axis=-2)
                float cs = 0.0f;
                for (uint j = 0; j < HC; ++j) { cs += c[j * HC + k]; }
                float den = cs + EPS;
                for (uint j = 0; j < HC; ++j) { c[j * HC + k] = c[j * HC + k] / den; }
            }
        }

        for (uint i = 0; i < N; ++i) { out[off + i] = c[i]; }
    