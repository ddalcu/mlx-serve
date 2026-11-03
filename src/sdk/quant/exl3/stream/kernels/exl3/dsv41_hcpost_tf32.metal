
    constexpr uint C = 4;
    constexpr uint DC = uint(D) / C;
    const uint g = thread_position_in_grid.x;
    const uint row = g / DC;
    const uint col = (g - row * DC) * C;
    float c[16];
    for (uint i = 0; i < 16u; ++i) c[i] = hcp_tf32(comb[ulong(row) * 16u + i]);
    float r[4][C];
    for (uint j = 0; j < 4u; ++j)
        for (uint i = 0; i < C; ++i) r[j][i] = hcp_tf32(static_cast<float>(res[(ulong(row) * 4u + j) * ulong(D) + col + i]));
    for (uint k = 0; k < 4u; ++k) {
        const float pk = post[ulong(row) * 4u + k];
        for (uint i = 0; i < C; ++i) {
            const float p0 = hcp_ftz(c[k] * r[0][i]);                  // comb[j, k] x residual[j], j = 0..3
            const float p1 = hcp_ftz(c[4u + k] * r[1][i]);
            const float p2 = hcp_ftz(c[8u + k] * r[2][i]);
            const float p3 = hcp_ftz(c[12u + k] * r[3][i]);
            const float m = (p0 + p1) + (p2 + p3);                      // the einsum's word
            const float t = pk * static_cast<float>(x[ulong(row) * ulong(D) + col + i]);
            h[(ulong(row) * 4u + k) * ulong(D) + col + i] = t + m;      // the compiled tail: post x, then + mixed
        }
    }
