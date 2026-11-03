
        uint nr = threadgroups_per_grid.y / 2u;
        uint half_up = threadgroup_position_in_grid.y / nr;          // 0 gate, 1 up (threadgroup-uniform)
        uint row = threadgroup_position_in_grid.y - half_up * nr;
        uint tn = threadgroup_position_in_grid.x;                   // output tile column (16 outputs)
        uint lane = thread_index_in_simdgroup;
        threadgroup float red[16][32];
        const device short* base = (half_up != 0u ? code_u : code_g) + ulong(ids[row]) * 1474560ul;
        const device float* xp = (half_up != 0u ? xh_u : xh_g) + row * 5120u;
        uint wa = 2u * (lane >> 1u); uint wb = (wa + 30u) % 32u; uint lp = 16u * ((lane + 1u) & 1u);
        float acc[8];
        for (uint m = 0; m < 8u; ++m) acc[m] = 0.0f;
        uint dr[8]; uint dc[8];
        for (uint m = 0; m < 8u; ++m) { uint q = m & 3u; dr[m] = 2u * (lane & 3u) + (q < 2u ? 9u : 3u) - q; dc[m] = (lane >> 2) + (m < 4u ? 8u : 0u); }
        #pragma clang loop unroll_count(8)
        for (uint tk = 0; tk < 320u; ++tk) {
            const device short* tile = base + (tk * 144u + tn) * 32u;
            const device uint* tile32 = reinterpret_cast<const device uint*>(tile);
            uint rd8 = tile32[wa >> 1u];
            uint rd9 = tile32[wb >> 1u];
            // The lane's 8 windows sit at bit offsets 2m (m = 0..7) of v = (rd9:rd8) >> lp, lp one of (0, 16):
            // window 7 ends at bit 29, so one funnel-shifted 32-bit word holds all eight (K = 2).
            uint v0 = (lp == 0u) ? rd8 : ((rd8 >> 16u) | (rd9 << 16u));
            for (uint m = 0; m < 8u; ++m) {
                uint window = (v0 >> (2u * m)) & 0xffffu;
                uint rr = window * 0x83DCD12Du;
                uint bs = (rr & 0x00FF00FFu) + ((rr >> 8u) & 0x00FF00FFu);
                half hs = as_type<half>(ushort((bs * 0x00010001u + 0x64000000u) >> 16u));
                float w = float(fma(hs, as_type<half>(ushort(0x1EEEu)), as_type<half>(ushort(0xC931u))));
                acc[m] = fma(xp[tk * 16u + dr[m]], w, acc[m]);
            }
        }
        for (uint c = 0; c < 16u; ++c) red[c][lane] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint m = 0; m < 8u; ++m) red[dc[m]][lane] += acc[m];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 16u) {
            float s = 0.0f;
            for (uint l = 0; l < 32u; ++l) s += red[lane][l];
            (half_up != 0u ? out_u : out_g)[row * 2304u + tn * 16u + lane] = s;
        }
    