
        uint row = threadgroup_position_in_grid.y;                  // assignment (this threadgroup's row)
        uint tn = threadgroup_position_in_grid.x;                   // output tile column (16 outputs)
        uint lane = thread_index_in_simdgroup;
        threadgroup float red[16][32];
        // Rows of one slot pair up in row order (ranks 0-1, 2-3, ...): the even rank leads and computes
        // both rows (one decode per weight), a row without a partner computes itself, the odd rank exits.
        // Every value below is threadgroup-uniform; the tile loops carry no predicate.
        uint nrows = threadgroups_per_grid.y;
        uint slot = ids[row];
        uint rank = 0u; uint cnt = 0u; uint mate = row;
        for (uint j = 0; j < nrows; ++j) {
            uint same = (ids[j] == slot) ? 1u : 0u;
            rank += (j < row) ? same : 0u;
            cnt += same;
            mate = (mate == row && j > row && same == 1u) ? j : mate;
        }
        if ((rank & 1u) != 0u) return;
        const device short* base = code + ulong(slot) * 1474560ul;
        const device float* xp = xh + row * 2304u;
        const device float* xq = xh + mate * 2304u;
        uint wa = 2u * (lane >> 1u); uint wb = (wa + 30u) % 32u; uint lp = 16u * ((lane + 1u) & 1u);
        float acc[8];
        for (uint m = 0; m < 8u; ++m) acc[m] = 0.0f;
        uint dr[8]; uint dc[8];
        for (uint m = 0; m < 8u; ++m) { uint q = m & 3u; dr[m] = 2u * (lane & 3u) + (q < 2u ? 9u : 3u) - q; dc[m] = (lane >> 2) + (m < 4u ? 8u : 0u); }
        if (rank + 1u < cnt) {                                   // pair leader: row and mate
            float accb[8];
            for (uint m = 0; m < 8u; ++m) accb[m] = 0.0f;
            #pragma clang loop unroll_count(4)
            for (uint tk = 0; tk < 144u; ++tk) {
                const device short* tile = base + (tk * 320u + tn) * 32u;
                const device uint* tile32 = reinterpret_cast<const device uint*>(tile);
                uint rd8 = tile32[wa >> 1u];
                uint rd9 = tile32[wb >> 1u];
                // The lane's 8 windows sit at bit offsets 2m (m = 0..7) of v = (rd9:rd8) >> lp, lp one of (0, 16):
                // window 7 ends at bit 29, so one funnel-shifted 32-bit word holds all eight (K = 2).
                uint v0 = (lp == 0u) ? rd8 : ((rd8 >> 16u) | (rd9 << 16u));
                for (uint m = 0; m < 8u; ++m) {
                    uint window = (v0 >> (2u * m)) & 0xffffu;
                    uint rr = window * 0x83DCD12Du;
                    uint q = (rr >> 8u) & 0x00FF00FFu;
                    uint z = (rr - q * 255u) * 0x00010001u + 0x64000000u;
                    float w = float(fma(as_type<half2>(z).y, as_type<half>(ushort(0x1EEEu)), as_type<half>(ushort(0xC931u))));
                    acc[m] = fma(xp[tk * 16u + dr[m]], w, acc[m]);
                    accb[m] = fma(xq[tk * 16u + dr[m]], w, accb[m]);
                }
            }
            for (uint c = 0; c < 16u; ++c) red[c][lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint m = 0; m < 8u; ++m) red[dc[m]][lane] += acc[m];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < 16u) {
                float s = 0.0f;
                for (uint l = 0; l < 32u; ++l) s += red[lane][l];
                out[row * 5120u + tn * 16u + lane] = s;
            }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint c = 0; c < 16u; ++c) red[c][lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint m = 0; m < 8u; ++m) red[dc[m]][lane] += accb[m];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < 16u) {
                float s = 0.0f;
                for (uint l = 0; l < 32u; ++l) s += red[lane][l];
                out[mate * 5120u + tn * 16u + lane] = s;
            }
            } else {                                                  // a row without a partner
            #pragma clang loop unroll_count(4)
            for (uint tk = 0; tk < 144u; ++tk) {
                const device short* tile = base + (tk * 320u + tn) * 32u;
                const device uint* tile32 = reinterpret_cast<const device uint*>(tile);
                uint rd8 = tile32[wa >> 1u];
                uint rd9 = tile32[wb >> 1u];
                // The lane's 8 windows sit at bit offsets 2m (m = 0..7) of v = (rd9:rd8) >> lp, lp one of (0, 16):
                // window 7 ends at bit 29, so one funnel-shifted 32-bit word holds all eight (K = 2).
                uint v0 = (lp == 0u) ? rd8 : ((rd8 >> 16u) | (rd9 << 16u));
                for (uint m = 0; m < 8u; ++m) {
                    uint window = (v0 >> (2u * m)) & 0xffffu;
                    uint rr = window * 0x83DCD12Du;
                    uint q = (rr >> 8u) & 0x00FF00FFu;
                    uint z = (rr - q * 255u) * 0x00010001u + 0x64000000u;
                    float w = float(fma(as_type<half2>(z).y, as_type<half>(ushort(0x1EEEu)), as_type<half>(ushort(0xC931u))));
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
                out[row * 5120u + tn * 16u + lane] = s;
            }
            }
