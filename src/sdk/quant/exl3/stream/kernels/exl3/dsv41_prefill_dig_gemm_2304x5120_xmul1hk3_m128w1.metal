    // q3_prefill_dig_gemm_2304x5120_xmul1hk3 at BM 128 x BN 128 with q3_prefill_dig_rot_widen1_5120 as its epilogue
    // (mlx-serve native, 2026-09-30). 16 simdgroups (WM 4 x WN 4) each run the 64-row text's 32 x 32 share (the same A
    // fragments, decoded B values and tile_matmad_nax sequence), so the accumulators are the 128-row text's z words. The
    // epilogue stages them 32 rows at a time and runs widen1's arithmetic on each 128-column block: its butterflies in
    // its bit order (bit 0 first), then (w * SCALE) * rout, fp contract off. z never leaves the threadgroup.
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int WN = 4;
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr int SK = 32;
    constexpr int TM = 2;
    constexpr int TN = 2;
    constexpr int TK = 2;
    constexpr int NSG = 16;
    constexpr int LDB = BK + 8;
    constexpr int KT = BK / 16;
    constexpr int NT = BN / 16;
    constexpr int JOBS = KT * NT / (2 * NSG);
    constexpr int KD = 2304;
    constexpr int WPT = 24;
    constexpr int SWPT = 24;
    constexpr int ND = 5120;
    constexpr int K16 = KD / 16;
    constexpr int N16 = ND / 16;
    constexpr int NTILES = ND / BN;
    constexpr int S = KD / BK;
    threadgroup uint4 dig_smem[BN * LDB * 2 / 16];
    threadgroup half* Bs = reinterpret_cast<threadgroup half*>(dig_smem);

    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = tid & 31u;
    const int g = int(threadgroup_position_in_grid.x);
    // this threadgroup's expert j: the last table entry whose first threadgroup is <= g (unused entries: INT_MAX)
    int j = 0;
    STEEL_PRAGMA_UNROLL
    for (int q = 1; q < 16; ++q) {
        j = tbl[48 + q] <= g ? q : j;
    }
    const int slot = tbl[j];
    const int row0 = tbl[16 + j];
    const int rows = tbl[32 + j];
    const int local = g - tbl[48 + j];
    const int tiles_m = (rows + BM - 1) / BM;
    const int nt_all = local / tiles_m;
    const int m0 = (local - nt_all * tiles_m) * BM;
    const int op = nt_all / NTILES;
    const int n0 = (nt_all - op * NTILES) * BN;
    const int tm = SM * int(sg / WN);
    const short tn = short(SN * int(sg % WN));
    const int sgp_sm = min(SM, max(0, min(BM, rows - m0) - tm));
    (void)op;
    const device int16_t* code = code0;
    // loader (the probe's map): job jj = trellis tile t (kt, nt) of the stage, column half h = sg & 1 (uniform per
    // simdgroup), column c7 = lane & 7 -> B^T row 16 nt + 8 h + c7, K slots 16 kt .. 16 kt + 15
    const uint h = sg & 1u;
    const uint c7 = lane & 7u;
    int dst_off[JOBS];
    int src_off[JOBS];
    STEEL_PRAGMA_UNROLL
    for (int jj = 0; jj < JOBS; ++jj) {
        const int t = int(lane >> 3u) + 4 * (int(sg >> 1u) + (NSG / 2) * jj);
        const int kt = t / NT;
        const int nt = t % NT;
        dst_off[jj] = (nt * 16 + 8 * int(h) + int(c7)) * LDB + kt * 16;
        src_off[jj] = (kt * N16 + nt) * WPT;
    }
    const uint o0 = (3u * c7 + 23u) % 24u;
    const uint o1 = (3u * c7 + 24u) % 24u;
    const uint o2 = (3u * c7 + 25u) % 24u;
    const uint o3 = (3u * c7 + 26u) % 24u;
    const device uint* cd = reinterpret_cast<const device uint*>(code)
        + size_t(slot) * size_t(K16 * N16) * size_t(SWPT) + size_t(n0 / 16) * size_t(WPT);
    const device half* xa = x + (size_t(row0) + size_t(m0 + tm)) * size_t(KD);
    const bool full = sgp_sm == SM;
    NAXTile<float, TM, TN> Dtile;
    Dtile.clear();
    for (int s = 0; s < S; ++s) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (int jj = 0; jj < JOBS; ++jj) {
            const device uint* tile = cd + size_t(s) * size_t(KT * N16 * WPT) + src_off[jj];
            if (h != 0u) {
                dig_dec_job<half, 1>(tile, o0, o1, o2, o3, Bs + dst_off[jj]);
            } else {
                dig_dec_job<half, 0>(tile, o0, o1, o2, o3, Bs + dst_off[jj]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgp_sm > 0) {
            STEEL_PRAGMA_NO_UNROLL
            for (int i = 0; i < 2; ++i) {
                NAXTile<half, TM, TK> Atile;
                NAXTile<half, TN, TK> Btile;
                volatile int compiler_barrier;
                if (full) {
                    Atile.load(xa + s * BK + i * SK, KD);
                } else {
                    Atile.load_safe(xa + s * BK + i * SK, KD, short2(SK, sgp_sm));
                }
                Btile.template load<half, LDB, 1>(Bs + tn * LDB + i * SK);
                tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
                (void)compiler_barrier;
            }
        }
    }
    // The epilogue, 32 of the tile's rows per pass: that row band's WN simdgroups stage their accumulators, then every
    // simdgroup takes whole rows (lane l holds columns 4 l .. 4 l + 3 of the 128-column block).
    threadgroup float* St = reinterpret_cast<threadgroup float*>(dig_smem);
    constexpr int LDS = BN + 4;
    const int valid = min(BM, rows - m0);
    STEEL_PRAGMA_NO_UNROLL
    for (int p = 0; p * SM < valid; ++p) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tm == p * SM) {
            Dtile.template store<float, LDS, 1>(St + tn);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int rr = int(sg); rr < SM && p * SM + rr < valid; rr += NSG) {
            #pragma METAL fp contract(off)
            const float SCALE = as_type<float>(1035273459u);
            const threadgroup float* src = St + rr * LDS + 4 * int(lane);
            float w[4];
            STEEL_PRAGMA_UNROLL
            for (int q = 0; q < 4; ++q) {
                w[q] = src[q];
            }
            // bits 0 and 1 in registers, then bits 2 .. 6 across lanes: widen1's pairs, each (lower + upper, lower - upper)
            const float a0 = w[0] + w[1];
            const float a1 = w[0] - w[1];
            const float a2 = w[2] + w[3];
            const float a3 = w[2] - w[3];
            w[0] = a0 + a2;
            w[2] = a0 - a2;
            w[1] = a1 + a3;
            w[3] = a1 - a3;
            STEEL_PRAGMA_UNROLL
            for (uint hh = 1u; hh < 32u; hh <<= 1u) {
                const bool upper = (lane & hh) != 0u;
                STEEL_PRAGMA_UNROLL
                for (int q = 0; q < 4; ++q) {
                    const float o = simd_shuffle_xor(w[q], (ushort) hh);
                    w[q] = upper ? (o - w[q]) : (w[q] + o);
                }
            }
            const size_t col = size_t(n0) + size_t(4u * lane);
            const size_t dst = (size_t(row0) + size_t(m0 + p * SM + rr)) * size_t(ND) + col;
            const size_t rb = size_t(slot) * size_t(ND) + col;
            STEEL_PRAGMA_UNROLL
            for (int q = 0; q < 4; ++q) {
                out[dst + size_t(q)] = (w[q] * SCALE) * float(rout[rb + size_t(q)]);
            }
        }
    }
