    // q3_prefill_dig_gemm_5120x2304_gu_xmul1hk3
    // at a 128-row M tile (mlx-serve native, 2026-09-30): 8 simdgroups (WM 4 x WN 2), each with the 64-row text's
    // 32 x 32 share (the same A fragments, decoded B values and tile_matmad_nax sequence per output), so one decoded
    // B stage feeds 128 rows. The decode jobs per lane follow NSG (JOBS = 8 / NSG); nothing else differs.
    constexpr int BM = 128;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int WN = 2;
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr int SK = 32;
    constexpr int TM = 2;
    constexpr int TN = 2;
    constexpr int TK = 2;
    constexpr int NSG = 8;
    constexpr int JOBS = 8 / NSG;
    constexpr int LDB = BK + 8;
    constexpr int KT = BK / 16;
    constexpr int NT = BN / 16;
    constexpr int KD = 5120;
    constexpr int WPT = 24;
    constexpr int SWPT = 24;
    constexpr int ND = 2304;
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
    const device int16_t* code = op == 0 ? code0 : code1;
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
    const device half* xa = (op == 0 ? x0 : x1) + (size_t(row0) + size_t(m0 + tm)) * size_t(KD);
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
    if (sgp_sm > 0) {
        device float* zp = (op == 0 ? z0 : z1) + (size_t(row0) + size_t(m0 + tm)) * size_t(ND) + n0 + tn;
        if (full) {
            Dtile.store(zp, ND);
        } else {
            Dtile.store_safe(zp, ND, short2(SN, sgp_sm));
        }
    }
