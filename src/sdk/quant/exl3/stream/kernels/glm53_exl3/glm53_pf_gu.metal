
    // GLM-5.3 EXL3 prompt, stage 1: the gate and up GEMMs of a wave's experts over the rows' x itself, the input prepare
    // folded into the weights: t128(x rin) W = x (rin t128 W) per 128-block of k. Each BK = 128 stage: every thread
    // decodes one column's 32 k of the stage (two trellis tiles), rotates them along k (t128: 32 in the thread, 4 across
    // lanes xor 1 and 2), scales them by the mini's rin and stores them in threadgroup memory; then the NAX MMA at a
    // 128-row M tile (8 simdgroups, WM 4 x WN 2, each 32 x 32 of 128 x 64), A each row's x (the call's f16 copy of it).
    // A threadgroup is (expert j of the wave table, M tile, op = 2 mini + projection, N tile): the table gives each
    // expert's slot, first row, rows and first threadgroup. rt: per wave row (x row, slot row). z: [rows, 4, MINI] f32
    // (times the rotation's scale).
    constexpr int BM = 128;
    constexpr int BN = 64;
    constexpr int BK = 128;
    constexpr int WN = 2;
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr int SK = 32;
    constexpr int TM = 2;
    constexpr int TN = 2;
    constexpr int TK = 2;
    constexpr int LDB = BK + 8;
    constexpr int NWT = 8 * K;
    constexpr int K16 = HID / 16;
    constexpr int N16 = MINI / 16;
    constexpr int NTILES = MINI / BN;
    constexpr int S = HID / BK;
    constexpr int LDZ = 4 * MINI;
    threadgroup half Bs[BN * LDB];

    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = tid & 31u;
    const int g = int(threadgroup_position_in_grid.x);
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
    const int r = op >> 1;
    const int proj = op & 1;
    const int tm = SM * int(sg / WN);
    const short tn = short(SN * int(sg % WN));
    const int sgp_sm = min(SM, max(0, min(BM, rows - m0) - tm));
    // the B loader: thread = (column n = 8 sg + lane / 4 of the N tile, quarter q = lane % 4 of the stage's k)
    const uint q = lane & 3u;
    const uint n = 8u * sg + (lane >> 2u);
    const uint hh = (n >> 3u) & 1u;
    const uint c7 = n & 7u;
    uint o[5];
    if (hh != 0u) {
        g53_col<K, 1>::offsets(c7, o);
    } else {
        g53_col<K, 0>::offsets(c7, o);
    }
    const size_t mrow = size_t(slot) * 4u + size_t(r);
    const device uint* cd = reinterpret_cast<const device uint*>(proj == 0 ? cg : cu) + mrow * size_t(K16 * N16 * NWT)
        + size_t(n0 / 16 + int(n >> 4u)) * size_t(NWT) + size_t(2u * q) * size_t(N16 * NWT);
    const device half* rin = (proj == 0 ? rg : ru) + mrow * size_t(HID) + 32u * q;
    threadgroup uint4* bdst = reinterpret_cast<threadgroup uint4*>(Bs + n * uint(LDB) + 32u * q);
    // the A rows: this lane's four of the simdgroup's 32 (frag i, element row e: tm + 16 i + fm + 8 e), each its x row
    const short2 fc = BaseNAXFrag::get_coord();
    const device half* xr[2][2];
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
        STEEL_PRAGMA_UNROLL
        for (int e = 0; e < 2; ++e) {
            const int row = m0 + tm + 16 * i + int(fc.y) + 8 * e;
            const int rr = row < rows ? row : 0;
            xr[i][e] = x + size_t(rt[2u * uint(row0 + rr)]) * size_t(HID) + size_t(fc.x);
        }
    }
    const bool full = sgp_sm == SM;
    NAXTile<float, TM, TN> Dtile;
    Dtile.clear();
    for (int s = 0; s < S; ++s) {
        float v[32];
        const device uint* t0 = cd + size_t(s) * size_t(8 * N16 * NWT);
        if (hh != 0u) {
            g53_col<K, 1>::decode_v(t0, o, v);
            g53_col<K, 1>::decode_v(t0 + N16 * NWT, o, v + 16);
        } else {
            g53_col<K, 0>::decode_v(t0, o, v);
            g53_col<K, 0>::decode_v(t0 + N16 * NWT, o, v + 16);
        }
        g53_t128_quarters(v, lane);
        const device half4* rv = reinterpret_cast<const device half4*>(rin + s * BK);
        uint pk[16];
        STEEL_PRAGMA_UNROLL
        for (int c = 0; c < 8; ++c) {
            const float4 w = float4(rv[c]);
            pk[2 * c] = uint(as_type<ushort>(half(v[4 * c] * w.x))) | (uint(as_type<ushort>(half(v[4 * c + 1] * w.y))) << 16u);
            pk[2 * c + 1] = uint(as_type<ushort>(half(v[4 * c + 2] * w.z))) | (uint(as_type<ushort>(half(v[4 * c + 3] * w.w))) << 16u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (int c = 0; c < 4; ++c) bdst[c] = uint4(pk[4 * c], pk[4 * c + 1], pk[4 * c + 2], pk[4 * c + 3]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgp_sm > 0) {
            STEEL_PRAGMA_NO_UNROLL
            for (int i = 0; i < 4; ++i) {
                NAXTile<half, TM, TK> Atile;
                const int k0 = s * BK + i * SK;
                STEEL_PRAGMA_UNROLL
                for (int fi = 0; fi < 2; ++fi) {
                    STEEL_PRAGMA_UNROLL
                    for (int kk = 0; kk < 2; ++kk) {
                        thread auto& fr = Atile.frag_at(fi, kk);
                        STEEL_PRAGMA_UNROLL
                        for (int e = 0; e < 2; ++e) {
                            const device half* p = xr[fi][e] + k0 + 16 * kk;
                            STEEL_PRAGMA_UNROLL
                            for (int jj = 0; jj < 4; ++jj) fr[4 * e + jj] = p[jj];
                        }
                    }
                }
                NAXTile<half, TN, TK> Btile;
                volatile int compiler_barrier;
                Btile.template load<half, LDB, 1>(Bs + tn * LDB + i * SK);
                tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
                (void)compiler_barrier;
            }
        }
    }
    if (sgp_sm > 0) {
        STEEL_PRAGMA_UNROLL
        for (int f = 0; f < TM * TN; ++f) Dtile.val_frags[f] *= G53_HSC;
        device float* zp = (proj == 0 ? zg : zu) + (size_t(row0) + size_t(m0 + tm)) * size_t(LDZ) + size_t(r) * size_t(MINI) + n0 + tn;
        if (full) {
            Dtile.store(zp, LDZ);
        } else {
            Dtile.store_safe(zp, LDZ, short2(SN, sgp_sm));
        }
    }
