
    // GLM-5.3 EXL3 prompt, stage 3: the down GEMM of a wave's experts at BM 64 x BN 128 (8 simdgroups, WM 2 x WN 4, each
    // a 32 x 32 share), each of the expert's four minis in turn: its BK = 32 stages decoded into threadgroup memory (one
    // 16-weight column per thread), double-buffered (a stage's trellis words read before its predecessor's MMA, decoded
    // after it, one barrier per stage), the NAX MMA, then the epilogue: each row's 128-column block rotated (t128: its
    // first five stages on the MMA fragments, then 32 rows at a time staged in threadgroup memory, the last two with lane
    // l holding columns l + 32 b) and scaled by the mini's rout, summed over the minis in registers. y: [rows, HID] bf16
    // in the wave's row order (rows grouped by expert).
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int WN = 4;
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr int SK = 32;
    constexpr int TM = 2;
    constexpr int TN = 2;
    constexpr int TK = 2;
    constexpr int NSG = 8;
    constexpr int LDB = BK + 8;
    constexpr int KT = BK / 16;
    constexpr int NT = BN / 16;
    constexpr int NWT = 8 * K;
    constexpr int K16 = MINI / 16;
    constexpr int N16 = HID / 16;
    constexpr int NTILES = HID / BN;
    constexpr int S = MINI / BK;
    constexpr int LDA = 4 * MINI;
    constexpr int LDS = BN + 4;
    constexpr int BUF = BN * LDB;
    // two B buffers; the epilogue's staging (32 rows of LDS floats) aliases them
    threadgroup uint4 smem[(2 * BUF * 2 > 32 * LDS * 4 ? 2 * BUF * 2 : 32 * LDS * 4) / 16];
    threadgroup half* Bs = reinterpret_cast<threadgroup half*>(smem);
    threadgroup float* St = reinterpret_cast<threadgroup float*>(smem);

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
    const int n0 = nt_all * BN;
    const int tm = SM * int(sg / WN);
    const short tn = short(SN * int(sg % WN));
    const int valid = min(BM, rows - m0);
    const int sgp_sm = min(SM, max(0, valid - tm));
    const uint h = sg & 1u;
    const uint c7 = lane & 7u;
    // 16 tiles x 16 columns = 256 jobs, one per thread: tile t = (lane >> 3) + 4 (sg >> 1)
    const int t = int(lane >> 3u) + 4 * int(sg >> 1u);
    const int kt = t / NT;
    const int nt = t - kt * NT;
    const int dst_off = (nt * 16 + 8 * int(h) + int(c7)) * LDB + kt * 16;
    const int src_off = (kt * N16 + nt) * NWT;
    uint o[5];
    if (h != 0u) {
        g53_col<K, 1>::offsets(c7, o);
    } else {
        g53_col<K, 0>::offsets(c7, o);
    }
    const bool full = sgp_sm == SM;
    float yacc[BM / SM][4][4];
    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < BM / SM; ++p) {
        STEEL_PRAGMA_UNROLL
        for (int q = 0; q < 4; ++q) {
            STEEL_PRAGMA_UNROLL
            for (int c = 0; c < 4; ++c) yacc[p][q][c] = 0.0f;
        }
    }
    STEEL_PRAGMA_NO_UNROLL
    for (int r = 0; r < 4; ++r) {
        const size_t mrow = size_t(slot) * 4u + size_t(r);
        const device uint* cd = reinterpret_cast<const device uint*>(cd0) + mrow * size_t(K16 * N16) * size_t(NWT) + size_t(n0 / 16) * size_t(NWT);
        const device half* xa = hm + (size_t(row0) + size_t(m0 + tm)) * size_t(LDA) + size_t(r) * size_t(MINI);
        NAXTile<float, TM, TN> Dtile;
        Dtile.clear();
        // a stage's B: its tile's words (read early), then decoded into buffer b
        uint P[5];
        auto fetch = [&](int s) {
            const device uint* tile = cd + size_t(s) * size_t(KT * N16 * NWT) + src_off;
            if (h != 0u) {
                g53_col<K, 1>::words(tile, o, P);
            } else {
                g53_col<K, 0>::words(tile, o, P);
            }
        };
        auto stage_b = [&](int b) {
            float v[16];
            if (h != 0u) {
                g53_col<K, 1>::decode_p(P, v);
                g53_col<K, 1>::template store<half>(v, Bs + b * BUF + dst_off);
            } else {
                g53_col<K, 0>::decode_p(P, v);
                g53_col<K, 0>::template store<half>(v, Bs + b * BUF + dst_off);
            }
        };
        fetch(0);
        // the previous mini's epilogue read the staging these buffers alias
        threadgroup_barrier(mem_flags::mem_threadgroup);
        stage_b(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = 0; s < S; ++s) {
            const bool more = s + 1 < S;
            if (more) fetch(s + 1);
            if (sgp_sm > 0) {
                NAXTile<half, TM, TK> Atile;
                NAXTile<half, TN, TK> Btile;
                volatile int compiler_barrier;
                if (full) {
                    Atile.load(xa + s * BK, LDA);
                } else {
                    Atile.load_safe(xa + s * BK, LDA, short2(SK, sgp_sm));
                }
                Btile.template load<half, LDB, 1>(Bs + (s & 1) * BUF + tn * LDB);
                tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
                (void)compiler_barrier;
            }
            if (more) stage_b((s + 1) & 1);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // the epilogue: the 128-column block's t128, its stages in order (the butterflies of
        // exl3/dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128w1.metal). Column bits 0-4 on the MMA fragments: 0-1 in
        // the thread, 2 and 3 across lanes xor 1 and xor 8, 4 across the two N fragments; then band p's WN simdgroups
        // stage their 32 rows and every simdgroup takes rows rr = sg + 8 q, lane l columns l + 32 b: bits 5-6 in the
        // thread, scaled by the mini's rout and summed over the minis in registers.
        STEEL_PRAGMA_UNROLL
        for (int f = 0; f < TM * TN; ++f) {
            STEEL_PRAGMA_UNROLL
            for (int er = 0; er < 2; ++er) {
                thread auto& d = Dtile.val_frags[f];
                const float a0 = d[4 * er] + d[4 * er + 1];
                const float a1 = d[4 * er] - d[4 * er + 1];
                const float a2 = d[4 * er + 2] + d[4 * er + 3];
                const float a3 = d[4 * er + 2] - d[4 * er + 3];
                d[4 * er] = a0 + a2;
                d[4 * er + 2] = a0 - a2;
                d[4 * er + 1] = a1 + a3;
                d[4 * er + 3] = a1 - a3;
            }
        }
        STEEL_PRAGMA_UNROLL
        for (ushort hh = 1; hh <= 8; hh <<= 3) {
            const bool upper = (lane & uint(hh)) != 0u;
            STEEL_PRAGMA_UNROLL
            for (int f = 0; f < TM * TN; ++f) {
                STEEL_PRAGMA_UNROLL
                for (int e = 0; e < 8; ++e) {
                    const float oth = simd_shuffle_xor(Dtile.val_frags[f][e], hh);
                    Dtile.val_frags[f][e] = upper ? (oth - Dtile.val_frags[f][e]) : (Dtile.val_frags[f][e] + oth);
                }
            }
        }
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < TM; ++i) {
            STEEL_PRAGMA_UNROLL
            for (int e = 0; e < 8; ++e) {
                const float lo = Dtile.frag_at(i, 0)[e];
                const float up = Dtile.frag_at(i, 1)[e];
                Dtile.frag_at(i, 0)[e] = lo + up;
                Dtile.frag_at(i, 1)[e] = lo - up;
            }
        }
        const device half* ro = rout + mrow * size_t(HID) + size_t(n0) + lane;
        float rv[4];
        STEEL_PRAGMA_UNROLL
        for (int c = 0; c < 4; ++c) rv[c] = float(ro[32 * c]);
        STEEL_PRAGMA_UNROLL
        for (int p = 0; p < BM / SM; ++p) {
            if (p * SM >= valid) break;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tm == p * SM) {
                Dtile.template store<float, LDS, 1>(St + tn);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            STEEL_PRAGMA_UNROLL
            for (int q = 0; q < 4; ++q) {
                const int rr = int(sg) + NSG * q;
                if (p * SM + rr >= valid) continue;
                const threadgroup float* src = St + rr * LDS + int(lane);
                float w[4];
                STEEL_PRAGMA_UNROLL
                for (int c = 0; c < 4; ++c) w[c] = src[32 * c];
                const float a0 = w[0] + w[1];
                const float a1 = w[0] - w[1];
                const float a2 = w[2] + w[3];
                const float a3 = w[2] - w[3];
                w[0] = a0 + a2;
                w[2] = a0 - a2;
                w[1] = a1 + a3;
                w[3] = a1 - a3;
                STEEL_PRAGMA_UNROLL
                for (int c = 0; c < 4; ++c) yacc[p][q][c] += w[c] * G53_HSC * rv[c];
            }
        }
    }
    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < BM / SM; ++p) {
        STEEL_PRAGMA_UNROLL
        for (int q = 0; q < 4; ++q) {
            const int rr = p * SM + int(sg) + NSG * q;
            if (rr >= valid) continue;
            device bfloat16_t* yp = y + (size_t(row0) + size_t(m0 + rr)) * size_t(HID) + size_t(n0) + lane;
            STEEL_PRAGMA_UNROLL
            for (int c = 0; c < 4; ++c) yp[32 * c] = bfloat16_t(yacc[p][q][c]);
        }
    }
