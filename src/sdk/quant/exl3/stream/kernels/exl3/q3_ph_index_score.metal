
    // q3_prefill_attnhalf idxscore: einsum("bshd,btd->bsht", q, index_k) as MLX runs it (matmul -> regular NAX
    // nt, bm64 bn128 bk256 wm2 wn4; steel_gemm_fused_nax.h tile / simdgroup / pointer arithmetic; MLX gemm_loop),
    // then the topksum head-sum + reach mask on the accumulator tile.  One threadgroup per (128 keys, 2 queries).
    constexpr short BM = 64, BN = 128, BK = 256, WN = 4, SM = 32, SN = 32, SK = 32;
    constexpr int HH = 32;
    constexpr int KD = 128;
    constexpr int PAD = 17;
    threadgroup float shm[8 * 32 * PAD];
    const int S = q_shape[1];
    const int N = k_shape[1];
    const int M = S * HH;
    const int tile_n = int(threadgroup_position_in_grid.x);
    const int tile_m = int(threadgroup_position_in_grid.y);
    const int bi = int(threadgroup_position_in_grid.z);
    const short sg = short(simdgroup_index_in_threadgroup);
    const short lane = short(thread_index_in_simdgroup);
    const int c_row = tile_m * BM;
    const int c_col = tile_n * BN;
    const short tm = SM * (sg / WN);
    const short tn = SN * (sg % WN);
    const int sgp_sm = min(int(SM), M - (c_row + tm));
    const int sgp_sn = min(int(SN), N - (c_col + tn));
    if (sgp_sm <= 0 || sgp_sn <= 0) {
        return;                                   // no output (M is a multiple of 32: a simdgroup is all in or out)
    }
    const device float* A = q + int64_t(bi) * M * KD + int64_t(c_row + tm) * KD;
    const device float* B = k + int64_t(bi) * N * KD + int64_t(c_col + tn) * KD;
    NAXTile<float, SM / 16, SN / 16> Dt;
    if (sgp_sn == SN) {
        Dt = gemm_loop<float, SM, SN, SK, BK, false, true, true, true, false, float>(
            A, B, KD, KD, KD, 0, short(sgp_sm), short(sgp_sn));
    } else {
        Dt = gemm_loop<float, SM, SN, SK, BK, false, true, true, false, false, float>(
            A, B, KD, KD, KD, 0, short(sgp_sm), short(sgp_sn));
    }
    // epilogue: this simdgroup's 32 x 32 tile = one query (heads 0..31) x 32 keys
    const int qi = (c_row + tm) / HH;
    const device float* wrow = w + (int64_t(bi) * S + qi) * HH;
    const int lim = int(clen[qi]);
    threadgroup float* sh = shm + int(sg) * (32 * PAD);
    const short2 sc = BaseNAXFrag::get_coord();
    device float* orow = out + (int64_t(bi) * S + qi) * int64_t(N) + c_col + tn;
    STEEL_PRAGMA_NO_UNROLL
    for (short fc = 0; fc < 2; fc++) {
        STEEL_PRAGMA_UNROLL
        for (short fr = 0; fr < 2; fr++) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                const short head = fr * 16 + sc.y + i * 8;
                const float wv = wrow[head];
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) {
                    // col_reduce_looped: tot = relu_w(v) + tot, tot = 0 (the stock "+ 0" canonicalises -0)
                    sh[head * PAD + sc.x + j] = q3pf_relu_w(Dt.frag_at(fr, fc)[i * 4 + j], wv) + 0.0f;
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float mine = 0.0f;
        for (short c = 0; c < 16; c++) {
            const float r = simd_sum(sh[lane * PAD + c]);   // lane = head, as the stock simd_sum(shv[lane * BN + .])
            if (lane == c) {
                mine = r;
            }
        }
        const int col = fc * 16 + lane;
        if (lane < 16 && col < sgp_sn) {
            orow[col] = (c_col + tn + col < lim) ? mine : -metal::numeric_limits<float>::infinity();
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
