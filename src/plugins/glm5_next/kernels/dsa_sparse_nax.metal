    constexpr int D = 512;
#ifdef MSV_DSA_ROPE
    constexpr int DR = 64;
#endif
    constexpr int BK = 128;
    const int L = params[0];
    const int Kn = params[1];
    const int TOPK = params[2];
    const int q_off = params[3];
    const float scale_log2 = scale[0] * 1.44269504088896341f;
    const int qi = int(threadgroup_position_in_grid.x);
    const int hh = int(threadgroup_position_in_grid.y);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    const int hg = int(sg) / 4;
    const int j4 = int(sg) % 4;   // QK + softmax: key quarter; PV: dim quarter
    const int q_abs = q_off + qi;
    const int head0 = hh * 32 + hg * 16;

    const short qid = short(lane >> 2);
    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    const short fn = short(((qid & 2) | (lane & 1)) * 4);
    // Lanes with fn == 0: one per row (fm) of a fragment after the
    // xor-1 / xor-8 row reductions.
    const bool row_writer = (lane & 9u) == 0u;

    threadgroup int sel[2][BK];
    threadgroup int live[2][BK / 32];
    threadgroup float red_max[2][4][16];
    threadgroup float red_sum[2][4][16];
    // P pieces per (head group, 16-key step, lane): the 8 values of the
    // lane's fragment slots (rows fm, fm + 8 x keys fn .. fn + 3).
    threadgroup half p_hi[2][BK / 16][32 * 8];
#ifdef MSV_DSA_TWO_P
    threadgroup half p_lo[2][BK / 16][32 * 8];
#endif

    constexpr auto qk_desc = matmul2d_descriptor(
        16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroup> qk_op;
    constexpr auto pv_desc = matmul2d_descriptor(
        16, 32, 16, false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroup> pv_op;

    auto pa = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
#ifdef MSV_DSA_TWO_P
    auto pm = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
#endif
    auto pb = pv_op.template get_right_input_cooperative_tensor<half, T, float>();
    auto o0 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o1 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o2 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o3 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    for (short e = 0; e < 16; ++e) {
        o0[e] = 0.0f;
        o1[e] = 0.0f;
        o2[e] = 0.0f;
        o3[e] = 0.0f;
    }
    float m_run[2] = {-FLT_MAX, -FLT_MAX};
    float l_run[2] = {0.0f, 0.0f};

    const device T* qr0 = q + (ulong(head0 + fm) * L + qi) * D + fn;
    const device T* qr1 = q + (ulong(head0 + fm + 8) * L + qi) * D + fn;
#ifdef MSV_DSA_ROPE
    const device T* qp0 = qpe + (ulong(head0 + fm) * L + qi) * DR + fn;
    const device T* qp1 = qpe + (ulong(head0 + fm + 8) * L + qi) * DR + fn;
#endif
    // `auto`: an index array of fewer than 8 slots is bound in the constant address space.
    const auto idx_row = idx + ulong(qi) * TOPK;

    auto stage = [&](int t, int b) {
        if (tid < uint(BK)) {
            const int slot = t * BK + int(tid);
            int kp = slot < TOPK ? int(idx_row[slot]) : -1;
            if (kp < 0 || kp >= Kn || kp > q_abs) {
                kp = -1;
            }
            sel[b][tid] = kp;
            const bool any_live = simd_any(kp >= 0);
            if (lane == 0) {
                live[b][sg] = any_live ? 1 : 0;
            }
        }
    };

    const int n_tiles = (TOPK + BK - 1) / BK;
    stage(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int t = 0; t < n_tiles; ++t) {
        const int buf = t & 1;
        const int tile_keys = min(BK, TOPK - t * BK);
        // Unused slots sort last in the indexer's top-k rows: skip tiles
        // with no live key (uniform across the threadgroup).
        if ((live[buf][0] | live[buf][1] | live[buf][2] | live[buf][3]) == 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (t + 1 < n_tiles) {
                stage(t + 1, buf ^ 1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        // ---- S = Q K^T for key quarter j4 (16 heads x 32 keys).
        auto qa = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
        auto kb = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
        auto sc = qk_op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kb)>, float>();
        for (short e = 0; e < 16; ++e) {
            sc[e] = 0.0f;
        }
        if (j4 * 32 < tile_keys) {
            const device T* kr[2][2];
            for (short tn = 0; tn < 2; ++tn) {
                for (short i = 0; i < 2; ++i) {
                    const int kp = sel[buf][j4 * 32 + tn * 16 + fm + i * 8];
                    kr[tn][i] = kv + ulong(max(kp, 0)) * D + fn;
                }
            }
            _Pragma("clang loop unroll_count(4)")
            for (short kk = 0; kk < D; kk += 16) {
                for (short j = 0; j < 4; ++j) {
                    qa[j] = qr0[kk + j];
                    qa[4 + j] = qr1[kk + j];
                }
                for (short tn = 0; tn < 2; ++tn) {
                    for (short i = 0; i < 2; ++i) {
                        for (short j = 0; j < 4; ++j) {
                            kb[tn * 8 + i * 4 + j] = kr[tn][i][kk + j];
                        }
                    }
                }
                qk_op.run(qa, kb, sc);
            }
#ifdef MSV_DSA_ROPE
            // The roped part: the query's against each selected row's roped key.
            const device T* pr[2][2];
            for (short tn = 0; tn < 2; ++tn) {
                for (short i = 0; i < 2; ++i) {
                    const int kp = sel[buf][j4 * 32 + tn * 16 + fm + i * 8];
                    pr[tn][i] = pe + ulong(max(kp, 0)) * DR + fn;
                }
            }
            for (short kk = 0; kk < DR; kk += 16) {
                for (short j = 0; j < 4; ++j) {
                    qa[j] = qp0[kk + j];
                    qa[4 + j] = qp1[kk + j];
                }
                for (short tn = 0; tn < 2; ++tn) {
                    for (short i = 0; i < 2; ++i) {
                        for (short j = 0; j < 4; ++j) {
                            kb[tn * 8 + i * 4 + j] = pr[tn][i][kk + j];
                        }
                    }
                }
                qk_op.run(qa, kb, sc);
            }
#endif
        }
        bool ok[2][4];
        for (short tn = 0; tn < 2; ++tn) {
            for (short j = 0; j < 4; ++j) {
                ok[tn][j] = sel[buf][j4 * 32 + tn * 16 + fn + j] >= 0;
            }
        }
        // Partial row max of the raw scores over the quarter (scale > 0 and
        // rounding is monotonic: max(s) * c == max(s * c) bitwise).
        float pmx[2];
        for (short i = 0; i < 2; ++i) {
            float m = -FLT_MAX;
            for (short tn = 0; tn < 2; ++tn) {
                for (short j = 0; j < 4; ++j) {
                    m = ok[tn][j] ? max(m, float(sc[tn * 8 + i * 4 + j])) : m;
                }
            }
            m = max(m, simd_shuffle_xor(m, ushort(1)));
            m = max(m, simd_shuffle_xor(m, ushort(8)));
            pmx[i] = m;
        }
        if (row_writer) {
            red_max[hg][j4][fm] = pmx[0];
            red_max[hg][j4][fm + 8] = pmx[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t + 1 < n_tiles) {
            stage(t + 1, buf ^ 1);
        }

        // ---- Tile row max (the same value in the 4 simdgroups of the group).
        float factor[2];
        float mnew[2];
        for (short i = 0; i < 2; ++i) {
            const int r = fm + i * 8;
            const float m = max(max(red_max[hg][0][r], red_max[hg][1][r]),
                                max(red_max[hg][2][r], red_max[hg][3][r]));
            // no usable key in the tile: keep the running max
            const float cand = m == -FLT_MAX ? -FLT_MAX : m * scale_log2;
            mnew[i] = max(m_run[i], cand);
            factor[i] = fast::exp2(m_run[i] - mnew[i]);
            m_run[i] = mnew[i];
        }
        // ---- P of this quarter, once: exp2, fp16 hi + lo pieces, row sums.
        float rs[2] = {0.0f, 0.0f};
        for (short tn = 0; tn < 2; ++tn) {
            vec<half, 8> h;
#ifdef MSV_DSA_TWO_P
            vec<half, 8> lo;
#endif
            for (short i = 0; i < 2; ++i) {
                for (short j = 0; j < 4; ++j) {
                    const float e = ok[tn][j]
                        ? fast::exp2(float(sc[tn * 8 + i * 4 + j]) * scale_log2 - mnew[i])
                        : 0.0f;
                    const half hi = half(e);
                    h[i * 4 + j] = hi;
#ifdef MSV_DSA_TWO_P
                    lo[i * 4 + j] = half(e - float(hi));
#endif
                    rs[i] += e;
                }
            }
            const int ks = j4 * 2 + tn;
            *(threadgroup vec<half, 8>*)(&p_hi[hg][ks][lane * 8]) = h;
#ifdef MSV_DSA_TWO_P
            *(threadgroup vec<half, 8>*)(&p_lo[hg][ks][lane * 8]) = lo;
#endif
        }
        for (short i = 0; i < 2; ++i) {
            rs[i] += simd_shuffle_xor(rs[i], ushort(1));
            rs[i] += simd_shuffle_xor(rs[i], ushort(8));
        }
        if (row_writer) {
            red_sum[hg][j4][fm] = rs[0];
            red_sum[hg][j4][fm + 8] = rs[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- O = O * factor + P V for dim quarter j4 over the whole tile.
        // (factor == 1 exactly for every row of the simdgroup: O * 1 == O.)
        if (!simd_all(factor[0] == 1.0f && factor[1] == 1.0f)) {
            for (short e = 0; e < 16; ++e) {
                const float f = factor[(e >> 2) & 1];
                o0[e] *= f;
                o1[e] *= f;
                o2[e] *= f;
                o3[e] *= f;
            }
        }
        for (short i = 0; i < 2; ++i) {
            const int r = fm + i * 8;
            const float tsum = ((red_sum[hg][0][r] + red_sum[hg][1][r]) + red_sum[hg][2][r]) + red_sum[hg][3][r];
            l_run[i] = l_run[i] * factor[i] + tsum;
        }
        const int n_ks = (tile_keys + 15) / 16;
        for (short ks = 0; ks < n_ks; ++ks) {
            const vec<half, 8> h = *(const threadgroup vec<half, 8>*)(&p_hi[hg][ks][lane * 8]);
#ifdef MSV_DSA_TWO_P
            const vec<half, 8> lo = *(const threadgroup vec<half, 8>*)(&p_lo[hg][ks][lane * 8]);
#endif
            for (short e = 0; e < 8; ++e) {
                pa[e] = h[e];
#ifdef MSV_DSA_TWO_P
                pm[e] = lo[e];
#endif
            }
            const int kp0 = sel[buf][ks * 16 + fm];
            const int kp1 = sel[buf][ks * 16 + fm + 8];
            const device T* v0 = kv + ulong(max(kp0, 0)) * D + j4 * 128 + fn;
            const device T* v1 = kv + ulong(max(kp1, 0)) * D + j4 * 128 + fn;
            for (short np = 0; np < 4; ++np) {
                for (short tn = 0; tn < 2; ++tn) {
                    for (short j = 0; j < 4; ++j) {
                        pb[tn * 8 + j] = v0[np * 32 + tn * 16 + j];
                        pb[tn * 8 + 4 + j] = v1[np * 32 + tn * 16 + j];
                    }
                }
                if (np == 0) {
                    pv_op.run(pa, pb, o0);
#ifdef MSV_DSA_TWO_P
                    pv_op.run(pm, pb, o0);
#endif
                } else if (np == 1) {
                    pv_op.run(pa, pb, o1);
#ifdef MSV_DSA_TWO_P
                    pv_op.run(pm, pb, o1);
#endif
                } else if (np == 2) {
                    pv_op.run(pa, pb, o2);
#ifdef MSV_DSA_TWO_P
                    pv_op.run(pm, pb, o2);
#endif
                } else {
                    pv_op.run(pa, pb, o3);
#ifdef MSV_DSA_TWO_P
                    pv_op.run(pm, pb, o3);
#endif
                }
            }
        }
    }

    for (short i = 0; i < 2; ++i) {
        device T* orow = out + (ulong(head0 + fm + i * 8) * L + qi) * D + j4 * 128 + fn;
        const float denom = l_run[i] > 0.0f ? l_run[i] : 1.0f;
        for (short tn = 0; tn < 2; ++tn) {
            for (short j = 0; j < 4; ++j) {
                const short e = tn * 8 + i * 4 + j;
                orow[tn * 16 + j] = T(o0[e] / denom);
                orow[32 + tn * 16 + j] = T(o1[e] / denom);
                orow[64 + tn * 16 + j] = T(o2[e] / denom);
                orow[96 + tn * 16 + j] = T(o3[e] / denom);
            }
        }
    }