    constexpr int D = 128;
    const int S = params[0];
    const int P = params[1];
    const int before = params[2];
    const int pool_len = params[3];
    const int ratio = params[4];
    const int H = params[5];
    const int q_row_stride = H * D;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int sm = int(threadgroup_position_in_grid.y) * (32 * WM) + int(sg / WN) * 32;
    const int sn = int(threadgroup_position_in_grid.x) * (32 * WN) + int(sg % WN) * 32;

    const short qid = short(lane >> 2);
    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    const short fn = short(((qid & 2) | (lane & 1)) * 4);

    // Every score of the block is masked when its first pooled key is not
    // yet complete for the block's last query row.
    const int s_last = min(sm + 31, S - 1);
    const bool live = sm < S && sn < P && sn < pool_len &&
        (sn + 1) * ratio - 1 <= before + s_last;

    float acc[2][16];
    for (short a = 0; a < 2; ++a) {
        for (short e = 0; e < 16; ++e) {
            acc[a][e] = 0.0f;
        }
    }

    int qrow[2][2];
    for (short tm = 0; tm < 2; ++tm) {
        for (short i = 0; i < 2; ++i) {
            qrow[tm][i] = min(sm + tm * 16 + fm + i * 8, S - 1);
        }
    }

    if (live) {
        constexpr auto desc = matmul2d_descriptor(
            16, 32, 16, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroup> op;

        int krow[2][2];
        for (short tn = 0; tn < 2; ++tn) {
            for (short i = 0; i < 2; ++i) {
                krow[tn][i] = min(sn + tn * 16 + fm + i * 8, P - 1);
            }
        }
        const device T* kbase[2][2];
        for (short tn = 0; tn < 2; ++tn) {
            for (short i = 0; i < 2; ++i) {
                kbase[tn][i] = k + ulong(krow[tn][i]) * D + fn;
            }
        }

        for (int h = 0; h < H; ++h) {
            auto ct_a0 = op.template get_left_input_cooperative_tensor<T, T, float>();
            auto ct_a1 = op.template get_left_input_cooperative_tensor<T, T, float>();
            auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
            auto c0 = op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ct_a0)>,
                metal::remove_addrspace_t<decltype(ct_b)>, float>();
            auto c1 = op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ct_a0)>,
                metal::remove_addrspace_t<decltype(ct_b)>, float>();
            for (short e = 0; e < 16; ++e) {
                c0[e] = 0.0f;
                c1[e] = 0.0f;
            }
            const device T* a00 = q + ulong(qrow[0][0]) * q_row_stride + h * D + fn;
            const device T* a01 = q + ulong(qrow[0][1]) * q_row_stride + h * D + fn;
            const device T* a10 = q + ulong(qrow[1][0]) * q_row_stride + h * D + fn;
            const device T* a11 = q + ulong(qrow[1][1]) * q_row_stride + h * D + fn;
            for (short kk = 0; kk < D; kk += 16) {
                for (short j = 0; j < 4; ++j) {
                    ct_a0[j] = a00[kk + j];
                    ct_a0[4 + j] = a01[kk + j];
                    ct_a1[j] = a10[kk + j];
                    ct_a1[4 + j] = a11[kk + j];
                }
                for (short tn = 0; tn < 2; ++tn) {
                    for (short i = 0; i < 2; ++i) {
                        for (short j = 0; j < 4; ++j) {
                            ct_b[tn * 8 + i * 4 + j] = kbase[tn][i][kk + j];
                        }
                    }
                }
                op.run(ct_a0, ct_b, c0);
                op.run(ct_a1, ct_b, c1);
            }
            // relu * head weight in fp32, heads accumulated in order.
            for (short i = 0; i < 2; ++i) {
                const float w0 = float(w[ulong(qrow[0][i]) * H + h]);
                const float w1 = float(w[ulong(qrow[1][i]) * H + h]);
                for (short tn = 0; tn < 2; ++tn) {
                    for (short j = 0; j < 4; ++j) {
                        const short e = tn * 8 + i * 4 + j;
                        acc[0][e] += max(float(c0[e]), 0.0f) * w0;
                        acc[1][e] += max(float(c1[e]), 0.0f) * w1;
                    }
                }
            }
        }
    }

    const O masked = O(-1e30f);
    for (short tm = 0; tm < 2; ++tm) {
        for (short i = 0; i < 2; ++i) {
            const int s = sm + tm * 16 + fm + i * 8;
            if (s >= S) {
                continue;
            }
            for (short tn = 0; tn < 2; ++tn) {
                for (short j = 0; j < 4; ++j) {
                    const int p = sn + tn * 16 + fn + j;
                    if (p >= P) {
                        continue;
                    }
                    const bool valid = p < pool_len && (p + 1) * ratio - 1 <= before + s;
                    out[ulong(s) * P + p] = valid ? O(acc[tm][tn * 8 + i * 4 + j]) : masked;
                }
            }
        }
    }