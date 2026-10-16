//! DeepSeek sparse attention on the tensor units (M5-class NAX): the indexer's head-summed
//! scores and the latent attention over each query's selected rows, as MLX custom kernels.
//! GLM-5-Next (`glm5_next.zig`) and plugins (through `mlx_host`) call them; the caller checks the
//! device has NAX (`transformer.verifyQmmNaxAvailable`), the kernels check the shapes.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

/// oMLX `patches/glm_moe_dsa/indexer_nax.py` (Apache-2.0, see NOTICE): the indexer's
/// head-summed `relu(q . k) * w` per (query, pool) on the tensor units, the causal pool mask
/// folded into the epilogue. Here the scores leave in f32 (`O`), as the reference ranks them.
const INDEXER_NAX_SOURCE =
    \\    constexpr int D = 128;
    \\    const int S = params[0];
    \\    const int P = params[1];
    \\    const int before = params[2];
    \\    const int pool_len = params[3];
    \\    const int ratio = params[4];
    \\    const int H = params[5];
    \\    const int q_row_stride = H * D;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint lane = thread_index_in_simdgroup;
    \\    const int sm = int(threadgroup_position_in_grid.y) * (32 * WM) + int(sg / WN) * 32;
    \\    const int sn = int(threadgroup_position_in_grid.x) * (32 * WN) + int(sg % WN) * 32;
    \\
    \\    const short qid = short(lane >> 2);
    \\    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    \\    const short fn = short(((qid & 2) | (lane & 1)) * 4);
    \\
    \\    // Every score of the block is masked when its first pooled key is not
    \\    // yet complete for the block's last query row.
    \\    const int s_last = min(sm + 31, S - 1);
    \\    const bool live = sm < S && sn < P && sn < pool_len &&
    \\        (sn + 1) * ratio - 1 <= before + s_last;
    \\
    \\    float acc[2][16];
    \\    for (short a = 0; a < 2; ++a) {
    \\        for (short e = 0; e < 16; ++e) {
    \\            acc[a][e] = 0.0f;
    \\        }
    \\    }
    \\
    \\    int qrow[2][2];
    \\    for (short tm = 0; tm < 2; ++tm) {
    \\        for (short i = 0; i < 2; ++i) {
    \\            qrow[tm][i] = min(sm + tm * 16 + fm + i * 8, S - 1);
    \\        }
    \\    }
    \\
    \\    if (live) {
    \\        constexpr auto desc = matmul2d_descriptor(
    \\            16, 32, 16, false, true, true,
    \\            matmul2d_descriptor::mode::multiply_accumulate);
    \\        matmul2d<desc, execution_simdgroup> op;
    \\
    \\        int krow[2][2];
    \\        for (short tn = 0; tn < 2; ++tn) {
    \\            for (short i = 0; i < 2; ++i) {
    \\                krow[tn][i] = min(sn + tn * 16 + fm + i * 8, P - 1);
    \\            }
    \\        }
    \\        const device T* kbase[2][2];
    \\        for (short tn = 0; tn < 2; ++tn) {
    \\            for (short i = 0; i < 2; ++i) {
    \\                kbase[tn][i] = k + ulong(krow[tn][i]) * D + fn;
    \\            }
    \\        }
    \\
    \\        for (int h = 0; h < H; ++h) {
    \\            auto ct_a0 = op.template get_left_input_cooperative_tensor<T, T, float>();
    \\            auto ct_a1 = op.template get_left_input_cooperative_tensor<T, T, float>();
    \\            auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
    \\            auto c0 = op.template get_destination_cooperative_tensor<
    \\                metal::remove_addrspace_t<decltype(ct_a0)>,
    \\                metal::remove_addrspace_t<decltype(ct_b)>, float>();
    \\            auto c1 = op.template get_destination_cooperative_tensor<
    \\                metal::remove_addrspace_t<decltype(ct_a0)>,
    \\                metal::remove_addrspace_t<decltype(ct_b)>, float>();
    \\            for (short e = 0; e < 16; ++e) {
    \\                c0[e] = 0.0f;
    \\                c1[e] = 0.0f;
    \\            }
    \\            const device T* a00 = q + ulong(qrow[0][0]) * q_row_stride + h * D + fn;
    \\            const device T* a01 = q + ulong(qrow[0][1]) * q_row_stride + h * D + fn;
    \\            const device T* a10 = q + ulong(qrow[1][0]) * q_row_stride + h * D + fn;
    \\            const device T* a11 = q + ulong(qrow[1][1]) * q_row_stride + h * D + fn;
    \\            for (short kk = 0; kk < D; kk += 16) {
    \\                for (short j = 0; j < 4; ++j) {
    \\                    ct_a0[j] = a00[kk + j];
    \\                    ct_a0[4 + j] = a01[kk + j];
    \\                    ct_a1[j] = a10[kk + j];
    \\                    ct_a1[4 + j] = a11[kk + j];
    \\                }
    \\                for (short tn = 0; tn < 2; ++tn) {
    \\                    for (short i = 0; i < 2; ++i) {
    \\                        for (short j = 0; j < 4; ++j) {
    \\                            ct_b[tn * 8 + i * 4 + j] = kbase[tn][i][kk + j];
    \\                        }
    \\                    }
    \\                }
    \\                op.run(ct_a0, ct_b, c0);
    \\                op.run(ct_a1, ct_b, c1);
    \\            }
    \\            // relu * head weight in fp32, heads accumulated in order.
    \\            for (short i = 0; i < 2; ++i) {
    \\                const float w0 = float(w[ulong(qrow[0][i]) * H + h]);
    \\                const float w1 = float(w[ulong(qrow[1][i]) * H + h]);
    \\                for (short tn = 0; tn < 2; ++tn) {
    \\                    for (short j = 0; j < 4; ++j) {
    \\                        const short e = tn * 8 + i * 4 + j;
    \\                        acc[0][e] += max(float(c0[e]), 0.0f) * w0;
    \\                        acc[1][e] += max(float(c1[e]), 0.0f) * w1;
    \\                    }
    \\                }
    \\            }
    \\        }
    \\    }
    \\
    \\    const O masked = O(-1e30f);
    \\    for (short tm = 0; tm < 2; ++tm) {
    \\        for (short i = 0; i < 2; ++i) {
    \\            const int s = sm + tm * 16 + fm + i * 8;
    \\            if (s >= S) {
    \\                continue;
    \\            }
    \\            for (short tn = 0; tn < 2; ++tn) {
    \\                for (short j = 0; j < 4; ++j) {
    \\                    const int p = sn + tn * 16 + fn + j;
    \\                    if (p >= P) {
    \\                        continue;
    \\                    }
    \\                    const bool valid = p < pool_len && (p + 1) * ratio - 1 <= before + s;
    \\                    out[ulong(s) * P + p] = valid ? O(acc[tm][tn * 8 + i * 4 + j]) : masked;
    \\                }
    \\            }
    \\        }
    \\    }
;
const INDEXER_WM: c_int = 2;
const INDEXER_WN: c_int = 2;
var indexer_nax_kernel: ?mlx.mlx_fast_metal_kernel = null;
var indexer_nax_engaged = false;

/// Masked scores f32 [1, S, P] of queries at positions `q_start..` (`q_idx` [1, S, H, 128],
/// `w` [1, S, H] scaled) against pooled keys `pk` [1, P, 128] (`pool_len` of them complete,
/// `kpool` tokens each); a pool a query cannot see scores -1e30. Null off the GPU or outside
/// the shape; the caller checks the device has NAX.
pub fn indexerScores(s: mlx.mlx_stream, q_idx: mlx.mlx_array, w: mlx.mlx_array, pk: mlx.mlx_array, q_start: c_int, kpool: c_int) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const qs = mlx.getShape(q_idx);
    const ps = mlx.getShape(pk);
    if (qs.len != 4 or qs[0] != 1 or qs[3] != 128 or ps.len != 3 or ps[0] != 1 or ps[2] != 128) return null;
    if (mlx.mlx_array_dtype(q_idx) != .bfloat16 or mlx.mlx_array_dtype(pk) != .bfloat16) return null;
    const sq = qs[1];
    const h = qs[2];
    const p = ps[1];
    if (sq < 1 or p < 1) return null;
    const kern = indexer_nax_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "q", "k", "w", "params" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("msv_glm5_indexer_nax", iv, ov, INDEXER_NAX_SOURCE, SPARSE_NAX_HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        indexer_nax_kernel = k;
        break :blk k;
    };
    const pdata = [6]i32{ sq, p, q_start, p, kpool, h };
    const params = mlx.mlx_array_new_data(&pdata, &[_]c_int{6}, 1, .int32);
    defer _ = mlx.mlx_array_free(params);
    const tg_x = @divTrunc(p + 32 * INDEXER_WN - 1, 32 * INDEXER_WN);
    const tg_y = @divTrunc(sq + 32 * INDEXER_WM - 1, 32 * INDEXER_WM);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, sq, p }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, tg_x * INDEXER_WM * INDEXER_WN * 32, tg_y, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, INDEXER_WM * INDEXER_WN * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "O", .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WM", INDEXER_WM));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WN", INDEXER_WN));
    const inputs = [_]mlx.mlx_array{ q_idx, pk, w, params };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    if (!indexer_nax_engaged) {
        indexer_nax_engaged = true;
        log.info("[dsa] NAX indexer scores engaged\n", .{});
    }
    return out;
}

/// oMLX `patches/glm_moe_dsa/sparse_mla_nax.py` (Apache-2.0, see NOTICE): the sparse
/// latent attention on the tensor units. One threadgroup per (query, 32-head half), 8
/// simdgroups; per 128-slot tile, scores read key rows through the indices (never gathered),
/// online exp2 softmax in f32, P as fp16 hi + lo pieces (MSV_DSA_TWO_P; the roped form one
/// piece) against bf16 values, f32 accumulate.
const SPARSE_NAX_HEADER =
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace mpp::tensor_ops;
;
const SPARSE_NAX_SOURCE =
    \\    constexpr int D = 512;
    \\#ifdef MSV_DSA_ROPE
    \\    constexpr int DR = 64;
    \\#endif
    \\    constexpr int BK = 128;
    \\    const int L = params[0];
    \\    const int Kn = params[1];
    \\    const int TOPK = params[2];
    \\    const int q_off = params[3];
    \\    const float scale_log2 = scale[0] * 1.44269504088896341f;
    \\    const int qi = int(threadgroup_position_in_grid.x);
    \\    const int hh = int(threadgroup_position_in_grid.y);
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint lane = thread_index_in_simdgroup;
    \\    const uint tid = sg * 32 + lane;
    \\    const int hg = int(sg) / 4;
    \\    const int j4 = int(sg) % 4;   // QK + softmax: key quarter; PV: dim quarter
    \\    const int q_abs = q_off + qi;
    \\    const int head0 = hh * 32 + hg * 16;
    \\
    \\    const short qid = short(lane >> 2);
    \\    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    \\    const short fn = short(((qid & 2) | (lane & 1)) * 4);
    \\    // Lanes with fn == 0: one per row (fm) of a fragment after the
    \\    // xor-1 / xor-8 row reductions.
    \\    const bool row_writer = (lane & 9u) == 0u;
    \\
    \\    threadgroup int sel[2][BK];
    \\    threadgroup int live[2][BK / 32];
    \\    threadgroup float red_max[2][4][16];
    \\    threadgroup float red_sum[2][4][16];
    \\    // P pieces per (head group, 16-key step, lane): the 8 values of the
    \\    // lane's fragment slots (rows fm, fm + 8 x keys fn .. fn + 3).
    \\    threadgroup half p_hi[2][BK / 16][32 * 8];
    \\#ifdef MSV_DSA_TWO_P
    \\    threadgroup half p_lo[2][BK / 16][32 * 8];
    \\#endif
    \\
    \\    constexpr auto qk_desc = matmul2d_descriptor(
    \\        16, 32, 16, false, true, true,
    \\        matmul2d_descriptor::mode::multiply_accumulate);
    \\    matmul2d<qk_desc, execution_simdgroup> qk_op;
    \\    constexpr auto pv_desc = matmul2d_descriptor(
    \\        16, 32, 16, false, false, true,
    \\        matmul2d_descriptor::mode::multiply_accumulate);
    \\    matmul2d<pv_desc, execution_simdgroup> pv_op;
    \\
    \\    auto pa = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
    \\#ifdef MSV_DSA_TWO_P
    \\    auto pm = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
    \\#endif
    \\    auto pb = pv_op.template get_right_input_cooperative_tensor<half, T, float>();
    \\    auto o0 = pv_op.template get_destination_cooperative_tensor<
    \\        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    \\    auto o1 = pv_op.template get_destination_cooperative_tensor<
    \\        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    \\    auto o2 = pv_op.template get_destination_cooperative_tensor<
    \\        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    \\    auto o3 = pv_op.template get_destination_cooperative_tensor<
    \\        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    \\    for (short e = 0; e < 16; ++e) {
    \\        o0[e] = 0.0f;
    \\        o1[e] = 0.0f;
    \\        o2[e] = 0.0f;
    \\        o3[e] = 0.0f;
    \\    }
    \\    float m_run[2] = {-FLT_MAX, -FLT_MAX};
    \\    float l_run[2] = {0.0f, 0.0f};
    \\
    \\    const device T* qr0 = q + (ulong(head0 + fm) * L + qi) * D + fn;
    \\    const device T* qr1 = q + (ulong(head0 + fm + 8) * L + qi) * D + fn;
    \\#ifdef MSV_DSA_ROPE
    \\    const device T* qp0 = qpe + (ulong(head0 + fm) * L + qi) * DR + fn;
    \\    const device T* qp1 = qpe + (ulong(head0 + fm + 8) * L + qi) * DR + fn;
    \\#endif
    \\    // `auto`: an index array of fewer than 8 slots is bound in the constant address space.
    \\    const auto idx_row = idx + ulong(qi) * TOPK;
    \\
    \\    auto stage = [&](int t, int b) {
    \\        if (tid < uint(BK)) {
    \\            const int slot = t * BK + int(tid);
    \\            int kp = slot < TOPK ? int(idx_row[slot]) : -1;
    \\            if (kp < 0 || kp >= Kn || kp > q_abs) {
    \\                kp = -1;
    \\            }
    \\            sel[b][tid] = kp;
    \\            const bool any_live = simd_any(kp >= 0);
    \\            if (lane == 0) {
    \\                live[b][sg] = any_live ? 1 : 0;
    \\            }
    \\        }
    \\    };
    \\
    \\    const int n_tiles = (TOPK + BK - 1) / BK;
    \\    stage(0, 0);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (int t = 0; t < n_tiles; ++t) {
    \\        const int buf = t & 1;
    \\        const int tile_keys = min(BK, TOPK - t * BK);
    \\        // Unused slots sort last in the indexer's top-k rows: skip tiles
    \\        // with no live key (uniform across the threadgroup).
    \\        if ((live[buf][0] | live[buf][1] | live[buf][2] | live[buf][3]) == 0) {
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            if (t + 1 < n_tiles) {
    \\                stage(t + 1, buf ^ 1);
    \\            }
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            continue;
    \\        }
    \\
    \\        // ---- S = Q K^T for key quarter j4 (16 heads x 32 keys).
    \\        auto qa = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
    \\        auto kb = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
    \\        auto sc = qk_op.template get_destination_cooperative_tensor<
    \\            metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kb)>, float>();
    \\        for (short e = 0; e < 16; ++e) {
    \\            sc[e] = 0.0f;
    \\        }
    \\        if (j4 * 32 < tile_keys) {
    \\            const device T* kr[2][2];
    \\            for (short tn = 0; tn < 2; ++tn) {
    \\                for (short i = 0; i < 2; ++i) {
    \\                    const int kp = sel[buf][j4 * 32 + tn * 16 + fm + i * 8];
    \\                    kr[tn][i] = kv + ulong(max(kp, 0)) * D + fn;
    \\                }
    \\            }
    \\            _Pragma("clang loop unroll_count(4)")
    \\            for (short kk = 0; kk < D; kk += 16) {
    \\                for (short j = 0; j < 4; ++j) {
    \\                    qa[j] = qr0[kk + j];
    \\                    qa[4 + j] = qr1[kk + j];
    \\                }
    \\                for (short tn = 0; tn < 2; ++tn) {
    \\                    for (short i = 0; i < 2; ++i) {
    \\                        for (short j = 0; j < 4; ++j) {
    \\                            kb[tn * 8 + i * 4 + j] = kr[tn][i][kk + j];
    \\                        }
    \\                    }
    \\                }
    \\                qk_op.run(qa, kb, sc);
    \\            }
    \\#ifdef MSV_DSA_ROPE
    \\            // The roped part: the query's against each selected row's roped key.
    \\            const device T* pr[2][2];
    \\            for (short tn = 0; tn < 2; ++tn) {
    \\                for (short i = 0; i < 2; ++i) {
    \\                    const int kp = sel[buf][j4 * 32 + tn * 16 + fm + i * 8];
    \\                    pr[tn][i] = pe + ulong(max(kp, 0)) * DR + fn;
    \\                }
    \\            }
    \\            for (short kk = 0; kk < DR; kk += 16) {
    \\                for (short j = 0; j < 4; ++j) {
    \\                    qa[j] = qp0[kk + j];
    \\                    qa[4 + j] = qp1[kk + j];
    \\                }
    \\                for (short tn = 0; tn < 2; ++tn) {
    \\                    for (short i = 0; i < 2; ++i) {
    \\                        for (short j = 0; j < 4; ++j) {
    \\                            kb[tn * 8 + i * 4 + j] = pr[tn][i][kk + j];
    \\                        }
    \\                    }
    \\                }
    \\                qk_op.run(qa, kb, sc);
    \\            }
    \\#endif
    \\        }
    \\        bool ok[2][4];
    \\        for (short tn = 0; tn < 2; ++tn) {
    \\            for (short j = 0; j < 4; ++j) {
    \\                ok[tn][j] = sel[buf][j4 * 32 + tn * 16 + fn + j] >= 0;
    \\            }
    \\        }
    \\        // Partial row max of the raw scores over the quarter (scale > 0 and
    \\        // rounding is monotonic: max(s) * c == max(s * c) bitwise).
    \\        float pmx[2];
    \\        for (short i = 0; i < 2; ++i) {
    \\            float m = -FLT_MAX;
    \\            for (short tn = 0; tn < 2; ++tn) {
    \\                for (short j = 0; j < 4; ++j) {
    \\                    m = ok[tn][j] ? max(m, float(sc[tn * 8 + i * 4 + j])) : m;
    \\                }
    \\            }
    \\            m = max(m, simd_shuffle_xor(m, ushort(1)));
    \\            m = max(m, simd_shuffle_xor(m, ushort(8)));
    \\            pmx[i] = m;
    \\        }
    \\        if (row_writer) {
    \\            red_max[hg][j4][fm] = pmx[0];
    \\            red_max[hg][j4][fm + 8] = pmx[1];
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        if (t + 1 < n_tiles) {
    \\            stage(t + 1, buf ^ 1);
    \\        }
    \\
    \\        // ---- Tile row max (the same value in the 4 simdgroups of the group).
    \\        float factor[2];
    \\        float mnew[2];
    \\        for (short i = 0; i < 2; ++i) {
    \\            const int r = fm + i * 8;
    \\            const float m = max(max(red_max[hg][0][r], red_max[hg][1][r]),
    \\                                max(red_max[hg][2][r], red_max[hg][3][r]));
    \\            // no usable key in the tile: keep the running max
    \\            const float cand = m == -FLT_MAX ? -FLT_MAX : m * scale_log2;
    \\            mnew[i] = max(m_run[i], cand);
    \\            factor[i] = fast::exp2(m_run[i] - mnew[i]);
    \\            m_run[i] = mnew[i];
    \\        }
    \\        // ---- P of this quarter, once: exp2, fp16 hi + lo pieces, row sums.
    \\        float rs[2] = {0.0f, 0.0f};
    \\        for (short tn = 0; tn < 2; ++tn) {
    \\            vec<half, 8> h;
    \\#ifdef MSV_DSA_TWO_P
    \\            vec<half, 8> lo;
    \\#endif
    \\            for (short i = 0; i < 2; ++i) {
    \\                for (short j = 0; j < 4; ++j) {
    \\                    const float e = ok[tn][j]
    \\                        ? fast::exp2(float(sc[tn * 8 + i * 4 + j]) * scale_log2 - mnew[i])
    \\                        : 0.0f;
    \\                    const half hi = half(e);
    \\                    h[i * 4 + j] = hi;
    \\#ifdef MSV_DSA_TWO_P
    \\                    lo[i * 4 + j] = half(e - float(hi));
    \\#endif
    \\                    rs[i] += e;
    \\                }
    \\            }
    \\            const int ks = j4 * 2 + tn;
    \\            *(threadgroup vec<half, 8>*)(&p_hi[hg][ks][lane * 8]) = h;
    \\#ifdef MSV_DSA_TWO_P
    \\            *(threadgroup vec<half, 8>*)(&p_lo[hg][ks][lane * 8]) = lo;
    \\#endif
    \\        }
    \\        for (short i = 0; i < 2; ++i) {
    \\            rs[i] += simd_shuffle_xor(rs[i], ushort(1));
    \\            rs[i] += simd_shuffle_xor(rs[i], ushort(8));
    \\        }
    \\        if (row_writer) {
    \\            red_sum[hg][j4][fm] = rs[0];
    \\            red_sum[hg][j4][fm + 8] = rs[1];
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        // ---- O = O * factor + P V for dim quarter j4 over the whole tile.
    \\        // (factor == 1 exactly for every row of the simdgroup: O * 1 == O.)
    \\        if (!simd_all(factor[0] == 1.0f && factor[1] == 1.0f)) {
    \\            for (short e = 0; e < 16; ++e) {
    \\                const float f = factor[(e >> 2) & 1];
    \\                o0[e] *= f;
    \\                o1[e] *= f;
    \\                o2[e] *= f;
    \\                o3[e] *= f;
    \\            }
    \\        }
    \\        for (short i = 0; i < 2; ++i) {
    \\            const int r = fm + i * 8;
    \\            const float tsum = ((red_sum[hg][0][r] + red_sum[hg][1][r]) + red_sum[hg][2][r]) + red_sum[hg][3][r];
    \\            l_run[i] = l_run[i] * factor[i] + tsum;
    \\        }
    \\        const int n_ks = (tile_keys + 15) / 16;
    \\        for (short ks = 0; ks < n_ks; ++ks) {
    \\            const vec<half, 8> h = *(const threadgroup vec<half, 8>*)(&p_hi[hg][ks][lane * 8]);
    \\#ifdef MSV_DSA_TWO_P
    \\            const vec<half, 8> lo = *(const threadgroup vec<half, 8>*)(&p_lo[hg][ks][lane * 8]);
    \\#endif
    \\            for (short e = 0; e < 8; ++e) {
    \\                pa[e] = h[e];
    \\#ifdef MSV_DSA_TWO_P
    \\                pm[e] = lo[e];
    \\#endif
    \\            }
    \\            const int kp0 = sel[buf][ks * 16 + fm];
    \\            const int kp1 = sel[buf][ks * 16 + fm + 8];
    \\            const device T* v0 = kv + ulong(max(kp0, 0)) * D + j4 * 128 + fn;
    \\            const device T* v1 = kv + ulong(max(kp1, 0)) * D + j4 * 128 + fn;
    \\            for (short np = 0; np < 4; ++np) {
    \\                for (short tn = 0; tn < 2; ++tn) {
    \\                    for (short j = 0; j < 4; ++j) {
    \\                        pb[tn * 8 + j] = v0[np * 32 + tn * 16 + j];
    \\                        pb[tn * 8 + 4 + j] = v1[np * 32 + tn * 16 + j];
    \\                    }
    \\                }
    \\                if (np == 0) {
    \\                    pv_op.run(pa, pb, o0);
    \\#ifdef MSV_DSA_TWO_P
    \\                    pv_op.run(pm, pb, o0);
    \\#endif
    \\                } else if (np == 1) {
    \\                    pv_op.run(pa, pb, o1);
    \\#ifdef MSV_DSA_TWO_P
    \\                    pv_op.run(pm, pb, o1);
    \\#endif
    \\                } else if (np == 2) {
    \\                    pv_op.run(pa, pb, o2);
    \\#ifdef MSV_DSA_TWO_P
    \\                    pv_op.run(pm, pb, o2);
    \\#endif
    \\                } else {
    \\                    pv_op.run(pa, pb, o3);
    \\#ifdef MSV_DSA_TWO_P
    \\                    pv_op.run(pm, pb, o3);
    \\#endif
    \\                }
    \\            }
    \\        }
    \\    }
    \\
    \\    for (short i = 0; i < 2; ++i) {
    \\        device T* orow = out + (ulong(head0 + fm + i * 8) * L + qi) * D + j4 * 128 + fn;
    \\        const float denom = l_run[i] > 0.0f ? l_run[i] : 1.0f;
    \\        for (short tn = 0; tn < 2; ++tn) {
    \\            for (short j = 0; j < 4; ++j) {
    \\                const short e = tn * 8 + i * 4 + j;
    \\                orow[tn * 16 + j] = T(o0[e] / denom);
    \\                orow[32 + tn * 16 + j] = T(o1[e] / denom);
    \\                orow[64 + tn * 16 + j] = T(o2[e] / denom);
    \\                orow[96 + tn * 16 + j] = T(o3[e] / denom);
    \\            }
    \\        }
    \\    }
;

var sparse_kernel: ?mlx.mlx_fast_metal_kernel = null;
var sparse_rope_kernel: ?mlx.mlx_fast_metal_kernel = null;
var sparse_engaged = false;
var sparse_rope_engaged = false;

/// `sparseLatentAttention` (glm5_next) on the tensor units, for queries at the LAST rows of
/// `latent`: `q_abs` [1, H, L, 512], `latent` [1, 1, K, 512], `idx` int32 [1, L, W]. An unused
/// slot is -1 and unused pools sort last, so whole tiles skip. Null off the GPU or outside the
/// shape; the caller checks the device has NAX.
pub fn sparseLatent(s: mlx.mlx_stream, q_abs: mlx.mlx_array, latent: mlx.mlx_array, idx: mlx.mlx_array, scale: f32) !?mlx.mlx_array {
    return sparse(s, q_abs, null, latent, null, idx, scale);
}

/// `sparseLatent` with a roped part (DeepSeek-V3.2's MLA, GLM-5.3): each score adds the
/// query's roped part against the selected row's roped key, `q_lat . kv + q_pe . k_pe`, the
/// values the latent rows. `q_pe` [1, H, L, 64], `k_pe` [1, 1, K, 64]. P meets the values as
/// one fp16 piece.
pub fn sparseLatentRope(s: mlx.mlx_stream, q_lat: mlx.mlx_array, q_pe: mlx.mlx_array, latent: mlx.mlx_array, k_pe: mlx.mlx_array, idx: mlx.mlx_array, scale: f32) !?mlx.mlx_array {
    return sparse(s, q_lat, q_pe, latent, k_pe, idx, scale);
}

fn sparse(s: mlx.mlx_stream, q_abs: mlx.mlx_array, q_pe: ?mlx.mlx_array, latent: mlx.mlx_array, k_pe: ?mlx.mlx_array, idx: mlx.mlx_array, scale: f32) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const qs = mlx.getShape(q_abs);
    const ls = mlx.getShape(latent);
    const is = mlx.getShape(idx);
    if (qs.len != 4 or ls.len != 4 or is.len != 3) return null;
    if (qs[0] != 1 or ls[0] != 1 or ls[1] != 1 or is[0] != 1) return null;
    if (qs[3] != 512 or ls[3] != 512 or @rem(qs[1], 32) != 0) return null;
    const dt = mlx.mlx_array_dtype(q_abs);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(latent) != dt or mlx.mlx_array_dtype(idx) != .int32) return null;
    const h = qs[1];
    const l = qs[2];
    const k = ls[2];
    const w = is[2];
    if (l < 1 or k < l or is[1] != l or w < 1) return null;
    const rope = q_pe != null;
    if (rope) {
        const ps = mlx.getShape(q_pe.?);
        const ks = mlx.getShape(k_pe.?);
        if (ps.len != 4 or ps[0] != 1 or ps[1] != h or ps[2] != l or ps[3] != 64) return null;
        if (ks.len != 4 or ks[0] != 1 or ks[1] != 1 or ks[2] != k or ks[3] != 64) return null;
        if (mlx.mlx_array_dtype(q_pe.?) != dt or mlx.mlx_array_dtype(k_pe.?) != dt) return null;
    }
    const slot = if (rope) &sparse_rope_kernel else &sparse_kernel;
    const kern = slot.* orelse blk: {
        const ins_plain = [_][*:0]const u8{ "q", "kv", "idx", "params", "scale" };
        const ins_rope = [_][*:0]const u8{ "q", "qpe", "kv", "pe", "idx", "params", "scale" };
        const ins: []const [*:0]const u8 = if (rope) &ins_rope else &ins_plain;
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const kn = if (rope)
            mlx.mlx_fast_metal_kernel_new("msv_dsa_sparse_mla_rope_nax", iv, ov, "#define MSV_DSA_ROPE\n" ++ SPARSE_NAX_SOURCE, SPARSE_NAX_HEADER, true, false)
        else
            mlx.mlx_fast_metal_kernel_new("msv_glm5_sparse_mla_nax", iv, ov, "#define MSV_DSA_TWO_P\n" ++ SPARSE_NAX_SOURCE, SPARSE_NAX_HEADER, true, false);
        if (kn.ctx == null) return error.MetalKernelCompileFailed;
        slot.* = kn;
        break :blk kn;
    };
    var arrays: [7]mlx.mlx_array = undefined;
    var n: usize = 0;
    defer for (arrays[0..n]) |x| {
        _ = mlx.mlx_array_free(x);
    };
    // The kernel's inputs in its order, each as the rows it indexes.
    const shapes = [5][3]c_int{ .{ h, l, 512 }, .{ h, l, 64 }, .{ k, 512, 0 }, .{ k, 64, 0 }, .{ l, w, 0 } };
    const ranks = [5]usize{ 3, 3, 2, 2, 2 };
    for ([5]?mlx.mlx_array{ q_abs, q_pe, latent, k_pe, idx }, shapes, ranks) |x_, sh, r| if (x_) |x| {
        arrays[n] = mlx.mlx_array_new();
        n += 1;
        try mlx.check(mlx.mlx_reshape(&arrays[n - 1], x, &sh, r, s));
    };
    const pdata = [4]i32{ l, k, w, k - l };
    arrays[n] = mlx.mlx_array_new_data(&pdata, &[_]c_int{4}, 1, .int32);
    n += 1;
    const sv = [1]f32{scale};
    arrays[n] = mlx.mlx_array_new_data(&sv, &[_]c_int{1}, 1, .float32);
    n += 1;

    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ h, l, 512 }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, l * 256, @divExact(h, 32), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    const iv = mlx.mlx_vector_array_new_data(&arrays, n);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var o3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o3);
    try mlx.check(mlx.mlx_vector_array_get(&o3, ov, 0));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, o3, &[_]c_int{ 1, h, l, 512 }, 4, s));
    const engaged = if (rope) &sparse_rope_engaged else &sparse_engaged;
    if (!engaged.*) {
        engaged.* = true;
        log.info("[dsa] NAX sparse latent attention{s} engaged\n", .{if (rope) " (roped part)" else ""});
    }
    return out;
}

const testing = std.testing;

fn testArray(alloc: std.mem.Allocator, rand: std.Random, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const v = try alloc.alloc(f32, n);
    defer alloc.free(v);
    for (v) |*x| x.* = rand.floatNorm(f32);
    const f = mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

fn hostF32(a: mlx.mlx_array, s: mlx.mlx_stream, alloc: std.mem.Allocator) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, a, .float32, s));
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, f, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const p = mlx.mlx_array_data_float32(c) orelse return error.MlxNoData;
    return alloc.dupe(f32, p[0..mlx.mlx_array_size(c)]);
}

test "dsa sparse latent attention with a roped part equals the latent attention over the joined rows" {
    if (mlx.noGpuBackend() or !@import("transformer.zig").verifyQmmNaxAvailable()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    // The second shape's 4 index slots are bound as a constant array.
    for ([_][4]c_int{ .{ 64, 300, 40, 260 }, .{ 32, 5, 1, 4 } }) |dims| try checkSparseRope(dims[0], dims[1], dims[2], dims[3]);
}

fn checkSparseRope(h: c_int, k: c_int, l: c_int, w: c_int) !void {
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(13);
    const rand = prng.random();
    const q = try testArray(alloc, rand, &.{ 1, h, l, 512 }, s);
    defer _ = mlx.mlx_array_free(q);
    const qp = try testArray(alloc, rand, &.{ 1, h, l, 64 }, s);
    defer _ = mlx.mlx_array_free(qp);
    const lat = try testArray(alloc, rand, &.{ 1, 1, k, 512 }, s);
    defer _ = mlx.mlx_array_free(lat);
    const kp = try testArray(alloc, rand, &.{ 1, 1, k, 64 }, s);
    defer _ = mlx.mlx_array_free(kp);
    // Each query reads a random subset of the keys at or before it, holes as -1 (some past it too).
    const iv = try alloc.alloc(i32, @intCast(l * w));
    defer alloc.free(iv);
    for (0..@intCast(l)) |qi| {
        const pos: i32 = k - l + @as(i32, @intCast(qi));
        for (0..@intCast(w)) |j| iv[qi * @as(usize, @intCast(w)) + j] = rand.intRangeAtMost(i32, -@divTrunc(pos, 4) - 1, pos + 3);
    }
    const idx = mlx.mlx_array_new_data(iv.ptr, &[_]c_int{ 1, l, w }, 3, .int32);
    defer _ = mlx.mlx_array_free(idx);
    const got = (try sparseLatentRope(s, q, qp, lat, kp, idx, 0.0625)) orelse return error.SkipZigTest;
    defer _ = mlx.mlx_array_free(got);
    // The reference: the roped part joined to each row (q | q_pe against latent | k_pe), its first 512 values.
    var qj = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qj);
    var kj = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kj);
    {
        const qv = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ q, qp }, 2);
        defer _ = mlx.mlx_vector_array_free(qv);
        try mlx.check(mlx.mlx_concatenate_axis(&qj, qv, -1, s));
        const kv = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ lat, kp }, 2);
        defer _ = mlx.mlx_vector_array_free(kv);
        try mlx.check(mlx.mlx_concatenate_axis(&kj, kv, -1, s));
    }
    // An unused slot or one past its query is -1 to the reference (the kernel drops both).
    for (0..@intCast(l)) |qi| {
        const pos: i32 = k - l + @as(i32, @intCast(qi));
        for (iv[qi * @as(usize, @intCast(w)) ..][0..@intCast(w)]) |*x| if (x.* > pos) {
            x.* = -1;
        };
    }
    const idx_ref = mlx.mlx_array_new_data(iv.ptr, &[_]c_int{ 1, l, w }, 3, .int32);
    defer _ = mlx.mlx_array_free(idx_ref);
    const joined = try @import("glm5_next.zig").sparseLatentAttention(s, qj, kj, idx_ref, 0.0625, 64);
    defer _ = mlx.mlx_array_free(joined);
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_slice(&ref, joined, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, h, l, 512 }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
    const a = try hostF32(got, s, alloc);
    defer alloc.free(a);
    const b = try hostF32(ref, s, alloc);
    defer alloc.free(b);
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    var maxd: f32 = 0;
    for (a, b) |x, y| {
        dot += x * y;
        na += x * x;
        nb += y * y;
        maxd = @max(maxd, @abs(x - y));
    }
    const cos = dot / @sqrt(na * nb);
    std.debug.print("[dsa nax rope] {d} heads, {d} keys, {d} queries x {d} slots: cos {d:.6} max|d| {d:.4} (|ref| {d:.3})\n", .{ h, k, l, w, cos, maxd, @sqrt(nb) });
    try testing.expect(cos > 0.9999 and maxd < 0.05);
}
