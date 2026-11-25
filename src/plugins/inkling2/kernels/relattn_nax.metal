// Inkling 2, a prompt's rows on Metal 4 NAX: causal flash attention under the relative bias and a sink logit per row.
// q [B, HQ, L, D] and k, v [B, HKV, S, D] (T, any strides over a unit last axis), the queries at the last L keys;
// rel [B, L, HQ, R] (T) each row's bias at the distances below R; sinks [B, L, HQ] f32; key_start [B] i32 the first
// key each batch row may read; sizes = { L, S }; scale the logit scale. WINDOW > 0 keeps keys fewer than WINDOW
// behind a row. grid (ceil(L / BQ) * TG, HQ, B), threadgroup (TG, 1, 1): a threadgroup takes BQ (64) queries of one
// head, 16 per SIMD group, against 32 keys at a time on 16x32x16 NAX tensor ops. out [B, L, HQ, D] (T).
constexpr float LOG2E = 1.4426950408889634f;
constexpr int BK = 32;
constexpr int SLOTS = D / 4;
static_assert(D == 128, "relattn_nax requires D == 128");
const int L = sizes[0];
const int S = sizes[1];
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int batch = threadgroup_position_in_grid.z;
const int head = threadgroup_position_in_grid.y;
const int kv_head = head / (HQ / HKV);
const int q0 = threadgroup_position_in_grid.x * BQ;
const int start = key_start[batch];
const float qscale = scale * LOG2E;

// 16x32x16 execution_simdgroup cooperative tensor lane layout: rows fm and fm + 8, cols fn..fn+3 per 16-col tile.
const short qid = short(lane >> 2);
const short fm = short((qid & 4) | ((lane >> 1) & 3));
const short fn = short(((qid & 2) | (lane & 1)) * 4);
const int sg_q0 = q0 + int(sg) * 16;
const int row0 = sg_q0 + fm;
const int row1 = row0 + 8;
const bool ok0 = row0 < L;
const bool ok1 = row1 < L;
const int diag[2] = {S - L + row0, S - L + row1};
const bool row_ok[2] = {ok0, ok1};
const int sg_first_diag = S - L + sg_q0;
const int sg_last_diag = S - L + min(sg_q0 + 16, L) - 1;

// Exactly 16 KB total (8 KB Ks + 8 KB Vs) with XOR-swizzled vec4 slots: 2 TGs/core and zero bank conflicts.
threadgroup vec<T, 4> Ks[BK * SLOTS];
threadgroup vec<T, 4> Vs[BK * SLOTS];

constexpr auto qk_desc = matmul2d_descriptor(
    16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<qk_desc, execution_simdgroup> qk_op;
constexpr auto pv_desc = matmul2d_descriptor(
    16, 32, 16, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<pv_desc, execution_simdgroup> pv_op;

auto qa = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
auto kb = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
auto sc = qk_op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kb)>, float>();
auto pa0 = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
auto pa1 = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
auto pb = pv_op.template get_right_input_cooperative_tensor<half, T, float>();
auto o0 = pv_op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(pa0)>, metal::remove_addrspace_t<decltype(pb)>, float>();
auto o1 = pv_op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(pa0)>, metal::remove_addrspace_t<decltype(pb)>, float>();
auto o2 = pv_op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(pa0)>, metal::remove_addrspace_t<decltype(pb)>, float>();
auto o3 = pv_op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(pa0)>, metal::remove_addrspace_t<decltype(pb)>, float>();
for (short e = 0; e < 16; ++e) {
    o0[e] = 0.0f;
    o1[e] = 0.0f;
    o2[e] = 0.0f;
    o3[e] = 0.0f;
}

const device T* q_head = q + batch * q_strides[0] + head * q_strides[1];
vec<T, 4> qr0[D / 16];
vec<T, 4> qr1[D / 16];
for (int kk = 0; kk < D / 16; ++kk) {
    qr0[kk] = ok0 ? *(const device vec<T, 4>*)(q_head + row0 * q_strides[2] + kk * 16 + fn) : vec<T, 4>(0);
    qr1[kk] = ok1 ? *(const device vec<T, 4>*)(q_head + row1 * q_strides[2] + kk * 16 + fn) : vec<T, 4>(0);
}

float row_max[2] = {
    ok0 ? float(sinks[(batch * L + row0) * HQ + head]) * LOG2E : 0.0f,
    ok1 ? float(sinks[(batch * L + row1) * HQ + head]) * LOG2E : 0.0f,
};
float row_sum[2] = {1.0f, 1.0f};
const device T* bias_ptr[2] = {
    rel + (size_t(batch * L + (ok0 ? row0 : 0)) * HQ + head) * R,
    rel + (size_t(batch * L + (ok1 ? row1 : 0)) * HQ + head) * R,
};

auto maxN = [](float a, float b) -> float {
    return isnan(a) ? a : (isnan(b) ? b : max(a, b));
};

const int first_diag = S - L + q0;
const int last_diag = S - L + min(q0 + BQ, L) - 1;
int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
first_key = max(first_key, start);
const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];
const int fn_slot = fn >> 2;

for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int r = int(sg); r < BK; r += TG / 32) {
        const int key = k0 + r;
        vec<T, 4> kv = vec<T, 4>(0);
        vec<T, 4> vv = vec<T, 4>(0);
        if (key < S) {
            kv = *(const device vec<T, 4>*)(k_head + key * k_strides[2] + lane * 4);
            vv = *(const device vec<T, 4>*)(v_head + key * v_strides[2] + lane * 4);
        }
        const int sw = int(lane) ^ (r & 7);
        Ks[r * SLOTS + sw] = kv;
        Vs[r * SLOTS + sw] = vv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_q0 >= L || k0 > sg_last_diag || (WINDOW > 0 && k0 + BK - 1 < sg_first_diag - WINDOW + 1)) continue;

    for (short e = 0; e < 16; ++e) sc[e] = 0.0f;
    for (int kk = 0; kk < D / 16; ++kk) {
        const vec<T, 4> a0 = qr0[kk];
        const vec<T, 4> a1 = qr1[kk];
        for (short j = 0; j < 4; ++j) {
            qa[j] = a0[j];
            qa[4 + j] = a1[j];
        }
        const int s_k = (kk * 4 + fn_slot) ^ (fm & 7);
        for (short tn = 0; tn < 2; ++tn) {
            const vec<T, 4> k0v = Ks[(tn * 16 + fm) * SLOTS + s_k];
            const vec<T, 4> k1v = Ks[(tn * 16 + fm + 8) * SLOTS + s_k];
            for (short j = 0; j < 4; ++j) {
                kb[tn * 8 + j] = k0v[j];
                kb[tn * 8 + 4 + j] = k1v[j];
            }
        }
        qk_op.run(qa, kb, sc);
    }

    const bool full_block = k0 >= start && (k0 + BK - 1 <= sg_first_diag) && (WINDOW == 0 || sg_last_diag - k0 < WINDOW) && (sg_q0 + 16 <= L);
    const bool no_bias = sg_first_diag - (k0 + BK - 1) >= R;
    float scores[16];
    float tile_max[2] = {-INFINITY, -INFINITY};
    for (short i = 0; i < 2; ++i) {
        float m = -INFINITY;
        if (full_block && no_bias) {
            for (short e = 0; e < 4; ++e) {
                const float s0 = float(sc[i * 4 + e]) * qscale;
                const float s1 = float(sc[8 + i * 4 + e]) * qscale;
                scores[i * 4 + e] = s0;
                scores[8 + i * 4 + e] = s1;
                m = maxN(m, maxN(s0, s1));
            }
        } else {
            for (short tn = 0; tn < 2; ++tn) {
                for (short j = 0; j < 4; ++j) {
                    const int key = k0 + tn * 16 + fn + j;
                    const int d = diag[i] - key;
                    const bool visible = row_ok[i] && d >= 0 && key >= start && (WINDOW == 0 || d < WINDOW);
                    float s = -INFINITY;
                    if (visible) {
                        s = float(sc[tn * 8 + i * 4 + j]) * qscale;
                        if (d < R) s += float(bias_ptr[i][d]) * LOG2E;
                    }
                    scores[tn * 8 + i * 4 + j] = s;
                    m = maxN(m, s);
                }
            }
        }
        m = maxN(m, simd_shuffle_xor(m, ushort(1)));
        m = maxN(m, simd_shuffle_xor(m, ushort(8)));
        tile_max[i] = m;
    }

    float new_max[2];
    float factor[2];
    float tile_sum[2] = {0.0f, 0.0f};
    for (short i = 0; i < 2; ++i) {
        new_max[i] = maxN(row_max[i], tile_max[i]);
        factor[i] = fast::exp2(row_max[i] - new_max[i]);
        row_max[i] = new_max[i];
    }
    for (short i = 0; i < 2; ++i) {
        for (short j = 0; j < 4; ++j) {
            const float p0 = fast::exp2(scores[i * 4 + j] - new_max[i]);
            const float p1 = fast::exp2(scores[8 + i * 4 + j] - new_max[i]);
            pa0[i * 4 + j] = half(p0);
            pa1[i * 4 + j] = half(p1);
            tile_sum[i] += p0 + p1;
        }
        tile_sum[i] += simd_shuffle_xor(tile_sum[i], ushort(1));
        tile_sum[i] += simd_shuffle_xor(tile_sum[i], ushort(8));
        row_sum[i] = row_sum[i] * factor[i] + tile_sum[i];
    }

    for (short e = 0; e < 16; ++e) {
        const float f = factor[(e >> 2) & 1];
        o0[e] *= f;
        o1[e] *= f;
        o2[e] *= f;
        o3[e] *= f;
    }

#define RELNAX_PV(PA, KH, OO, NP)                                                \
    {                                                                            \
        const int r0 = (KH) * 16 + fm;                                           \
        const int r1 = r0 + 8;                                                   \
        for (short tn = 0; tn < 2; ++tn) {                                       \
            const int sw = ((NP) * 8 + tn * 4 + fn_slot) ^ (fm & 7);             \
            const vec<T, 4> x0 = Vs[r0 * SLOTS + sw];                            \
            const vec<T, 4> x1 = Vs[r1 * SLOTS + sw];                            \
            for (short j = 0; j < 4; ++j) {                                      \
                pb[tn * 8 + j] = x0[j];                                          \
                pb[tn * 8 + 4 + j] = x1[j];                                      \
            }                                                                    \
        }                                                                        \
        pv_op.run(PA, pb, OO);                                                   \
    }
    RELNAX_PV(pa0, 0, o0, 0)
    RELNAX_PV(pa0, 0, o1, 1)
    RELNAX_PV(pa0, 0, o2, 2)
    RELNAX_PV(pa0, 0, o3, 3)
    RELNAX_PV(pa1, 1, o0, 0)
    RELNAX_PV(pa1, 1, o1, 1)
    RELNAX_PV(pa1, 1, o2, 2)
    RELNAX_PV(pa1, 1, o3, 3)
#undef RELNAX_PV
}

const float inv[2] = {1.0f / row_sum[0], 1.0f / row_sum[1]};
const int rows_out[2] = {row0, row1};
#define RELNAX_STORE(OO, NP)                                                                           \
    for (short e = 0; e < 16; ++e) {                                                                   \
        const short i = (e >> 2) & 1;                                                                  \
        if (!row_ok[i]) continue;                                                                      \
        const short tn = e >> 3;                                                                       \
        const short j = e & 3;                                                                         \
        const int d = (NP) * 32 + tn * 16 + fn + j;                                                    \
        out[(size_t(batch * L + rows_out[i]) * HQ + head) * D + d] = T(OO[e] * inv[i]);                \
    }
RELNAX_STORE(o0, 0)
RELNAX_STORE(o1, 1)
RELNAX_STORE(o2, 2)
RELNAX_STORE(o3, 3)
#undef RELNAX_STORE
