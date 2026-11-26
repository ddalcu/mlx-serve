// Inkling 2, a prompt's rows: causal flash attention under the relative bias and a sink logit per row, no mask
// built. Ported from mlx-lm's `relative_attention` (`mlx_lm/models/inkling2/kernels.py`, MIT; see NOTICE).
// q [B, HQ, L, D] and k, v [B, HKV, S, D] (T, any strides over a unit last axis), the queries at the last L keys;
// rel [B, L, HQ, R] (T) each row's bias at the distances below R; sinks [B, L, HQ] f32; key_start [B] i32 the first
// key each batch row may read; sizes = { L, S }; scale the logit scale. WINDOW > 0 keeps the keys fewer than WINDOW
// behind a row. grid (ceil(L / BQ) * TG, HQ, B), threadgroup (TG, 1, 1): a threadgroup takes BQ queries of one
// head, 8 per SIMD group, against 16 keys at a time, as 8x8 fragments. out [B, L, HQ, D] (T).
// Each row's online softmax starts from its sink logit, so the zero-valued sink needs no key.
constexpr float LOG2E = 1.4426950408889634f;
constexpr int BK = 16;
constexpr int LDK = BK + 8;
constexpr int LDV = D + 8;
const int L = sizes[0];
const int S = sizes[1];
const uint tid = thread_position_in_threadgroup.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int batch = threadgroup_position_in_grid.z;
const int head = threadgroup_position_in_grid.y;
const int kv_head = head / (HQ / HKV);
const int q0 = threadgroup_position_in_grid.x * BQ;
const int start = key_start[batch];
const float qscale = scale * LOG2E;

// The lane holds row fm and columns fn, fn + 1 of each 8x8 fragment.
const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
const int row = q0 + int(sg) * 8 + fm;
const bool row_ok = row < L;
const int diag = S - L + row;

// q and k are clipped and v has unit RMS, so half holds them.
threadgroup half Ks[D * LDK];
threadgroup half Vs[BK * LDV];

simdgroup_half8x8 q_frag[D / 8];
for (int db = 0; db < D / 8; ++db) {
    half2 qv = half2(0.0h);
    if (row_ok) {
        const device T* qp = q + batch * q_strides[0] + head * q_strides[1]
            + row * q_strides[2] + db * 8 + fn;
        qv = half2(float(qp[0]), float(qp[1]));
    }
    q_frag[db].thread_elements()[0] = qv.x;
    q_frag[db].thread_elements()[1] = qv.y;
}
simdgroup_float8x8 acc[D / 8];
for (int db = 0; db < D / 8; ++db) {
    acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}
float row_max = row_ok ? float(sinks[(batch * L + row) * HQ + head]) * LOG2E : 0.0f;
float row_sum = 1.0f;
const int bias_row = batch * L + (row_ok ? row : 0);
const device T* bias = rel + (size_t(bias_row) * HQ + head) * R;

const int first_diag = S - L + q0;
const int last_diag = S - L + min(q0 + BQ, L) - 1;
int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
first_key = max(first_key, start);
const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];

for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = tid; c < BK * D / 8; c += TG) {
        const int r = c / (D / 8);
        const int col = (c % (D / 8)) * 8;
        const int key = k0 + r;
        vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
        vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
        if (key < S) {
            auto kp = (const device vec<T, 4>*)(k_head + key * k_strides[2] + col);
            auto vp = (const device vec<T, 4>*)(v_head + key * v_strides[2] + col);
            kv[0] = kp[0];
            kv[1] = kp[1];
            vv[0] = vp[0];
            vv[1] = vp[1];
        }
        for (int j = 0; j < 8; ++j) {
            Ks[(col + j) * LDK + r] = half(float(kv[j / 4][j % 4]));
            Vs[r * LDV + col + j] = half(float(vv[j / 4][j % 4]));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (q0 + int(sg) * 8 >= L) {
        continue;
    }

    simdgroup_float8x8 scores[BK / 8];
    for (int kb = 0; kb < BK / 8; ++kb) {
        scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int db = 0; db < D / 8; ++db) {
            simdgroup_half8x8 k_frag;
            simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
            simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
        }
    }

    float tile_max = -INFINITY;
    for (int kb = 0; kb < BK / 8; ++kb) {
        for (int e = 0; e < 2; ++e) {
            const int key = k0 + kb * 8 + fn + e;
            const int d = diag - key;
            const bool visible = row_ok && d >= 0 && key >= start
                && (WINDOW == 0 || d < WINDOW);
            float s = -INFINITY;
            if (visible) {
                s = scores[kb].thread_elements()[e] * qscale;
                if (d < R) {
                    s += float(bias[d]) * LOG2E;
                }
            }
            scores[kb].thread_elements()[e] = s;
            tile_max = max(tile_max, s);
        }
    }
    // The 4 lanes of a row are lane ^ 1 and lane ^ 8 apart.
    tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
    tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
    const float new_max = max(row_max, tile_max);
    const float factor = fast::exp2(row_max - new_max);
    float tile_sum = 0.0f;
    simdgroup_half8x8 probs[BK / 8];
    for (int kb = 0; kb < BK / 8; ++kb) {
        for (int e = 0; e < 2; ++e) {
            const float p = fast::exp2(scores[kb].thread_elements()[e] - new_max);
            probs[kb].thread_elements()[e] = half(p);
            tile_sum += p;
        }
    }
    tile_sum += simd_shuffle_xor(tile_sum, 1);
    tile_sum += simd_shuffle_xor(tile_sum, 8);
    row_sum = row_sum * factor + tile_sum;
    row_max = new_max;
    for (int db = 0; db < D / 8; ++db) {
        acc[db].thread_elements()[0] *= factor;
        acc[db].thread_elements()[1] *= factor;
        for (int kb = 0; kb < BK / 8; ++kb) {
            simdgroup_half8x8 v_frag;
            simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
            simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
        }
    }
}

if (row_ok) {
    device T* op = out + (size_t(batch * L + row) * HQ + head) * D;
    for (int db = 0; db < D / 8; ++db) {
        op[db * 8 + fn] = T(acc[db].thread_elements()[0] / row_sum);
        op[db * 8 + fn + 1] = T(acc[db].thread_elements()[1] / row_sum);
    }
}
