// Inkling 2 DFlash2: sharded exact top-K over [R_TOTAL, STRIDE] logits (first V columns), replacing full-vocab sort.
// grid (SHARDS * 256, R_TOTAL, 1), threadgroup (256, 1, 1).
// Outputs: PIDS [R_TOTAL, SHARDS, K] int32, PVALS [R_TOTAL, SHARDS, K] float32 (descending, lowest id on ties).
const uint row = threadgroup_position_in_grid.y;
const uint shard = threadgroup_position_in_grid.x;
const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;

const uint chunk = (V + SHARDS - 1u) / SHARDS;
const uint lo = shard * chunk;
const uint hi = min(uint(V), lo + chunk);
const device T* src = LOGITS + size_t(row) * STRIDE;

float bv[K];
int bi[K];
for (uint k = 0; k < K; ++k) { bv[k] = -INFINITY; bi[k] = int(V); }

for (uint col = lo + tid; col < hi; col += 256u) {
    const float v = float(src[col]);
    if (isnan(v)) continue;
    if (v > bv[K - 1] || (v == bv[K - 1] && int(col) < bi[K - 1])) {
        int pos = int(K) - 1;
        while (pos > 0 && (v > bv[pos - 1] || (v == bv[pos - 1] && int(col) < bi[pos - 1]))) {
            bv[pos] = bv[pos - 1];
            bi[pos] = bi[pos - 1];
            --pos;
        }
        bv[pos] = v;
        bi[pos] = int(col);
    }
}

threadgroup float sg_v[8];
threadgroup int sg_i[8];
threadgroup uint sg_t[8];
threadgroup uint win_tid;

uint head = 0;
const size_t out_base = (size_t(row) * SHARDS + shard) * K;
for (uint m = 0; m < K; ++m) {
    float cur_v = (head < K) ? bv[head] : -INFINITY;
    int cur_i = (head < K) ? bi[head] : int(V);
    uint cur_t = tid;
    // Simd-level max reduction (value descending, token id ascending on ties).
    for (uint off = 16u; off > 0u; off >>= 1) {
        const float ov = simd_shuffle_xor(cur_v, off);
        const int oi = simd_shuffle_xor(cur_i, off);
        const uint ot = simd_shuffle_xor(cur_t, off);
        if (ov > cur_v || (ov == cur_v && oi < cur_i)) {
            cur_v = ov;
            cur_i = oi;
            cur_t = ot;
        }
    }
    if (lane == 0u) {
        sg_v[sg] = cur_v;
        sg_i[sg] = cur_i;
        sg_t[sg] = cur_t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float best_v = sg_v[0];
        int best_i = sg_i[0];
        uint best_t = sg_t[0];
        for (uint s = 1u; s < 8u; ++s) {
            if (sg_v[s] > best_v || (sg_v[s] == best_v && sg_i[s] < best_i)) {
                best_v = sg_v[s];
                best_i = sg_i[s];
                best_t = sg_t[s];
            }
        }
        PIDS[out_base + m] = best_i;
        PVALS[out_base + m] = best_v;
        win_tid = best_t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == win_tid && head < K) ++head;
}
