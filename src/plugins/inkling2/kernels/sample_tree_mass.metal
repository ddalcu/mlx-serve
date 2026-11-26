// Inkling 2: sharded max / exp-mass / argmax over [B * W, STRIDE] verify logits (first V columns).
// Handles both greedy (temp <= 0) and sampled (temp > 0) lanes in one batched launch.
// grid (SHARDS * 256, B * W, 1), threadgroup (256, 1, 1).
// LPARAMS [B, 4] f32 = { temp, float(top_k), top_p, min_p }.
// PMASS [B * W, SHARDS, 4] f32 = { max_scaled_logit, sum_exp, float(admitted), as_type<float>(argmax_id) }.
const uint row = threadgroup_position_in_grid.y;
const uint shard = threadgroup_position_in_grid.x;
const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint b = row / uint(W);
const float temp = LPARAMS[b * 4u + 0u];
const bool sampled = temp > 0.0f;
const float inv_t = sampled ? (1.0f / temp) : 1.0f;

const uint chunk = (uint(V) + uint(SHARDS) - 1u) / uint(SHARDS);
const uint lo = shard * chunk;
const uint hi = min(uint(V), lo + chunk);
const device T* src = LOGITS + size_t(row) * size_t(STRIDE);

float mx = -INFINITY;
int best_id = int(V);
float sum = 0.0f;
uint adm = 0u;

for (uint col = lo + tid; col < hi; col += 256u) {
    if constexpr (MASK != 0) {
        if (SUPPRESS[col]) continue;
    }
    const float raw = float(src[col]);
    if (isnan(raw) || raw == -INFINITY) continue;
    const float v = raw * inv_t;
    ++adm;
    if (v > mx || (v == mx && int(col) < best_id)) {
        if (sampled && mx > -INFINITY && v > mx) sum *= metal::exp(mx - v);
        mx = v;
        best_id = int(col);
    }
    if (sampled) sum += metal::exp(v - mx);
}

// SIMD reduce max + argmax.
float smx = mx;
int sbi = best_id;
for (uint off = 16u; off > 0u; off >>= 1) {
    const float om = simd_shuffle_xor(smx, off);
    const int oi = simd_shuffle_xor(sbi, off);
    if (om > smx || (om == smx && oi < sbi)) { smx = om; sbi = oi; }
}
float ssum = sampled && mx > -INFINITY ? (sum * metal::exp(mx - smx)) : 0.0f;
if (sampled) ssum = simd_sum(ssum);
uint sadm = simd_sum(adm);

threadgroup float tg_mx[8];
threadgroup int tg_id[8];
threadgroup float tg_sum[8];
threadgroup uint tg_adm[8];
if (lane == 0u) {
    tg_mx[sg] = smx;
    tg_id[sg] = sbi;
    tg_sum[sg] = ssum;
    tg_adm[sg] = sadm;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid == 0u) {
    float gmx = tg_mx[0];
    int gid = tg_id[0];
    uint gadm = tg_adm[0];
    for (uint s = 1u; s < 8u; ++s) {
        gadm += tg_adm[s];
        if (tg_mx[s] > gmx || (tg_mx[s] == gmx && tg_id[s] < gid)) {
            gmx = tg_mx[s];
            gid = tg_id[s];
        }
    }
    float gsum = 0.0f;
    if (sampled && gmx > -INFINITY) {
        for (uint s = 0u; s < 8u; ++s) {
            if (tg_mx[s] > -INFINITY) gsum += tg_sum[s] * metal::exp(tg_mx[s] - gmx);
        }
    }
    const size_t out = (size_t(row) * uint(SHARDS) + shard) * 4u;
    PMASS[out + 0u] = gmx;
    PMASS[out + 1u] = gsum;
    PMASS[out + 2u] = float(gadm);
    PMASS[out + 3u] = as_type<float>(gid);
}
