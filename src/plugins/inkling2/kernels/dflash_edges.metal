// Inkling 2 DFlash2: full R x K x K edge score + softcap + temperature-1 softmax lattice per lane.
// T1 [B] i32, CANDS [B, R, K] i32, UNARY [B, R, K] f32, PROJ [B, R, RANK] (PT),
// PRED [V, RANK] (CT), SUCC [V, RANK] (CT), CAP [1] f32 (softcap, 0 = none).
// PROBS [B, R, K, K] f32: PROBS[b, pos, pred_i, :] is the probability distribution over CANDS[b, pos, 0..K-1]
// given predecessor T1[b] (pos == 0, pred_i == 0) or CANDS[b, pos - 1, pred_i] (pos > 0).
// grid (K * 32, R, B), threadgroup (32, 1, 1).
const uint b = threadgroup_position_in_grid.z;
const uint pos = threadgroup_position_in_grid.y;
const uint pred_i = threadgroup_position_in_grid.x;
const uint lane = thread_index_in_simdgroup;

const size_t out_off = (((size_t(b) * R + pos) * K) + pred_i) * K;
if (pos == 0u && pred_i > 0u) {
    if (lane < K) PROBS[out_off + lane] = 0.0f;
    return;
}

const int prev_id = (pos == 0u) ? T1[b] : CANDS[(size_t(b) * R + (pos - 1u)) * K + pred_i];
const device CT* pred_row = PRED + size_t(prev_id) * RANK;
const device PT* proj_row = PROJ + (size_t(b) * R + pos) * RANK;
constexpr uint ELEMS = (RANK + 31u) / 32u;
float pp[ELEMS];
for (uint j = 0; j < ELEMS; ++j) {
    const uint c = j * 32u + lane;
    pp[j] = (c < RANK) ? float(pred_row[c]) * float(proj_row[c]) : 0.0f;
}

const float cap = CAP[0];
const size_t cand_off = (size_t(b) * R + pos) * K;
float scores[K];
float mx = -INFINITY;
for (uint s = 0; s < K; ++s) {
    const int sid = CANDS[cand_off + s];
    const device CT* succ_row = SUCC + size_t(sid) * RANK;
    float dot_v = 0.0f;
    for (uint j = 0; j < ELEMS; ++j) {
        const uint c = j * 32u + lane;
        if (c < RANK) dot_v += pp[j] * float(succ_row[c]);
    }
    const float edge = simd_sum(dot_v);
    const float raw = UNARY[cand_off + s] + edge;
    const float sc = (cap > 0.0f && raw > -INFINITY) ? (cap * precise::tanh(raw / cap)) : raw;
    scores[s] = sc;
    mx = max(mx, sc);
}

float mass = 0.0f;
float w[K];
for (uint s = 0; s < K; ++s) {
    w[s] = precise::exp(scores[s] - mx);
    mass += w[s];
}
if (lane < K) {
    PROBS[out_off + lane] = (mass > 0.0f) ? (w[lane] / mass) : (1.0f / float(K));
}
