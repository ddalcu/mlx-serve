// Inkling 2 DFlash2: per-lane GPU best-first draft tree builder (up to W <= 16 verify rows per lane).
// Expands top-C candidates (greedy) or Gumbel-top-C without replacement (sampled) from the R x K x K lattice,
// calibrated via logit(p) = slope * logit(q) + intercept[pos], and merges any host lookup/copy chain into the tree.
// grid (B, 1, 1), threadgroup (1, 1, 1).
// Outputs:
//   TOKENS [B, W] i32, PARENT [B, W] i32, TMETA [B * W, 2] i32 ({ depth, ancestors_mask }),
//   NEXT_Q [B, W, K] f32 (normalized q_T at each node over its next position's K candidates),
//   DRAWS  [B, W, 4] u8  (placed drawn candidate indices 0..K-1 in draw order, 255 past placed count),
//   SCORES [B, W] f32    (node's draft probability q at its position, -1 for root / pure copy).
const uint b = threadgroup_position_in_grid.x;
const uint B = T1_shape[0];
if (b >= B) return;

// SplitMix64 helpers for Gumbel-top-k without replacement.
auto mix64 = [](ulong v) -> ulong {
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ul;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebul;
    return v ^ (v >> 31);
};
auto node_key = [&](ulong parent_key, uint tok) -> ulong {
    return mix64(parent_key + 0x9e3779b97f4a7c15ul * (ulong(tok) + 1ul));
};
auto gumbel = [&](ulong key, uint tok) -> float {
    const ulong h = mix64(key ^ (0xd1b54a32d192ed03ul * (ulong(tok) + 1ul)));
    const float u = (float(h >> 41) + 0.5f) * 0x1p-23f;
    return -metal::log(-metal::log(u));
};
auto est_prob = [&](float q, uint pos) -> float {
    const float qc = clamp(q, 1e-6f, 1.0f - 1e-6f);
    const float lq = metal::log(qc / (1.0f - qc));
    const float slope = CALIB[b * (1u + R) + 0u];
    const float inter = CALIB[b * (1u + R) + 1u + pos];
    const float z = slope * lq + inter;
    return 1.0f / (1.0f + metal::exp(-z));
};

const float temp = TEMPS[b];
const bool sampled = temp > 0.0f;
const ulong seed0 = (ulong(SEEDS[b * 2u + 1u]) << 32) | ulong(SEEDS[b * 2u + 0u]);
const uint copy_n = uint(max(0, COPY_LEN[b]));
const float copy_r = COPY_RATE[b];
const uint n_stops = uint(PARAMS[0]);
const uint max_ch = uint(PARAMS[1]);
const uint want_w = min(uint(W), uint(PARAMS[2]));

auto is_stop = [&](int tok) -> bool {
    for (uint i = 0; i < n_stops; ++i) if (STOPS[i] == tok) return true;
    return false;
};

int r_tok[W];
int r_par[W];
int r_dep[W];
uint r_anc[W];
bool r_on_copy[W];
float r_q[W];
ulong r_key[W];
uchar r_draw[W][4];
uchar r_placed[W];
float r_next[W][K];

for (uint i = 0; i < W; ++i) {
    r_tok[i] = T1[b];
    r_par[i] = -1;
    r_dep[i] = 0;
    r_anc[i] = 1u;
    r_on_copy[i] = false;
    r_q[i] = -1.0f;
    r_key[i] = seed0;
    r_placed[i] = 0;
    for (uint c = 0; c < 4u; ++c) r_draw[i][c] = 255u;
    for (uint k = 0; k < K; ++k) r_next[i][k] = 0.0f;
}

// Max-heap over pending expansions (up to 96 entries).
constexpr uint HCAP = 96;
float h_score[HCAP];
uint h_parent[HCAP];
uint h_pos[HCAP];
uint h_idx[HCAP];   // candidate index in 0..K-1, or 255 for pure copy node
uint h_order[HCAP]; // draw rank 0..3 within parent (or 255 for copy)
uint h_size = 0;

auto h_push = [&](float sc, uint par, uint pos, uint idx, uint ord) {
    if (h_size >= HCAP) return;
    uint cur = h_size++;
    while (cur > 0u) {
        const uint p = (cur - 1u) >> 1;
        if (h_score[p] >= sc) break;
        h_score[cur] = h_score[p];
        h_parent[cur] = h_parent[p];
        h_pos[cur] = h_pos[p];
        h_idx[cur] = h_idx[p];
        h_order[cur] = h_order[p];
        cur = p;
    }
    h_score[cur] = sc;
    h_parent[cur] = par;
    h_pos[cur] = pos;
    h_idx[cur] = idx;
    h_order[cur] = ord;
};

auto expand_node = [&](uint row, uint pos, uint pred_i, float path_score) {
    if (pos >= R) return;
    const size_t p_off = (((size_t(b) * R + pos) * K) + pred_i) * K;
    const size_t c_off = (size_t(b) * R + pos) * K;
    float q[K];
    float top = 0.0f;
    for (uint i = 0; i < K; ++i) {
        q[i] = PROBS[p_off + i];
        top = max(top, q[i]);
    }
    float mass = 0.0f;
    for (uint i = 0; i < K; ++i) {
        const float wt = (!sampled || !(q[i] > 0.0f)) ? q[i] : metal::exp((metal::log(q[i]) - metal::log(top)) / temp);
        r_next[row][i] = wt;
        mass += wt;
    }
    if (mass > 0.0f) for (uint i = 0; i < K; ++i) r_next[row][i] /= mass;

    float gkey[K];
    if (sampled) {
        for (uint i = 0; i < K; ++i) {
            gkey[i] = (r_next[row][i] > 0.0f) ? (metal::log(r_next[row][i]) + gumbel(r_key[row], uint(CANDS[c_off + i]))) : -INFINITY;
        }
    }

    const bool has_copy = r_on_copy[row] && pos < copy_n;
    const int copy_tok = has_copy ? COPY_IDS[b * MAX_COPY + pos] : -1;
    const float c_rate = has_copy ? copy_r : 0.0f;
    uint taken = 0u;
    float bound = path_score;
    float left = 1.0f - c_rate;
    float reserved = c_rate;
    bool copy_joined = false;
    const uint ch_lim = sampled ? min(max_ch, 4u) : min(max_ch, 4u);
    for (uint ch = 0; ch < ch_lim; ++ch) {
        uint picked = K;
        if (!sampled) {
            for (uint i = 0; i < K; ++i) {
                if ((taken & (1u << i)) == 0u && q[i] > 0.0f) {
                    if (picked == K || q[i] > q[picked]) picked = i;
                }
            }
        } else {
            for (uint i = 0; i < K; ++i) {
                if ((taken & (1u << i)) == 0u && gkey[i] > -INFINITY) {
                    if (picked == K || gkey[i] > gkey[picked]) picked = i;
                }
            }
        }
        if (picked == K) break;
        taken |= (1u << picked);
        const int tok = CANDS[c_off + picked];
        // Greedy never verifies a stop token (the trunk emits it as next_token); sampled keeps it for stopPending.
        if (!sampled && is_stop(tok)) continue;
        const bool joins = has_copy && (tok == copy_tok);
        if (joins) copy_joined = true;
        const float cap = joins ? (left + reserved) : left;
        const float own = min(est_prob(r_next[row][picked], pos), max(cap, 0.0f));
        const float eff = joins ? max(own, reserved) : own;
        left = joins ? max(0.0f, cap - eff) : max(0.0f, left - own);
        if (joins) reserved = 0.0f;
        bound = min(bound, path_score * eff);
        r_draw[row][ch] = uchar(picked);
        h_push(bound, row, pos, picked, ch);
    }
    if (has_copy && !copy_joined && (!sampled ? !is_stop(copy_tok) : true)) {
        // Only add pure copy nodes for greedy or when not sampling (sampled tree only verifies drawn children).
        if (!sampled) h_push(path_score * c_rate, row, pos, 255u, 255u);
    }
};

// Root = row 0 (t1).
r_tok[0] = T1[b];
r_par[0] = -1;
r_dep[0] = 0;
r_anc[0] = 1u;
r_on_copy[0] = (copy_n > 0u);
r_q[0] = 1.0f;
r_key[0] = seed0;

uint made = 1u;
if (want_w > 1u) expand_node(0u, 0u, 0u, 1.0f);

while (h_size > 0u && made < want_w) {
    // Pop max.
    const float sc = h_score[0];
    const uint par = h_parent[0];
    const uint pos = h_pos[0];
    const uint idx = h_idx[0];
    const uint ord = h_order[0];
    --h_size;
    if (h_size > 0u) {
        const float ls = h_score[h_size];
        const uint lp = h_parent[h_size];
        const uint lpos = h_pos[h_size];
        const uint li = h_idx[h_size];
        const uint lo = h_order[h_size];
        uint cur = 0u;
        while (true) {
            uint l = cur * 2u + 1u;
            if (l >= h_size) break;
            uint r = l + 1u;
            uint best = (r < h_size && h_score[r] > h_score[l]) ? r : l;
            if (ls >= h_score[best]) break;
            h_score[cur] = h_score[best];
            h_parent[cur] = h_parent[best];
            h_pos[cur] = h_pos[best];
            h_idx[cur] = h_idx[best];
            h_order[cur] = h_order[best];
            cur = best;
        }
        h_score[cur] = ls;
        h_parent[cur] = lp;
        h_pos[cur] = lpos;
        h_idx[cur] = li;
        h_order[cur] = lo;
    }
    // For sampled trees, children of a node must be placed in draw order (exact without-replacement prefix).
    if (sampled && ord != 255u && ord != uint(r_placed[par])) continue;
    const uint row = made++;
    const bool is_copy = (idx == 255u);
    const int tok = is_copy ? COPY_IDS[b * MAX_COPY + pos] : CANDS[(size_t(b) * R + pos) * K + idx];
    r_tok[row] = tok;
    r_par[row] = int(par);
    r_dep[row] = int(pos + 1u);
    r_anc[row] = r_anc[par] | (1u << row);
    r_on_copy[row] = r_on_copy[par] && (pos < copy_n) && (tok == COPY_IDS[b * MAX_COPY + pos]);
    r_q[row] = is_copy ? -1.0f : r_next[par][idx];
    r_key[row] = node_key(r_key[par], uint(tok));
    if (ord != 255u) r_placed[par] = uchar(ord + 1u);

    if (!is_stop(tok)) {
        if (!is_copy && pos + 1u < R) {
            expand_node(row, pos + 1u, idx, sc);
        } else if (r_on_copy[row] && pos + 1u < copy_n && !sampled) {
            const int ntok = COPY_IDS[b * MAX_COPY + pos + 1u];
            if (!is_stop(ntok)) h_push(sc * copy_r, row, pos + 1u, 255u, 255u);
        }
    }
}

// Clear unplaced draw slots so sampled rejection tests ONLY the placed prefix 0..r_placed[row]-1.
for (uint row = 0; row < W; ++row) {
    for (uint ch = uint(r_placed[row]); ch < 4u; ++ch) r_draw[row][ch] = 255u;
}

// Write outputs.
for (uint row = 0; row < W; ++row) {
    const size_t at = size_t(b) * W + row;
    TOKENS[at] = r_tok[row];
    PARENT[at] = r_par[row];
    TMETA[at * 2u + 0u] = r_dep[row];
    TMETA[at * 2u + 1u] = int(r_anc[row]);
    SCORES[at] = r_q[row];
    for (uint ch = 0; ch < 4u; ++ch) DRAWS[at * 4u + ch] = r_draw[row][ch];
    for (uint k = 0; k < K; ++k) NEXT_Q[at * K + k] = r_next[row][k];
}
