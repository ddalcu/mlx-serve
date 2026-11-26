// Inkling 2: walk each lane's verified tree matching SELECTED[b, row] against child TOKENS, stopping before stop ids.
// grid (B, 1, 1), threadgroup (1, 1, 1).
// TOKENS [B, W] i32, PARENT [B, W] i32, SELECTED [B * W] i32, ACCEPT_CAP [B] u32, STOPS [MAX_STOPS] i32, N_STOPS [1] u32.
// RESULT [B, 2 + 2 * W] i32: { accepted, next_token, path_rows[0..W-1], emitted_tokens[0..W-1] }.
const uint b = threadgroup_position_in_grid.x;
const uint B = ACCEPT_CAP_shape[0];
if (b >= B) return;

const uint n_stops = N_STOPS[0];
auto is_stop = [&](int tok) -> bool {
    for (uint i = 0u; i < n_stops; ++i) if (STOPS[i] == tok) return true;
    return false;
};

const uint cap = min(ACCEPT_CAP[b], uint(W - 1));
const size_t base = size_t(b) * uint(W);
const size_t out = size_t(b) * uint(2 + 2 * W);

int path[W];
int emitted[W];
for (uint i = 0u; i < uint(W); ++i) { path[i] = 0; emitted[i] = 0; }

uint cur = 0u;
uint accepted = 0u;
path[0] = 0;
emitted[0] = TOKENS[base + 0u];
int next_tok = SELECTED[base + 0u];

while (true) {
    const int tok = SELECTED[base + cur];
    next_tok = tok;
    if (accepted >= cap || is_stop(tok)) break;
    uint matched = 0u;
    for (uint child = cur + 1u; child < uint(W); ++child) {
        if (PARENT[base + child] == int(cur) && TOKENS[base + child] == tok) {
            matched = child;
            break;
        }
    }
    if (matched == 0u) break;
    ++accepted;
    path[accepted] = int(matched);
    emitted[accepted] = tok;
    cur = matched;
}

RESULT[out + 0u] = int(accepted);
RESULT[out + 1u] = next_tok;
for (uint i = 0u; i < uint(W); ++i) {
    RESULT[out + 2u + i] = path[i];
    RESULT[out + 2u + uint(W) + i] = emitted[i];
}
