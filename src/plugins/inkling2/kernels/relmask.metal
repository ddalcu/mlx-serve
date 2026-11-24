// Inkling 2, a few rows' additive attention mask: each row's relative bias where a key is visible, -inf elsewhere,
// and with SINK its sink logit in a last column (the key there is zero). A handful of rows reading a long cache are bound by
// the flash kernel's serial walk over the keys; MLX's attention under this mask is not.
// rel [1, L, HQ, R] (T) each row's bias at the distances below R; sinks [1, L, HQ] f32; sizes = { L, S } with the
// rows at the last L of the S keys. WINDOW > 0 keeps the keys fewer than WINDOW behind a row.
// grid (S + SINK rounded up to the threadgroup, L, HQ). M [1, HQ, L, S + SINK] (T).
const int L = sizes[0];
const int S = sizes[1];
const int key = thread_position_in_grid.x;
const int row = thread_position_in_grid.y;
const int head = thread_position_in_grid.z;
if (key >= S + SINK) {
    return;
}
T m = T(sinks[row * HQ + head]);
if (key < S) {
    const int d = S - L + row - key;
    const bool visible = d >= 0 && (WINDOW == 0 || d < WINDOW);
    m = visible ? (d < R ? rel[(row * HQ + head) * R + d] : T(0)) : T(-INFINITY);
}
M[(size_t(head) * L + row) * (S + SINK) + key] = m;
