// Inkling 2, a few rows: the router's logits SC [S, E] f32, the MLP's normed input X [1, S, D] f32 over the
// router's rows RT [E, D] f32. MLX's f32 matmul at two rows and more is several times a one-row launch's latency.
// grid (32 * E, S), threadgroup (32, 1): a SIMD group sums one expert's row for one row of X, a lane every 32nd
// element.
const uint lane = thread_index_in_simdgroup;
const uint ex = thread_position_in_grid.x / 32;
const uint row = thread_position_in_grid.y;
float dot = 0.0f;
for (uint i = lane; i < D; i += 32) dot += X[row * D + i] * RT[ex * D + i];
dot = simd_sum(dot);
if (lane == 0) SC[row * E + ex] = dot;
