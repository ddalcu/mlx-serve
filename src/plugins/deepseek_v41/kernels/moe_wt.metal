// A sorted pair's activation times its routing weight: the f32 product rounded once to T, as MLX's promoted multiply
// then cast gives it, in one pass.
const uint i = thread_position_in_grid.x;
const uint row = i / N;
if (row >= uint(a_shape[0])) return;
out[i] = T(float(a[i]) * wts[row]);
