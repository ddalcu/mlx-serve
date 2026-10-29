const uint k = thread_position_in_grid.x;
const uint m = thread_position_in_grid.y;
const size_t i = size_t(m) * x_shape[1] + k;
out[i] = m < uint(x_shape[0]) ? float(x[i]) : 0.0f;