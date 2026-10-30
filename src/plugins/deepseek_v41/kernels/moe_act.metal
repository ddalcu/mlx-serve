const uint i = thread_position_in_grid.x;
const uint row = i / N, c = i % N;
if (row >= uint(y_shape[0])) return;
const device T* yr = y + size_t(row) * 2 * N;
float g = float(yr[c]), u = float(yr[N + c]);
if (lim[0] > 0.0f) { g = metal::min(g, lim[0]); u = metal::clamp(u, -lim[0], lim[0]); }
const float s = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
out[i] = T(g * (g < 0.0f ? s : 1.0f - s) * u * wts[row]);