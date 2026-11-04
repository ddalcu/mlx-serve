const uint d = thread_position_in_grid.x;
const uint t = thread_position_in_grid.y;
const uint h = thread_position_in_grid.z;
const int n_tiles = meta[3];
const int n = vbs[t];
const device T* src = x + (size_t(h) * n_tiles + t) * TILE * 128 + d;
float acc = 0.0f;
for (int j = 0; j < n; j++) acc += float(src[size_t(j) * 128]);
out[(size_t(h) * n_tiles + t) * 128 + d] = acc / float(n);