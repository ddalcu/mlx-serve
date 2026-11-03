const uint c = thread_position_in_grid.x * 4;
const uint r = thread_position_in_grid.y;
const int n_dense = shape[0];
const int n_tiles = shape[1];
const int n_video_rows = shape[2];
const int hd = shape[3];
const uint h = c / 128;
const uint dd = c % 128;
const int src = out_src[r];
const device T* a = src < n_dense ? dense + (size_t(h) * n_dense + src) * 128 + dd
    : video + (size_t(h) * n_video_rows + (src - n_dense)) * 128 + dd;
const device float* m = comp + (size_t(h) * n_tiles + row_tile[r]) * 128 + dd;
const device T* g = gate + size_t(r) * hd + c;
for (int i = 0; i < 4; i++) {
    const T gv = static_cast<T>(m[i] * float(g[i]));
    out[size_t(r) * hd + c + i] = static_cast<T>(float(a[i]) + float(gv));
}