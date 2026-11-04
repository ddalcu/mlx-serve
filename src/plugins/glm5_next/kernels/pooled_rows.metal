const uint d = thread_position_in_grid.x;
const int r = int(thread_position_in_grid.y);
const int b = int(thread_position_in_grid.z);
const int S = int(threads_per_grid.y);
const int pos0 = dims[0];
const int prev_len = dims[1];
const device T* crow = chunk + (size_t(b) * S + r) * (2 * DH);
device T* orow = out + (size_t(b) * S + r) * (2 * DH);
orow[d] = crow[d];
const int pos = pos0 + r;
if ((pos + 1) % KPOOL != 0) {
    orow[DH + d] = crow[DH + d];
    return;
}
float lg[KPOOL];
T kv[KPOOL];
float mx = -INFINITY;
for (int j = 0; j < KPOOL; ++j) {
    const int q = pos - KPOOL + 1 + j;
    const device T* row = q >= pos0 ? chunk + (size_t(b) * S + (q - pos0)) * (2 * DH)
                                    : prev + (size_t(b) * prev_len + (q - pos0 + prev_len)) * (2 * DH);
    kv[j] = row[d];
    lg[j] = float(row[DH + d]) + ape[j * DH + d];
    mx = metal::max(mx, lg[j]);
}
float sum = 0.0f;
for (int j = 0; j < KPOOL; ++j) { lg[j] = metal::precise::exp(lg[j] - mx); sum += lg[j]; }
float acc = 0.0f;
for (int j = 0; j < KPOOL; ++j) acc += float(T(T(lg[j] / sum) * kv[j]));
orow[DH + d] = T(acc);