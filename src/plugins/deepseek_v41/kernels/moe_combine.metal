const uint i = thread_position_in_grid.x;
const uint row = i / D, c = i % D;
if (row >= uint(sh_shape[0])) return;
float acc = 0.0f;
#if TOK
// MoeParts.tok: the down rows in pair order, read in place.
for (int j = 0; j < K; ++j) acc += float(dn[size_t(row * K + j) * D + c]);
#else
for (int j = 0; j < K; ++j) acc += float(dn[size_t(inv[row * K + j]) * D + c]);
#endif
out[i] = T(acc + float(sh[i]));