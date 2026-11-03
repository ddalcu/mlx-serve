const uint N = Y_shape[1];
const size_t i = size_t(thread_position_in_grid.x) * 4;
if (i >= size_t(Y_shape[0]) * N) return;
const uint c = uint(i % N);
const vec<T, 4> x = *(const device vec<T, 4>*)(X + i);
const vec<T, 4> y = *(const device vec<T, 4>*)(Y + i);
vec<T, 4> r;
for (int e = 0; e < 4; e++) {
    // Each step rounds to T, as the chain's add, multiply and add do.
    const T yb = T(float(y[e]) + float(B[c + e]));
    const T yg = T(float(yb) * float(G[c + e]));
    r[e] = T(float(x[e]) + float(yg));
}
*(device vec<T, 4>*)(O + i) = r;
