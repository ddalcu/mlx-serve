const uint N = Y_shape[1];
const size_t i = size_t(thread_position_in_grid.x) * 4;
if (i >= size_t(Y_shape[0]) * N) return;
const vec<T, 4> y = *(const device vec<T, 4>*)(Y + i);
const uint c = uint(i % N);
const vec<T, 4> b = vec<T, 4>(B[c], B[c + 1], B[c + 2], B[c + 3]);
vec<T, 4> r;
for (int e = 0; e < 4; e++) {
    const float x = float(T(float(y[e]) + float(b[e])));
    const float a = 0.7978845608f * (x + 0.044715f * x * x * x);
    r[e] = T(x / (1.0f + metal::precise::exp(-2.0f * a)));
}
*(device vec<T, 4>*)(O + i) = r;
