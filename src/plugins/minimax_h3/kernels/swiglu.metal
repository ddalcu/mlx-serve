const uint idx = thread_position_in_grid.x;
const uint f4 = uint(F) / 4;
const uint row = idx / f4;
const uint c = (idx - row * f4) * 4;
const size_t g0 = size_t(row) * (2 * uint(F)) + c;
for (uint i = 0; i < 4; ++i) {
    const T g = Y[g0 + i];
    const T u = Y[g0 + uint(F) + i];
    const T sg = msv_sigmoid<T>(g);
    const T gs = g * sg;
    OUT[size_t(row) * uint(F) + c + i] = gs * u;
}