const uint r = thread_position_in_grid.x;
float s = 0.0f;
for (int i = 1; i < K; i++) s += x[r * K + i];
y[r] = s;
const_cast<device float*>(x)[r * K] = s;