const uint r = thread_position_in_grid.x;
float s = 0.0f;
for (int i = 0; i < K; i++) s += x[r * K + i];
y[r] = s;
if (r == 0) const_cast<device float*>(x)[-1] = s;