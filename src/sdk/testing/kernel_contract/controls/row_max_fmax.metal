const uint r = thread_position_in_grid.x;
float m = -INFINITY;
for (int i = 0; i < K; i++) m = fmax(m, x[r * K + i]);
y[r] = m;