const uint r = thread_position_in_grid.x;
half s = 0.0h;
for (int i = 0; i < K; i++) s += half(x[r * K + i]) * half(w[i]);
y[r] = float(s);