const uint r = thread_position_in_grid.x;
float s = 0.0f;
int i = 0;
for (; i + 32 <= K; i += 32)
    for (int j = 0; j < 32; j++) s += x[r * K + i + j];
for (; i < K - 1; i++) s += x[r * K + i];
y[r] = s;