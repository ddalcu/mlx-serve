const uint t = thread_position_in_grid.x;
for (int r = int(t) * 4; r < int(t) * 4 + 4 && r < ROWS; r++) {
    float s = 0.0f;
    for (int i = 0; i < K; i++) s += x[r * K + i];
    y[r] = s;
}