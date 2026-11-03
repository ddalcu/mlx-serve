const int k = int(thread_position_in_grid.x);
const uint row = thread_position_in_grid.y;
h[row * HC * D + k] = T(HVAL(k));
