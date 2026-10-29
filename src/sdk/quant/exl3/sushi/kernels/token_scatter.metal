uint col = uint(thread_position_in_grid.x);
uint slot = uint(thread_position_in_grid.y);
if (col >= uint(DIM)) return;
const uint orig = uint(order[slot]);
y[(size_t)orig * (size_t)(DIM) + col] = x[(size_t)slot * (size_t)(DIM) + col];