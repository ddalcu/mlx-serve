const uint i = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
if (i >= uint(DIM) / 4u) return;
const uint g = meta[uint(eids[row]) * 8u + 3u];
const size_t at = (size_t)row * (uint(DIM) / 4u) + i;
