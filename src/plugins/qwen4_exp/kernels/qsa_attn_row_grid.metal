
const int qL = q_shape[2];
const int kL = kq_shape[2];
const int Hq = q_shape[1];
const int Hk = kq_shape[1];
const int gqa = Hq / Hk;
const int KB = blocks_shape[2];

const int s = int(threadgroup_position_in_grid.x);
const int hk = int(threadgroup_position_in_grid.y);
// One view pins batch to 1 (qs[0] != n declines), so the grid's z is
// free to carry the split index instead of the batch index.
const int bb = 0;
const int jsp = int(threadgroup_position_in_grid.z);
