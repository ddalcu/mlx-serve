// Inkling 2: gather a slot's new conv state [1, KT - 1, SW] f32 from an open verify window WIN [NSLOT, KT - 1 + RPS, SW]
// along the accepted tree path indices IDX [KT - 1] int32 (each entry in 0 .. KT - 1 + RPS - 1).
// grid ((KT - 1) * SW, 1, 1), threadgroup (64, 1, 1).
const uint e = thread_position_in_grid.x;
const uint SW = WIN_shape[2];
const uint row = e / SW;
const uint col = e % SW;
const uint src = uint(IDX[row]);
const uint stride = WIN_shape[1];
NST[e] = WIN[(size_t(SLOT) * stride + src) * SW + col];
