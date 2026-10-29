const bool upper = threadgroup_position_in_grid.z != 0u;
const device half* x = upper ? xu : xg;
const device ushort* trellis = upper ? tu : tg;
device half* y = upper ? yu : yg;
