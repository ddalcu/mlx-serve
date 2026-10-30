const uint slot = uint(threadgroup_position_in_grid.y);
const uint sg = uint(threadgroup_position_in_grid.x);
const uint lane = uint(thread_index_in_simdgroup);
const uint eid = uint(slots[slot]);
constexpr uint SGS = uint(IDIM) / 128u;