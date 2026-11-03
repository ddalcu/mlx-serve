threadgroup float stage[9216];
const uint r = thread_position_in_grid.x;
stage[r] = x[r];
threadgroup_barrier(mem_flags::mem_threadgroup);
y[r] = stage[r];