uint i = thread_position_in_grid.x;
if (i >= uint(N_size)) return;
T act = sigtab[as_type<ushort>(gate[i])];
y[i] = act * up[i];