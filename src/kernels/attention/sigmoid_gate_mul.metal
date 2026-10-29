uint i = thread_position_in_grid.x;
if (i >= uint(N_size)) return;
T sig = sigtab[as_type<ushort>(gate[i / uint(GROUP)])];
y[i] = x[i] * sig;