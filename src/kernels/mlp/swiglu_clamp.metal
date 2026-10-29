uint i = thread_position_in_grid.x;
if (i >= uint(N_size)) return;
T g = metal::min(gate[i], lim);
T u = metal::max(metal::min(up[i], lim), T(-lim));
T sig = sigtab[as_type<ushort>(g)];
T act = g * sig;
y[i] = act * u;