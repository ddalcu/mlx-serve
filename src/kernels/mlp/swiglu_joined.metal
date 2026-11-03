const uint i = thread_position_in_grid.x;
if (i >= uint(N_size)) return;
const uint r = i / uint(F), c = i % uint(F);
T g = both[r * uint(2 * F) + c];
T sig = sigtab[as_type<ushort>(g)];
T act = g * sig;
y[i] = act * both[r * uint(2 * F) + uint(F) + c];