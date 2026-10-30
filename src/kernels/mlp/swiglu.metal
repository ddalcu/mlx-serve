uint i = thread_position_in_grid.x;
if (i >= uint(N_size)) return;
T g = gate[i];
T sig = sigtab[as_type<ushort>(g)];
// mlx Multiply (binary_ops.h), twice: silu(g) = g * sigmoid(g), then * up.
T act = g * sig;
y[i] = act * up[i];