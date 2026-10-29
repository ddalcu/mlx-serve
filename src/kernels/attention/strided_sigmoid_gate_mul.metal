uint i = thread_position_in_grid.x;
if (i >= uint(N_size)) return;
const uint D = x_shape[3], H = x_shape[2], S = x_shape[1];
const uint d = i % D, h = (i / D) % H, sq = (i / (D * H)) % S, b = i / (D * H * S);
const T xv = x[b * x_strides[0] + sq * x_strides[1] + h * x_strides[2] + d * x_strides[3]];
const T gv = gate[b * gate_strides[0] + sq * gate_strides[1] + h * gate_strides[2] + d * gate_strides[3]];
y[i] = xv * sigtab[as_type<ushort>(gv)];