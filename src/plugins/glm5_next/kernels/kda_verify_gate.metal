// A KDA layer's bounded forget gate (sdk/blocks/kda/gate.zig `kdaGateChain`) in one launch, each element through the
// MLX kernels the chain runs, in their operand order and types: Add in AddT, astype to f32, Multiply by exp(A_log),
// unary_ops.h Sigmoid (precise exp), Multiply by the f32 bound, Exp (precise). Grid (N, rows, 1).
const uint c = thread_position_in_grid.x;
const uint t = thread_position_in_grid.y;
const size_t at = size_t(t) * N + c;
const AddT added = static_cast<AddT>(a[at]) + static_cast<AddT>(dt[c]);
const float scaled = ea[c] * static_cast<float>(added);
const float sy = 1 / (1 + metal::precise::exp(metal::abs(scaled)));
const float sg = (scaled < 0) ? sy : 1 - sy;
g[at] = metal::precise::exp(sg * bound);
