// A KDA layer's bounded forget gate (sdk/blocks/kda/gate.zig `kdaGateChain`) in one launch, each element through the
// MLX kernels the chain runs, in their operand order and types: Add in AddT, astype to f32, Multiply by exp(A_log),
// unary_ops.h Sigmoid (precise exp), Multiply by the f32 bound, Exp (precise). With EXP, `ea` is A_log and its
// exp(astype(A_log, f32)) is made here (Exp: precise exp). Grid (N, rows, 1).
const uint c = thread_position_in_grid.x;
const uint t = thread_position_in_grid.y;
const size_t at = size_t(t) * N + c;
const AddT added = static_cast<AddT>(a[at]) + static_cast<AddT>(dt[c]);
const float e = EXP ? metal::precise::exp(static_cast<float>(ea[c])) : static_cast<float>(ea[c]);
const float scaled = e * static_cast<float>(added);
const float sy = 1 / (1 + metal::precise::exp(metal::abs(scaled)));
const float sg = (scaled < 0) ? sy : 1 - sy;
g[at] = metal::precise::exp(sg * bound);
