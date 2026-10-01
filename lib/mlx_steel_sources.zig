// MLX's steel NAX tile headers, embedded for the kernels that reuse its tile
// matmul in their own `metal_kernel` source (src/lane_qmm.zig). Order matters:
// each header only uses the ones before it; their `#include "mlx/...` lines are
// stripped by the consumer.

pub const defines: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/defines.h");
pub const type_traits: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/utils/type_traits.h");
pub const integral_constant: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/utils/integral_constant.h");
pub const nax: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/gemm/nax.h");
