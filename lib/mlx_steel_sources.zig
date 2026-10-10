// MLX's steel NAX tile headers, embedded for the kernels that reuse its tile
// matmul in their own `metal_kernel` source (src/lane_qmm.zig). Order matters:
// each header only uses the ones before it; their `#include "mlx/...` lines are
// stripped by the consumer.

pub const defines: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/defines.h");
pub const type_traits: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/utils/type_traits.h");
pub const integral_constant: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/utils/integral_constant.h");
pub const nax: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/gemm/nax.h");

/// MLX kernel headers by their `#include "mlx/..."` path, for src/jangtq2.zig, which inlines them
/// itself: every file reachable from utils.h, steel/gemm/{gemm,nax,loader}.h and quantized{,_utils,_nax}.h.
pub const Header = struct { path: []const u8, src: []const u8 };
pub const kernel_headers = [_]Header{
    header("utils.h"),
    header("bf16.h"),
    header("bf16_math.h"),
    header("complex.h"),
    header("defines.h"),
    header("logging.h"),
    header("steel/gemm/gemm.h"),
    header("steel/gemm/loader.h"),
    header("steel/defines.h"),
    header("steel/gemm/mma.h"),
    header("steel/gemm/transforms.h"),
    header("steel/utils.h"),
    header("steel/utils/integral_constant.h"),
    header("steel/utils/type_traits.h"),
    header("steel/gemm/params.h"),
    header("steel/gemm/nax.h"),
    header("quantized_nax.h"),
    header("quantized_utils.h"),
    header("quantized.h"),
};

fn header(comptime rel: []const u8) Header {
    return .{ .path = "mlx/backend/metal/kernels/" ++ rel, .src = @embedFile("mlx-src/mlx/backend/metal/kernels/" ++ rel) };
}
