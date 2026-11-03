//! The server's own foreign formats, in the order a weight is tried. The plugin kit stages an empty list in place of this
//! file (build/kit.zig), so the blocks it exposes reach none of these formats' code.

const mlx = @import("../../mlx/ffi.zig");
const gguf = @import("gguf/root.zig");
const fp8_block128 = @import("fp8_block128.zig");
const sushi_pack = @import("exl3/sushi_pack.zig");
const Format = @import("foreign.zig").Format;

const Gguf = struct {
    fn claims(w: mlx.mlx_array, sc: mlx.mlx_array) bool {
        return gguf.kernels.infoOf(w, sc) != null;
    }
    fn linear(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, s: mlx.mlx_stream) anyerror!mlx.mlx_array {
        return gguf.kernels.linear(gguf.kernels.infoOf(w, sc).?, x, w, s);
    }
    fn gather(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, lhs: mlx.mlx_array, rhs: mlx.mlx_array, s: mlx.mlx_stream) anyerror!mlx.mlx_array {
        return gguf.kernels.gatherLinear(gguf.kernels.infoOf(w, sc).?, x, w, lhs, rhs, s);
    }
};

const Fp8Block128 = struct {
    fn linear(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, s: mlx.mlx_stream) anyerror!mlx.mlx_array {
        return fp8_block128.linear(x, w, sc, s);
    }
};

const SushiRowFp8 = struct {
    fn linear(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, s: mlx.mlx_stream) anyerror!mlx.mlx_array {
        return sushi_pack.fp8Linear(x, w, sc, s);
    }
};

pub const formats = [_]Format{
    .{ .name = "gguf", .claims = &Gguf.claims, .linear = &Gguf.linear, .gather = &Gguf.gather },
    .{ .name = "fp8_block128", .claims = &fp8_block128.isBlockScaled, .linear = &Fp8Block128.linear },
    .{ .name = "sushi_row_fp8", .claims = &sushi_pack.isRowScaledFp8, .linear = &SushiRowFp8.linear },
};
