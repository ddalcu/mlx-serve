//! `fp8_block128`'s claim over the MLX bindings alone (no kernel, no format table): U8 e4m3 codes beside an F32 scale
//! grid of one scale per 64 output rows x 128 input columns (`fp8_block128_gemv.zig`'s `scale_rows`, `scale_cols`), so a
//! base reads it without the format.

const mlx = @import("../../mlx/ffi.zig");
const plan = @import("fp8_block128_gemv.zig");

pub const block_rows: u32 = plan.scale_rows;
pub const block_cols: u32 = plan.scale_cols;

/// U8 e4m3 codes `[out, in]` beside an F32 scale grid `[ceil(out / 64), ceil(in / 128)]` (Sushi's row-scaled e4m3 has
/// one scale row per code row).
pub fn isBlockScaled(w: mlx.mlx_array, sc: mlx.mlx_array) bool {
    if (w.ctx == null or sc.ctx == null) return false;
    if (mlx.mlx_array_dtype(w) != .uint8 or mlx.mlx_array_dtype(sc) != .float32) return false;
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    return ws.len == 2 and ss.len == 2 and ws[0] > 1 and
        ss[0] == @divFloor(ws[0] + block_rows - 1, block_rows) and ss[1] == @divFloor(ws[1] + block_cols - 1, block_cols);
}
