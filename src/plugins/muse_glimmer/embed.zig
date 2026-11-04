//! Muse's token embedding: the table's rows, then a weight-less RMS norm (the reference's
//! `MuseGlimmerTextNormedEmbedding`, deliberately not folded into the table).
//! copied_from: sdk/bases/blocks/embed.zig (`TableEmbed`), sdk/blocks/embedding.zig (`embedding`'s normed arm).

const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const block = @import("../../sdk/bases/block.zig");
const embed = @import("../../sdk/bases/blocks/embed.zig");
const Linear = @import("../../sdk/bases/blocks/linear.zig").Linear;

const Array = ops.Array;

pub const NormedEmbed = struct {
    pub const kind: block.Kind = .embed;
    pub const supports_batched_rows = true;
    pub const Weights = struct {
        table: Linear,
        hidden: c_int,
        act: mlx.mlx_dtype = .bfloat16,
        /// bf16 ones `[hidden]`: the norm's weight.
        ones: Array,
    };

    pub fn forward(w: *const Weights, c: *block.Ctx, ids: Array) !Array {
        const g = c.g;
        const sh = ops.shape(ids);
        const flat = try g.reshape(ids, &.{sh[0] * sh[1]});
        const rows = try embed.gatherRows(g, &w.table, flat, if (w.table.s.ctx == null) .bfloat16 else w.act);
        return g.rmsNorm(try g.reshape(rows, &.{ sh[0], sh[1], w.hidden }), w.ones, c.shared.eps);
    }
};
