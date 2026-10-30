//! copied_from: sdk/blocks/embedding.zig (`rawEmbedding`, `embedding` without the normed-embedding arm; `dequantTake`;
//! sdk/quant/hadamard.zig `unrotateEmbedding`). Token embeddings: a table gather, dequantized per row, optionally
//! scaled.

const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const block = @import("../block.zig");
const mlx_gguf = @import("../../quant/gguf/root.zig");
const rht = @import("../../quant/hadamard/rht.zig");
const Linear = @import("linear.zig").Linear;

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;

/// Rows `ids` (any shape, flattened by the caller) of `table`, as `dt`: a dense table casts, a quantized one
/// dequantizes only the gathered rows.
pub fn gatherRows(parent: *Graph, table: *const Linear, ids: Array, dt: mlx.mlx_dtype) !Array {
    var c = parent.scope();
    defer c.deinit();
    const g = &c;
    const taken = try g.take(table.w, ids, 0);
    if (table.s.ctx == null) return parent.keep(try g.astype(taken, dt));
    if (mlx_gguf.kernels.infoOf(table.w, table.s)) |info| return parent.adopt(try mlx_gguf.kernels.dequantize(info.ty, taken, dt, g.s));
    const ts = try g.take(table.s, ids, 0);
    const tb = if (table.b.ctx != null) try g.take(table.b, ids, 0) else Array{};
    return parent.keep(try g.call(mlx.mlx_dequantize, .{
        taken,
        ts,
        tb,
        mlx.mlx_optional_int.some(@intCast(table.group_size)),
        mlx.mlx_optional_int.some(@intCast(table.bits)),
        table.mode.cstr(),
        Array{},
        mlx.mlx_optional_dtype{ .value = dt, .has_value = true },
    }));
}

/// A Hadamard pack's gathered rows back to the residual basis (`signs * H(x)`); other tables' rows as they are.
pub fn unrotate(g: *Graph, table: *const Linear, rows: Array) !Array {
    const reg = table.rht orelse return rows;
    if (table.signs.ctx == null) return rows;
    return g.adopt(try rht.transform(rows, table.signs, reg.block, true, g.s));
}

/// A decoder's token embedding: `[slots, rows]` ids to `[slots, rows, hidden]`.
pub const TableEmbed = struct {
    pub const kind: block.Kind = .embed;
    pub const supports_batched_rows = true;
    pub const Weights = struct {
        table: Linear,
        hidden: c_int,
        /// The activation dtype a quantized table dequantizes to.
        act: mlx.mlx_dtype = .bfloat16,
        /// A multiplier on every row (Gemma's sqrt(hidden)); null = none.
        scale: Array = .{},
    };

    pub fn forward(w: *const Weights, c: *Ctx, ids: Array) !Array {
        const g = c.g;
        const sh = ops.shape(ids);
        const flat = try g.reshape(ids, &.{sh[0] * sh[1]});
        const rows = if (w.table.s.ctx == null)
            try gatherRows(g, &w.table, flat, .bfloat16)
        else
            try gatherRows(g, &w.table, flat, w.act);
        const emb = try unrotate(g, &w.table, try g.reshape(rows, &.{ sh[0], sh[1], w.hidden }));
        return if (w.scale.ctx != null) g.mul(emb, w.scale) else emb;
    }
};
