//! The decoder base: embed, then per layer norm → mixer → seam (residual + the MLP's norm) → MLP → seam (residual +
//! the next layer's norm when the seam fuses it), then the final norm and the head. Composed at comptime from block
//! types; hooks run after the embedding, after each layer and before the head. Serves prefill, decode, verify and,
//! when every block does, batched rows.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const abi = @import("../../abi/plugin.zig");
const block = @import("block.zig");
const run_mod = @import("run.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;

pub const Run = run_mod.Run;
pub const Out = run_mod.Out;

/// A block's output and the norm the seam applies to it: a `PostNormed` block's own (folded into the seam's dispatch).
fn blockOut(comptime B: type, w: *const B.Weights, c: *Ctx, x: Array) !struct { x: Array, norm: Array } {
    if (@hasDecl(B, "Inner")) return .{ .x = try B.Inner.forward(&w.inner, c, x), .norm = w.norm };
    return .{ .x = try B.forward(w, c, x), .norm = .{} };
}

/// `spec`: `.embed`, `.norm`, `.mixer`, `.mlp`, `.seam`, `.head` (block types) and optionally `.hooks`, a type with
/// any of `Extra` (weights the hooks read), `afterEmbed`, `perLayer`, `beforeHead`, each
/// `fn (x: *const Extra, c: *Ctx, h: Array) !Array`. A `perLayer` hook rewrites the residual after the layer's last
/// seam, so the seam then never carries the next layer's norm.
pub fn Decoder(comptime spec: anytype) type {
    const Embed = spec.embed;
    const Norm = spec.norm;
    const Mixer = spec.mixer;
    const Mlp = spec.mlp;
    const Seam = spec.seam;
    const Head = spec.head;
    const Hooks = if (@hasField(@TypeOf(spec), "hooks")) spec.hooks else struct {};
    comptime {
        block.check(Embed, &.{.embed});
        block.check(Norm, &.{.norm});
        block.check(Mixer, &.{.mixer});
        block.check(Mlp, &.{.mlp});
        block.check(Seam, &.{.seam});
        block.check(Head, &.{.head});
    }
    const Extra = if (@hasDecl(Hooks, "Extra")) Hooks.Extra else void;
    const per_layer = @hasDecl(Hooks, "perLayer");

    return struct {
        /// Every block serves `batched_rows` (a plugin never declares it by hand).
        pub const batches_decode = block.allBatch(.{ Embed, Norm, Mixer, Mlp, Seam, Head });

        pub const Layer = struct {
            input_norm: Norm.Weights,
            mixer: Mixer.Weights,
            /// The MLP's input norm (`post_attention_layernorm`).
            post_norm: Norm.Weights,
            mlp: Mlp.Weights,
        };

        pub const Weights = struct {
            embed: Embed.Weights,
            layers: []const Layer,
            final_norm: Norm.Weights,
            head: Head.Weights,
            extra: Extra = if (Extra == void) {} else undefined,
        };

        /// One forward of `ids` (`[slots, rows]` int32). Results are owned by `g`.
        pub fn forward(w: *const Weights, shared: *const block.Shared, g: *Graph, ids: Array, run: Run) !Out {
            if (run.mode == .batched_rows and !batches_decode) return error.BatchedRowsUnsupported;
            var c = run.ctx(g, shared);
            var memo: block.Memo = .{ .g = g };
            c.memo = &memo;
            var h = try run.input(g, try Embed.forward(&w.embed, &c, ids), ids);
            if (@hasDecl(Hooks, "afterEmbed")) h = try Hooks.afterEmbed(&w.extra, &c, h);
            var dt = mlx.DtypeTrace.begin("decoder", h, null);

            var carried: ?Array = null;
            for (w.layers, 0..) |*L, i| {
                var lg = g.scope();
                defer lg.deinit();
                c.g = &lg;
                c.layer = @intCast(i);
                const normed = carried orelse try Norm.forward(&L.input_norm, &c, h);
                const mixed = try blockOut(Mixer, &L.mixer, &c, normed);
                const mid = try Seam.mid(&c, h, mixed.x, mixed.norm, Norm, &L.post_norm);
                const m = try blockOut(Mlp, &L.mlp, &c, mid.normed);
                const next: ?*const Norm.Weights = if (per_layer) null else if (i + 1 < w.layers.len) &w.layers[i + 1].input_norm else &w.final_norm;
                var end = try Seam.end(&c, mid.h, m.x, m.norm, Norm, next);
                if (per_layer) end.h = try Hooks.perLayer(&w.extra, &c, end.h);

                c.g = g;
                const old_h = h;
                h = try g.keep(end.h);
                g.drop(old_h);
                if (carried) |old| g.drop(old);
                carried = if (end.normed) |n| try g.keep(n) else null;
                dt.layer(h, i);
                try run.capture(g, @intCast(i), h);
                try run.cadence(i, w.layers.len, h);
            }
            dt.end(h);
            c.g = g;
            var final = carried orelse try Norm.forward(&w.final_norm, &c, h);
            if (@hasDecl(Hooks, "beforeHead")) final = try Hooks.beforeHead(&w.extra, &c, final);
            return run.finish(g, &c, final, Head, &w.head);
        }

        /// The model table's `forward` over a `Model` of this base.
        pub fn tableForward(m: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
            const Self = @This();
            return run_mod.tableForward(Self, run_mod.Model(Self).of(m), f, out);
        }
    };
}
