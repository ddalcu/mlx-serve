//! The hybrid base: one operation per layer (gated conv, Mamba2, attention, an MLP or an MoE: an `Either` of
//! blocks), optionally followed by its own MLP, each layer closing on a seam that carries the next layer's norm when
//! it fuses it. An operation block may fold its output into the closing seam (`fold`), as Nemotron-H's MoE combine
//! does. The final norm is optional.

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

/// What an operation's `fold` returns: the new residual, and the next layer's normed input when it made it.
pub const Fold = block.End;

/// `spec`: `.embed`, `.norm`, `.op` (a mixer or MLP block, or an `Either` of them), `.mlp`, `.seam`, `.head`, and
/// optionally `.hooks` as in the decoder base. An `.op` may declare
/// `fold(w, c, normed, h, next_norm: Array) !?Fold`: the layer's whole closing seam, or null to decline.
pub fn Hybrid(comptime spec: anytype) type {
    const Embed = spec.embed;
    const Norm = spec.norm;
    const Op = spec.op;
    const Mlp = spec.mlp;
    const Seam = spec.seam;
    const Head = spec.head;
    const Hooks = if (@hasField(@TypeOf(spec), "hooks")) spec.hooks else struct {};
    comptime {
        block.check(Embed, &.{.embed});
        block.check(Norm, &.{.norm});
        block.check(Op, &.{ .mixer, .mlp });
        block.check(Mlp, &.{.mlp});
        block.check(Seam, &.{.seam});
        block.check(Head, &.{.head});
    }
    const Extra = if (@hasDecl(Hooks, "Extra")) Hooks.Extra else void;
    const per_layer = @hasDecl(Hooks, "perLayer");

    return struct {
        pub const batches_decode = block.allBatch(.{ Embed, Norm, Op, Mlp, Seam, Head });

        pub const Layer = struct {
            input_norm: Norm.Weights,
            op: Op.Weights,
            /// The MLP's input norm and the MLP; both null on an operation-only layer.
            post_norm: ?Norm.Weights = null,
            mlp: ?Mlp.Weights = null,
        };

        pub const Weights = struct {
            embed: Embed.Weights,
            layers: []const Layer,
            final_norm: ?Norm.Weights = null,
            head: Head.Weights,
            extra: Extra = if (Extra == void) {} else undefined,
        };

        pub fn forward(w: *const Weights, shared: *const block.Shared, g: *Graph, ids: Array, run: Run) !Out {
            if (run.mode == .batched_rows and !batches_decode) return error.BatchedRowsUnsupported;
            var c = run.ctx(g, shared);
            var memo: block.Memo = .{ .g = g };
            c.memo = &memo;
            var h = try run.input(g, try Embed.forward(&w.embed, &c, ids), ids);
            if (@hasDecl(Hooks, "afterEmbed")) h = try Hooks.afterEmbed(&w.extra, &c, h);
            var dt = mlx.DtypeTrace.begin("hybrid", h, null);

            var carried: ?Array = null;
            for (w.layers, 0..) |*L, i| {
                var lg = g.scope();
                defer lg.deinit();
                c.g = &lg;
                c.layer = @intCast(i);
                const normed = carried orelse try Norm.forward(&L.input_norm, &c, h);
                const next: ?*const Norm.Weights = if (per_layer) null else if (i + 1 < w.layers.len) &w.layers[i + 1].input_norm else if (w.final_norm) |*f| f else null;
                var end = try closeLayer(L, &c, normed, h, next);
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
            var final = if (w.final_norm) |*f| (carried orelse try Norm.forward(f, &c, h)) else h;
            if (@hasDecl(Hooks, "beforeHead")) final = try Hooks.beforeHead(&w.extra, &c, final);
            var out = try run.finish(g, &c, final, Head, &w.head);
            // `Hooks.residual_hidden`: a hidden captured beside the logits is the residual before the final norm; one
            // that replaces them (`logits = .none`) stays final-normed.
            if (@hasDecl(Hooks, "residual_hidden") and run.want_hidden and run.logits != .none) out.hidden = h;
            return out;
        }

        fn closeLayer(L: *const Layer, c: *Ctx, normed: Array, h: Array, next: ?*const Norm.Weights) !block.End {
            if (L.mlp) |*mlp| {
                const out = try Op.forward(&L.op, c, normed);
                const mid = try Seam.mid(c, h, out, .{}, Norm, &L.post_norm.?);
                return Seam.end(c, mid.h, try Mlp.forward(mlp, c, mid.normed), .{}, Norm, next);
            }
            if (@hasDecl(Op, "fold")) if (next) |nw| if (Norm.fusable) {
                if (try Op.fold(&L.op, c, normed, h, nw.*)) |f| return f;
            };
            return Seam.end(c, h, try Op.forward(&L.op, c, normed), .{}, Norm, next);
        }

        pub fn tableForward(m: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
            const Self = @This();
            return run_mod.tableForward(Self, run_mod.Model(Self).of(m), f, out);
        }
    };
}
