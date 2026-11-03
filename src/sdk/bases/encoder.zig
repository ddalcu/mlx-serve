//! The encoder base (post-norm, BERT-shaped): embed, then per layer `h = norm(h + attention(h))`,
//! `h = norm(h + mlp(h))`; the forward returns the hidden rows, which the engine pools (`EmbedFacts.pooling`), and an
//! optional projection head (`EmbedHooks.project`) maps pooled rows. Pre-norm bidirectional encoders (EmbeddingGemma)
//! are the decoder base with a bidirectional mixer and no head.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const abi = @import("../../abi/plugin.zig");
const block = @import("block.zig");
const run_mod = @import("run.zig");
const Projection = @import("blocks/head.zig").Projection;

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;

pub const Run = run_mod.Run;
pub const Out = run_mod.Out;

/// `spec`: `.embed`, `.attention` (a mixer reading no cache), `.mlp`, `.norm` (block types).
pub fn Encoder(comptime spec: anytype) type {
    const Embed = spec.embed;
    const Attn = spec.attention;
    const Mlp = spec.mlp;
    const Norm = spec.norm;
    comptime {
        block.check(Embed, &.{.embed});
        block.check(Attn, &.{.mixer});
        block.check(Mlp, &.{.mlp});
        block.check(Norm, &.{.norm});
    }
    return struct {
        pub const batches_decode = false;

        pub const Layer = struct {
            attention: Attn.Weights,
            attn_norm: Norm.Weights,
            mlp: Mlp.Weights,
            mlp_norm: Norm.Weights,
        };

        pub const Weights = struct {
            embed: Embed.Weights,
            layers: []const Layer,
            projection: ?Projection = null,
        };

        /// The hidden rows `[B, T, hidden]` of `ids` (`[B, T]`); `run.mask` keeps padded keys out.
        pub fn forward(w: *const Weights, shared: *const block.Shared, g: *Graph, ids: Array, run: Run) !Out {
            if (run.mode != .prefill) return error.EncoderPrefillOnly;
            var c = run.ctx(g, shared);
            var h = try Embed.forward(&w.embed, &c, ids);
            var dt = mlx.DtypeTrace.begin("encoder", h, null);
            for (w.layers, 0..) |*L, i| {
                var lg = g.scope();
                defer lg.deinit();
                c.g = &lg;
                c.layer = @intCast(i);
                const a = try Norm.forward(&L.attn_norm, &c, try lg.add(h, try Attn.forward(&L.attention, &c, h)));
                const m = try Norm.forward(&L.mlp_norm, &c, try lg.add(a, try Mlp.forward(&L.mlp, &c, a)));
                c.g = g;
                const old = h;
                h = try g.keep(m);
                g.drop(old);
                dt.layer(h, i);
                try run.capture(g, @intCast(i), h);
                try run.cadence(i, w.layers.len, h);
            }
            dt.end(h);
            return .{ .hidden = h };
        }

        /// Pooled rows `[B, hidden]` through the projection head, f32; the pooled rows unchanged without one.
        pub fn project(w: *const Weights, g: *Graph, pooled: Array) !Array {
            const p = w.projection orelse return pooled;
            return p.forward(g, pooled);
        }

        pub fn tableForward(m: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
            const Self = @This();
            return run_mod.tableForward(Self, run_mod.Model(Self).of(m), f, out);
        }
    };
}
