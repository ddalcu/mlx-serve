//! Nemotron-H's MoE (mlx-lm `NemotronHMoE`): a sigmoid router with a selection-only bias (`groupLimitedRouting`),
//! ReLU² experts read in place for a short window (`gatherQmv`, ReLU² fused) or through one sorted gather past it,
//! the f32 routing weights applied before the K-sum, then the ungated shared expert. Puzzle trunks run the experts
//! in a latent width and route on the full hidden. At seam width the combine folds into the residual add and the
//! next layer's norm (`fold`). copied_from: plugins/nemotron_h/moe.zig (`nemotronMoe`, `nemotronMoeExpertsRouted`,
//! `nemotronMoeDecodeExperts`, `nemotronSharedExpert`, `nemotronMoeCombine`, without the row-exact arms) and
//! engines/mlx/forward/hybrid.zig (the fused combine seam).

const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const block = @import("../block.zig");
const Linear = @import("linear.zig").Linear;
const row_exact = @import("row_exact.zig");
const seam = @import("seam.zig");
const add_norm = @import("../../blocks/norms/add_norm.zig");
const routing = @import("../../blocks/moe/routing.zig");
const gather_qmv = @import("../../blocks/moe/gather_qmv.zig");
const gatherExpertMm = @import("../../blocks/moe/experts.zig").gatherExpertMm;
const verifyQmmNaxAvailable = @import("../../blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;

/// Widest token window that reads its experts in place (decode steps, verify windows); wider windows sort.
pub const GATHER_MAX_ROWS: c_int = 8;

pub const NemotronMoe = struct {
    pub const kind: block.Kind = .mlp;
    pub const supports_batched_rows = false;

    pub const Routing = struct {
        k: c_int,
        n_group: c_int,
        topk_group: c_int,
        renorm: bool,
        scale: f32,
    };

    pub const Weights = struct {
        /// Dense router `[hidden, experts]` pre-transposed, or quantized.
        router: Linear,
        /// A dense router's stored `[experts, hidden]` rows, for the row-exact router; null = none.
        router_rows: Array = .{},
        /// f32 `[experts]`, added for selection only.
        expert_bias: Array,
        /// Expert banks `[E, inter, in]` / `[E, in, inter]`, each with its own quant geometry.
        fc1: Linear,
        fc2: Linear,
        shared_up: Linear = .{},
        shared_down: Linear = .{},
        /// The latent pair (Puzzle); null `w` = none.
        latent_in: Linear = .{},
        latent_out: Linear = .{},
        routing: Routing,
    };

    const Parts = struct { y: Array, scores: Array };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        if (c.mode == .batched_rows) return error.BatchedRowsUnsupported;
        const g = c.g;
        const lane = c.row_exact;
        if (w.latent_in.w.ctx == null) return combine(w, lane, g, x, try experts(w, lane, g, x, x), x);
        const sh = ops.shape(x);
        const t = sh[0] * sh[1];
        const lat = try row_exact.linear(lane, g, &w.latent_in, try g.reshape(x, &.{ t, sh[2] }));
        const x_lat = try g.reshape(lat, &.{ sh[0], sh[1], ops.shape(lat)[1] });
        return combine(w, lane, g, x_lat, try experts(w, lane, g, x_lat, x), x);
    }

    /// The layer's closing seam at `AddNorm` width: combine + shared + residual add + the next norm in one dispatch
    /// where the kernel serves, else the combine and a plain add. Null (the latent trunk, or a wider window) = the
    /// base closes the layer as usual.
    pub fn fold(w: *const Weights, c: *Ctx, normed: Array, h: Array, next_norm: Array) !?block.End {
        if (w.latent_in.w.ctx != null or !seam.AddNorm.fuses(c, h)) return null;
        const g = c.g;
        const parts = try experts(w, c.row_exact, g, normed, normed);
        const sx = try sharedExpert(w, c.row_exact, g, normed);
        if (try add_norm.moeCombineAddNorm(h, parts.y, parts.scores, sx, next_norm, c.shared.eps_arr, g.s)) |fo| {
            const nh = g.adopt(fo.h) catch |e| {
                _ = mlx.mlx_array_free(fo.normed);
                return e;
            };
            return .{ .h = nh, .normed = try g.adopt(fo.normed) };
        }
        return .{ .h = try g.add(h, try combine(w, c.row_exact, g, normed, parts, normed)) };
    }

    /// The routed experts' outputs before the K-sum (`y` `[T, K, D]`) and the f32 routing weights `[T, K]`. The
    /// experts read `x`; the router reads `route_x`.
    fn experts(w: *const Weights, lane: row_exact.Lane, g: *Graph, x: Array, route_x: Array) !Parts {
        const sh = ops.shape(x);
        const t = sh[0] * sh[1];
        const d = sh[2];
        const k = w.routing.k;
        const x_flat = try g.reshape(x, &.{ t, d });
        const route_flat = if (route_x.ctx == x.ctx) x_flat else try g.reshape(route_x, &.{ t, ops.shape(route_x)[2] });
        const logits = (try row_exact.router(lane, g, w.router_rows, route_flat)) orelse try row_exact.linear(lane, g, &w.router, route_flat);
        const r = w.routing;
        const rt = try routing.groupLimitedRouting(logits, w.expert_bias, k, r.n_group, r.topk_group, r.renorm, r.scale, g.s);
        const scores = g.adopt(rt.norm_scores) catch |e| {
            _ = mlx.mlx_array_free(rt.inds);
            return e;
        };
        const inds = try g.adopt(rt.inds);
        const y = if (t <= GATHER_MAX_ROWS or (lane != .off and t <= row_exact.MAX_ROWS))
            try inPlace(w, lane, g, x_flat, inds, t, k, d)
        else
            try sorted(w, g, x_flat, inds, t, k, d);
        return .{ .y = y, .scores = scores };
    }

    /// Every (token, expert) pair one `gatherQmv` row (fc1 with ReLU² fused, then fc2); unsorted gather_qmm where a
    /// bank is outside the kernel. None of the sort machinery runs.
    fn inPlace(w: *const Weights, lane: row_exact.Lane, parent: *Graph, x_flat: Array, inds: Array, t: c_int, k: c_int, d: c_int) !Array {
        var sc = parent.scope();
        defer sc.deinit();
        const g = &sc;
        const pairs = t * k;
        const inds_flat = try g.reshape(inds, &.{pairs});
        if (lane != .off) if (try row_exact.experts(lane, g, x_flat, try g.astype(inds_flat, .uint32), k, &w.fc1, &w.fc2)) |y|
            return parent.keep(try g.reshape(y, &.{ t, k, d }));
        if (w.fc1.s.ctx != null and w.fc2.s.ctx != null) fused: {
            const inds_u32 = try g.astype(inds_flat, .uint32);
            const x_in = if (t == 1)
                try g.reshape(x_flat, &.{d})
            else
                try g.reshape(try g.broadcast(try g.reshape(x_flat, &.{ t, 1, d }), &.{ t, k, d }), &.{ pairs, d });
            const a = w.fc1;
            const act = try g.adopt((try gather_qmv.gatherQmvAct(g.s, x_in, a.w, a.s, a.b, inds_u32, a.bits, a.group_size, a.mode, t > 1, true)) orelse break :fused);
            const o = w.fc2;
            const y = try g.adopt((try gather_qmv.gatherQmv(g.s, act, o.w, o.s, o.b, inds_u32, o.bits, o.group_size, o.mode, true)) orelse break :fused);
            return parent.keep(try g.reshape(y, &.{ t, k, d }));
        }
        const x_rep = try g.reshape(x_flat, &.{ t, 1, 1, d });
        const inds_tk = try g.reshape(inds, &.{ t, k });
        const h = try gatherMm(g, x_rep, &w.fc1, .{}, inds_tk, false);
        const y = try gatherMm(g, try reluSq(g, h), &w.fc2, .{}, inds_tk, false);
        return parent.keep(try g.reshape(y, &.{ t, k, d }));
    }

    /// One global sort so gather_qmm streams each expert's rows once; a short sorted window is padded up to the
    /// expert-streaming kernel where that pays (`streamPadRows`).
    fn sorted(w: *const Weights, parent: *Graph, x_flat: Array, inds: Array, t: c_int, k: c_int, d: c_int) !Array {
        var sc = parent.scope();
        defer sc.deinit();
        const g = &sc;
        const total = t * k;
        var flat = try g.reshape(inds, &.{total});
        const n_experts = ops.shape(w.fc1.w)[0];
        const pad = if (w.fc1.s.ctx != null and verifyQmmNaxAvailable()) streamPadRows(total, n_experts) else 0;
        if (pad > 0) {
            // Pad rows sort past every real row: the inverse order slice drops them.
            const pad_e = try g.call(mlx.mlx_remainder, .{ try g.arange(0, @floatFromInt(pad), 1, .int32), try int(g, n_experts) });
            flat = try g.concat(&.{ flat, try g.astype(pad_e, ops.dtype(flat)) }, 0);
        }
        const order = try g.argsort(flat, 0);
        var inv_order = try g.argsort(order, 0);
        const sorted_inds = try g.take(flat, order, 0);
        var lhs = try g.call(mlx.mlx_floor_divide, .{ order, try int(g, k) });
        if (pad > 0) lhs = try g.minimum(lhs, try int(g, t - 1));
        const x_rep = try g.reshape(try g.take(x_flat, lhs, 0), &.{ total + pad, 1, d });
        const h = try gatherMm(g, x_rep, &w.fc1, .{}, sorted_inds, true);
        const y = try gatherMm(g, try reluSq(g, h), &w.fc2, .{}, sorted_inds, true);
        if (pad > 0) inv_order = try g.sliceAxis(inv_order, 0, 0, total);
        return parent.keep(try g.reshape(try g.take(y, inv_order, 0), &.{ t, k, d }));
    }

    /// `(y * scores).sum(K)` in f32, back in y's dtype, the latent out-projection, plus the shared expert on
    /// `shared_x`: `[B, S, D]`.
    fn combine(w: *const Weights, lane: row_exact.Lane, parent: *Graph, x: Array, p: Parts, shared_x: Array) !Array {
        var sc = parent.scope();
        defer sc.deinit();
        const g = &sc;
        const sh = ops.shape(x);
        const summed = try g.sum(try g.mul(p.y, try g.expandDims(p.scores, -1)), 1, false);
        var routed = try g.astype(summed, ops.dtype(p.y));
        if (w.latent_out.w.ctx != null) routed = try row_exact.linear(lane, g, &w.latent_out, routed);
        const combined = if (try sharedExpert(w, lane, g, shared_x)) |sx| try g.add(routed, sx) else routed;
        return parent.keep(try g.reshape(combined, &.{ sh[0], sh[1], ops.shape(routed)[1] }));
    }

    /// `down(relu(up(x))^2)` over x `[B, S, hidden]` as `[B*S, hidden]`; null without a shared expert.
    fn sharedExpert(w: *const Weights, lane: row_exact.Lane, g: *Graph, x: Array) !?Array {
        if (w.shared_up.w.ctx == null) return null;
        const sh = ops.shape(x);
        const x_flat = try g.reshape(x, &.{ sh[0] * sh[1], sh[2] });
        const act = (try row_exact.reluSqUp(lane, g, &w.shared_up, x_flat)) orelse try reluSq(g, try row_exact.linear(lane, g, &w.shared_up, x_flat));
        return try row_exact.linear(lane, g, &w.shared_down, act);
    }
};

/// Rows to append so a sorted quantized gather reaches the expert-streaming kernel (4+ rows per expert); below 2
/// rows per expert the re-reads stay cheaper than the pad.
fn streamPadRows(rows: c_int, experts: c_int) c_int {
    return if (rows >= 2 * experts and rows < 4 * experts) 4 * experts - rows else 0;
}

fn int(g: *Graph, v: c_int) !Array {
    return g.adopt(mlx.mlx_array_new_int(v));
}

fn reluSq(parent: *Graph, x: Array) !Array {
    var sc = parent.scope();
    defer sc.deinit();
    const g = &sc;
    return parent.keep(try g.square(try g.maximum(x, try g.scalar(0, .bfloat16))));
}

fn gatherMm(g: *Graph, x: Array, bank: *const Linear, lhs: Array, rhs: Array, is_sorted: bool) !Array {
    var r = mlx.mlx_array_new();
    gatherExpertMm(&r, x, bank.w, bank.s, bank.b, lhs, rhs, bank.bits, bank.group_size, bank.mode, is_sorted, g.s) catch |e| {
        _ = mlx.mlx_array_free(r);
        return e;
    };
    return g.adopt(r);
}
