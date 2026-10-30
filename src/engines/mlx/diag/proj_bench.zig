//! engines/mlx/diag/proj_bench.
//! Owns projLadderFits, ProjRung, ProjOrder, diagProjBench, projRungOnce.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const MoeLayerWeights = @import("../layer_weights.zig").MoeLayerWeights;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const Transformer = @import("../transformer.zig").Transformer;

/// Whether `diagProjBench`'s hand-rebuilt attention can actually reshape
/// this q_proj back. `q_shape` is the loaded (pre-transposed) [in, out]
/// matrix; `head_width` is `num_attention_heads * head_dim`, the width the
/// rungs reshape through.
pub fn projLadderFits(q_shape: []const c_int, head_width: c_int) bool {
    if (q_shape.len != 2) return false;
    return q_shape[1] == head_width;
}

/// How much of the real attention chain a `diagProjBench` rung rebuilds.
/// The rungs are cumulative, so the ladder walks FORWARD from the bare
/// matmul loop to the full attention block — the direction that localizes
/// the cost. Walking backward from the whole forward made every component
/// measure cheap while the parts summed to a third of the whole.
pub const ProjRung = enum {
    matmul_only, // q/k/v/o back to back, nothing between them
    reshape, // + [B,S,H,D] reshape
    qk_norm, // + per-head QK RMS-norm
    transpose, // + [B,H,S,D] transpose
    rope, // + RoPE
    sdpa, // + SDPA over this step's own K/V (no cache)
    gate, // + softplus per-head output gate
    norms, // + input/post-attention RMS-norms and both residual adds
    mlp, // + the layer's real MLP (MoE or dense) — a whole layer
    yarn_rope, // + laguna's REAL rope arguments (YaRN freqs on full layers)
};

/// Whether a rung issues each projection immediately before its own
/// dependent chain (how the attention arms build the graph) or issues q/k/v
/// together and only then post-processes them.
pub const ProjOrder = enum { interleaved, hoisted };

/// DIAGNOSTIC: walk a rung ladder over the model's OWN loaded attention
/// weights, from a raw `mlx_matmul` loop up to the full attention block, in
/// both issue orders. This forks "the weights are slow" from "the call site
/// is slow": the same weights hit 490 GB/s in the bare loop and ~85 GB/s
/// inside the real graph, and the rung where the rate collapses names the
/// op responsible. Dense-bf16 attention only (quantized attention is
/// skipped — `qmatmul` is a different path).
pub fn diagProjBench(self: *Transformer, iters: usize, ctx: *ForwardCtx) void {
    const io_u = @import("../../../sdk/util/io_util.zig");
    const tio = std.Io.Threaded.global_single_threaded.io();
    const ml = self.moe_layers orelse return;
    const HID: c_int = @intCast(self.config.hidden_size);
    // NOTE (honest caveat): this input is CONSTANT, so the MoE router
    // picks the same top-K experts in every layer and every iteration and
    // the `mlp` rung reads a resident expert bank — it is a LOWER BOUND on
    // the real MoE cost, not the real cost. Feeding the ladder eight
    // varying inputs was tried and reverted: the rungs then SIGBUS (the
    // lazy astype nodes outlive their f32 sources in a way the
    // single-input form never exposed), and this is a diagnostic, not
    // shipped behaviour. Size the MoE share off the REAL forward
    // (diag decode-fwd-ubench) instead, which routes through the
    // shipped kernels with real embeddings.
    const xsh = [_]c_int{ 1, 1, HID };
    const xb = self.allocator.alloc(f32, @intCast(HID)) catch return;
    defer self.allocator.free(xb);
    for (xb) |*v| v.* = 0.01;
    const x32 = mlx.mlx_array_new_data(xb.ptr, &xsh, 3, .float32);
    defer _ = mlx.mlx_array_free(x32);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, self.s)) catch return;

    var bytes: f64 = 0;
    var eligible: usize = 0;
    var declined_shape = false;
    for (ml, 0..) |*lw, li| switch (lw.attn) {
        .full => |fa| {
            if (fa.q_s.ctx != null) continue;
            // The rungs rebuild attention by hand and reshape q back
            // through [1, 1, h_count * hd]. A checkpoint whose q_proj is
            // wider than that product raises an MLX reshape error, which
            // is an uncatchable process kill — decline instead, loudly,
            // because a silent skip is indistinguishable from "this model
            // has no dense attention".
            //
            // The head count is PER LAYER (`cfg.layerNumHeads`, which is
            // what the rung itself uses) — Laguna runs 48 heads on its
            // full-attention layers and 72 on its sliding ones, so a
            // global `num_attention_heads * head_dim` would decline 30 of
            // its 40 layers and quietly under-report the ladder's bytes.
            const layer_width: c_int = @intCast(self.config.layerNumHeads(@intCast(li)) * self.config.head_dim);
            if (!projLadderFits(mlx.getShape(fa.q_w), layer_width)) {
                declined_shape = true;
                continue;
            }
            eligible += 1;
            for ([_]mlx.mlx_array{ fa.q_w, fa.k_w, fa.v_w, fa.o_w }) |w| {
                if (w.ctx == null) continue;
                const sh = mlx.getShape(w);
                if (sh.len == 2) bytes += @as(f64, @floatFromInt(sh[0])) * @as(f64, @floatFromInt(sh[1])) * 2;
            }
        },
        else => {},
    };
    if (declined_shape) {
        log.info("[proj-bench] declined one or more layers: q_proj output != that layer's heads * head_dim — the rung reconstruction cannot rebuild this shape\n", .{});
    }
    if (eligible == 0) return;

    log.info("[proj-bench] ladder over {d} dense-bf16 attention layers ({d:.2} GB/pass)\n", .{ eligible, bytes / 1.0e9 });
    const rungs = [_]ProjRung{ .matmul_only, .reshape, .qk_norm, .transpose, .rope, .sdpa, .gate, .norms, .mlp, .yarn_rope };
    const orders = [_]ProjOrder{ .interleaved, .hoisted };
    for (rungs) |rung| {
        var line: [2]f64 = .{ 0, 0 };
        var ops: u64 = 0;
        for (orders, 0..) |order, oi| {
            for (0..3) |_| self.projRungOnce(ml, x, rung, order, ctx) catch break;
            const ops_before = mlx.op_count.load(.monotonic);
            var sw = io_u.Stopwatch.init(tio);
            var done: usize = 0;
            for (0..iters) |_| {
                self.projRungOnce(ml, x, rung, order, ctx) catch break;
                done += 1;
            }
            line[oi] = @as(f64, @floatFromInt(sw.read())) / 1.0e6 / @as(f64, @floatFromInt(@max(done, 1)));
            if (oi == 0) ops = (mlx.op_count.load(.monotonic) - ops_before) / @max(done, 1);
        }
        log.info("[proj-bench]   {s:<12} interleaved {d:.3} ms ({d:.0} GB/s) | hoisted {d:.3} ms ({d:.0} GB/s) | {d} ops\n", .{
            @tagName(rung),
            line[0],
            bytes / (line[0] / 1000.0) / 1.0e9,
            line[1],
            bytes / (line[1] / 1000.0) / 1.0e9,
            ops,
        });
    }
}

/// One pass of a `diagProjBench` rung. Everything built is folded into the
/// running residual, so MLX's lazy graph cannot drop a chain the rung is
/// supposed to be paying for.
pub fn projRungOnce(
    self: *Transformer,
    layers: []MoeLayerWeights,
    xin: mlx.mlx_array,
    rung: ProjRung,
    order: ProjOrder,
    ctx: *ForwardCtx,
) !void {
    _ = ctx;
    const cfg = &self.config;
    const hd: c_int = @intCast(cfg.head_dim);
    const kv_h: c_int = @intCast(cfg.num_key_value_heads);
    const perm = [_]c_int{ 0, 2, 1, 3 };
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);

    var acc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(acc);
    try mlx.check(mlx.mlx_add(&acc, xin, xin, self.s));

    for (layers, 0..) |*lw, li| switch (lw.attn) {
        .full => |fa| {
            if (fa.q_s.ctx != null) continue;
            const h_count: c_int = @intCast(cfg.layerNumHeads(@intCast(li)));
            const q_shape = [_]c_int{ 1, 1, h_count, hd };
            const kv_shape = [_]c_int{ 1, 1, kv_h, hd };
            const flat_shape = [_]c_int{ 1, 1, h_count * hd };

            // Scratch slots; each holds the chain's current tensor and is
            // replaced in place as the rung adds ops.
            var q = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q);
            var k = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k);
            var v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v);
            const step = struct {
                fn to(t: *Transformer, slot: *mlx.mlx_array, next: mlx.mlx_array) void {
                    _ = t;
                    _ = mlx.mlx_array_free(slot.*);
                    slot.* = next;
                }
            }.to;

            // Q's post-projection chain, run either right after q_proj
            // (interleaved) or only once all three are issued (hoisted).
            const post = struct {
                fn q_chain(t: *Transformer, slot: *mlx.mlx_array, r: ProjRung, sh: []const c_int, nrm: mlx.mlx_array, pm: []const c_int, dims: c_int, base: mlx.mlx_optional_float, freqs: mlx.mlx_array) !void {
                    if (@backingInt(r) < @backingInt(ProjRung.reshape)) return;
                    var o = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_reshape(&o, slot.*, sh.ptr, sh.len, t.s));
                    _ = mlx.mlx_array_free(slot.*);
                    slot.* = o;
                    if (@backingInt(r) >= @backingInt(ProjRung.qk_norm) and nrm.ctx != null) {
                        const n = try t.rmsNorm(slot.*, nrm);
                        _ = mlx.mlx_array_free(slot.*);
                        slot.* = n;
                    }
                    if (@backingInt(r) >= @backingInt(ProjRung.transpose)) {
                        var tr = mlx.mlx_array_new();
                        try mlx.check(mlx.mlx_transpose_axes(&tr, slot.*, pm.ptr, @intCast(pm.len), t.s));
                        _ = mlx.mlx_array_free(slot.*);
                        slot.* = tr;
                    }
                    if (@backingInt(r) >= @backingInt(ProjRung.rope)) {
                        var rp = mlx.mlx_array_new();
                        try mlx.check(mlx.mlx_fast_rope(&rp, slot.*, dims, false, base, 1.0, 0, freqs, t.s));
                        _ = mlx.mlx_array_free(slot.*);
                        slot.* = rp;
                    }
                }
            }.q_chain;

            // From the `norms` rung on, attention reads the real normed
            // residual rather than the residual itself — the same array
            // shape, but produced by the same op the forward produces it
            // with.
            // At the `yarn_rope` rung the ladder switches to the exact
            // RoPE arguments the laguna attention arm used: full layers take a
            // precomputed YaRN freqs ARRAY (and only the partial rotary
            // width), sliding layers the local base frequency.
            const l_full = cfg.isGlobalLayer(@intCast(li));
            const l_yarn = @backingInt(rung) >= @backingInt(ProjRung.yarn_rope) and l_full and self.rope_freqs_yarn != null;
            const l_dims: c_int = if (@backingInt(rung) < @backingInt(ProjRung.yarn_rope))
                hd
            else if (l_yarn)
                @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor_global)
            else
                @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);
            const l_base = mlx.mlx_optional_float{
                .value = if (@backingInt(rung) < @backingInt(ProjRung.yarn_rope))
                    cfg.rope_theta
                else if (l_full) cfg.rope_theta else cfg.rope_local_base_freq,
                .has_value = !l_yarn,
            };
            const l_freqs: mlx.mlx_array = if (l_yarn) self.rope_freqs_yarn.? else .{ .ctx = null };

            var xin_l = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xin_l);
            if (@backingInt(rung) >= @backingInt(ProjRung.norms) and lw.input_norm.ctx != null) {
                const n = try self.rmsNorm(acc, lw.input_norm);
                _ = mlx.mlx_array_free(xin_l);
                xin_l = n;
            } else {
                try mlx.check(mlx.mlx_array_set(&xin_l, acc));
            }

            var tmp = mlx.mlx_array_new();
            switch (order) {
                .interleaved => {
                    try mlx.check(mlx.mlx_matmul(&tmp, xin_l, fa.q_w, self.s));
                    step(self, &q, tmp);
                    try post(self, &q, rung, &q_shape, fa.q_norm, &perm, l_dims, l_base, l_freqs);
                    tmp = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_matmul(&tmp, xin_l, fa.k_w, self.s));
                    step(self, &k, tmp);
                    try post(self, &k, rung, &kv_shape, fa.k_norm, &perm, l_dims, l_base, l_freqs);
                    tmp = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_matmul(&tmp, xin_l, fa.v_w, self.s));
                    step(self, &v, tmp);
                    try post(self, &v, rung, &kv_shape, .{ .ctx = null }, &perm, l_dims, l_base, l_freqs);
                },
                .hoisted => {
                    try mlx.check(mlx.mlx_matmul(&tmp, xin_l, fa.q_w, self.s));
                    step(self, &q, tmp);
                    tmp = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_matmul(&tmp, xin_l, fa.k_w, self.s));
                    step(self, &k, tmp);
                    tmp = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_matmul(&tmp, xin_l, fa.v_w, self.s));
                    step(self, &v, tmp);
                    try post(self, &q, rung, &q_shape, fa.q_norm, &perm, l_dims, l_base, l_freqs);
                    try post(self, &k, rung, &kv_shape, fa.k_norm, &perm, l_dims, l_base, l_freqs);
                    try post(self, &v, rung, &kv_shape, .{ .ctx = null }, &perm, l_dims, l_base, l_freqs);
                },
            }

            // Collapse the q side back to [B,S,H*D] for o_proj. Below the
            // SDPA rung, K and V are folded in through a scalar sum so the
            // rung still pays for their chains.
            var head: mlx.mlx_array = undefined;
            if (@backingInt(rung) >= @backingInt(ProjRung.sdpa)) {
                var att = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&att, q, k, v, attn_scale, "", none_mask, .{ .ctx = null }, false, self.s));
                var back = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_transpose_axes(&back, att, &perm, 4, self.s));
                _ = mlx.mlx_array_free(att);
                head = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_reshape(&head, back, &flat_shape, 3, self.s));
                _ = mlx.mlx_array_free(back);
            } else {
                head = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_reshape(&head, q, &flat_shape, 3, self.s));
                for ([_]mlx.mlx_array{ k, v }) |t| {
                    var s = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(s);
                    try mlx.check(mlx.mlx_sum(&s, t, false, self.s));
                    var h2 = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&h2, head, s, self.s));
                    _ = mlx.mlx_array_free(head);
                    head = h2;
                }
            }
            defer _ = mlx.mlx_array_free(head);

            if (@backingInt(rung) >= @backingInt(ProjRung.gate) and fa.g_w.ctx != null) {
                var gl = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(gl);
                try mlx.check(mlx.mlx_matmul(&gl, xin_l, fa.g_w, self.s));
                var gf = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(gf);
                try mlx.check(mlx.mlx_astype(&gf, gl, .float32, self.s));
                const zero = mlx.mlx_array_new_float(0.0);
                defer _ = mlx.mlx_array_free(zero);
                var sp = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(sp);
                try mlx.check(mlx.mlx_logaddexp(&sp, zero, gf, self.s));
                var gb = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(gb);
                try mlx.check(mlx.mlx_astype(&gb, sp, .bfloat16, self.s));
                const g_shape = [_]c_int{ 1, 1, h_count, 1 };
                var gr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(gr);
                try mlx.check(mlx.mlx_reshape(&gr, gb, &g_shape, 4, self.s));
                var h4 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(h4);
                try mlx.check(mlx.mlx_reshape(&h4, head, &q_shape, 4, self.s));
                var gated = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(gated);
                try mlx.check(mlx.mlx_multiply(&gated, h4, gr, self.s));
                var flat = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_reshape(&flat, gated, &flat_shape, 3, self.s));
                _ = mlx.mlx_array_free(head);
                head = flat;
            }

            var op = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(op);
            try mlx.check(mlx.mlx_matmul(&op, head, fa.o_w, self.s));
            var s1 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&s1, acc, op, self.s));
            _ = mlx.mlx_array_free(acc);
            acc = s1;

            // Layer tail: post-attention norm, the real MLP, second
            // residual. At the top rung this loop IS the forward's layer
            // loop, so a jump here localizes to the op that caused it.
            if (@backingInt(rung) >= @backingInt(ProjRung.norms) and lw.post_attn_norm.ctx != null) {
                const ffn = try self.rmsNorm(acc, lw.post_attn_norm);
                defer _ = mlx.mlx_array_free(ffn);
                var mlp_out = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(mlp_out);
                if (@backingInt(rung) >= @backingInt(ProjRung.mlp)) {
                    _ = mlx.mlx_array_free(mlp_out);
                    mlp_out = switch (lw.mlp) {
                        .moe => |*mw| try self.moeMLP(ffn, mw),
                        .dense => |*dw| try self.denseMLP(ffn, dw),
                    };
                } else {
                    try mlx.check(mlx.mlx_array_set(&mlp_out, ffn));
                }
                var s2 = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&s2, acc, mlp_out, self.s));
                _ = mlx.mlx_array_free(acc);
                acc = s2;
            }
        },
        else => {},
    };
    try mlx.check(mlx.mlx_array_eval(acc));
}

// ── tests ──

test "projLadderFits declines a shape its reconstruction cannot rebuild" {
    // `diagProjBench` rebuilds attention by hand and reshapes q back through
    // [1, 1, n_heads * head_dim]. When a checkpoint's q_proj is wider than
    // that product the reshape raises an MLX error — and an MLX error is not
    // a Zig error, it KILLS the process. Live on Qwen3.5-2B-bf16: q_proj
    // emits 4096 against a computed 2048, so one diagnostic boot took the
    // whole server down.
    //
    // q_w is stored [in, out] (pre-transposed at load).
    try std.testing.expect(Transformer.projLadderFits(&.{ 2048, 2048 }, 2048));
    try std.testing.expect(!Transformer.projLadderFits(&.{ 2048, 4096 }, 2048));
    // Anything that is not a plain 2-D matrix is not a shape this can rebuild.
    try std.testing.expect(!Transformer.projLadderFits(&.{ 1, 2048, 2048 }, 2048));
    try std.testing.expect(!Transformer.projLadderFits(&.{}, 2048));
}
