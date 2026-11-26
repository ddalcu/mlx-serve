//! MLX_SERVE_DSV41_CED_LAZY against the eager replay at the release's widths: a CED trunk (two encoder layers, two
//! decoder layers, the DSpark targets among them) with two DSpark stages, over random weights in both expert formats,
//! fed the 17.6K card's chunks and a prompt whose deferred replay a full-forward chunk and a snapshot read.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const abi = @import("../../../abi/plugin.zig");
const env = @import("../../../env.zig");
const resident = @import("../../../sdk/weights/resident.zig");
const sdk_fixture = @import("../../../sdk/testing/fixture.zig");
const config = @import("../config.zig");
const dsv41 = @import("../deepseek_v41.zig");
const fixture = @import("../decode/fixture.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };
const testing = std.testing;

/// The release's widths and kinds of layer: 0 sliding (as 0-1), 1 a pooling KV and index source (as 2), 2 the
/// decoder's KV and index source (as 20, the CED split), 3 an index source reading it (as 24); 16 experts a layer.
const cfg_json =
    \\{"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "bos_token_id": 0, "eos_token_id": 1,
    \\ "text_config": {"model_type": "deepseek_v41_text", "vocab_size": 4096, "hidden_size": 5120,
    \\  "moe_intermediate_size": 2304, "num_hidden_layers": 4, "num_attention_heads": 64, "num_key_value_heads": 1,
    \\  "head_dim": 512, "qk_rope_head_dim": 64, "q_lora_rank": 1280, "o_lora_rank": 1024, "o_groups": 8,
    \\  "hidden_act": "silu", "swiglu_limit": 10.0, "rms_norm_eps": 1e-20, "max_position_embeddings": 1048576,
    \\  "rope_theta": 10000, "rope_scaling": {"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1,
    \\  "original_max_position_embeddings": 65536}, "n_routed_experts": 16, "n_shared_experts": 1,
    \\  "num_experts_per_tok": 6, "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc", "norm_topk_prob": true,
    \\  "routed_scaling_factor": 1.5, "sliding_window": 128, "compress_ratios": [0, 2, 1, 1, 0, 0],
    \\  "compress_rope_theta": 160000, "kv_source_layer_ids": [1, 2], "index_source_layer_ids": [1, 2, 3],
    \\  "index_n_heads": 32, "index_head_dim": 128, "index_topk": 512, "candidate_source_layer_id": 2,
    \\  "candidate_topk_blocks": 2048, "candidate_block_size": 8, "hc_mult": 4, "hc_sinkhorn_iters": 20,
    \\  "hc_eps": 1e-06, "num_nextn_predict_layers": 2, "dspark_block_size": 5, "dspark_noise_token_id": 4000,
    \\  "dspark_target_layer_ids": [2, 3], "dspark_markov_rank": 256, "dspark_n_routed_experts": 8,
    \\  "dspark_num_experts_per_tok": 3}}
;

fn miniConfig() !config.Config {
    const mem: sdk_fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = cfg_json }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    if (!config.parse(&dir, abi.Str.of(cfg_json), .{}, null, false, &p).isOk()) return error.ConfigRefused;
    defer config.freeParsed(&p);
    return config.of(&p).*;
}

fn put(fx: *fixture.Fixture, comptime fmt: []const u8, args: anytype, shape: []const c_int, mean: f32, std_: f32) !void {
    var buf: [160]u8 = undefined;
    try fx.w.put(try std.fmt.bufPrint(&buf, fmt, args), try fx.normal(shape, .bfloat16, mean, std_));
}

/// The whole model (embedding, trunk, head, DSpark stages and heads) on random weights; NVFP4 experts with ModelOpt's
/// per-expert scales, else MXFP4.
fn build(gpa: std.mem.Allocator, nv: bool) !*fixture.Fixture {
    var cfg = try miniConfig();
    const fx = try gpa.create(fixture.Fixture);
    errdefer gpa.destroy(fx);
    fx.* = .{ .gpa = gpa, .m = undefined, .w = resident.Weights.init(gpa), .seed = 11, .nvfp4 = nv };
    errdefer fx.w.deinit();
    const d: c_int = @intCast(cfg.hidden_size);
    const V: c_int = @intCast(cfg.vocab_size);
    const rank: c_int = @intCast(cfg.dsv4_dspark_markov_rank);
    const targets: c_int = @intCast(cfg.dsv4_n_dspark_target_layers);
    const E = cfg.num_experts;
    for (0..cfg.num_hidden_layers) |li| {
        try fx.putLayer(&cfg, "layers.{d}.", li, E, !nv, null);
        if (nv) inline for (.{ "gate_proj.expert_scale", "up_proj.expert_scale", "expert_scale" }) |n| {
            const u = try fx.uniform(&.{@intCast(E)}, 0.5, 2);
            defer _ = mlx.mlx_array_free(u);
            var v = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&v, u, .bfloat16, mlx.gpuStream()));
            try mlx.check(mlx.mlx_array_eval(v));
            var buf: [96]u8 = undefined;
            try fx.w.put(try std.fmt.bufPrint(&buf, "layers.{d}.ffn.experts." ++ n, .{li}), v);
        };
    }
    const stages = cfg.dsv4_n_compress_ratios - cfg.num_hidden_layers;
    for (0..stages) |si| try fx.putLayer(&cfg, "mtp.{d}.", si, cfg.dsv41_dspark_experts, true, null);
    const last = stages - 1;
    try put(fx, "embed.weight", .{}, &.{ V, d }, 0, 1);
    try put(fx, "head.weight", .{}, &.{ V, d }, 0, 1 / @sqrt(@as(f32, @floatFromInt(d))));
    try put(fx, "norm.weight", .{}, &.{d}, 1, 0.1);
    try put(fx, "mtp.0.main_proj.weight", .{}, &.{ d, d * targets }, 0, 1 / @sqrt(@as(f32, @floatFromInt(d * targets))));
    try put(fx, "mtp.0.main_norm.weight", .{}, &.{d}, 1, 0.1);
    try put(fx, "mtp.{d}.norm.weight", .{last}, &.{d}, 1, 0.1);
    try put(fx, "mtp.{d}.markov_head.embed.weight", .{last}, &.{ V, rank }, 0, 0.01);
    try put(fx, "mtp.{d}.markov_head.head.weight", .{last}, &.{ V, rank }, 0, 0.01);
    try put(fx, "mtp.{d}.confidence_head.proj.weight", .{last}, &.{ 1, d + rank }, 0, 0.1);
    fx.m = try dsv41.init(gpa, &cfg, &fx.w, mlx.gpuStream(), .{ .dspark = true });
    return fx;
}

/// `x`'s words, contiguous (caller frees).
fn words(x: A) ![]u8 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, x, false, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(c));
    const p = mlx.mlx_array_data_uint8(c) orelse return error.NoData;
    return testing.allocator.dupe(u8, p[0 .. mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c)]);
}

/// What a run leaves: its logits and the state's words by part, a ring's window rows apart from its whole buffer.
const Run = struct {
    names: std.ArrayList([]u8) = .empty,
    parts: std.ArrayList([]u8) = .empty,
    /// The rings' slots outside the window at `n` (`dead[i]`: slot i of a ring of `cap`).
    n: usize = 0,

    fn add(r: *Run, comptime fmt: []const u8, args: anytype, x: A) !void {
        if (x.ctx == null) return;
        try r.names.append(testing.allocator, try std.fmt.allocPrint(testing.allocator, fmt, args));
        try r.parts.append(testing.allocator, try words(x));
    }

    fn deinit(r: *Run) void {
        for (r.names.items, r.parts.items) |n, p| {
            testing.allocator.free(n);
            testing.allocator.free(p);
        }
        r.names.deinit(testing.allocator);
        r.parts.deinit(testing.allocator);
    }
};

/// The window rows of `ring` at state length `n` (positions [n - window, n)) and its whole buffer, into `r`.
fn addRing(r: *Run, sc: *dsv41.Scope, comptime what: []const u8, i: usize, ring: dsv41.Ring, n: usize, window: usize) !void {
    var idx: [256]i32 = undefined;
    const lo = n - window;
    for (lo..n, 0..) |p, k| idx[k] = @intCast(p % ring.cap);
    try r.add(what ++ " {d} window", .{i}, try sc.take(ring.buf, try sc.ints(idx[0..window], &.{@intCast(window)}), 0));
    try r.add(what ++ " {d} ring", .{i}, ring.buf);
}

fn addState(r: *Run, m: *dsv41.Dsv41Model, st: *const dsv41.State) !void {
    var sc = dsv41.Scope.init(testing.allocator, m.s);
    defer sc.deinit();
    const w: usize = m.cfg.sliding_window;
    r.n = st.n;
    for (st.layers, 0..) |*ls, li| {
        try addRing(r, &sc, "win", li, ls.win, st.n, w);
        if (ls.comp) |c| try r.add("comp {d}", .{li}, try sc.slice(c.buf, 0, 0, c.used));
        if (ls.idx_k) |c| try r.add("idx_k {d}", .{li}, try sc.slice(c.buf, 0, 0, c.used));
        try r.add("pend_kv {d}", .{li}, ls.pend_kv);
        try r.add("pend_gate {d}", .{li}, ls.pend_gate);
    }
    for (st.ds, 0..) |ring, si| try addRing(r, &sc, "ds", si, ring, st.n, w);
    try r.add("mh", .{}, st.mh);
}

/// The switches an arm runs under.
const Sw = struct { lazy: bool = false, async_end: bool = false, ced_hc: bool = false, rope_pairs: bool = false, rms_rows: bool = false, ced_comp: bool = false };

/// After each chunk of an arm: `State.ced_lazy`, and whether the first decoder layer's ring was ready once the GPU
/// went idle (`ran`).
const Obs = struct { flags: [8]bool = @splat(false), ran: [8]bool = @splat(false) };

/// One arm: a fresh state armed with `ids` (a capture at its last token when `capture`), fed `plan`'s chunks (a negative
/// width is a lead, `.encode`), then the last token and `next`; the state after the chunks, every logits row after.
/// `snap_after`: the chunk after which an abandoned prompt's capture is taken (its words into `snap`).
fn arm(fx: *fixture.Fixture, ids: []const u32, plan: []const isize, next: []const u32, sw: Sw, capture: bool, snap_after: ?usize, out: *Run, snap: *Run, obs: *Obs) !void {
    const gpa = testing.allocator;
    const m = fx.m;
    const lz = env.override(.MLX_SERVE_DSV41_CED_LAZY, if (sw.lazy) "1" else "0");
    defer lz.restore();
    const ae = env.override(.MLX_SERVE_DSV41_PROMPT_ASYNC_EVAL, if (sw.async_end) "1" else "0");
    defer ae.restore();
    const ch = env.override(.MLX_SERVE_DSV41_PF_CED_HC, if (sw.ced_hc) "1" else "0");
    defer ch.restore();
    const rp = env.override(.MLX_SERVE_DSV41_PF_ROPE_PAIRS, if (sw.rope_pairs) "1" else "0");
    defer rp.restore();
    const rr = env.override(.MLX_SERVE_DSV41_PF_RMS_ROWS, if (sw.rms_rows) "1" else "0");
    defer rr.restore();
    const cc = env.override(.MLX_SERVE_DSV41_PF_CED_COMP, if (sw.ced_comp) "1" else "0");
    defer cc.restore();
    try dsv41.reset(m, gpa);
    try dsv41.armPrompt(m, ids, null, capture);
    const st = &m.dec_state.?;
    const split = m.ced_split.?;
    var at: usize = 0;
    for (plan, 0..) |p, k| {
        const n: usize = @abs(p);
        const l = try dsv41.extend(m, gpa, st, ids[at .. at + n], if (p < 0) .encode else .last);
        if (l.ctx != null) _ = mlx.mlx_array_free(l);
        at += n;
        obs.flags[k] = st.ced_lazy;
        try mlx.check(mlx.mlx_synchronize(m.s));
        try mlx.check(mlx._mlx_array_is_available(&obs.ran[k], st.layers[split].win.buf));
        if (snap_after == k) {
            try dsv41.snapshotStateAtChunk(m, st, m.state_gen);
            const sn = if (st.snap) |*s| s else return error.NoSnapshot;
            var sc = dsv41.Scope.init(gpa, m.s);
            defer sc.deinit();
            for (sn.layers, 0..) |l2, li| try addRing(snap, &sc, "snap win", li, ringOf(l2.win), sn.n, m.cfg.sliding_window);
            for (sn.ds, 0..) |b, si| try addRing(snap, &sc, "snap ds", si, ringOf(b), sn.n, m.cfg.sliding_window);
            try snap.add("snap mh", .{}, sn.mh);
        }
    }
    try testing.expectEqual(ids.len - 1, at);
    try addState(out, m, st);
    for ([_][]const u32{ ids[at..], next }) |toks| for (toks) |t| {
        const l = try dsv41.extend(m, gpa, st, &.{t}, .last);
        defer _ = mlx.mlx_array_free(l);
        try out.add("logits at {d}", .{st.n}, l);
    };
}

/// A snapshot's ring buffer `[cap, width]` as a `Ring`.
fn ringOf(buf: A) dsv41.Ring {
    const sh = mlx.getShape(buf);
    return .{ .buf = buf, .cap = @intCast(sh[0]), .width = @intCast(sh[1]) };
}

/// Every part of `got` against `want`: the same words, but for a ring's slots outside its window (counted in `dead`).
fn expectSame(label: []const u8, got: *const Run, want: *const Run, window: usize, dead: *usize) !void {
    try testing.expectEqual(want.names.items.len, got.names.items.len);
    for (want.names.items, want.parts.items, got.names.items, got.parts.items) |wn, wp, gn, gp| {
        try testing.expectEqualStrings(wn, gn);
        try testing.expectEqual(wp.len, gp.len);
        if (std.mem.eql(u8, wp, gp)) continue;
        if (!std.mem.endsWith(u8, wn, " ring")) {
            std.debug.print("[ced_lazy] {s}: {s} differs\n", .{ label, wn });
            return error.NotByteIdentical;
        }
        // A ring `[cap, width]` (its window and 16 rows of slack): only slots no window read reaches may differ.
        const cap: usize = window + 16;
        const row = wp.len / cap;
        for (0..cap) |s| {
            if (std.mem.eql(u8, wp[s * row ..][0..row], gp[s * row ..][0..row])) continue;
            const live = for (want.n - window..want.n) |p| {
                if (p % cap == s) break true;
            } else false;
            if (live) {
                std.debug.print("[ced_lazy] {s}: {s} differs at live slot {d}\n", .{ label, wn, s });
                return error.NotByteIdentical;
            }
            dead.* += 1;
        }
    }
}

test "dsv41 ced_lazy: deferred and dropped replays, alone and with an unwaited pass end, leave the eager path's logits and window rows, the 17.6K card's chunks and a read-back prompt, NVFP4 with per-expert scales and MXFP4 experts at the release's widths" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var rng = std.Random.DefaultPrng.init(41);
    const arms = [_]Sw{ .{}, .{ .lazy = true }, .{ .lazy = true, .async_end = true } };
    for ([_]bool{ true, false }) |nv| {
        const fx = try build(gpa, nv);
        defer fx.deinit();
        const m = fx.m;
        try testing.expectEqual(@as(?usize, 2), m.ced_split);
        try testing.expect(m.n_mtp == 2);
        const w: usize = m.cfg.sliding_window;
        const next = [_]u32{ 7, 11, 13 };
        const fmt = if (nv) "nvfp4" else "mxfp4";

        // The card: 17,639 tokens, the engine's chunks 8,192 (its lead and rest), 8,192 and 1,254, then the last token.
        const card = try gpa.alloc(u32, 17_639);
        defer gpa.free(card);
        for (card) |*t| t.* = 1 + rng.random().uintLessThan(u32, 3999);
        const card_plan = [_]isize{ -4096, 4096, 8192, 1254 };
        var runs: [arms.len]Run = @splat(.{});
        defer for (&runs) |*r| r.deinit();
        var no_snap: Run = .{};
        defer no_snap.deinit();
        var obs: [arms.len]Obs = @splat(.{});
        for (arms, &runs, &obs) |sw, *r, *o| try arm(fx, card, &card_plan, &next, sw, false, null, r, &no_snap, o);
        const n = card_plan.len;
        // The eager path runs every replay.
        try testing.expectEqualSlices(bool, &.{ true, true, true, true }, obs[0].ran[0..n]);
        for (arms[1..], obs[1..], runs[1..]) |sw, o, *r| {
            // The rest's and the second chunk's replays waited and never ran; the last chunk's ran, as the token after
            // it reads it.
            try testing.expectEqualSlices(bool, if (sw.lazy) &.{ false, true, true, false } else &.{ false, false, false, false }, o.flags[0..n]);
            try testing.expectEqualSlices(bool, if (sw.lazy) &.{ true, false, false, true } else &.{ true, true, true, true }, o.ran[0..n]);
            var dead: usize = 0;
            try expectSame("card", r, &runs[0], w, &dead);
            std.debug.print("[ced_lazy] {s} card lazy={} async_end={}: logits and window rows the same; {d} ring slots outside the window differ\n", .{ fmt, sw.lazy, sw.async_end, dead });
            // Two replays dropped: each decoder ring and DSpark ring keeps zeros in the 16 slots outside the last window.
            try testing.expectEqual(@as(usize, if (sw.lazy) 16 * (2 + 2) else 0), dead);
        }

        // A planted difference: one token of the last chunk changed.
        var planted: Run = .{};
        defer planted.deinit();
        card[17_000] = 1 + card[17_000] % 3999;
        var po: Obs = .{};
        try arm(fx, card, &card_plan, &next, .{}, false, null, &planted, &no_snap, &po);
        var pdead: usize = 0;
        try testing.expectError(error.NotByteIdentical, expectSame("planted", &planted, &runs[0], w, &pdead));

        // A replay deferred and dropped at a lead, then one a capture and a full-forward chunk read, then the last.
        const back = try gpa.alloc(u32, 4 * 2048 + 256 + 1024 + 1);
        defer gpa.free(back);
        for (back) |*t| t.* = 1 + rng.random().uintLessThan(u32, 3999);
        const back_plan = [_]isize{ -2048, 2048, -2048, 2048, 256, 1024 };
        const bn = back_plan.len;
        var bruns: [arms.len]Run = @splat(.{});
        defer for (&bruns) |*r| r.deinit();
        var snaps: [arms.len]Run = @splat(.{});
        defer for (&snaps) |*r| r.deinit();
        var bobs: [arms.len]Obs = @splat(.{});
        for (arms, &bruns, &snaps, &bobs) |sw, *r, *sn, *o| try arm(fx, back, &back_plan, &next, sw, true, 3, r, sn, o);
        for (arms[1..], bobs[1..], bruns[1..], snaps[1..]) |sw, o, *r, *sn| {
            try testing.expectEqualSlices(bool, if (sw.lazy) &.{ false, true, false, true, false, false } else &.{ false, false, false, false, false, false }, o.flags[0..bn]);
            // The full-forward chunk ran the second rest's replay; the first one's, dropped at the lead, never ran.
            if (sw.lazy) try testing.expectEqualSlices(bool, &.{ true, false, true, false, true, true }, o.ran[0..bn]);
            var dead: usize = 0;
            sn.n = 4 * 2048;
            snaps[0].n = 4 * 2048;
            try expectSame("capture", sn, &snaps[0], w, &dead);
            try testing.expectEqual(@as(usize, if (sw.lazy) 16 * (2 + 2) else 0), dead);
            dead = 0;
            try expectSame("read back", r, &bruns[0], w, &dead);
            // The full-forward chunk rewrote every slot before the last replay: nothing differs.
            try testing.expectEqual(@as(usize, 0), dead);
        }
    }
}

test "dsv41 pf ced_hc: the CED compressor's hc pre-norm over the pass's whole stream leaves every cache, window row and logits row of the span-by-span path word for word, the 17.6K card's chunks, NVFP4 with per-expert scales and MXFP4 experts; spans read a row late fail" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var rng = std.Random.DefaultPrng.init(43);
    for ([_]bool{ true, false }) |nv| {
        const fx = try build(gpa, nv);
        defer fx.deinit();
        const w: usize = fx.m.cfg.sliding_window;
        const next = [_]u32{ 7, 11, 13 };
        const card = try gpa.alloc(u32, 17_639);
        defer gpa.free(card);
        for (card) |*t| t.* = 1 + rng.random().uintLessThan(u32, 3999);
        const plan = [_]isize{ -4096, 4096, 8192, 1254 };
        var off: Run = .{};
        defer off.deinit();
        var on: Run = .{};
        defer on.deinit();
        var planted: Run = .{};
        defer planted.deinit();
        var no_snap: Run = .{};
        defer no_snap.deinit();
        var obs: Obs = .{};
        try arm(fx, card, &plan, &next, .{}, false, null, &off, &no_snap, &obs);
        const before = dsv41.ced_hc_passes.load(.monotonic);
        try arm(fx, card, &plan, &next, .{ .ced_hc = true }, false, null, &on, &no_snap, &obs);
        // Every chunk is a CED pass of several spans: each took the whole stream.
        try testing.expectEqual(before + plan.len, dsv41.ced_hc_passes.load(.monotonic));
        var dead: usize = 0;
        try expectSame("ced_hc", &on, &off, w, &dead);
        try testing.expectEqual(@as(usize, 0), dead);
        dsv41.ced_hc_plant.store(1, .monotonic);
        defer dsv41.ced_hc_plant.store(0, .monotonic);
        try arm(fx, card, &plan, &next, .{ .ced_hc = true }, false, null, &planted, &no_snap, &obs);
        try testing.expectError(error.NotByteIdentical, expectSame("ced_hc planted", &planted, &off, w, &dead));
        std.debug.print("[ced_hc] {s}: {d} passes on the whole stream, every part word for word; the planted late row differs\n", .{ if (nv) "nvfp4" else "mxfp4", plan.len });
    }
}

test "dsv41 pf rope_pairs: the query tail's rope a pair a thread over the heads leaves every cache, window row and logits row word for word, the 17.6K card's chunks, NVFP4 with per-expert scales and MXFP4 experts" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var rng = std.Random.DefaultPrng.init(53);
    for ([_]bool{ true, false }) |nv| {
        const fx = try build(gpa, nv);
        defer fx.deinit();
        const w: usize = fx.m.cfg.sliding_window;
        const next = [_]u32{ 7, 11, 13 };
        const card = try gpa.alloc(u32, 17_639);
        defer gpa.free(card);
        for (card) |*t| t.* = 1 + rng.random().uintLessThan(u32, 3999);
        const plan = [_]isize{ -4096, 4096, 8192, 1254 };
        var off: Run = .{};
        defer off.deinit();
        var on: Run = .{};
        defer on.deinit();
        var no_snap: Run = .{};
        defer no_snap.deinit();
        var obs: Obs = .{};
        const pin = env.override(.MLX_SERVE_DSV41_PF_ROPE_PAIRS, "0");
        try arm(fx, card, &plan, &next, .{}, false, null, &off, &no_snap, &obs);
        pin.restore();
        const before = dsv41.rope_pairs_calls.load(.monotonic);
        try arm(fx, card, &plan, &next, .{ .rope_pairs = true }, false, null, &on, &no_snap, &obs);
        try testing.expect(dsv41.rope_pairs_calls.load(.monotonic) > before);
        var dead: usize = 0;
        try expectSame("rope_pairs", &on, &off, w, &dead);
        try testing.expectEqual(@as(usize, 0), dead);
        std.debug.print("[rope_pairs] {s}: {d} pair launches, every part word for word\n", .{ if (nv) "nvfp4" else "mxfp4", dsv41.rope_pairs_calls.load(.monotonic) - before });
    }
}

test "dsv41 pf rms_rows: the query latent's rms read in place in q_kv's output leaves every cache, window row and logits row word for word, the 17.6K card's chunks, NVFP4 with per-expert scales and MXFP4 experts" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var rng = std.Random.DefaultPrng.init(59);
    for ([_]bool{ true, false }) |nv| {
        const fx = try build(gpa, nv);
        defer fx.deinit();
        const w: usize = fx.m.cfg.sliding_window;
        const next = [_]u32{ 7, 11, 13 };
        const card = try gpa.alloc(u32, 17_639);
        defer gpa.free(card);
        for (card) |*t| t.* = 1 + rng.random().uintLessThan(u32, 3999);
        const plan = [_]isize{ -4096, 4096, 8192, 1254 };
        var off: Run = .{};
        defer off.deinit();
        var on: Run = .{};
        defer on.deinit();
        var no_snap: Run = .{};
        defer no_snap.deinit();
        var obs: Obs = .{};
        const pin = env.override(.MLX_SERVE_DSV41_PF_RMS_ROWS, "0");
        try arm(fx, card, &plan, &next, .{}, false, null, &off, &no_snap, &obs);
        pin.restore();
        const before = dsv41.rms_rows_calls.load(.monotonic);
        try arm(fx, card, &plan, &next, .{ .rms_rows = true }, false, null, &on, &no_snap, &obs);
        try testing.expect(dsv41.rms_rows_calls.load(.monotonic) > before);
        var dead: usize = 0;
        try expectSame("rms_rows", &on, &off, w, &dead);
        try testing.expectEqual(@as(usize, 0), dead);
        std.debug.print("[rms_rows] {s}: {d} strided rms launches, every part word for word\n", .{ if (nv) "nvfp4" else "mxfp4", dsv41.rms_rows_calls.load(.monotonic) - before });
    }
}

test "dsv41 pf ced_comp: the CED compressor's pooling, index keys and rope over the pass in one compressPass (each span's projection as before) leave every cache, window row and logits row of the span-by-span compressor word for word, the 17.6K card's chunks, NVFP4 with per-expert scales and MXFP4 experts; the groups' rope a row late fails" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var rng = std.Random.DefaultPrng.init(61);
    for ([_]bool{ true, false }) |nv| {
        const fx = try build(gpa, nv);
        defer fx.deinit();
        const w: usize = fx.m.cfg.sliding_window;
        const next = [_]u32{ 7, 11, 13 };
        const card = try gpa.alloc(u32, 17_639);
        defer gpa.free(card);
        for (card) |*t| t.* = 1 + rng.random().uintLessThan(u32, 3999);
        const plan = [_]isize{ -4096, 4096, 8192, 1254 };
        var off: Run = .{};
        defer off.deinit();
        var on: Run = .{};
        defer on.deinit();
        var planted: Run = .{};
        defer planted.deinit();
        var no_snap: Run = .{};
        defer no_snap.deinit();
        var obs: Obs = .{};
        try arm(fx, card, &plan, &next, .{ .ced_hc = true }, false, null, &off, &no_snap, &obs);
        const before = dsv41.ced_comp_passes.load(.monotonic);
        try arm(fx, card, &plan, &next, .{ .ced_hc = true, .ced_comp = true }, false, null, &on, &no_snap, &obs);
        try testing.expectEqual(before + plan.len, dsv41.ced_comp_passes.load(.monotonic));
        var dead: usize = 0;
        try expectSame("ced_comp", &on, &off, w, &dead);
        try testing.expectEqual(@as(usize, 0), dead);
        dsv41.comp_pass_plant = true;
        defer dsv41.comp_pass_plant = false;
        try arm(fx, card, &plan, &next, .{ .ced_hc = true, .ced_comp = true }, false, null, &planted, &no_snap, &obs);
        try testing.expectError(error.NotByteIdentical, expectSame("ced_comp planted", &planted, &off, w, &dead));
        std.debug.print("[ced_comp] {s}: {d} passes in one compressPass, every part word for word; the planted late rope differs\n", .{ if (nv) "nvfp4" else "mxfp4", plan.len });
    }
}
