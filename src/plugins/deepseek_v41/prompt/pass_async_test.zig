//! MLX_SERVE_DSV41_PROMPT_ASYNC_EVAL against the waited pass end at the release's widths: a CED trunk (two encoder
//! layers, two decoder layers, the DSpark targets among them) with two DSpark stages, over random weights in both
//! expert formats, fed the 17.6K card's chunks and a prompt with a capture and a full-forward chunk.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const abi = @import("../../../abi/plugin.zig");
const env = @import("../../../env.zig");
const resident = @import("../../../sdk/weights/resident.zig");
const sdk_fixture = @import("../../../sdk/testing/fixture.zig");
const config = @import("../config.zig");
const dsv41 = @import("../deepseek_v41.zig");
const fixture = @import("../decode/fixture.zig");
const pass_async = @import("pass_async.zig");

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

/// What a run leaves: its logits and every part of the state, each ring's whole buffer.
const Run = struct {
    names: std.ArrayList([]u8) = .empty,
    parts: std.ArrayList([]u8) = .empty,

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

fn addState(r: *Run, m: *dsv41.Dsv41Model, st: *const dsv41.State) !void {
    var sc = dsv41.Scope.init(testing.allocator, m.s);
    defer sc.deinit();
    for (st.layers, 0..) |*ls, li| {
        try r.add("win {d}", .{li}, ls.win.buf);
        if (ls.comp) |c| try r.add("comp {d}", .{li}, try sc.slice(c.buf, 0, 0, c.used));
        if (ls.idx_k) |c| try r.add("idx_k {d}", .{li}, try sc.slice(c.buf, 0, 0, c.used));
        try r.add("pend_kv {d}", .{li}, ls.pend_kv);
        try r.add("pend_gate {d}", .{li}, ls.pend_gate);
    }
    for (st.ds, 0..) |ring, si| try r.add("ds {d}", .{si}, ring.buf);
    try r.add("mh", .{}, st.mh);
}

/// After each chunk of an arm: whether the last encoder layer's ring was ready as the chunk returned, before any wait.
const Obs = struct { early: [8]bool = @splat(false) };

/// One arm: a fresh state armed with `ids` (a capture at its last token when `capture`), fed `plan`'s chunks (a negative
/// width is a lead, `.encode`), then the last token and `next`; the state after the chunks, every logits row after.
/// `snap_after`: the chunk after which an abandoned prompt's capture is taken (its words into `snap`). Returns the
/// passes the switch left unwaited.
fn arm(fx: *fixture.Fixture, ids: []const u32, plan: []const isize, next: []const u32, async_end: bool, capture: bool, snap_after: ?usize, out: *Run, snap: *Run, obs: *Obs) !usize {
    const gpa = testing.allocator;
    const m = fx.m;
    const ae = env.override(.MLX_SERVE_DSV41_PROMPT_ASYNC_EVAL, if (async_end) "1" else "0");
    defer ae.restore();
    const passes0 = pass_async.shared_state.get().*;
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
        try mlx.check(mlx._mlx_array_is_available(&obs.early[k], st.layers[split - 1].win.buf));
        try mlx.check(mlx.mlx_synchronize(m.s));
        if (snap_after == k) {
            try dsv41.snapshotStateAtChunk(m, st, m.state_gen);
            const sn = if (st.snap) |*s| s else return error.NoSnapshot;
            for (sn.layers, 0..) |l2, li| try snap.add("snap win {d}", .{li}, l2.win);
            for (sn.ds, 0..) |b, si| try snap.add("snap ds {d}", .{si}, b);
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
    return pass_async.shared_state.get().* - passes0;
}

/// Every part of `got` has `want`'s words.
fn expectSame(label: []const u8, got: *const Run, want: *const Run) !void {
    try testing.expectEqual(want.names.items.len, got.names.items.len);
    for (want.names.items, want.parts.items, got.names.items, got.parts.items) |wn, wp, gn, gp| {
        try testing.expectEqualStrings(wn, gn);
        if (std.mem.eql(u8, wp, gp)) continue;
        std.debug.print("[pass_async] {s}: {s} differs\n", .{ label, wn });
        return error.NotByteIdentical;
    }
}

/// The plan's wide passes: each chunk of more than one span runs one (its encoder pass, or its whole forward).
fn widePasses(plan: []const isize) usize {
    var n: usize = 0;
    for (plan) |p| n += @intFromBool(@abs(p) > dsv41.SPAN);
    return n;
}

test "dsv41 prompt async eval: an unwaited pass end leaves the waited path's logits and every state word, the 17.6K card's chunks and a captured prompt, NVFP4 with per-expert scales and MXFP4 experts at the release's widths" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var rng = std.Random.DefaultPrng.init(41);
    const next = [_]u32{ 7, 11, 13 };
    for ([_]bool{ true, false }) |nv| {
        const fx = try build(gpa, nv);
        defer fx.deinit();
        const m = fx.m;
        try testing.expectEqual(@as(?usize, 2), m.ced_split);
        try testing.expect(m.n_mtp == 2);
        const fmt = if (nv) "nvfp4" else "mxfp4";

        // The card: 17,639 tokens, the engine's chunks 4,096 (a lead), 4,096 (its rest), 8,192 and 1,254, then the last token.
        const card = try gpa.alloc(u32, 17_639);
        defer gpa.free(card);
        for (card) |*t| t.* = 1 + rng.random().uintLessThan(u32, 3999);
        const card_plan = [_]isize{ -4096, 4096, 8192, 1254 };
        var waited: Run = .{};
        defer waited.deinit();
        var unwaited: Run = .{};
        defer unwaited.deinit();
        var no_snap: Run = .{};
        defer no_snap.deinit();
        var wo: Obs = .{};
        var uo: Obs = .{};
        try testing.expectEqual(@as(usize, 0), try arm(fx, card, &card_plan, &next, false, false, null, &waited, &no_snap, &wo));
        try testing.expectEqual(widePasses(&card_plan), try arm(fx, card, &card_plan, &next, true, false, null, &unwaited, &no_snap, &uo));
        // The waited path's last encoder layer is done at every chunk's return.
        try testing.expectEqualSlices(bool, &.{ true, true, true, true }, wo.early[0..card_plan.len]);
        std.debug.print("[pass_async] {s} card: every word the same; last encoder layer ready at return {any} (waited {any})\n", .{ fmt, uo.early[0..card_plan.len], wo.early[0..card_plan.len] });
        try expectSame("card", &unwaited, &waited);

        // A planted difference: one token of the last chunk changed.
        var planted: Run = .{};
        defer planted.deinit();
        card[17_000] = 1 + card[17_000] % 3999;
        var po: Obs = .{};
        _ = try arm(fx, card, &card_plan, &next, true, false, null, &planted, &no_snap, &po);
        try testing.expectError(error.NotByteIdentical, expectSame("planted", &planted, &waited));

        // Two leads, a capture after the second rest, a full-forward chunk and a wide last chunk.
        const back = try gpa.alloc(u32, 4 * 2048 + 256 + 1024 + 1);
        defer gpa.free(back);
        for (back) |*t| t.* = 1 + rng.random().uintLessThan(u32, 3999);
        const back_plan = [_]isize{ -2048, 2048, -2048, 2048, 256, 1024 };
        var bw: Run = .{};
        defer bw.deinit();
        var bu: Run = .{};
        defer bu.deinit();
        var sw: Run = .{};
        defer sw.deinit();
        var su: Run = .{};
        defer su.deinit();
        var bwo: Obs = .{};
        var buo: Obs = .{};
        try testing.expectEqual(@as(usize, 0), try arm(fx, back, &back_plan, &next, false, true, 3, &bw, &sw, &bwo));
        try testing.expectEqual(widePasses(&back_plan), try arm(fx, back, &back_plan, &next, true, true, 3, &bu, &su, &buo));
        try expectSame("capture", &su, &sw);
        try expectSame("captured prompt", &bu, &bw);
        std.debug.print("[pass_async] {s} captured prompt: every word the same, the capture's too\n", .{fmt});
    }
}
