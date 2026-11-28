//! Hy4-flash's numeric oracle: each block against the REFERENCE PyTorch module on tiny random weights,
//! dumped by `tests/dump_hyv4_fixtures.py` into `fixtures/num/`. The fixtures never touch the 440 GB
//! checkpoint; tiny widths catch a sign, a scale or an ordering bug that a coherent greedy answer hides.
//! Bar: cos >= 0.999 and |rms_ratio - 1| <= 0.02 (bf16 rounding where the reference runs bf16).

const std = @import("std");
const testing = std.testing;
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const block = @import("../../../sdk/bases/block.zig");
const cache_ops = @import("../../../sdk/testing/cache_ops.zig");
const ihc = @import("ihc.zig");
const mla = @import("mla.zig");
const indexer = @import("indexer.zig");
const moe = @import("moe.zig");
const engram = @import("engram.zig");
const proj = @import("proj.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Proj = proj.Proj;

fn bytes(comptime path: []const u8) []const u8 {
    return @embedFile(path);
}

fn f32s(a: std.mem.Allocator, comptime path: []const u8) ![]f32 {
    const b = bytes(path);
    const out = try a.alloc(f32, b.len / 4);
    for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, b[i * 4 ..][0..4], .little));
    return out;
}

fn i32s(a: std.mem.Allocator, comptime path: []const u8) ![]i32 {
    const b = bytes(path);
    const out = try a.alloc(i32, b.len / 4);
    for (out, 0..) |*v, i| v.* = std.mem.readInt(i32, b[i * 4 ..][0..4], .little);
    return out;
}

fn dims(comptime path: []const u8) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, testing.allocator, bytes(path), .{});
}

fn f32Of(v: std.json.Value, key: []const u8) !f32 {
    return @floatCast(v.object.get(key).?.float);
}

fn intOf(v: std.json.Value, key: []const u8) !u32 {
    return @intCast(v.object.get(key).?.integer);
}

/// The array as f32 values, casting whatever the forward produced.
fn read(a: std.mem.Allocator, x: Array) ![]f32 {
    try ops.eval(x);
    const n = mlx.mlx_array_size(x);
    if (mlx.mlx_array_dtype(x) == .float32) {
        const p = mlx.mlx_array_data_float32(x) orelse return error.NoData;
        const out = try a.alloc(f32, n);
        @memcpy(out, p[0..n]);
        return out;
    }
    var cast = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cast);
    try mlx.check(mlx.mlx_astype(&cast, x, .float32, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(cast));
    const p = mlx.mlx_array_data_float32(cast) orelse return error.NoData;
    const out = try a.alloc(f32, n);
    @memcpy(out, p[0..n]);
    return out;
}

fn checkParity(name: []const u8, got: []const f32, want: []const f32) !void {
    try testing.expectEqual(want.len, got.len);
    var dot: f64 = 0;
    var ng: f64 = 0;
    var nw: f64 = 0;
    for (got, want) |g, w| {
        dot += @as(f64, g) * w;
        ng += @as(f64, g) * g;
        nw += w * w;
    }
    const cos = dot / @sqrt(ng * nw);
    const rms_ratio = @sqrt(ng / nw);
    std.debug.print("[hy_v4 oracle] {s}: cos {d:.6} rms_ratio {d:.6}\n", .{ name, cos, rms_ratio });
    try testing.expect(cos >= 0.999);
    try testing.expect(@abs(rms_ratio - 1.0) <= 0.02);
}

fn dense(g: *Graph, hf_w: []const f32, out_dim: usize, in_dim: usize, a: std.mem.Allocator) !Proj {
    const t = try a.alloc(f32, hf_w.len);
    for (0..out_dim) |o| {
        for (0..in_dim) |i| t[i * out_dim + o] = hf_w[o * in_dim + i];
    }
    return .{ .w = try g.fromSlice(f32, t, &.{ @intCast(in_dim), @intCast(out_dim) }) };
}

fn vec(g: *Graph, x: []const f32, a: std.mem.Allocator) !Array {
    const t = try a.alloc(f32, x.len);
    @memcpy(t, x);
    return g.fromSlice(f32, t, &.{@intCast(x.len)});
}

test "hy_v4 oracle: iHC pre/post and the head collapse match the reference dump" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    var g = Graph.init(testing.allocator, mlx.mlx_default_cpu_stream_new());
    defer g.deinit();
    const d = try dims("../fixtures/num/ihc.json");
    defer d.deinit();
    const lanes: usize = @intCast(try intOf(d.value, "lanes"));
    const hidden: usize = @intCast(try intOf(d.value, "hidden"));
    const rows: usize = @intCast(try intOf(d.value, "rows"));
    const width = lanes * hidden;

    const fn_hf = try f32s(aa, "../fixtures/num/ihc_fn.bin");
    var fn_t = try aa.alloc(f32, fn_hf.len);
    for (0..2 * lanes) |i| {
        for (0..width) |j| fn_t[j * 2 * lanes + i] = fn_hf[i * width + j];
    }
    const base = try f32s(aa, "../fixtures/num/ihc_base.bin");
    const scale = try f32s(aa, "../fixtures/num/ihc_scale.bin");
    const streams_hf = try f32s(aa, "../fixtures/num/ihc_streams.bin");
    const want_post = try f32s(aa, "../fixtures/num/ihc_post.bin");
    const want_mixed = try f32s(aa, "../fixtures/num/ihc_mixed.bin");

    const streams = try g.fromSlice(f32, streams_hf, &.{ 1, @intCast(rows), @intCast(lanes), @intCast(hidden) });
    defer g.drop(streams);
    const p: ihc.Params = .{
        .fn_t = try g.fromSlice(f32, fn_t, &.{ @intCast(width), @intCast(2 * lanes) }),
        .base = try vec(&g, base, aa),
        .scale = try vec(&g, scale, aa),
        .lanes = @intCast(lanes),
        .hidden = @intCast(hidden),
        .magnitude = try f32Of(d.value, "magnitude"),
        .eps = try f32Of(d.value, "eps"),
        .rms_eps = try f32Of(d.value, "rms_eps"),
    };
    const r = try ihc.pre(&g, streams, &p);
    defer r.deinit();
    try checkParity("ihc mixed", try read(aa, r.mixed), want_mixed);
    try checkParity("ihc post", try read(aa, r.post), want_post);

    const head_fn_hf = try f32s(aa, "../fixtures/num/ihc_head_fn.bin");
    var head_fn_t = try aa.alloc(f32, head_fn_hf.len);
    for (0..lanes) |i| {
        for (0..width) |j| head_fn_t[j * lanes + i] = head_fn_hf[i * width + j];
    }
    const hp: ihc.HeadParams = .{
        .fn_t = try g.fromSlice(f32, head_fn_t, &.{ @intCast(width), @intCast(lanes) }),
        .base = try vec(&g, try f32s(aa, "../fixtures/num/ihc_head_base.bin"), aa),
        .scale = try vec(&g, try f32s(aa, "../fixtures/num/ihc_head_scale.bin"), aa),
        .lanes = @intCast(lanes),
        .hidden = @intCast(hidden),
        .eps = try f32Of(d.value, "eps"),
        .rms_eps = try f32Of(d.value, "rms_eps"),
    };
    const out = try ihc.head(&g, streams, &hp);
    defer _ = mlx.mlx_array_free(out);
    try checkParity("ihc head", try read(aa, out), try f32s(aa, "../fixtures/num/ihc_head_out.bin"));
}

test "hy_v4 oracle: gated MLA with sinks matches the reference dump" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    var g = Graph.init(testing.allocator, mlx.mlx_default_cpu_stream_new());
    defer g.deinit();
    const d = try dims("../fixtures/num/mla.json");
    defer d.deinit();
    const hidden: usize = @intCast(try intOf(d.value, "hidden"));
    const heads: usize = @intCast(try intOf(d.value, "heads"));
    const nope: usize = @intCast(try intOf(d.value, "nope"));
    const rope_dim: usize = @intCast(try intOf(d.value, "rope"));
    const vd: usize = @intCast(try intOf(d.value, "v_dim"));
    const rank: usize = @intCast(try intOf(d.value, "rank"));
    const q_lora: usize = @intCast(try intOf(d.value, "q_lora"));
    const rows: usize = @intCast(try intOf(d.value, "rows"));
    const qk = nope + rope_dim;
    const eps: f32 = try f32Of(d.value, "eps");

    const kv_b = try f32s(aa, "../fixtures/num/mla_kv_b.bin");
    const wk = try aa.alloc(f32, heads * nope * rank);
    const wv = try aa.alloc(f32, heads * vd * rank);
    for (0..heads) |h| {
        for (0..nope) |i| @memcpy(wk[h * nope * rank + i * rank ..][0..rank], kv_b[h * (nope + vd) * rank + i * rank ..][0..rank]);
        for (0..vd) |i| @memcpy(wv[h * vd * rank + i * rank ..][0..rank], kv_b[h * (nope + vd) * rank + (nope + i) * rank ..][0..rank]);
    }

    const attn = mla.Attn{
        .q_a = try dense(&g, try f32s(aa, "../fixtures/num/mla_q_a.bin"), q_lora, hidden, aa),
        .q_a_norm = try vec(&g, try f32s(aa, "../fixtures/num/mla_q_a_norm.bin"), aa),
        .q_b = try dense(&g, try f32s(aa, "../fixtures/num/mla_q_b.bin"), heads * qk, q_lora, aa),
        .kv_a = try dense(&g, try f32s(aa, "../fixtures/num/mla_kv_a.bin"), rank + rope_dim, hidden, aa),
        .kv_a_norm = try vec(&g, try f32s(aa, "../fixtures/num/mla_kv_a_norm.bin"), aa),
        .wk = try g.fromSlice(f32, wk, &.{ @intCast(heads), @intCast(nope), @intCast(rank) }),
        .wv = try g.fromSlice(f32, wv, &.{ @intCast(heads), @intCast(vd), @intCast(rank) }),
        .linear_gate = try dense(&g, try f32s(aa, "../fixtures/num/mla_gate.bin"), heads * vd, hidden, aa),
        .o_proj = try dense(&g, try f32s(aa, "../fixtures/num/mla_o.bin"), hidden, heads * vd, aa),
        .post_proj_norm = try vec(&g, try f32s(aa, "../fixtures/num/mla_post_norm.bin"), aa),
        .sink = try vec(&g, try f32s(aa, "../fixtures/num/mla_sink.bin"), aa),
        .heads = @intCast(heads),
        .nope = @intCast(nope),
        .rope_dim = @intCast(rope_dim),
        .v_dim = @intCast(vd),
        .rank = @intCast(rank),
        .theta = try f32Of(d.value, "theta"),
        .window = 0,
        .rms_eps = eps,
    };

    var cache = try cache_ops.TestCache.init(testing.allocator, 1);
    defer cache.deinit();
    var shared = block.Shared.init(eps, eps);
    defer shared.deinit();
    var pos: [16]u32 = undefined;
    for (0..rows) |i| pos[i] = @intCast(i);
    const x = try g.fromSlice(f32, try f32s(aa, "../fixtures/num/mla_x.bin"), &.{ 1, @intCast(rows), @intCast(hidden) });
    defer g.drop(x);
    var ctx: block.Ctx = .{
        .g = &g,
        .mode = .prefill,
        .rows = @intCast(rows),
        .positions = pos[0..rows],
        .caches = &.{cache.handle()},
        .cache_ops = &cache_ops.ops,
        .shared = &shared,
    };
    const out = try attn.forwards(&ctx, x);
    defer g.drop(out);
    try checkParity("mla", try read(aa, out), try f32s(aa, "../fixtures/num/mla_out.bin"));
}

test "hy_v4 oracle: the compressed indexer's recall matches the reference dump" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    var g = Graph.init(testing.allocator, mlx.mlx_default_cpu_stream_new());
    defer g.deinit();
    const d = try dims("../fixtures/num/indexer.json");
    defer d.deinit();
    const hidden: usize = @intCast(try intOf(d.value, "hidden"));
    const q_lora: usize = @intCast(try intOf(d.value, "q_lora"));
    const heads: usize = @intCast(try intOf(d.value, "heads"));
    const dim: usize = @intCast(try intOf(d.value, "dim"));
    const ratio: usize = @intCast(try intOf(d.value, "ratio"));
    const group_topk: usize = @intCast(try intOf(d.value, "group_topk"));
    const rope_dim: usize = @intCast(try intOf(d.value, "rope"));
    const kv_len: usize = @intCast(try intOf(d.value, "kv_len"));
    const rows: usize = @intCast(try intOf(d.value, "rows"));
    const base_pos: c_int = @intCast(try intOf(d.value, "base_pos"));

    const ix = indexer.Indexer{
        .query = try dense(&g, try f32s(aa, "../fixtures/num/idx_query.bin"), heads * dim, q_lora, aa),
        .head_weight = try dense(&g, try f32s(aa, "../fixtures/num/idx_head_weight.bin"), heads, hidden, aa),
        .token_key = try dense(&g, try f32s(aa, "../fixtures/num/idx_token_key.bin"), dim, hidden, aa),
        .token_key_norm = try vec(&g, try f32s(aa, "../fixtures/num/idx_token_key_norm.bin"), aa),
        .value_proj = try dense(&g, try f32s(aa, "../fixtures/num/idx_value.bin"), dim, dim, aa),
        .gate_proj = try dense(&g, try f32s(aa, "../fixtures/num/idx_gate.bin"), dim, dim, aa),
        .key_norm = try vec(&g, try f32s(aa, "../fixtures/num/idx_key_norm.bin"), aa),
        .ape = try g.fromSlice(f32, try f32s(aa, "../fixtures/num/idx_ape.bin"), &.{ @intCast(ratio), @intCast(dim) }),
        .heads = @intCast(heads),
        .dim = @intCast(dim),
        .ratio = @intCast(ratio),
        .group_topk = @intCast(group_topk),
        .budget = @intCast(group_topk * ratio),
        .rope_dim = @intCast(rope_dim),
        .theta = try f32Of(d.value, "theta"),
        .rms_eps = try f32Of(d.value, "eps"),
    };

    const tk_hf = try f32s(aa, "../fixtures/num/idx_token_keys_all.bin");
    const tk = try g.fromSlice(f32, tk_hf, &.{ 1, 1, @intCast(kv_len), @intCast(dim) });
    defer g.drop(tk);
    const x = try g.fromSlice(f32, try f32s(aa, "../fixtures/num/idx_hidden_rows.bin"), &.{ 1, @intCast(rows), @intCast(hidden) });
    defer g.drop(x);
    const q_resid = try g.fromSlice(f32, try f32s(aa, "../fixtures/num/idx_q_resid.bin"), &.{ 1, @intCast(rows), @intCast(q_lora) });
    defer g.drop(q_resid);
    const pos_i = try i32s(aa, "../fixtures/num/idx_positions.bin");
    var pos: [32]u32 = undefined;
    for (pos_i, 0..) |p, i| pos[i] = @intCast(p);
    const mask = try ix.recallMask(&g, x, q_resid, tk, @intCast(kv_len), base_pos, pos[0..rows]);
    defer _ = mlx.mlx_array_free(mask);
    const got = try read(aa, mask);
    const recalled = try i32s(aa, "../fixtures/num/idx_recalled.bin");

    for (0..rows) |r| {
        var allowed: [256]bool = @splat(false);
        for (0..group_topk) |k| {
            const grp: usize = @intCast(recalled[r * group_topk + k]);
            for (0..ratio) |i| allowed[grp * ratio + i] = true;
        }
        const qpos = kv_len - rows + r;
        for (0..kv_len) |slot| {
            const want = allowed[slot] and slot <= qpos;
            const is_open = got[r * kv_len + slot] > -1e30;
            if (want != is_open) std.debug.print("indexer mismatch row={d} slot={d} want={} got={} v={d:.3}\n", .{ r, slot, want, is_open, got[r * kv_len + slot] });
            try testing.expectEqual(want, is_open);
        }
    }
    std.debug.print("[hy_v4 oracle] indexer: {d} rows, {d} groups, recall mask pinned\n", .{ rows, kv_len / ratio });
}

test "hy_v4 oracle: the latent MoE, shared expert and combine norm match the reference dump" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    var g = Graph.init(testing.allocator, mlx.mlx_default_cpu_stream_new());
    defer g.deinit();
    const d = try dims("../fixtures/num/moe.json");
    defer d.deinit();
    const hidden: usize = @intCast(try intOf(d.value, "hidden"));
    const latent: usize = @intCast(try intOf(d.value, "latent"));
    const mi: usize = @intCast(try intOf(d.value, "mi"));
    const experts: usize = @intCast(try intOf(d.value, "experts"));
    const top_k: usize = @intCast(try intOf(d.value, "top_k"));
    const rows: usize = @intCast(try intOf(d.value, "rows"));
    const shared_mi: usize = @intCast(try intOf(d.value, "shared_mi"));

    const gate_hf = try f32s(aa, "../fixtures/num/moe_gate.bin");
    var gate_t = try aa.alloc(f32, gate_hf.len);
    for (0..experts) |e| {
        for (0..hidden) |i| gate_t[i * experts + e] = gate_hf[e * hidden + i];
    }

    const m = moe.Moe{
        .router = .{
            .gate_t = try g.fromSlice(f32, gate_t, &.{ @intCast(hidden), @intCast(experts) }),
            .bias = try vec(&g, try f32s(aa, "../fixtures/num/moe_bias.bin"), aa),
            .experts = @intCast(experts),
            .hidden = @intCast(hidden),
            .top_k = @intCast(top_k),
            .norm = true,
            .scale = try f32Of(d.value, "scale"),
            .n_group = 1,
            .topk_group = 1,
        },
        .routed_down = try dense(&g, try f32s(aa, "../fixtures/num/moe_routed_down.bin"), latent, hidden, aa),
        .routed_up = try dense(&g, try f32s(aa, "../fixtures/num/moe_routed_up.bin"), hidden, latent, aa),
        .gate_up = .{ .w = try g.fromSlice(f32, try f32s(aa, "../fixtures/num/moe_gate_up.bin"), &.{ @intCast(experts), @intCast(2 * mi), @intCast(latent) }) },
        .down = .{ .w = try g.fromSlice(f32, try f32s(aa, "../fixtures/num/moe_down.bin"), &.{ @intCast(experts), @intCast(latent), @intCast(mi) }) },
        .shared_gate = try dense(&g, try f32s(aa, "../fixtures/num/moe_shared_gate.bin"), shared_mi, hidden, aa),
        .shared_up = try dense(&g, try f32s(aa, "../fixtures/num/moe_shared_up.bin"), shared_mi, hidden, aa),
        .shared_down = try dense(&g, try f32s(aa, "../fixtures/num/moe_shared_down.bin"), hidden, shared_mi, aa),
        .combine_output_norm = try vec(&g, try f32s(aa, "../fixtures/num/moe_combine_norm.bin"), aa),
        .rms_eps = try f32Of(d.value, "eps"),
        .swiglu_limit = try f32Of(d.value, "swiglu_limit"),
    };
    const x = try g.fromSlice(f32, try f32s(aa, "../fixtures/num/moe_x.bin"), &.{ 1, @intCast(rows), @intCast(hidden) });
    defer g.drop(x);
    const out = try m.forward(&g, x);
    defer g.drop(out);
    try checkParity("moe", try read(aa, out), try f32s(aa, "../fixtures/num/moe_out.bin"));
}

test "hy_v4 oracle: the Engram gate and its per-lane branch match the reference dump" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    var g = Graph.init(testing.allocator, mlx.mlx_default_cpu_stream_new());
    defer g.deinit();
    const d = try dims("../fixtures/num/engram.json");
    defer d.deinit();
    const hidden: usize = @intCast(try intOf(d.value, "hidden"));
    const lanes: usize = @intCast(try intOf(d.value, "lanes"));
    const heads: usize = @intCast(try intOf(d.value, "heads"));
    const head_dim: usize = @intCast(try intOf(d.value, "head_dim"));
    const emb_width: usize = @intCast(try intOf(d.value, "emb_width"));
    const rows: usize = @intCast(try intOf(d.value, "rows"));

    var e: engram.Engram = .{
        .tbl = undefined,
        .lookup_host = &.{},
        .mult = .{ 0, 0, 0 },
        .moduli = @splat(0),
        .head_off = @splat(0),
        .heads = @intCast(heads),
        .head_dim = @intCast(head_dim),
        .emb_width = @intCast(emb_width),
        .key_proj = undefined,
        .key_bias = undefined,
        .norm_key = undefined,
        .norm_query = undefined,
        .value_proj = try dense(&g, try f32s(aa, "../fixtures/num/eng_value_w.bin"), hidden, emb_width, aa),
        .value_bias = try vec(&g, try f32s(aa, "../fixtures/num/eng_value_b.bin"), aa),
        .head_norm_w = try g.fromSlice(f32, try f32s(aa, "../fixtures/num/eng_head_norm.bin"), &.{ @intCast(heads), @intCast(head_dim) }),
        .pre_gamma = try vec(&g, try f32s(aa, "../fixtures/num/eng_pre_gamma.bin"), aa),
        .hc_scales = undefined,
        .pad_id = 0,
        .image_token_id = 0,
        .video_token_id = 0,
        .lookup_size = 0,
        .fp8 = false,
        .hc_mult = @intCast(lanes),
        .hidden = @intCast(hidden),
    };
    inline for (0..4) |k| {
        const kw = comptime std.fmt.comptimePrint("../fixtures/num/eng_key_w{d}.bin", .{k});
        const kb = comptime std.fmt.comptimePrint("../fixtures/num/eng_key_b{d}.bin", .{k});
        const nk = comptime std.fmt.comptimePrint("../fixtures/num/eng_norm_key{d}.bin", .{k});
        const nq = comptime std.fmt.comptimePrint("../fixtures/num/eng_norm_query{d}.bin", .{k});
        e.key_proj[k] = try dense(&g, try f32s(aa, kw), hidden, emb_width, aa);
        e.key_bias[k] = try vec(&g, try f32s(aa, kb), aa);
        e.norm_key[k] = try vec(&g, try f32s(aa, nk), aa);
        e.norm_query[k] = try vec(&g, try f32s(aa, nq), aa);
    }
    const scales = try f32s(aa, "../fixtures/num/eng_hc_scales.bin");
    inline for (0..4) |k| {
        e.hc_scales[k] = try g.fromSlice(f32, scales[k * hidden ..][0..hidden], &.{@intCast(hidden)});
    }

    const emb = try g.fromSlice(f32, try f32s(aa, "../fixtures/num/eng_emb.bin"), &.{ @intCast(rows), @intCast(emb_width) });
    defer g.drop(emb);
    const streams = try g.fromSlice(f32, try f32s(aa, "../fixtures/num/eng_streams.bin"), &.{ 1, @intCast(rows), @intCast(lanes), @intCast(hidden) });
    defer g.drop(streams);
    const out = try e.gateResidual(&g, emb, streams);
    defer g.drop(out);
    try checkParity("engram gate", try read(aa, out), try f32s(aa, "../fixtures/num/eng_gate.bin"));
}
