//! The qwen3_5 MTP head (`SpecHooks.head`) served through the driver against the legacy sidecar arm bound to the
//! legacy shell, over one tiny dense pack and a qwen-layout head file: history appends, a merged last-row step, an
//! every-row step and the draft pick bit-equal, then a pinned-depth greedy round loop with the same tokens and
//! per-round accepted counts. The table lends its trunk head (`trunk_weight`); the proof turns layer capture off, so
//! the head alone borrows it.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const mtp_mod = @import("../../sdk/spec/mtp/mtp.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const pack = @import("../../sdk/bases/proof/qwen35_pack.zig");
const plugin_head = @import("../../engines/mlx/plugin/head.zig");
const depth_bounds = @import("../../engines/mlx/spec/mtp_depth_bounds.zig");
const dflash_mod = @import("../../engines/mlx/spec/dflash/dflash.zig");
const MtpHeadRef = @import("../../engines/mlx/spec/mtp_head_ref.zig").MtpHeadRef;
const MtpCacheRef = @import("../../engines/mlx/spec/mtp_head_ref.zig").MtpCacheRef;
const Generator = @import("../../engines/mlx/generate/generator.zig").Generator;
const Tokenizer = @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const transformer_mod = @import("../../engines/mlx/transformer.zig");
const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;
const SSMCacheEntry = transformer_mod.SSMCacheEntry;
const ForwardCtx = transformer_mod.ForwardCtx;
const chain_build = @import("../../engines/mlx/spec/mtp/chain_build.zig");
const served_test = @import("served_test.zig");
const qwen3_5 = @import("plugin.zig");

const testing = std.testing;
const Array = mlx.mlx_array;

const H = 256;

/// A head's geometry: the trunk's width, its attention heads (hd 256) and the MLP width.
pub const Geo = struct { hidden: usize = H, heads: usize = 4, kv_heads: usize = 2, inter: usize = 512 };

/// A qwen-layout dense head (`mtp.*`, the q projection carrying the output gate) at the tiny pack's geometry.
fn writeHead(path: []const u8, geo: Geo) !void {
    const h = geo.hidden;
    const s = fixture.stream();
    const Fx = dflash_mod.TinyFix;
    const map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(map);
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try Fx.putW(map, "mtp.fc.weight", h, 2 * h, 301, s);
    for ([_][]const u8{ "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight", "mtp.layers.0.input_layernorm.weight", "mtp.layers.0.post_attention_layernorm.weight" }) |k|
        try Fx.putNorm(map, k, h, s);
    try Fx.putNorm(map, "mtp.layers.0.self_attn.q_norm.weight", 256, s);
    try Fx.putNorm(map, "mtp.layers.0.self_attn.k_norm.weight", 256, s);
    try Fx.putW(map, "mtp.layers.0.self_attn.q_proj.weight", 2 * geo.heads * 256, h, 302, s);
    try Fx.putW(map, "mtp.layers.0.self_attn.k_proj.weight", geo.kv_heads * 256, h, 303, s);
    try Fx.putW(map, "mtp.layers.0.self_attn.v_proj.weight", geo.kv_heads * 256, h, 304, s);
    try Fx.putW(map, "mtp.layers.0.self_attn.o_proj.weight", h, geo.heads * 256, 305, s);
    try Fx.putW(map, "mtp.layers.0.mlp.gate_proj.weight", geo.inter, h, 306, s);
    try Fx.putW(map, "mtp.layers.0.mlp.up_proj.weight", geo.inter, h, 307, s);
    try Fx.putW(map, "mtp.layers.0.mlp.down_proj.weight", h, geo.inter, 308, s);
    const file = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/model-mtp.safetensors", .{path}, 0);
    defer testing.allocator.free(file);
    try mlx.check(mlx.mlx_save_safetensors(file.ptr, map, meta));
}

/// A MoE trunk's head (4-bit): the router, eight stacked experts and the gated shared expert in place of the SwiGLU.
fn writeMoeHead(path: []const u8) !void {
    var hp = fixture.Pack.init(0x47b0);
    defer hp.deinit();
    try hp.weight("mtp.fc", H, 2 * H, true);
    for ([_][]const u8{ "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight", "mtp.layers.0.input_layernorm.weight", "mtp.layers.0.post_attention_layernorm.weight" }) |k|
        try hp.norm(k, H);
    try hp.norm("mtp.layers.0.self_attn.q_norm.weight", 256);
    try hp.norm("mtp.layers.0.self_attn.k_norm.weight", 256);
    try hp.weight("mtp.layers.0.self_attn.q_proj", 2 * 4 * 256, H, true);
    try hp.weight("mtp.layers.0.self_attn.k_proj", 2 * 256, H, true);
    try hp.weight("mtp.layers.0.self_attn.v_proj", 2 * 256, H, true);
    try hp.weight("mtp.layers.0.self_attn.o_proj", H, 4 * 256, true);
    try hp.weight("mtp.layers.0.mlp.gate", 8, H, true);
    try hp.tensor("mtp.layers.0.mlp.switch_mlp.gate_proj", &.{ 8, 128, H }, true);
    try hp.tensor("mtp.layers.0.mlp.switch_mlp.up_proj", &.{ 8, 128, H }, true);
    try hp.tensor("mtp.layers.0.mlp.switch_mlp.down_proj", &.{ 8, H, 128 }, true);
    try hp.weight("mtp.layers.0.mlp.shared_expert.gate_proj", 128, H, true);
    try hp.weight("mtp.layers.0.mlp.shared_expert.up_proj", 128, H, true);
    try hp.weight("mtp.layers.0.mlp.shared_expert.down_proj", H, 128, true);
    try hp.weight("mtp.layers.0.mlp.shared_expert_gate", 1, H, true);
    const file = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/model-mtp.safetensors", .{path}, 0);
    defer testing.allocator.free(file);
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_save_safetensors(file.ptr, hp.map, meta));
}

/// `rows` trunk hidden rows of width `h`, in `dt` (the trunk's activations).
fn hiddenRows(g: *ops.Graph, rows: u32, h: u32, dt: mlx.mlx_dtype, seed: u64) !Array {
    var prng = std.Random.DefaultPrng.init(seed);
    const buf = try testing.allocator.alloc(f32, rows * h);
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = prng.random().floatNorm(f32) * 0.5;
    const f = try g.adopt(mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, @intCast(rows), @intCast(h) }, 3, .float32));
    return g.astype(f, dt);
}

fn idArr(g: *ops.Graph, ids: []const i32) !Array {
    return g.adopt(mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(ids.len)}, 1, .int32));
}

/// A greedy pinned-depth MTP run: the emitted stream and each round's accepted count.
fn mtpRun(xfm: *Transformer, head: MtpHeadRef, tokens: []u32, accepted: *std.ArrayList(u32)) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tok_dummy: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28 };
    var gen = try Generator.initWithOptions(io, a, xfm, &tok_dummy, &prompt, @intCast(tokens.len), .{ .temperature = 0.0 }, &.{}, .{
        .mtp_enabled = true,
        .mtp = head,
        .model_has_mtp = true,
        .mtp_depth = 2,
    });
    defer gen.deinit(a);
    var n: usize = 0;
    while (try gen.nextMtp(a)) |res| {
        defer a.free(res.tokens);
        @memcpy(tokens[n..][0..res.tokens.len], res.tokens);
        n += res.tokens.len;
        try accepted.append(a, res.accepted_tokens);
    }
    try testing.expectEqual(tokens.len, n);
    try testing.expect(gen.mtp_attempted > 0);
}

fn noCapture(parsed: *@import("../../abi/plugin.zig").Parsed) void {
    parsed.caps.dflash_capture = false;
}

test "qwen3_5 MTP head: head steps, draft picks and a pinned-depth round loop equal the legacy sidecar arm" {
    var pk = fixture.Pack.init(0x47a1);
    defer pk.deinit();
    try pack.writePack(&pk, false, true, 8);
    try expectHead(&qwen3_5.table, &pk, pack.config(false, true, 8), false, "mtp-dense");
}

test "qwen3_5 MTP head: a MoE trunk's routed-expert head equals the legacy arm (steps, draft picks, rounds)" {
    var pk = fixture.Pack.init(0x47a2);
    defer pk.deinit();
    try pack.writePack(&pk, true, true, 8);
    try expectHead(&qwen3_5.table, &pk, pack.config(true, true, 8), true, "mtp-moe");
}

test "qwen3_5 MTP head: a qwen3_next trunk's routed-expert head equals the legacy arm" {
    const next = @import("../qwen3_next/proof_test.zig");
    var pk = fixture.Pack.init(0x47a3);
    defer pk.deinit();
    try next.writePack(&pk, .{});
    const json = try next.config(testing.allocator, .{});
    defer testing.allocator.free(json);
    try expectHead(&@import("../qwen3_next/plugin.zig").table, &pk, json, true, "mtp-next");
}

pub fn expectHead(table: *const @import("../../abi/plugin.zig").ModelTable, pk: *fixture.Pack, json: []const u8, comptime moe: bool, comptime tape: []const u8) !void {
    return expectHeadAt(table, pk, json, moe, .{}, "qwen3_5", tape);
}

/// `expectHead` with a dense head of geometry `geo`, its tape under `family`.
pub fn expectHeadAt(table: *const @import("../../abi/plugin.zig").ModelTable, pk: *fixture.Pack, json: []const u8, comptime moe: bool, geo: Geo, comptime family: []const u8, comptime tape: []const u8) !void {
    const a = testing.allocator;
    const s = fixture.stream();
    var p: served_test.Pair = undefined;
    // Without layer capture: the head reads the trunk the shell borrows for it alone.
    try p.openWith(table, pk, json, &noCapture, try served_test.oracle(family, tape));
    defer p.close();
    try testing.expect(!p.xfm.plugin.?.drafters and p.xfm.plugin.?.trunk);
    if (moe) try writeMoeHead(p.dir) else try writeHead(p.dir, geo);

    var heads = try Heads.open(&p);
    defer heads.deinit();
    const loaded: *qwen3_5.Loaded = @ptrCast(@alignCast(p.xfm.plugin.?.model));
    try testing.expectEqual(moe, loaded.head.?.moe != null);

    const hs = heads.refs();
    const xs = p.shells();
    var caches: [2]MtpCacheRef = undefined;
    for (hs, caches[0..hs.len]) |h, *c| c.* = try h.makeCache(a);
    defer for (caches[0..hs.len]) |*c| c.deinit();
    {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        const width = p.cfg.hidden_size;
        const dt = p.cfg.actDtype();
        const hist = try hiddenRows(&g, 3, width, dt, 1);
        const step = try hiddenRows(&g, 2, width, dt, 2);
        const one = try hiddenRows(&g, 1, width, dt, 3);
        const many = try hiddenRows(&g, 3, width, dt, 4);
        var outs: [2][5]Array = undefined;
        for (hs, xs, caches[0..hs.len], 0..) |h, x, *c, k| {
            try h.appendHistory(x, c, &.{ 5, 9, 40 }, hist, 0, null, a);
            const merged = try h.forward(x, c, try idArr(&g, &.{ 7, 11 }), step, 3, .logits, null, null);
            const mixed = try h.forward(x, c, try idArr(&g, &.{60}), one, 5, .mixed, null, null);
            const rows = try h.forward(x, c, try idArr(&g, &.{ 61, 2, 90 }), many, 6, .mixed, null, null);
            outs[k] = .{ try g.adopt(merged.logits), try g.adopt(merged.hidden_next), try g.adopt(mixed.hidden_next), try g.adopt(rows.hidden_next), try g.adopt(try h.draftSelect(x, mixed.hidden_next, null)) };
            try testing.expectEqual(@as(usize, 9), c.step());
        }
        try served_test.checkRows(&p, &outs, xs.len);
    }

    depth_bounds.active = .{ .min = 2, .max = 2 };
    defer depth_bounds.active = .{};
    var tokens: [2][16]u32 = undefined;
    var accepted: [2]std.ArrayList(u32) = .{ .empty, .empty };
    defer for (&accepted) |*l| l.deinit(a);
    for (hs, xs, 0..) |h, x, k| try mtpRun(x, h, &tokens[k], &accepted[k]);
    const last = xs.len - 1;
    try p.oracle.stream(&tokens[last]);
    try p.oracle.stream(accepted[last].items);
    try p.oracle.done();
}

/// The served head beside the pair's shell.
const Heads = struct {
    served: *plugin_head.Head,
    buf: [1]MtpHeadRef = undefined,

    fn open(p: *served_test.Pair) !Heads {
        const io = std.Io.Threaded.global_single_threaded.io();
        return .{ .served = plugin_head.load(io, testing.allocator, &p.xfm, p.dir) orelse return error.HeadNotLoaded };
    }

    fn deinit(h: *Heads) void {
        plugin_head.destroy(h.served, testing.allocator);
    }

    /// The heads in `Pair.shells` order.
    fn refs(h: *Heads) []const MtpHeadRef {
        h.buf = .{.{ .plugin = h.served }};
        return &h.buf;
    }
};

/// A dense head in the measured layout: bf16 fc and norms, every projection 4-bit/gs64.
fn writeQuantHead(path: []const u8) !void {
    var hp = fixture.Pack.init(0x47b1);
    defer hp.deinit();
    try hp.weight("mtp.fc", H, 2 * H, false);
    for ([_][]const u8{ "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight", "mtp.layers.0.input_layernorm.weight", "mtp.layers.0.post_attention_layernorm.weight" }) |k|
        try hp.norm(k, H);
    try hp.norm("mtp.layers.0.self_attn.q_norm.weight", 256);
    try hp.norm("mtp.layers.0.self_attn.k_norm.weight", 256);
    try hp.weight("mtp.layers.0.self_attn.q_proj", 2 * 4 * 256, H, true);
    try hp.weight("mtp.layers.0.self_attn.k_proj", 2 * 256, H, true);
    try hp.weight("mtp.layers.0.self_attn.v_proj", 2 * 256, H, true);
    try hp.weight("mtp.layers.0.self_attn.o_proj", H, 4 * 256, true);
    try hp.weight("mtp.layers.0.mlp.gate_proj", 512, H, true);
    try hp.weight("mtp.layers.0.mlp.up_proj", 512, H, true);
    try hp.weight("mtp.layers.0.mlp.down_proj", H, 512, true);
    const file = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/model-mtp.safetensors", .{path}, 0);
    defer testing.allocator.free(file);
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_save_safetensors(file.ptr, hp.map, meta));
}

test "qwen3_5 MTP head: the measured-cost fingerprint, trunk projections and profile are the legacy sidecar arm's" {
    var pk = fixture.Pack.init(0x47a4);
    defer pk.deinit();
    try pack.writePack(&pk, false, true, 8);
    var p: served_test.Pair = undefined;
    try p.openWith(&qwen3_5.table, &pk, pack.config(false, true, 8), &noCapture, try served_test.oracle("qwen3_5", "mtp-profile"));
    defer p.close();
    try writeQuantHead(p.dir);
    var heads = try Heads.open(&p);
    defer heads.deinit();
    const served = heads.served;
    try testing.expectEqual(@as(?mtp_mod.HeadFingerprint, .{ .bits = 4, .group_size = 64 }), served.fingerprint);
    const got = [_]u32{ @intFromEnum(served.m5NaxCostProfile(&p.xfm)), @intFromBool(served.draft_head != null), @intFromBool(served.rerank_coarse != null), @intFromBool(p.xfm.mtp_uniform_affine_trunk), @intFromBool(p.xfm.mtp_oqe_affine_trunk) };
    try p.oracle.stream(&got);
    try p.oracle.done();
}

/// One request on `xfm` with its own trunk cache, drafting on `head`.
const Slot = struct {
    cache: KVCache,
    off: usize = 0,
    entries: []SSMCacheEntry,
    gen: Generator = undefined,

    fn init(a: std.mem.Allocator, xfm: *Transformer, tok: *Tokenizer, head: MtpHeadRef, prompt: []const u32) !*Slot {
        const io = std.Io.Threaded.global_single_threaded.io();
        const slot = try a.create(Slot);
        errdefer a.destroy(slot);
        slot.* = .{ .cache = try KVCache.init(a, xfm.config.num_hidden_layers), .entries = try fixture.newEntries(xfm.config.num_hidden_layers) };
        errdefer {
            fixture.freeEntries(slot.entries);
            slot.cache.deinit();
        }
        const ctx: ForwardCtx = .{ .cache = &slot.cache, .moe_seq_offset = &slot.off, .ssm_entries = slot.entries, .capture_hidden = null, .vision_embeddings = null };
        slot.gen = try Generator.initWithOptions(io, a, xfm, tok, prompt, 64, .{ .temperature = 0 }, &.{}, .{
            .ctx = ctx,
            .mtp = head,
            .mtp_enabled = true,
            .model_has_mtp = true,
            .mtp_depth = 2,
            .skip_lazy_preforward = true,
        });
        return slot;
    }

    fn deinit(slot: *Slot, a: std.mem.Allocator) void {
        slot.gen.deinit(a);
        fixture.freeEntries(slot.entries);
        slot.cache.deinit();
        a.destroy(slot);
    }
};

/// Two requests' draft chains built as lanes of one head forward (`mtpChainBuildLanes`) on each arm: the same drafts
/// and the same chained hidden rows, two rounds deep (the second merges each request's committed history first).
fn expectLanes(comptime moe: bool, comptime seed: u64) !void {
    const a = testing.allocator;
    var pk = fixture.Pack.init(seed);
    defer pk.deinit();
    try pack.writePack(&pk, moe, true, 8);
    var p: served_test.Pair = undefined;
    try p.openWith(&qwen3_5.table, &pk, pack.config(moe, true, 8), &noCapture, try served_test.oracle("qwen3_5", std.fmt.comptimePrint("mtp-lanes-{x}", .{seed})));
    defer p.close();
    if (moe) try writeMoeHead(p.dir) else try writeHead(p.dir, .{});
    var hb = try Heads.open(&p);
    defer hb.deinit();
    const heads = hb.refs();
    for (heads) |h| try testing.expect(h.lanesWith(h));
    if (heads.len == 2) try testing.expect(!heads[1].lanesWith(heads[0]));

    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    depth_bounds.active = .{ .min = 2, .max = 2 };
    defer depth_bounds.active = .{};
    const prompts = [_][]const u32{ &.{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28 }, &.{ 40, 2, 9, 61, 7, 33, 5, 19, 8 } };
    const xs = p.shells();
    var slots: [2][2]*Slot = undefined;
    for (heads, xs, 0..) |h, x, k| for (prompts, 0..) |pr, r| {
        slots[k][r] = try Slot.init(a, x, &tok, h, pr);
    };
    defer for (slots[0..xs.len]) |arm| for (arm) |sl| sl.deinit(a);
    var g = ops.Graph.init(a, fixture.stream());
    defer g.deinit();
    for (0..2) |_| {
        var drafts: [2][2][2]Array = undefined;
        var hidden: [2][2]Array = undefined;
        for (0..xs.len) |k| {
            var gens: [2]*Generator = undefined;
            var opens: [2]Generator.MtpRoundOpen = undefined;
            var chains: [2]Generator.MtpPreDraft = undefined;
            for (0..2) |r| {
                gens[r] = &slots[k][r].gen;
                gens[r].mtp_depth_current = 2;
                gens[r].spec_cost_solo = false;
                // A group row: the round opens for the group's build and never pre-drafts alone.
                gens[r].mtp_batch_head = true;
                opens[r] = switch (try gens[r].mtpRoundBegin(a)) {
                    .open => |o| o,
                    else => return error.TestExpectedOpenRound,
                };
                chains[r] = opens[r].chain;
            }
            try chain_build.mtpChainBuildLanes(&gens, &chains, 0, 2);
            for (0..2) |r| {
                for (0..2) |d| drafts[k][r][d] = try g.adopt(try Transformer.axisView(fixture.stream(), chains[r].draft_arrs[d], 0, 0));
                hidden[k][r] = try g.adopt(try Transformer.axisView(fixture.stream(), chains[r].h_chain.?, 0, 0));
                opens[r].chain = chains[r];
                var st = try gens[r].mtpRoundContinue(a, opens[r]);
                defer st.deinit(a);
                try gens[r].mtpRoundVerify(&st);
                if (try gens[r].mtpRoundFinish(a, &st)) |res| a.free(res.tokens);
            }
        }
        for (0..2) |r| {
            for (0..2) |d| {
                const col = [2]Array{ drafts[0][r][d], drafts[1][r][d] };
                try p.check(col[0..xs.len]);
            }
            const col = [2]Array{ hidden[0][r], hidden[1][r] };
            try p.check(col[0..xs.len]);
        }
    }
    try p.oracle.done();
}

test "qwen3_5 MTP head: a group drafts as lanes of one head forward, equal to the legacy sidecar's lanes" {
    try expectLanes(false, 0x47a5);
}

test "qwen3_5 MTP head: a MoE trunk's head drafts lanes equal to the legacy arm's" {
    try expectLanes(true, 0x47a6);
}

test "qwen3_5 MTP head: an image turn's steps rope as the legacy arm ropes them, inside and past the M-RoPE table" {
    const a = testing.allocator;
    var pk = fixture.Pack.init(0x47a7);
    defer pk.deinit();
    try pack.writePack(&pk, false, true, 8);
    var p: served_test.Pair = undefined;
    try p.openWith(&qwen3_5.table, &pk, pack.config(false, true, 8), &noCapture, try served_test.oracle("qwen3_5", "mtp-image"));
    defer p.close();
    try writeHead(p.dir, .{});
    var hb = try Heads.open(&p);
    defer hb.deinit();
    const heads = hb.refs();
    const xs = p.shells();
    for (xs) |x| x.config.mrope_section = .{ 11, 11, 10 };

    // Six prompt positions (an image's rows share t), then generated text at `absolute + delta`.
    const table = [_]i32{ 0, 1, 2, 2, 2, 3 } ++ [_]i32{ 0, 1, 2, 3, 3, 4 } ++ [_]i32{ 0, 1, 2, 3, 4, 5 };
    const mctx: mtp_mod.MropeContext = .{ .pos = &table, .total = 6, .delta = -2 };
    var caches: [2]MtpCacheRef = undefined;
    for (heads, caches[0..heads.len]) |h, *c| c.* = try h.makeCache(a);
    defer for (caches[0..heads.len]) |*c| c.deinit();
    var g = ops.Graph.init(a, fixture.stream());
    defer g.deinit();
    const width = p.cfg.hidden_size;
    const dt = p.cfg.actDtype();
    const hist = try hiddenRows(&g, 4, width, dt, 11);
    const straddle = try hiddenRows(&g, 3, width, dt, 12);
    const one = try hiddenRows(&g, 1, width, dt, 13);
    const many = try hiddenRows(&g, 3, width, dt, 14);
    var outs: [2][5]Array = undefined;
    for (heads, xs, caches[0..heads.len], 0..) |h, x, *c, k| {
        try h.appendHistory(x, c, &.{ 5, 9, 40, 7 }, hist, 0, mctx, a);
        // Rows 4 and 5 sit inside the table, row 6 past it.
        const merged = try h.forward(x, c, try idArr(&g, &.{ 11, 12, 13 }), straddle, 4, .logits, mctx, null);
        const mixed = try h.forward(x, c, try idArr(&g, &.{60}), one, 7, .mixed, mctx, null);
        const rows = try h.forward(x, c, try idArr(&g, &.{ 61, 2, 90 }), many, 8, .mixed, mctx, null);
        outs[k] = .{ try g.adopt(merged.logits), try g.adopt(merged.hidden_next), try g.adopt(mixed.hidden_next), try g.adopt(rows.hidden_next), try g.adopt(try h.draftSelect(x, mixed.hidden_next, null)) };
        try testing.expectEqual(@as(usize, 11), c.step());
    }
    try served_test.checkRows(&p, &outs, xs.len);
    try p.oracle.done();
}
