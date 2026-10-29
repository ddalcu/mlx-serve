//! Parity proof for hy_v3's MTP head behind `SpecHooks.head`: a `model-mtp.safetensors` head (all 4-bit; or 8-bit
//! group-32 projections over 3-bit experts, Hunyuan's own mix; or bf16) beside a 4-bit trunk, run by the plugin and by
//! the legacy `MtpModel` (`appendKvOnly`, `forwardWithMrope`, its values recorded to `fixtures/`) on caches of their
//! own, the head output bit-equal at every step; then served through the driver against the legacy sidecar arm.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const mtp_mod = @import("../../sdk/spec/mtp/mtp.zig");
const cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const FsPackDir = @import("../../host/registry/pack_dir.zig").FsPackDir;
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const plugin_head = @import("../../engines/mlx/plugin/head.zig");
const mtpRun = @import("../../engines/mlx/plugin/proof_mtp.zig").mtpRun;
const MtpHeadRef = @import("../../engines/mlx/spec/mtp_head_ref.zig").MtpHeadRef;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const hy3 = @import("plugin.zig");
const trunk = @import("proof_test.zig");

const testing = std.testing;
const H: c_int = 256;
const HD: c_int = 128;
const EXPERTS: c_int = 8;
const MOE_INTER: c_int = 128;

const Quant = enum { q4, mixed, bf16 };

fn proj(p: *fixture.Pack, base: []const u8, shape: []const c_int, q: Quant, expert: bool) !void {
    switch (q) {
        .q4 => try p.tensor(base, shape, true),
        .bf16 => try p.tensor(base, shape, false),
        .mixed => if (expert) try p.affine(base, shape, 3, 64) else try p.affine(base, shape, 8, 32),
    }
}

fn writeHead(p: *fixture.Pack, q: Quant) !void {
    try proj(p, "mtp.eh_proj", &.{ H, 2 * H }, q, false);
    for ([_][]const u8{ "mtp.enorm.weight", "mtp.hnorm.weight", "mtp.final_layernorm.weight", "mtp.layer.input_layernorm.weight", "mtp.layer.post_attention_layernorm.weight" }) |n| try p.norm(n, H);
    try p.norm("mtp.layer.self_attn.q_norm.weight", HD);
    try p.norm("mtp.layer.self_attn.k_norm.weight", HD);
    try proj(p, "mtp.layer.self_attn.q_proj", &.{ 4 * HD, H }, q, false);
    try proj(p, "mtp.layer.self_attn.k_proj", &.{ 2 * HD, H }, q, false);
    try proj(p, "mtp.layer.self_attn.v_proj", &.{ 2 * HD, H }, q, false);
    try proj(p, "mtp.layer.self_attn.o_proj", &.{ H, 4 * HD }, q, false);
    try proj(p, "mtp.layer.mlp.router.gate", &.{ EXPERTS, H }, q, false);
    try p.denseAs("mtp.layer.mlp.expert_bias", &.{EXPERTS}, 0.3, 0, .float32);
    try proj(p, "mtp.layer.mlp.experts.gate_proj", &.{ EXPERTS, MOE_INTER, H }, q, true);
    try proj(p, "mtp.layer.mlp.experts.up_proj", &.{ EXPERTS, MOE_INTER, H }, q, true);
    try proj(p, "mtp.layer.mlp.experts.down_proj", &.{ EXPERTS, H, MOE_INTER }, q, true);
    try proj(p, "mtp.layer.mlp.shared_mlp.gate_proj", &.{ MOE_INTER, H }, q, false);
    try proj(p, "mtp.layer.mlp.shared_mlp.up_proj", &.{ MOE_INTER, H }, q, false);
    try proj(p, "mtp.layer.mlp.shared_mlp.down_proj", &.{ H, MOE_INTER }, q, false);
}

fn saveHead(hp: *fixture.Pack, dir: []const u8) !void {
    const file = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/model-mtp.safetensors", .{dir}, 0);
    defer testing.allocator.free(file);
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_save_safetensors(file.ptr, hp.map, meta));
}

fn rows(g: *ops.Graph, n: c_int, seed: u64) !mlx.mlx_array {
    var prng = std.Random.DefaultPrng.init(seed);
    const buf = try testing.allocator.alloc(f32, @intCast(n * H));
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = prng.random().floatNorm(f32) * 0.5;
    return g.astype(try g.adopt(mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, n, H }, 3, .float32)), .bfloat16);
}

const Step = struct { ids: []const i32, logits: abi.Logits, want_hidden: bool };

/// The trunk served on the legacy config, the legacy values recorded to `fixtures/head-<seed>.tape`.
fn open(p: *fixture.Pack, comptime seed: u64) !both.Both {
    const name = std.fmt.comptimePrint("fixtures/head-{x}.tape", .{seed});
    return both.Both.replay(&hy3.table, p, trunk.configJson(true), true, @embedFile(name));
}

fn headCase(q: Quant, comptime seed: u64) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try trunk.writePack(&p, true);
    var hp = fixture.Pack.init(seed + 1);
    defer hp.deinit();
    try writeHead(&hp, q);
    var b = try open(&p, seed);
    defer b.close();
    try saveHead(&hp, b.dir);
    const s = b.xfm.s;
    var fs = try FsPackDir.open(io, b.dir);
    defer fs.close();
    const pd = fs.packDir();
    const hooks = hy3.table.spec.?.head.?;
    const m = b.xfm.plugin.?.model;
    try testing.expect(hooks.load(m, &.{ .model_dir = .of(b.dir), .dir = &pd, .parsed = &b.cfg.plugin.?.parsed, .stream = .{ .ctx = s.ctx } }).isOk());

    var pcache = try KVCache.init(a, 1);
    defer pcache.deinit();
    var slot: cache_ops.Slot = .{ .kv = &pcache };
    const handles = [_]abi.Handle{slot.handle()};
    const steps = [_]Step{
        .{ .ids = &.{ 3, 17, 42, 99, 7 }, .logits = .none, .want_hidden = false },
        .{ .ids = &.{ 250, 1, 8 }, .logits = .last_row, .want_hidden = true },
        .{ .ids = &.{66}, .logits = .last_row, .want_hidden = true },
        .{ .ids = &.{ 31, 120 }, .logits = .none, .want_hidden = true },
        .{ .ids = &.{5}, .logits = .last_row, .want_hidden = true },
    };
    var pos: c_int = 0;
    for (steps, 0..) |st, i| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        const n: c_int = @intCast(st.ids.len);
        const ids = try g.fromSlice(i32, st.ids, &.{n});
        const hid = try rows(&g, n, seed + 10 + i);
        var out: abi.ForwardOut = .{};
        const v = hooks.forward(m, &.{
            .mode = if (n == 1) .decode else .prefill,
            .logits = st.logits,
            .want_hidden = st.want_hidden,
            .rows = @intCast(n),
            .tokens = .{},
            .ids = .{ .ctx = ids.ctx },
            .positions = .of(&.{@intCast(pos)}),
            .caches = .of(&handles),
            .cache_ops = &cache_ops.ops,
            .hidden_in = .{ .ctx = hid.ctx },
        }, &out);
        if (!v.isOk()) {
            std.debug.print("head forward: {s}\n", .{v.why.slice()});
            return error.PluginForward;
        }
        if (st.logits == .none and !st.want_hidden) {
            try testing.expect(out.hidden.ctx == null);
        } else try b.check(try g.adopt(.{ .ctx = out.hidden.ctx }), false);
        pos += n;
    }
    try b.expectDone();
}

/// Served: the driver's shell on the legacy config with the head beside it; the engine's head calls through
/// `MtpHeadRef.plugin` bit-equal to the legacy sidecar arm (`.qwen`), and a pinned-depth MTP round loop with equal
/// tokens and accepted counts. The trunk itself: steps past a chunk and greedy decode.
fn servedCase(q: Quant, comptime seed: u64) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try trunk.writePack(&p, true);
    var hp = fixture.Pack.init(seed + 1);
    defer hp.deinit();
    try writeHead(&hp, q);
    var b = try open(&p, seed);
    defer b.close();
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try saveHead(&hp, b.dir);
    const served = plugin_head.load(io, a, &b.xfm, b.dir) orelse return error.TestHeadNotLoaded;
    defer plugin_head.destroy(served, a);
    const h: MtpHeadRef = .{ .plugin = served };
    const x = &b.xfm;
    {
        // The engine's head calls: history, a multi-row `.logits` step, a `.mixed` step and its draft pick.
        var g = ops.Graph.init(a, b.xfm.s);
        defer g.deinit();
        const hist = try rows(&g, 3, seed + 2);
        const step = try rows(&g, 2, seed + 3);
        const one = try rows(&g, 1, seed + 4);
        var cr = try h.makeCache(a);
        defer cr.deinit();
        try h.appendHistory(x, &cr, &.{ 5, 9, 40 }, hist, 0, null, a);
        const merged = try h.forward(x, &cr, try g.fromSlice(i32, &.{ 7, 11 }, &.{2}), step, 3, .logits, null, null);
        const mixed = try h.forward(x, &cr, try g.fromSlice(i32, &.{60}, &.{1}), one, 5, .mixed, null, null);
        const outs = [_]mlx.mlx_array{ try g.adopt(merged.logits), try g.adopt(merged.hidden_next), try g.adopt(mixed.hidden_next), try g.adopt(try h.draftSelect(x, mixed.hidden_next, null)) };
        try testing.expectEqual(@as(usize, 6), cr.step());
        for (outs) |got| try b.check(got, false);
    }
    var tokens: [16]u32 = undefined;
    var accepted: std.ArrayList(u32) = .empty;
    defer accepted.deinit(a);
    try mtpRun(x, h, &tokens, &accepted);
    for ([_][]const u32{ &tokens, accepted.items }) |got| {
        const g_arr = mlx.mlx_array_new_data(got.ptr, &[_]c_int{@intCast(got.len)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(g_arr);
        try b.check(g_arr, true);
    }
    try b.expectDone();
}

// ── tests ──

test "hy_v3 MTP head: a 4-bit head is bit-equal to the legacy head" {
    try headCase(.q4, 0x4b01);
}

test "hy_v3 MTP head: 8-bit group-32 projections over 3-bit experts are bit-equal to the legacy head" {
    try headCase(.mixed, 0x4b02);
}

test "hy_v3 MTP head: a bf16 head (dense experts, which the legacy head cannot run) is refused by name" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var p = fixture.Pack.init(0x4b03);
    defer p.deinit();
    try trunk.writePack(&p, true);
    var hp = fixture.Pack.init(0x4b04);
    defer hp.deinit();
    try writeHead(&hp, .bf16);
    var b = try both.Both.openAs(&hy3.table, &p, trunk.configJson(true), true, .none);
    defer b.close();
    try saveHead(&hp, b.dir);
    var fs = try FsPackDir.open(io, b.dir);
    defer fs.close();
    const pd = fs.packDir();
    const v = hy3.table.spec.?.head.?.load(b.xfm.plugin.?.model, &.{ .model_dir = .of(b.dir), .dir = &pd, .parsed = &b.cfg.plugin.?.parsed, .stream = .{ .ctx = b.xfm.s.ctx } });
    try testing.expectEqualStrings("DenseExpertHead", v.why.slice());
}

test "hy_v3 served: the trunk and its MTP head through the driver equal the legacy path" {
    try servedCase(.q4, 0x4b11);
    try servedCase(.mixed, 0x4b12);
}
