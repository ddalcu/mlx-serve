//! Parity proof for nemotron_h's MTP head behind `SpecHooks.head`: a head in sevren-ai's layout (bare keys, a dense
//! `eh_proj` and router) and in TensorFold's (`layers.0/1`, every projection 4-bit), beside a 4-bit trunk, run by the
//! plugin against the deleted legacy `MtpModel` (`appendKvOnly`, `forwardWithMrope`), replayed from
//! `fixtures/head-*.tape`: history rows, multi-row and one-row `.last_row` steps and an every-row `want_hidden` step,
//! the head output bit-equal each time.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const FsPackDir = @import("../../host/registry/pack_dir.zig").FsPackDir;
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const nemotron = @import("plugin.zig");
const trunk = @import("proof_test.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const plugin_head = @import("../../engines/mlx/plugin/head.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const mtpRun = @import("../../engines/mlx/plugin/proof_mtp.zig").mtpRun;
const MtpHeadRef = @import("../../engines/mlx/spec/mtp_head_ref.zig").MtpHeadRef;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;

const testing = std.testing;
const H: c_int = 128;

fn writeHead(p: *fixture.Pack, tf: bool) !void {
    var buf: [160]u8 = undefined;
    const attn = if (tf) "layers.0" else "attn";
    const moe = if (tf) "layers.1" else "moe";
    const k = struct {
        fn f(b: []u8, base: []const u8, rest: []const u8) ![]const u8 {
            return std.fmt.bufPrint(b, "{s}.{s}", .{ base, rest });
        }
    }.f;
    if (tf) try p.weight("layers.0.eh_proj", H, 2 * H, true) else try p.dense("eh_proj", &.{ H, 2 * H }, 0.1, 0);
    try p.norm(if (tf) "layers.0.enorm.weight" else "enorm", H);
    try p.norm(if (tf) "layers.0.hnorm.weight" else "hnorm", H);
    try p.norm(if (tf) "layers.1.final_layernorm.weight" else "final_norm", H);
    try p.norm(try k(&buf, attn, "norm.weight"), H);
    try p.weight(try k(&buf, attn, "mixer.q_proj"), 4 * 32, H, true);
    try p.weight(try k(&buf, attn, "mixer.k_proj"), 2 * 32, H, true);
    try p.weight(try k(&buf, attn, "mixer.v_proj"), 2 * 32, H, true);
    try p.weight(try k(&buf, attn, "mixer.o_proj"), H, 4 * 32, true);
    try p.norm(try k(&buf, moe, "norm.weight"), H);
    if (tf) try p.weight(try k(&buf, moe, "mixer.gate"), 8, H, true) else try p.dense(try k(&buf, moe, "mixer.gate.weight"), &.{ 8, H }, 0.5, 0);
    try p.denseAs(try k(&buf, moe, "mixer.gate.e_score_correction_bias"), &.{8}, 0.2, 0, .float32);
    try p.tensor(try k(&buf, moe, "mixer.switch_mlp.fc1"), &.{ 8, 64, H }, true);
    try p.tensor(try k(&buf, moe, "mixer.switch_mlp.fc2"), &.{ 8, H, 64 }, true);
    try p.weight(try k(&buf, moe, "mixer.shared_experts.up_proj"), 128, H, true);
    try p.weight(try k(&buf, moe, "mixer.shared_experts.down_proj"), H, 128, true);
}

fn rows(g: *ops.Graph, n: c_int, seed: u64) !mlx.mlx_array {
    var prng = std.Random.DefaultPrng.init(seed);
    const buf = try testing.allocator.alloc(f32, @intCast(n * H));
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = prng.random().floatNorm(f32) * 0.5;
    return g.astype(try g.adopt(mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, n, H }, 3, .float32)), .bfloat16);
}

const Step = struct { ids: []const i32, logits: abi.Logits, want_hidden: bool };

/// The 4-bit trunk served on the legacy config with the head (`tf`: TensorFold's file, else sevren-ai's) beside it.
fn open(p: *fixture.Pack, hp: *fixture.Pack, comptime tf: bool, comptime seed: u64) !both.Both {
    const a = testing.allocator;
    try trunk.writeTrunk(p);
    try writeHead(hp, tf);
    var cbuf: [2048]u8 = undefined;
    const json = try trunk.trunkConfig(&cbuf);
    const name = std.fmt.comptimePrint("fixtures/head-{x}.tape", .{seed});
    var b = try both.Both.replay(&nemotron.table, p, json, true, @embedFile(name));
    errdefer b.close();
    const file = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ b.dir, if (tf) "mtp-4bit.safetensors" else "mtp_head.safetensors" }, 0);
    defer a.free(file);
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_save_safetensors(file.ptr, hp.map, meta));
    return b;
}

/// The table's head hooks against the recorded legacy `MtpModel` (`appendKvOnly`, `forwardWithMrope`).
fn headCase(comptime tf: bool, comptime seed: u64) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    var hp = fixture.Pack.init(seed + 1);
    defer hp.deinit();
    var b = try open(&p, &hp, tf, seed);
    defer b.close();
    const s = b.xfm.s;
    var fs = try FsPackDir.open(io, b.dir);
    defer fs.close();
    const pd = fs.packDir();
    const m = b.xfm.plugin.?.model;
    const hooks = nemotron.table.spec.?.head.?;
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
        } else {
            try b.check(try g.adopt(.{ .ctx = out.hidden.ctx }), false);
        }
        pos += n;
    }
    try b.expectDone();
}

/// Served: the engine's head calls through `MtpHeadRef.plugin` against the recorded legacy sidecar arm (`.qwen`), and
/// a pinned-depth MTP round loop with its tokens and accepted counts.
fn servedCase(comptime tf: bool, comptime seed: u64) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    var hp = fixture.Pack.init(seed + 1);
    defer hp.deinit();
    var b = try open(&p, &hp, tf, seed);
    defer b.close();
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
    var g = ops.Graph.init(a, b.xfm.s);
    defer g.deinit();
    for ([_][]const u32{ &tokens, accepted.items }) |got| try b.check(try g.fromSlice(u32, got, &.{@intCast(got.len)}), true);
    try b.expectDone();
}

// ── tests ──

test "nemotron_h MTP head: sevren-ai's layout (dense eh_proj and router) is bit-equal to the recorded legacy head" {
    try headCase(false, 0x6e01);
}

test "nemotron_h MTP head: TensorFold's 4-bit layout is bit-equal to the recorded legacy head" {
    try headCase(true, 0x6e02);
}

test "nemotron_h served: an MTP round loop through the engine's head equals the recorded legacy sidecar arm (sevren-ai's layout)" {
    try servedCase(false, 0x6e11);
}

test "nemotron_h served: an MTP round loop through the engine's head equals the recorded legacy sidecar arm (TensorFold's layout)" {
    try servedCase(true, 0x6e12);
}

const Generator = @import("../../engines/mlx/generate/generator.zig").Generator;
const SamplingParams = @import("../../engines/mlx/sampling/params.zig").SamplingParams;
const Tokenizer = @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer;

/// `want` tokens after `prompt`: serial decode, or MTP rounds through `head` (every token through a real round).
fn generate(x: *Transformer, head: ?MtpHeadRef, sp: SamplingParams, out: []u32) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tok_dummy: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 5, 9, 4, 2, 11, 6, 14, 8 };
    var gen = try Generator.initWithOptions(io, a, x, &tok_dummy, &prompt, @intCast(out.len), sp, &.{}, if (head) |h| .{
        .mtp_enabled = true,
        .mtp = h,
        .model_has_mtp = true,
        .mtp_depth = 2,
    } else .{ .skip_lazy_preforward = true });
    defer gen.deinit(a);
    var n: usize = 0;
    if (head == null) {
        while (n < out.len) : (n += 1) out[n] = (try gen.next(a)) orelse break;
    } else while (true) {
        const rounds_before = gen.mtp_attempted + gen.mtp_lookup_rounds;
        const res = (try gen.nextMtp(a)) orelse break;
        defer a.free(res.tokens);
        @memcpy(out[n..][0..res.tokens.len], res.tokens);
        n += res.tokens.len;
        try testing.expect(gen.mtp_attempted + gen.mtp_lookup_rounds != rounds_before);
    }
    try testing.expectEqual(out.len, n);
    if (head != null) try testing.expect(gen.mtp_attempted > 0 and !gen.spec_disabled_runtime);
}

// Bar: the spec-decode contract on this trunk family. On the row-exact kernels with a drafter bound (exact
// acceptance), greedy and seeded MTP through the served head emit exactly the serial stream, through the hidden
// capture, the fused-window per-row SSM capture and the capture-based rollback.
test "nemotron_h served: MTP rounds through the plugin's head emit the serial stream, greedy and seeded" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var p = fixture.Pack.init(0x6e21);
    defer p.deinit();
    var hp = fixture.Pack.init(0x6e22);
    defer hp.deinit();
    try trunk.writeRowExactTrunk(&p);
    try writeHead(&hp, false);
    var cbuf: [2048]u8 = undefined;
    var b = try both.Both.openAs(&nemotron.table, &p, try trunk.rowExactTrunkConfig(&cbuf), true, .none);
    defer b.close();
    const file = try std.fmt.allocPrintSentinel(a, "{s}/mtp_head.safetensors", .{b.dir}, 0);
    defer a.free(file);
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_save_safetensors(file.ptr, hp.map, meta));
    for ([_]SamplingParams{ .{ .temperature = 0.0 }, .{ .temperature = 1.0, .seed = 7 } }) |sp| {
        var out: [2][16]u32 = undefined;
        for (&out, 0..) |*o, k| {
            // A fresh trunk per run: a generation leaves its caches on the shell.
            var x = try driver.load(io, a, &b.cfg, b.dir, fixture.stream());
            defer x.deinit();
            x.config.dflash_bound = true;
            try testing.expect(x.config.row_exact_covered);
            if (k == 0) try generate(&x, null, sp, o) else {
                const served = plugin_head.load(io, a, &x, b.dir) orelse return error.TestHeadNotLoaded;
                defer plugin_head.destroy(served, a);
                try generate(&x, .{ .plugin = served }, sp, o);
            }
        }
        try testing.expectEqualSlices(u32, &out[0], &out[1]);
    }
}
