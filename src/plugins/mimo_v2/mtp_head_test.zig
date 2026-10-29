//! Parity proof for MiMo-V2's MTP heads behind `SpecHooks.head`: three heads in a 4-bit and a bf16 pack, run by the
//! plugin over the head cache's recurrent entries against the deleted legacy `mimo_mtp.Head` (replayed from
//! `fixtures/*.tape`) through history appends past the window, round openings, draft steps 1..3 (the last past the
//! final head), a re-opening below the committed rows, bit-equal at every step, the engine's history length (KV layer
//! 0) the legacy `State.step()`.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
const PluginRef = @import("../../engines/mlx/spec/mtp_head_ref.zig").MtpCacheRef.PluginRef;
const FsPackDir = @import("../../host/registry/pack_dir.zig").FsPackDir;
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const mimo = @import("plugin.zig");
const trunk = @import("proof_test.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const plugin_head = @import("../../engines/mlx/plugin/head.zig");
const MtpHeadRef = @import("../../engines/mlx/spec/mtp_head_ref.zig").MtpHeadRef;
const mtpRun = @import("../../engines/mlx/plugin/proof_mtp.zig").mtpRun;

const testing = std.testing;
const H: c_int = 256;
const HEADS: c_int = 8;
const KV: c_int = 2;
const HD: c_int = 192;
const VD: c_int = 128;
const INTER: c_int = 512;

fn writeHeads(p: *fixture.Pack, q: bool) !void {
    var b: [160]u8 = undefined;
    for (0..3) |k| {
        const n = struct {
            fn f(buf: []u8, kk: usize, leaf: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, "model.mtp.layers.{d}.{s}", .{ kk, leaf });
            }
        }.f;
        for ([_][]const u8{ "enorm.weight", "hnorm.weight", "input_layernorm.weight", "pre_mlp_layernorm.weight", "final_layernorm.weight" }) |leaf| try p.norm(try n(&b, k, leaf), H);
        // Head 1 ships no sinks (the legacy reads them as optional).
        if (k != 1) try p.dense(try n(&b, k, "self_attn.attention_sink_bias"), &.{HEADS}, 0.5, 0);
        try p.weight(try n(&b, k, "eh_proj"), H, 2 * H, q);
        try p.weight(try n(&b, k, "self_attn.q_proj"), HEADS * HD, H, q);
        try p.weight(try n(&b, k, "self_attn.k_proj"), KV * HD, H, q);
        try p.weight(try n(&b, k, "self_attn.v_proj"), KV * VD, H, q);
        try p.weight(try n(&b, k, "self_attn.o_proj"), H, HEADS * VD, q);
        try p.weight(try n(&b, k, "mlp.gate_proj"), INTER, H, q);
        try p.weight(try n(&b, k, "mlp.up_proj"), INTER, H, q);
        try p.weight(try n(&b, k, "mlp.down_proj"), H, INTER, q);
    }
}

fn rows(g: *ops.Graph, n: c_int, seed: u64) !mlx.mlx_array {
    var prng = std.Random.DefaultPrng.init(seed);
    const buf = try testing.allocator.alloc(f32, @intCast(n * H));
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = prng.random().floatNorm(f32) * 0.5;
    return g.astype(try g.adopt(mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, n, H }, 3, .float32)), .bfloat16);
}

/// Committed tokens from row `p0`.
const Rows = struct { p0: usize, ids: []const u32 };

/// One engine call: committed rows (an append or an opening), or a draft step on `id`.
const Call = union(enum) {
    history: Rows,
    open: Rows,
    draft: struct { step: usize, id: u32 },
    /// The engine's rollback to `len` committed rows.
    truncate: usize,
};

fn headCase(comptime quantized: bool, comptime seed: u64) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try trunk.writePack(&p, quantized);
    try writeHeads(&p, quantized);
    const base = trunk.configJson(quantized);
    const json = try std.mem.concat(a, u8, &.{ "{\"num_nextn_predict_layers\":3,", base[1..] });
    defer a.free(json);
    var b = try trunk.openCase(&p, json, false, seed);
    defer b.close();
    const parsed = &b.cfg.plugin.?.parsed;
    try testing.expectEqual(@as(u32, 3), parsed.spec.mtp_heads);
    const s = b.xfm.s;

    var fs = try FsPackDir.open(io, b.dir);
    defer fs.close();
    const pd = fs.packDir();
    const m = b.xfm.plugin.?.model;
    const hooks = mimo.table.spec.?.head.?;
    try testing.expect(hooks.load(m, &.{ .model_dir = .of(b.dir), .dir = &pd, .parsed = parsed, .stream = .{ .ctx = s.ctx } }).isOk());
    var ref = try PluginRef.init(a, 3);
    defer ref.deinit();
    var slot: cache_ops.Slot = .{ .kv = &ref.kv, .ssm = ref.ssm };
    const handles = [_]abi.Handle{slot.handle()};

    const calls = [_]Call{
        .{ .history = .{ .p0 = 0, .ids = &.{ 3, 17, 42, 99, 7, 250 } } },
        .{ .open = .{ .p0 = 6, .ids = &.{ 1, 8 } } },
        .{ .draft = .{ .step = 1, .id = 66 } },
        .{ .draft = .{ .step = 2, .id = 31 } },
        .{ .draft = .{ .step = 3, .id = 120 } },
        .{ .truncate = 8 },
        .{ .open = .{ .p0 = 8, .ids = &.{ 66, 5 } } },
        .{ .draft = .{ .step = 1, .id = 77 } },
        .{ .truncate = 6 },
        .{ .open = .{ .p0 = 6, .ids = &.{ 2, 18, 19 } } },
        .{ .draft = .{ .step = 1, .id = 200 } },
        .{ .draft = .{ .step = 2, .id = 9 } },
        // Past the 8-row window: the leading rows never reach a later query.
        .{ .history = .{ .p0 = 9, .ids = &.{ 33, 77, 140, 2, 18, 19, 20, 21, 22, 23, 24, 25 } } },
        .{ .open = .{ .p0 = 21, .ids = &.{88} } },
        .{ .draft = .{ .step = 1, .id = 64 } },
        .{ .draft = .{ .step = 2, .id = 12 } },
    };
    var trunk_hidden: mlx.mlx_array = .{ .ctx = null };
    defer if (trunk_hidden.ctx != null) {
        _ = mlx.mlx_array_free(trunk_hidden);
    };
    for (calls, 0..) |call, i| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        switch (call) {
            .truncate => |len| {
                try ref.kv.truncate(len, s);
                continue;
            },
            else => {},
        }
        var f: abi.Forward = .{ .mode = .decode, .rows = 1, .tokens = .{}, .caches = .of(&handles), .cache_ops = &cache_ops.ops, .positions = .of(&.{0}) };
        var pos: [1]u32 = undefined;
        switch (call) {
            .history, .open => |c| {
                const n: c_int = @intCast(c.ids.len);
                const hid = try rows(&g, n, seed + 10 + i);
                const ids = try g.fromSlice(u32, c.ids, &.{n});
                pos = .{@intCast(c.p0)};
                f.rows = @intCast(n);
                f.mode = if (n == 1) .decode else .prefill;
                f.ids = .{ .ctx = ids.ctx };
                f.positions = .of(&pos);
                f.hidden_in = .{ .ctx = hid.ctx };
                if (call == .open) {
                    f.want_hidden = true;
                    f.logits = .last_row;
                    // A round's chained steps read the trunk's last hidden row.
                    if (trunk_hidden.ctx != null) _ = mlx.mlx_array_free(trunk_hidden);
                    trunk_hidden = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_array_set(&trunk_hidden, try g.slice(hid, &.{ 0, n - 1, 0 }, &.{ 1, n, H })));
                }
            },
            .draft => |d| {
                const id = try g.fromSlice(u32, &.{d.id}, &.{1});
                pos = .{@intCast(ref.kv.step + d.step - 1)};
                f.ids = .{ .ctx = id.ctx };
                f.positions = .of(&pos);
                f.hidden_in = .{ .ctx = trunk_hidden.ctx };
                f.want_hidden = true;
                f.logits = .last_row;
                f.head = @intCast(@min(d.step, 2));
            },
            .truncate => unreachable,
        }
        var out: abi.ForwardOut = .{};
        const v = hooks.forward(m, &f, &out);
        if (!v.isOk()) {
            std.debug.print("call {d}: {s}\n", .{ i, v.why.slice() });
            return error.PluginForward;
        }
        if (call == .open or call == .draft) {
            try b.check(try g.adopt(.{ .ctx = out.hidden.ctx }), false);
        } else try testing.expect(out.hidden.ctx == null);
        // The engine's history length is the legacy `State.step()`.
        const len: [1]u32 = .{@intCast(ref.kv.step)};
        try b.check(try g.fromSlice(u32, &len, &.{1}), true);
    }
    try b.expectDone();
}

/// Served: the driver's shell on the legacy config, its steps, greedy and batched decode; the engine's head calls
/// (`MtpHeadRef.plugin`) against the recorded legacy heads (`.mimo`) bit-equal; a pinned-depth MTP round loop, tokens
/// and accepted counts equal.
fn servedCase(comptime quantized: bool, comptime seed: u64) !void {
    const a = testing.allocator;
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try trunk.writePack(&p, quantized);
    try writeHeads(&p, quantized);
    const base = trunk.configJson(quantized);
    const json = try std.mem.concat(a, u8, &.{ "{\"num_nextn_predict_layers\":3,", base[1..] });
    defer a.free(json);
    var b = try trunk.openCase(&p, json, true, seed);
    defer b.close();
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try b.expectBatched();
    try expectServedHeads(&b, seed);
    try b.expectDone();
}

/// The engine's MTP head calls through `MtpHeadRef.plugin` on `b`'s served shell bit-equal to the legacy heads
/// (`.mimo`) on its legacy one, and a pinned-depth round loop with equal tokens and accepted counts.
pub fn expectServedHeads(b: *both.Both, seed: u64) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const served = plugin_head.load(io, a, &b.xfm, b.dir) orelse return error.TestHeadNotLoaded;
    defer plugin_head.destroy(served, a);
    const h: MtpHeadRef = .{ .plugin = served };
    const x = &b.xfm;
    {
        // The engine's head calls: a history append, an opening that wants nothing, an opening with two chained
        // drafts (the readout's input under `.mixed`, the trunk head's logits under `.logits`), a partial accept.
        var cr = try h.makeCache(a);
        defer cr.deinit();
        var g = ops.Graph.init(a, b.xfm.s);
        defer g.deinit();
        const hist = try rows(&g, 6, seed + 1);
        const open = try rows(&g, 1, seed + 2);
        const quiet = try rows(&g, 2, seed + 3);
        const again = try rows(&g, 1, seed + 4);
        // The rows waiting for their tokens are never persisted: the head history rebuilds from the tail.
        try testing.expect(cr.kv() == null);
        try h.appendHistory(x, &cr, &.{ 3, 17, 42, 99, 7, 250 }, hist, 0, null, a);
        try testing.expectEqual(@as(usize, 6), cr.step());
        // An opening that wants nothing appends history.
        const q = try h.forward(x, &cr, try g.fromSlice(i32, &.{ 1, 8 }, &.{2}), quiet, 6, .none, null, &.{ 1, 8 });
        const o0 = try h.forward(x, &cr, try g.fromSlice(i32, &.{66}, &.{1}), open, 8, .mixed, null, &.{66});
        const o1 = try h.forward(x, &cr, try g.fromSlice(i32, &.{31}, &.{1}), o0.hidden_next, 9, .logits, null, null);
        const o2 = try h.forward(x, &cr, try g.fromSlice(i32, &.{120}, &.{1}), o1.hidden_next, 10, .mixed, null, null);
        // A partial accept: one draft kept, the round rolled back, a new opening on the committed rows.
        try cr.truncate(9, x.s);
        const o3 = try h.forward(x, &cr, try g.fromSlice(i32, &.{5}, &.{1}), again, 9, .mixed, null, &.{5});
        const o4 = try h.forward(x, &cr, try g.fromSlice(i32, &.{77}, &.{1}), o3.hidden_next, 10, .mixed, null, null);
        _ = try g.adopt(q.hidden_next);
        const outs = [_]mlx.mlx_array{ try g.adopt(o0.rerank_x), try g.adopt(o1.logits), try g.adopt(o2.rerank_x), try g.adopt(o3.rerank_x), try g.adopt(o4.rerank_x), try g.adopt(o0.hidden_next), try g.adopt(o1.hidden_next), try g.adopt(o2.hidden_next), try g.adopt(o3.hidden_next), try g.adopt(o4.hidden_next) };
        for (outs) |got| try b.check(got, false);
    }
    var tokens: [16]u32 = undefined;
    var accepted: std.ArrayList(u32) = .empty;
    defer accepted.deinit(a);
    try mtpRun(x, h, &tokens, &accepted);
    var g = ops.Graph.init(a, b.xfm.s);
    defer g.deinit();
    for ([_][]const u32{ &tokens, accepted.items }) |got| try b.check(try g.fromSlice(u32, got, &.{@intCast(got.len)}), true);
}

// ── tests ──

test "mimo_v2 MTP heads: a 4-bit pack's history, openings and draft steps are bit-equal to the legacy heads" {
    try headCase(true, 0x313a);
}

test "mimo_v2 MTP heads: a bf16 pack's are too" {
    try headCase(false, 0x313b);
}

test "mimo_v2 served: steps, greedy and batched decode equal legacy, the engine's MTP head calls too" {
    try servedCase(true, 0x313c);
}
