//! ANE prefill (`--ane-prefill`) on the qwen3_5 plugin trunk: the engine's build reads the same layers, weight bytes,
//! GPU rest slices and plan off the served shell as off the legacy shell, and the table's forward hands the offload
//! each dense MLP and GDN q|k|v / z projection at the legacy forward's sites, with the legacy seam's inputs.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const pack = @import("../../sdk/bases/proof/qwen35_pack.zig");
const ane_prefill = @import("../../engines/mlx/ane_prefill.zig");
const cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const maxAbsDiffF32 = @import("../../sdk/testing/parity.zig").maxAbsDiffF32;
const served = @import("served_test.zig");
const qwen3_5 = @import("plugin.zig");

const testing = std.testing;
const Array = mlx.mlx_array;
const AneLayer = ane_prefill.AneLayer;

fn openPair(p: *served.Pair, comptime moe: bool, comptime seed: u64) !void {
    var pk = fixture.Pack.init(seed);
    defer pk.deinit();
    try pack.writePack(&pk, moe, true, 8);
    try p.open(&qwen3_5.table, &pk, pack.config(moe, true, 8), try served.oracle("qwen3_5", std.fmt.comptimePrint("ane-{x}", .{seed})));
}

/// What the build reads off shell `x`, in order: the plan at four shares and unit counts, then per layer which programs
/// it offers, each program's dequantized weights and the GPU rest slices of the 0.25 plan. `rests` keeps those alive.
fn buildInputs(x: *Transformer, g: *ops.Graph, out: *std.ArrayList(Array), rests: *std.ArrayList(ane_prefill.AneChanRest)) !ane_prefill.AnePlan {
    const a = testing.allocator;
    const layers = (try x.aneLayers(a)) orelse return error.PluginOffersNoAneLayers;
    defer a.free(layers);
    const cfg = &x.config;
    for ([_]f32{ 0.25, 0.5 }) |share| for ([_]u32{ 1, 2 }) |units| {
        const pl = ane_prefill.anePlan(cfg, layers, share, units);
        const v = [_]u32{ @intCast(pl.dense_layers), @intCast(pl.gdn_layers), pl.eng_ffn, pl.eng_qkv, pl.eng_z, pl.qkv_out, pl.z_out, @intCast(pl.rest_bytes) };
        try out.append(a, try g.fromSlice(u32, &v, &.{v.len}));
    };
    const plan = ane_prefill.anePlan(cfg, layers, 0.25, 1);
    const h = cfg.hidden_size;
    const ffn = cfg.intermediate_size;
    for (layers) |*l| {
        try out.append(a, try g.fromSlice(u32, &.{ @intFromBool(l.mlp != null), @intFromBool(l.gdn != null) }, &.{2}));
        if (l.mlp) |*m| {
            try out.append(a, try hostWeights(x, g, .{ m.gate_w, m.gate_s, m.gate_b }, h, ffn));
            try out.append(a, try hostWeights(x, g, .{ m.up_w, m.up_s, m.up_b }, h, ffn));
            try out.append(a, try hostWeights(x, g, .{ m.down_w, m.down_s, m.down_b }, ffn, h));
            try rests.append(a, .{ .mlp = try x.aneBuildMlpRest(m, plan.eng_ffn, ffn) });
            const r = rests.items[rests.items.len - 1].mlp.?;
            try appendSet(g, out, .{ r.gate_w, r.gate_s, r.gate_b, r.up_w, r.up_s, r.up_b, r.down_w, r.down_s, r.down_b });
        }
        if (l.gdn) |*gd| {
            try out.append(a, try hostWeights(x, g, .{ gd.qkv_w, gd.qkv_s, gd.qkv_b }, h, plan.qkv_out));
            try out.append(a, try hostWeights(x, g, .{ gd.z_w, gd.z_s, gd.z_b }, h, plan.z_out));
            try rests.append(a, .{});
            const r = &rests.items[rests.items.len - 1];
            try x.aneBuildGdnRest(r, gd, plan.eng_qkv, plan.eng_z);
            try appendSet(g, out, .{ r.gdn_qkv_w, r.gdn_qkv_s, r.gdn_qkv_b, r.gdn_z_w, r.gdn_z_s, r.gdn_z_b });
        }
    }
    return plan;
}

/// A program's weights dequantized to f32 on the host, as the build hands them to the compiler.
fn hostWeights(x: *Transformer, g: *ops.Graph, w: [3]Array, in_dim: u32, out_dim: u32) !Array {
    const v = try x.dequantToHostF32(testing.allocator, w[0], w[1], w[2], in_dim, out_dim);
    defer testing.allocator.free(v);
    return g.fromSlice(f32, v, &.{@intCast(v.len)});
}

/// The set's present arrays after a mask of which are.
fn appendSet(g: *ops.Graph, out: *std.ArrayList(Array), set: anytype) !void {
    const a = testing.allocator;
    var mask: [set.len]u32 = undefined;
    inline for (set, &mask) |x, *m| m.* = @intFromBool(x.ctx != null);
    try out.append(a, try g.fromSlice(u32, &mask, &.{set.len}));
    inline for (set) |x| if (x.ctx != null) try out.append(a, x);
}

/// Every shell's build inputs equal, the served one's against the legacy shell's or the tape.
fn expectSameInputs(p: *served.Pair, dense: u64, gdn: u64) !void {
    const a = testing.allocator;
    const xs = p.shells();
    var g = ops.Graph.init(a, fixture.stream());
    defer g.deinit();
    var lists: [2]std.ArrayList(Array) = .{ .empty, .empty };
    defer for (&lists) |*l| l.deinit(a);
    var rests: std.ArrayList(ane_prefill.AneChanRest) = .empty;
    defer {
        for (rests.items) |*r| r.deinit();
        rests.deinit(a);
    }
    for (xs, lists[0..xs.len]) |x, *l| {
        const plan = try buildInputs(x, &g, l, &rests);
        try testing.expectEqual(dense, plan.dense_layers);
        try testing.expectEqual(gdn, plan.gdn_layers);
        try testing.expect(plan.eng_qkv > 0 and plan.eng_z > 0 and (dense == 0 or plan.eng_ffn > 0));
    }
    if (xs.len == 2) try testing.expectEqual(lists[0].items.len, lists[1].items.len);
    const last = xs.len - 1;
    for (lists[last].items) |got| try p.oracle.check(got, false);
    try p.oracle.done();
}

/// An offload that computes what the legacy seam's GPU arm computes from its input, counting calls per layer.
const Probe = struct {
    xfm: *Transformer,
    layers: []const AneLayer,
    mlp: [4]u32 = @splat(0),
    gdn: [4]u32 = @splat(0),
    /// Hand back zeros for the MLP (the forward must read what the offload returns).
    zero_mlp: bool = false,

    fn offload(p: *Probe) abi.Offload {
        return .{ .ctx = p, .mlp = &mlp_, .gdn_proj = &gdn_ };
    }

    fn mlp_(ctx: *anyopaque, layer: u32, x: abi.Array, out: *abi.Array) callconv(.c) bool {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        const dw = &(p.layers[layer].mlp orelse return false);
        p.mlp[layer] += 1;
        const y = p.xfm.denseMLP(.{ .ctx = x.ctx }, dw) catch return false;
        if (p.zero_mlp) {
            defer _ = mlx.mlx_array_free(y);
            const sh = mlx.getShape(y);
            var z = mlx.mlx_array_new();
            _ = mlx.mlx_zeros(&z, sh.ptr, sh.len, mlx.mlx_array_dtype(y), p.xfm.s);
            out.* = .{ .ctx = z.ctx };
            return true;
        }
        out.* = .{ .ctx = y.ctx };
        return true;
    }

    fn gdn_(ctx: *anyopaque, layer: u32, x: abi.Array, qkv: *abi.Array, z: *abi.Array) callconv(.c) bool {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        const g = &(p.layers[layer].gdn orelse return false);
        p.gdn[layer] += 1;
        const r = p.xfm.gdnProjArrays(.{ .ctx = x.ctx }, g.qkv_w, g.qkv_s, g.qkv_b, g.z_w, g.z_s, g.z_b) catch return false;
        qkv.* = .{ .ctx = r.qkv.ctx };
        z.* = .{ .ctx = r.z.ctx };
        return true;
    }
};

/// A prompt chunk's logits through the table with `probe` as the offload, into `g`.
fn seamLogits(p: *served.Pair, probe: *Probe, g: *ops.Graph) !Array {
    const a = testing.allocator;
    const n = p.cfg.num_hidden_layers;

    var kv = try KVCache.init(a, n);
    defer kv.deinit();
    const ssm = try fixture.newEntries(n);
    defer fixture.freeEntries(ssm);
    var slot: cache_ops.Slot = .{ .kv = &kv, .ssm = ssm };
    const handles = [_]abi.Handle{slot.handle()};
    var host: [seam_ids.len]u32 = undefined;
    for (seam_ids, &host) |x, *y| y.* = @intCast(x);
    const pos = [_]u32{0};
    const off = probe.offload();
    var out: abi.ForwardOut = .{};
    const v = qwen3_5.table.forward(p.xfm.plugin.?.model, &.{ .mode = .prefill, .logits = .all_rows, .rows = seam_ids.len, .tokens = .of(&host), .positions = .of(&pos), .caches = .of(&handles), .cache_ops = &cache_ops.ops, .offload = &off }, &out);
    if (!v.isOk()) {
        std.debug.print("forward: {s}\n", .{v.why.slice()});
        return error.PluginForward;
    }
    return g.adopt(.{ .ctx = out.logits.ctx });
}

const seam_ids = [_]i32{ 3, 17, 42, 99, 7, 250, 1, 64, 12, 9, 33, 77, 140, 2, 5, 61, 200, 88, 6, 30 };

/// A prompt chunk with the probe as the offload equals the legacy forward of the chunk (or its tape); with the probe
/// handing back zeros it differs.
fn seamCase(comptime moe: bool, comptime seed: u64) !void {
    const a = testing.allocator;
    var p: served.Pair = undefined;
    try openPair(&p, moe, seed);
    defer p.close();
    try testing.expect(p.cfg.plugin.?.parsed.caps.ane_prefill);
    const layers = (try p.xfm.aneLayers(a)).?;
    defer a.free(layers);
    var probe: Probe = .{ .xfm = &p.xfm, .layers = layers };
    var g = ops.Graph.init(a, fixture.stream());
    defer g.deinit();
    const got = try seamLogits(&p, &probe, &g);
    try p.oracle.check(got, false);
    try p.oracle.done();
    // Layers 0 and 2 are linear attention; a MoE trunk has no dense MLP to offload.
    try testing.expectEqualSlices(u32, &.{ 1, 0, 1, 0 }, &probe.gdn);
    try testing.expectEqualSlices(u32, if (moe) &.{ 0, 0, 0, 0 } else &.{ 1, 1, 1, 1 }, &probe.mlp);
    if (moe) return;
    probe.zero_mlp = true;
    try testing.expect(try maxAbsDiffF32(try seamLogits(&p, &probe, &g), got, fixture.stream()) > 0);
}

test "qwen3_5 ANE prefill: the served shell gives the build the legacy shell's layers, weights, rest slices and plan" {
    inline for (.{ false, true }, .{ 0xa7e1, 0xa7e2 }, .{ 4, 0 }) |moe, seed, dense| {
        var p: served.Pair = undefined;
        try openPair(&p, moe, seed);
        defer p.close();
        try expectSameInputs(&p, dense, 2);
    }
}

test "qwen3_5 ANE prefill: a prompt chunk hands the offload each dense MLP and GDN projection at the legacy seam" {
    try seamCase(false, 0xa7e3);
    try seamCase(true, 0xa7e4);
}
