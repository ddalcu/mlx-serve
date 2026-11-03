//! The qwen3_5 MoE trunk with its routed experts streamed from the pack as downloaded (the `stream_experts` load
//! setting: `Streamed.openPack` and the MoE block's streamed arm) against the same pack's experts in memory: logits bit
//! for bit at every width the trunk runs (prompt chunks on the sorted gathers, a call's experts a fetch's worth at a
//! time; one-token decode on the affine kernels; verify rows, past one fetch's experts in chunks; two slots' batched
//! rows), 4-bit and 8-bit experts, a few slot rows (evictions and slots spread over banks every call) and every expert
//! resident. Owns tests only.

const std = @import("std");
const build_options = @import("build_options");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const sdk = @import("../../sdk/root.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const pack = @import("../../sdk/bases/proof/qwen35_pack.zig");
const model = @import("../../sdk/config/model.zig");
const cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const qwen3_5 = @import("plugin.zig");

const testing = std.testing;
const LAYERS = 4;

/// The plugin over the pack in `dir`, loaded with `settings` (its load settings JSON).
const Served = struct {
    res: Resident,
    parsed: abi.Parsed = .{},
    m: *anyopaque,

    fn open(dir: []const u8, config_json: []const u8, settings: []const u8) !*Served {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        var cfg = try model.parseConfigFromJson(a, config_json);
        defer cfg.deinit(a);
        const t = try a.create(Served);
        errdefer a.destroy(t);
        t.* = .{ .res = Resident.adopt(try model.loadModelWeights(io, a, dir, &cfg, false)), .m = undefined };
        errdefer t.res.deinit();
        const files = [_]sdk.fixture.File{.{ .name = "config.json", .bytes = config_json }};
        const mem: sdk.fixture.MemDir = .{ .files = &files };
        const pd = mem.packDir();
        if (!qwen3_5.table.parse(&pd, .of(config_json), .{}, null, false, &t.parsed).isOk()) return error.PluginParse;
        errdefer qwen3_5.table.free_parsed(&t.parsed);
        const src = t.res.source();
        var why: abi.Verdict = .{};
        t.m = qwen3_5.table.load(&.{ .model_dir = .of(dir), .dir = &pd, .parsed = &t.parsed, .settings_json = .of(settings), .stream = .{ .ctx = fixture.stream().ctx }, .weights = &src }, &why) orelse {
            std.debug.print("load: {s}\n", .{why.why.slice()});
            return error.PluginLoad;
        };
        return t;
    }

    fn close(t: *Served) void {
        qwen3_5.table.unload(t.m);
        qwen3_5.table.free_parsed(&t.parsed);
        t.res.deinit();
        testing.allocator.destroy(t);
    }

    fn loaded(t: *const Served) *qwen3_5.Loaded {
        return @ptrCast(@alignCast(t.m));
    }
};

/// One request: its KV cache, recurrent entries and next position.
const Side = struct {
    kv: KVCache,
    ssm: []SSMCacheEntry,
    pos: u32 = 0,
    slot: cache_ops.Slot = undefined,

    fn init(s: *Side) !void {
        s.* = .{ .kv = try KVCache.init(testing.allocator, LAYERS), .ssm = try fixture.newEntries(LAYERS) };
        s.slot = .{ .kv = &s.kv, .ssm = s.ssm };
    }

    fn deinit(s: *Side) void {
        s.kv.deinit();
        fixture.freeEntries(s.ssm);
    }
};

const Step = struct { mode: abi.Mode, ids: []const u32 };

/// A prompt, a warm chunk, decodes, verify rows (3, 5 and 8: past one fetch's 48 experts at top-8), every logit row's
/// bits.
const solo = [_]Step{
    .{ .mode = .prefill, .ids = &.{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31, 120, 5, 77, 200, 9, 33, 77, 140, 2, 18 } },
    .{ .mode = .prefill, .ids = &.{ 61, 4, 190, 11, 23, 87, 140, 6, 222, 15, 45 } },
    .{ .mode = .decode, .ids = &.{88} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 200 } },
    .{ .mode = .decode, .ids = &.{12} },
    .{ .mode = .verify, .ids = &.{ 9, 33, 77, 140, 2 } },
    .{ .mode = .verify, .ids = &.{ 6, 62, 201, 5, 12, 91, 14, 8 } },
    .{ .mode = .decode, .ids = &.{64} },
};

fn forward(t: *const Served, f: abi.Forward) ![]u32 {
    var g = ops.Graph.init(testing.allocator, fixture.stream());
    defer g.deinit();
    var out: abi.ForwardOut = .{};
    var fw = f;
    fw.logits = .all_rows;
    fw.cache_ops = &cache_ops.ops;
    const v = qwen3_5.table.forward(t.m, &fw, &out);
    if (!v.isOk()) {
        std.debug.print("forward: {s}\n", .{v.why.slice()});
        return error.PluginForward;
    }
    const y = try g.contiguous(try g.astype(try g.adopt(.{ .ctx = out.logits.ctx }), .float32));
    try mlx.check(mlx.mlx_array_eval(y));
    const n = mlx.mlx_array_size(y);
    return testing.allocator.dupe(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes((mlx.mlx_array_data_float32(y) orelse return error.NoData)[0..n])));
}

/// Every step's logits, then two slots' batched rows (a decode tick, three verify rows each), in order.
fn run(t: *const Served, out: *std.ArrayList([]u32)) !void {
    var s: Side = undefined;
    try s.init();
    defer s.deinit();
    for (solo) |st| {
        const handles = [_]abi.Handle{s.slot.handle()};
        try out.append(testing.allocator, try forward(t, .{ .mode = st.mode, .rows = @intCast(st.ids.len), .tokens = .of(st.ids), .positions = .of(&.{s.pos}), .caches = .of(&handles) }));
        s.pos += @intCast(st.ids.len);
    }
    var two: [2]Side = undefined;
    for (&two) |*sd| try sd.init();
    defer for (&two) |*sd| sd.deinit();
    const prompts = [2][]const u32{ &.{ 42, 101, 7, 9, 250, 3, 77 }, &.{ 9, 33, 77, 140, 2, 18 } };
    for (&two, prompts) |*sd, pr| {
        const handles = [_]abi.Handle{sd.slot.handle()};
        try out.append(testing.allocator, try forward(t, .{ .mode = .prefill, .rows = @intCast(pr.len), .tokens = .of(pr), .positions = .of(&.{0}), .caches = .of(&handles) }));
        sd.pos = @intCast(pr.len);
    }
    const ticks = [_][]const u32{ &.{ 77, 3 }, &.{ 5, 61, 200, 9, 33, 140 } };
    for (ticks) |tk| {
        const rows: u32 = @intCast(tk.len / 2);
        const handles = [_]abi.Handle{ two[0].slot.handle(), two[1].slot.handle() };
        try out.append(testing.allocator, try forward(t, .{ .mode = .batched_rows, .rows = rows, .tokens = .of(tk), .slots = .of(&.{ 0, 1 }), .positions = .of(&.{ two[0].pos, two[1].pos }), .caches = .of(&handles) }));
        for (&two) |*sd| sd.pos += rows;
    }
}

fn parity(comptime experts: c_int, comptime bits: c_int, rows: ?u32, seed: u64) !void {
    const a = testing.allocator;
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try pack.writePack(&p, true, true, experts);
    // 8-bit experts at group 64: the pack's other tensors stay 4-bit (the geometry names each width).
    if (bits != 4) for (0..LAYERS) |i| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, pi| {
        var nb: [128]u8 = undefined;
        const shape: [3]c_int = if (pi == 2) .{ experts, 256, 128 } else .{ experts, 128, 256 };
        try p.affine(try std.fmt.bufPrint(&nb, "language_model.model.layers.{d}.mlp.switch_mlp.{s}", .{ i, proj }), &shape, bits, 64);
    };
    const config = try std.mem.replaceOwned(u8, a, pack.config(true, true, experts), "\"num_experts_per_tok\":2", "\"num_experts_per_tok\":8");
    defer a.free(config);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
    try p.save(dir);

    var native: std.ArrayList([]u32) = .empty;
    defer {
        for (native.items) |x| a.free(x);
        native.deinit(a);
    }
    {
        const t = try Served.open(dir, config, "{}");
        defer t.close();
        try testing.expectEqual(@as(u32, bits), t.loaded().body.moe.layers[0].mlp.gate.qp.bits);
        try run(t, &native);
    }
    var settings_buf: [48]u8 = undefined;
    const settings = try std.fmt.bufPrint(&settings_buf, "{{\"stream_experts\":{d}}}", .{rows orelse experts});
    const t = try Served.open(dir, config, settings);
    defer t.close();
    var got: std.ArrayList([]u32) = .empty;
    defer {
        for (got.items) |x| a.free(x);
        got.deinit(a);
    }
    try run(t, &got);
    try testing.expectEqual(native.items.len, got.items.len);
    for (native.items, got.items) |want, have| try testing.expectEqualSlices(u32, want, have);
    const st = t.loaded().experts.?.src.pack.stats();
    std.debug.print("qwen3_5 streamed ({d}-bit experts, {d} rows of {d} a layer): {d} routes, {d} hits, {d} misses, {d} evictions\n", .{ bits, rows orelse experts, experts, st.route_calls, st.expert_cache_hits, st.expert_cache_misses, st.expert_cache_evictions });
    if (rows != null) try testing.expect(st.expert_cache_evictions > 0);
}

// ── tests ──

test "qwen3_5 streamed: 4-bit experts streamed from the pack (3 of 64 slot rows a layer) give the resident's logits bit for bit" {
    if (!build_options.mlx_stream or mlx.noGpuBackend()) return error.SkipZigTest;
    try parity(64, 4, 3, 0x57e1);
}

test "qwen3_5 streamed: 8-bit experts streamed from the pack (every one of 256 in slot rows) give the resident's logits bit for bit" {
    if (!build_options.mlx_stream or mlx.noGpuBackend()) return error.SkipZigTest;
    try parity(256, 8, null, 0x57e2);
}
