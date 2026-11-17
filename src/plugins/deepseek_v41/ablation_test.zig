//! The directional ablation's spec (`ablation.zig`): the forward's table read slot by slot, the ops that take the
//! direction out of a write against host math, the armed arrays, the batching and resume rules, the calibration tap's
//! records and file header; and the plugin's forward on the fixture pack, where an ablated prompt, decode step and
//! DSpark draft leave the stream nothing along the direction at any layer, a strength of 0 changes no bit, and a
//! batched tick runs each slot as its own decode step.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const dsv41 = @import("deepseek_v41.zig");
const ablation = @import("ablation.zig");
const v41 = @import("plugin.zig");
const sdk = @import("../../sdk/root.zig");
const fixture = @import("../deepseek_v4/fixture.zig");

/// The plugin's table over the tiny fixture pack, without the proof tape. The shared harness in
/// `../deepseek_v4/proof_test.zig` pulls `engines/mlx/plugin/tape.zig`, which would leave this plugin's unit closure;
/// this keeps the same surface the tests use (`open`, `close`, `m`, `s`, `plugin`).
const Twin = struct {
    tmp: testing.TmpDir,
    parsed: abi.Parsed = .{},
    m: *anyopaque,
    s: mlx.mlx_stream,

    fn open(files: []const sdk.fixture.File, _: []const u8) !*Twin {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        const t = try a.create(Twin);
        errdefer a.destroy(t);
        t.tmp = testing.tmpDir(.{});
        errdefer t.tmp.cleanup();
        for (files) |f| try t.tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
        var buf: [512]u8 = undefined;
        const dir = buf[0..try t.tmp.dir.realPath(io, &buf)];
        t.s = mlx.gpuStream();
        const mem: sdk.fixture.MemDir = .{ .files = files };
        const pd = mem.packDir();
        try testing.expect(v41.table.parse(&pd, .of(files[0].bytes), .{}, null, false, &t.parsed).isOk());
        var why: abi.Verdict = .{};
        t.m = v41.table.load(&.{ .model_dir = .of(dir), .dir = &pd, .parsed = &t.parsed, .stream = .{ .ctx = t.s.ctx } }, &why) orelse {
            std.debug.print("load: {s}\n", .{why.why.slice()});
            return error.PluginLoad;
        };
        return t;
    }

    fn close(t: *Twin) void {
        v41.table.unload(t.m);
        v41.table.free_parsed(&t.parsed);
        t.tmp.cleanup();
        testing.allocator.destroy(t);
    }

    fn plugin(t: *Twin, mode: abi.Mode, ids: []const u32, pos: u32) !mlx.mlx_array {
        var out: abi.ForwardOut = .{};
        const v = v41.table.forward(t.m, &.{ .mode = mode, .logits = if (mode == .verify) .all_rows else .last_row, .rows = @intCast(ids.len), .tokens = .of(ids), .positions = .of(&.{pos}) }, &out);
        if (!v.isOk()) {
            std.debug.print("forward: {s}\n", .{v.why.slice()});
            return error.PluginForward;
        }
        return .{ .ctx = out.logits.ctx };
    }
};

const testing = std.testing;
const A = mlx.mlx_array;
const Scope = dsv41.Scope;

/// `data` as an f32 array of `shape`, kept by `sc`.
fn f32s(sc: *Scope, data: []const f32, shape: []const c_int) !A {
    return sc.keep(mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32));
}

/// An armed ablation along the unit vector `r` at strength `s`, its arrays kept by `sc`.
fn along(sc: *Scope, r: []const f32, s: f32) !ablation.Ablation {
    const n: c_int = @intCast(r.len);
    return .{ .r = try f32s(sc, r, &.{n}), .col = try f32s(sc, r, &.{ n, 1 }), .s = try sc.f(s) };
}

/// `y - s (y . r) r` on the host, into `out`.
fn ablatedHost(y: []const f32, r: []const f32, s: f32, out: []f32) void {
    var dot: f32 = 0;
    for (y, r) |a, b| dot += a * b;
    for (out, y, r) |*o, a, b| o.* = a - s * dot * b;
}

fn expectClose(want: []const f32, got: []const f32) !void {
    try testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try testing.expectApproxEqAbs(w, g, 1e-4);
}

test "deepseek_v41 ablation: a forward without the field, a null one, or strength 0 is no ablation; bad tables refuse" {
    if (comptime !ablation.supported) return error.SkipZigTest;
    const one = [_]u32{0};
    var f: abi.Forward = .{ .mode = .decode, .rows = 1, .tokens = .{}, .slots = .of(&one) };
    try testing.expect((try ablation.of(&f)) == null);
    const zero = [_]f32{0};
    var ab: abi.Ablation = .{ .strengths = .of(&zero) };
    f.ablation = &ab;
    try testing.expect((try ablation.of(&f)) == null);
    const half = [_]f32{0.5};
    ab.strengths = .of(&half);
    const got = (try ablation.of(&f)) orelse return error.TestExpectedAblation;
    try testing.expectEqual(@as(f32, 0.5), got.strength);
    const big = [_]f32{1.5};
    ab.strengths = .of(&big);
    try testing.expectError(error.AblationStrength, ablation.of(&f));
    const two = [_]f32{ 1, 1 };
    ab.strengths = .of(&two);
    try testing.expectError(error.AblationSlots, ablation.of(&f));
    // A host whose `Forward` ends before the field: no ablation, whatever the bytes past its end hold.
    ab.strengths = .of(&half);
    f.size = @offsetOf(abi.Forward, "ablation");
    try testing.expect((try ablation.of(&f)) == null);
}

test "deepseek_v41 ablation: a batch reads each slot's own row, strength and key; a slot at strength 0 runs plain" {
    if (comptime !ablation.supported) return error.SkipZigTest;
    const slots = [_]u32{ 3, 4 };
    var f: abi.Forward = .{ .mode = .batched_rows, .rows = 1, .tokens = .{}, .slots = .of(&slots) };
    const strengths = [_]f32{ 0, 0.25 };
    const keys = [_]u64{ 0, 0xab1e };
    var ab: abi.Ablation = .{ .strengths = .of(&strengths), .keys = .of(&keys) };
    f.ablation = &ab;
    try testing.expect((try ablation.ofSlot(&f, 0)) == null);
    const second = (try ablation.ofSlot(&f, 1)) orelse return error.TestExpectedAblation;
    try testing.expectEqual(@as(usize, 1), second.row);
    try testing.expectEqual(@as(f32, 0.25), second.strength);
    try testing.expectEqual(@as(u64, 0xab1e), second.key);
    // A bad strength in any slot refuses the batch, whichever slot asks.
    const bad = [_]f32{ 0, 2 };
    ab.strengths = .of(&bad);
    try testing.expectError(error.AblationStrength, ablation.ofSlot(&f, 0));
}

test "deepseek_v41 ablation: a write loses s times its component along r, in its own dtype" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const r = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    const y = [_]f32{ 1, 2, 3, 4 };
    const ab = try along(&sc, &r, 0.5);
    // y . r = 5: every element loses 0.5 * 5 * 0.5.
    var want: [4]f32 = undefined;
    ablatedHost(&y, &r, 0.5, &want);
    try testing.expectApproxEqAbs(@as(f32, -0.25), want[0], 1e-6);
    const got = try dsv41.hostF32(try ablation.ablateWith(&sc, ab, try f32s(&sc, &y, &.{ 1, 4 })));
    defer testing.allocator.free(got);
    try expectClose(&want, got);
    const yb = try sc.astype(try f32s(&sc, &y, &.{ 1, 4 }), .bfloat16);
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(try ablation.ablateWith(&sc, ab, yb)));
}

test "deepseek_v41 ablation: a stream the FFN output was expanded into loses that output's share along r only" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    // x2 = post y + comb^T x1 over one row, hc = 2 copies of dim 4; the old stream x1's share along r stays.
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const r = [_]f32{ 0.5, -0.5, 0.5, -0.5 };
    const y = [_]f32{ 2, -1, 0.5, 3 };
    const x1 = [_]f32{ 1, 0, 2, -1, 0.5, 1.5, -2, 1 }; // [hc, dim]
    const post = [_]f32{ 0.75, 1.25 };
    const comb = [_]f32{ 0.5, 0.25, -0.5, 1 }; // [j, k]
    var y_ablated: [4]f32 = undefined;
    ablatedHost(&y, &r, 1, &y_ablated);
    var x2: [8]f32 = undefined;
    var want: [8]f32 = undefined;
    for (0..2) |k| for (0..4) |d| {
        var mix: f32 = 0;
        for (0..2) |j| mix += comb[j * 2 + k] * x1[j * 4 + d];
        x2[k * 4 + d] = post[k] * y[d] + mix;
        want[k * 4 + d] = post[k] * y_ablated[d] + mix;
    };
    const ab = try along(&sc, &r, 1);
    const out = try ablation.ablateExpandedWith(&sc, ab, try f32s(&sc, &x1, &.{ 1, 2, 4 }), try f32s(&sc, &x2, &.{ 1, 2, 4 }), try f32s(&sc, &comb, &.{ 1, 2, 2 }));
    const got = try dsv41.hostF32(out);
    defer testing.allocator.free(got);
    try expectClose(&want, got);
}

test "deepseek_v41 ablation: an Engram row keeps its per-copy keys and loses the value's share along r" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const r = [_]f32{ 0, 0.6, 0, 0.8 };
    // hc = 2 keys of 4, then the value.
    const kv = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 3, 1, -2, 2 };
    var want = kv;
    ablatedHost(kv[8..12], &r, 1, want[8..12]);
    const ab = try along(&sc, &r, 1);
    const got = try dsv41.hostF32(try ablation.ablateValueWith(&sc, ab, try f32s(&sc, &kv, &.{ 1, 12 }), 8));
    defer testing.allocator.free(got);
    try expectClose(&want, got);
}

test "deepseek_v41 ablation: an armed ablation bills its r, col and s arrays" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const table = [_]f32{ 0.6, 0.8 };
    const dirs = try f32s(&sc, &table, &.{ 1, 2 });
    const ab = try ablation.armed(.{ .directions = .{ .ctx = dirs.ctx }, .strength = 1 }, 2, sc.s);
    defer ab.deinit();
    // r [2], col [2, 1] and s [1] as f32: a state or snapshot that retains them bills these bytes.
    try testing.expectEqual(@as(u64, 2 * 4 + 2 * 4 + 4), ablation.retainedBytes(ab));
    try testing.expectEqual(@as(u64, 0), ablation.retainedBytes(null));
}

test "deepseek_v41 ablation: arming takes the slot's row of the table as r and as a column, and its strength" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    // [slots, dim]: slot 1's direction is (0, 0.6, 0, 0.8).
    const table = [_]f32{ 1, 0, 0, 0, 0, 0.6, 0, 0.8 };
    const dirs = try f32s(&sc, &table, &.{ 2, 4 });
    const spec: ablation.Spec = .{ .directions = .{ .ctx = dirs.ctx }, .row = 1, .strength = 0.75, .key = 9 };
    const ab = try ablation.armed(spec, 4, sc.s);
    defer ab.deinit();
    try testing.expectEqual(@as(u64, 9), ab.key);
    const r = try dsv41.hostF32(ab.r);
    defer testing.allocator.free(r);
    try expectClose(table[4..8], r);
    try testing.expectEqualSlices(c_int, &.{ 4, 1 }, mlx.getShape(ab.col));
    const s = try dsv41.hostF32(ab.s);
    defer testing.allocator.free(s);
    try expectClose(&.{0.75}, s);
    // A table that is not `[slots, dim]` f32 refuses.
    try testing.expectError(error.AblationShape, ablation.armed(spec, 8, sc.s));
}

test "deepseek_v41 ablation: states batch together only unablated or under one keyed ablation" {
    const none: A = .{ .ctx = null };
    const k7: ablation.Ablation = .{ .r = none, .col = none, .s = none, .key = 7 };
    const k8: ablation.Ablation = .{ .r = none, .col = none, .s = none, .key = 8 };
    const unkeyed: ablation.Ablation = .{ .r = none, .col = none, .s = none };
    try testing.expect(ablation.same(null, null));
    try testing.expect(ablation.same(k7, k7));
    try testing.expect(!ablation.same(k7, k8));
    try testing.expect(!ablation.same(k7, null));
    try testing.expect(!ablation.same(null, k7));
    try testing.expect(!ablation.same(unkeyed, unkeyed));
}

test "deepseek_v41 ablation: a state takes a forward of its own variant; a fresh or disk-tier state takes any" {
    const none: ablation.Variant = .none;
    const k7: ablation.Variant = .{ .keyed = 7 };
    const plain: ablation.Spec = .{ .directions = .{}, .strength = 1 };
    const seven: ablation.Spec = .{ .directions = .{}, .strength = 1, .key = 7 };
    try testing.expect(std.meta.eql(none, ablation.variantOf(null)));
    try testing.expect(std.meta.eql(k7, ablation.variantOf(seven)));
    try testing.expect(std.meta.eql(ablation.Variant.unkeyed, ablation.variantOf(plain)));
    // Nothing computed yet.
    try testing.expect(ablation.fits(0, k7, plain));
    // Read back from the disk tier: the host keyed the entry by its variant.
    try testing.expect(ablation.fits(12, null, seven));
    try testing.expect(ablation.fits(12, k7, seven));
    // An unablated request on an ablated state, and the reverse.
    try testing.expect(!ablation.fits(12, k7, null));
    try testing.expect(!ablation.fits(12, none, seven));
    // The host's own keys, including its odd hash and a caller's even one, continue their own rows.
    const odd: u64 = std.math.maxInt(u64);
    const even: u64 = 2;
    try testing.expect(ablation.fits(12, .{ .keyed = odd }, .{ .directions = .{}, .strength = 1, .key = odd }));
    try testing.expect(ablation.fits(12, .{ .keyed = even }, .{ .directions = .{}, .strength = 1, .key = even }));
    try testing.expect(!ablation.fits(12, .{ .keyed = odd }, .{ .directions = .{}, .strength = 1, .key = even }));
    // An unkeyed ablation matches only itself, so no state continues rows under it.
    try testing.expect(!ablation.fits(12, ablation.Variant.unkeyed, plain));
}

test "deepseek_v41 ablation: a tap directory's next seq is one past its index, so a restart cannot reuse a file" {
    try testing.expectEqual(@as(u64, 0), ablation.seqFromIndex(""));
    try testing.expectEqual(@as(u64, 1), ablation.seqFromIndex("{\"seq\":0,\"file\":\"tap-000000.npy\"}\n"));
    const two = "{\"seq\":0,\"file\":\"tap-000000.npy\"}\n{\"seq\":1,\"file\":\"tap-000001.npy\"}\n";
    try testing.expectEqual(@as(u64, 2), ablation.seqFromIndex(two));
    // A torn last line (a crash mid-write) gives the highest complete seq.
    try testing.expectEqual(@as(u64, 2), ablation.seqFromIndex(two ++ "{\"seq\":"));
}

test "deepseek_v41 ablation: two writers racing in one tap directory claim distinct files and leave whole records" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try td.dir.realPath(io, &path_buf)];
    // Eight records each, in two threads, so the two writers meet in the directory: each must claim its own seq (the
    // exclusive create steps past a name the other took), and the two appends must not interleave inside a line.
    const Writer = struct {
        dir: []const u8,
        tag: []const u8,
        seqs: [8]u64 = undefined,
        failed: ?anyerror = null,
        fn run(w: *@This()) void {
            for (&w.seqs, 0..) |*seq, i| {
                seq.* = ablation.writeTapRecord(testing.allocator, w.dir, w.tag, i * 4, &.{ 10, 11, 12 }, 2, &.{.{ 0, 2 }}) catch |e| {
                    w.failed = e;
                    return;
                };
            }
        }
    };
    var a: Writer = .{ .dir = dir, .tag = "writer-a's tensor" };
    var b: Writer = .{ .dir = dir, .tag = "writer-b's tensor" };
    // A tensor the index does not name: both writers read seq 0 from the empty index and must step past this name
    // (exclusive create), so the claim fallback runs deterministically, not only when the threads happen to collide.
    try td.dir.writeFile(io, .{ .sub_path = "tap-000000.npy", .data = "sentinel-not-a-tensor" });
    const ta = try std.Thread.spawn(.{}, Writer.run, .{&a});
    const tb = try std.Thread.spawn(.{}, Writer.run, .{&b});
    ta.join();
    tb.join();
    const sentinel = try td.dir.readFileAlloc(io, "tap-000000.npy", testing.allocator, .unlimited);
    defer testing.allocator.free(sentinel);
    try testing.expectEqualStrings("sentinel-not-a-tensor", sentinel);
    try testing.expectEqual(@as(?anyerror, null), a.failed);
    try testing.expectEqual(@as(?anyerror, null), b.failed);

    const index = try td.dir.readFileAlloc(io, "index.jsonl", testing.allocator, .unlimited);
    defer testing.allocator.free(index);
    var by_tag = [_]usize{ 0, 0 };
    var seqs: [16]u64 = undefined;
    var n: usize = 0;
    var lines = std.mem.splitScalar(u8, index, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, line, .{});
        defer parsed.deinit();
        const rec = parsed.value.object;
        const seq: u64 = @intCast(rec.get("seq").?.integer);
        const file = rec.get("file").?.string;
        const tail = rec.get("tail_ids").?.array;
        try testing.expectEqual(@as(usize, 3), tail.items.len);
        const ranges = rec.get("site_ranges").?.array;
        try testing.expectEqual(@as(usize, 1), ranges.items.len);
        try testing.expectEqual(@as(i64, 0), ranges.items[0].array.items[0].integer);
        try testing.expectEqual(@as(i64, 2), ranges.items[0].array.items[1].integer);
        const bytes = try td.dir.readFileAlloc(io, file, testing.allocator, .unlimited);
        defer testing.allocator.free(bytes);
        if (std.mem.eql(u8, bytes, a.tag)) by_tag[0] += 1 else if (std.mem.eql(u8, bytes, b.tag)) by_tag[1] += 1 else return error.TestUnexpectedResult;
        for (seqs[0..n]) |seen| try testing.expect(seen != seq);
        seqs[n] = seq;
        n += 1;
    }
    // Every record of both writers landed once, on its own file, with its own seq.
    try testing.expectEqual(@as(usize, 16), n);
    try testing.expectEqualSlices(usize, &.{ 8, 8 }, &by_tag);
}

test "deepseek_v41 ablation: a partial tap dump writes the sites it recorded and names their layer runs" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    var tap = try ablation.StreamTap.init(testing.allocator, 3, 2); // sites 0..3, one more than layers
    defer tap.deinit();
    tap.on = true;
    const row = [_]f32{ 1, 2, 3, 4 }; // [rows, hc, dim]
    const span = try f32s(&sc, &row, &.{ 1, 1, 4 });
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    var path_buf: [512]u8 = undefined;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = path_buf[0..try td.dir.realPath(io, &path_buf)];

    // The middle layers only, as a pair's worker records them: the dump holds those sites and names the run.
    try tap.record(sc.s, 1, &.{span});
    try tap.record(sc.s, 2, &.{span});
    try testing.expectEqual(@as(?u64, 0), try ablation.dumpTap(testing.allocator, &tap, dir, 0, &.{ 7, 8 }, sc.s));
    const first = try td.dir.readFileAlloc(io, "tap-000000.npy", testing.allocator, .unlimited);
    defer testing.allocator.free(first);
    try testing.expect(std.mem.indexOf(u8, first, "'shape': (2, 1, 4)") != null);
    var index = try td.dir.readFileAlloc(io, "index.jsonl", testing.allocator, .unlimited);
    defer testing.allocator.free(index);
    try testing.expect(std.mem.indexOf(u8, index, "\"site_ranges\":[[1,3]]") != null);
    testing.allocator.free(index);

    // The outer layers only, as a pair's leader records them: two runs, named in order.
    try tap.record(sc.s, 0, &.{span});
    try tap.record(sc.s, 3, &.{span});
    try testing.expectEqual(@as(?u64, 1), try ablation.dumpTap(testing.allocator, &tap, dir, 2, &.{9}, sc.s));
    index = try td.dir.readFileAlloc(io, "index.jsonl", testing.allocator, .unlimited);
    try testing.expect(std.mem.indexOf(u8, index, "\"site_ranges\":[[0,1],[3,4]]") != null);
}

test "deepseek_v41 ablation: the tap dumps a prompt forward, not a decode step or a verify block" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    var tp: Tapped = undefined;
    try tp.attach(t);
    defer tp.detach();
    const io = std.Io.Threaded.global_single_threaded.io();
    // A prompt forward records and dumps one tensor.
    const prompt = try t.plugin(.prefill, &short_prompt, 0);
    defer _ = mlx.mlx_array_free(prompt);
    try testing.expectEqual(@as(usize, 1), try tapRecords(tp.tmp, io));
    // A decode step and a verify block record nothing and dump nothing. Gating by mode is what keeps them off: a pair
    // runs its decode steps through the prompt path, and a verify block is all rows, so both reach `runPrompt`.
    const step = try t.plugin(.decode, &.{44}, @intCast(short_prompt.len));
    defer _ = mlx.mlx_array_free(step);
    // The verify block sits where the decode step left the state.
    const verify = try t.plugin(.verify, &.{ 45, 2 }, @intCast(short_prompt.len + 1));
    defer _ = mlx.mlx_array_free(verify);
    try testing.expectEqual(@as(usize, 1), try tapRecords(tp.tmp, io));
}

test "deepseek_v41 ablation: the stream tap keeps the hc mean of a pass's last rows across its spans, only while on" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    var tap = try ablation.StreamTap.init(testing.allocator, 1, 3);
    defer tap.deinit();
    // A pass of two spans, [2, hc = 2, dim = 2] and [1, 2, 2]: its last three rows, each row's copies averaged.
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]f32{ 9, 10, 11, 12 };
    const spans = [_]A{ try f32s(&sc, &a, &.{ 2, 2, 2 }), try f32s(&sc, &b, &.{ 1, 2, 2 }) };
    try tap.record(sc.s, 0, &spans);
    try testing.expectEqual(@as(usize, 0), tap.sites[0].items.len);
    tap.on = true;
    try tap.record(sc.s, 0, &spans);
    try tap.record(sc.s, 1, spans[1..]);
    try testing.expectEqual(@as(usize, 1), tap.sites[0].items.len);
    try testing.expectEqual(@as(usize, 1), tap.sites[1].items.len);
    const got = try dsv41.hostF32(tap.sites[0].items[0]);
    defer testing.allocator.free(got);
    try expectClose(&.{ 2, 3, 6, 7, 10, 11 }, got);
    try testing.expectEqualSlices(c_int, &.{ 1, 2 }, mlx.getShape(tap.sites[1].items[0]));
    tap.clear();
    try testing.expectEqual(@as(usize, 0), tap.sites[0].items.len);
}

test "deepseek_v41 ablation: the npy header is a 64-byte multiple that names the shape" {
    var buf: [256]u8 = undefined;
    const h = try ablation.npyHeader(&buf, 44, 5, 7168);
    try testing.expectEqual(@as(usize, 0), h.len % 64);
    try testing.expect(std.mem.startsWith(u8, h, "\x93NUMPY\x01\x00"));
    try testing.expect(std.mem.indexOf(u8, h, "'shape': (44, 5, 7168)") != null);
    try testing.expectEqual(@as(u8, '\n'), h[h.len - 1]);
    try testing.expectEqual(@as(u16, @intCast(h.len - 10)), std.mem.readInt(u16, h[8..10], .little));
}

// ── the plugin's forward on the fixture pack ──────────────────────────────────
//
// `fixture.v41Files` (8 layers of hidden 128, no Engram) and `v41DsparkFiles` (two DSpark stages more), loaded by the
// plugin's table. A table row is the unit vector along stream column `col`: a pass that ablates every write to the
// stream at strength 1 leaves that column exactly 0 (a write's product with the unit column is its own value there,
// subtracted whole, and the hyper-connection mixes only scale and sum the column), and a write it misses leaves the
// column nonzero. The calibration tap (`ablation.StreamTap`) reads the column at every layer of a trunk pass, a probe
// confidence head at a DSpark draft's last stage.

const dkern = @import("decode/decode.zig");

const col = 5;
const key: u64 = 0x7e57;
/// The last rows of a pass the tap keeps.
const keep = 5;
/// One span each: 11 rows run every layer's compiled region path (`dsv41.block`: within `region_rows`, past the decode
/// kernels' `max_rows`), 21 rows the plain path (`blockAttn` and the MoE).
const short_prompt = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60, 11, 2, 90 };
const long_prompt = short_prompt ++ [_]u32{ 44, 2, 71, 8, 55, 5, 61, 20, 12, 4 };

/// A forward's table: per slot the unit vector along `col` (`[slots, dim]` f32), with the slots' strengths and keys
/// (borrowed: they outlive the table).
const Table = struct {
    dirs: A,
    abl: abi.Ablation,

    fn init(strengths: []const f32, keys: []const u64, dim: usize) !Table {
        const host = try testing.allocator.alloc(f32, strengths.len * dim);
        defer testing.allocator.free(host);
        @memset(host, 0);
        for (0..strengths.len) |i| host[i * dim + col] = 1;
        const dirs = mlx.mlx_array_new_data(host.ptr, &[_]c_int{ @intCast(strengths.len), @intCast(dim) }, 2, .float32);
        return .{ .dirs = dirs, .abl = .{ .directions = .{ .ctx = dirs.ctx }, .strengths = .of(strengths), .keys = .of(keys) } };
    }

    fn deinit(self: Table) void {
        _ = mlx.mlx_array_free(self.dirs);
    }
};

/// The plugin's logits `[rows, V]` (host f32, the caller frees) for `toks` of `slots` at `positions`, under `abl`.
fn forwardUnder(t: *Twin, mode: abi.Mode, slots: []const u32, toks: []const u32, positions: []const u32, abl: ?*const abi.Ablation) ![]f32 {
    var out: abi.ForwardOut = .{};
    const rows: u32 = if (mode == .batched_rows) 1 else @intCast(toks.len);
    const v = v41.table.forward(t.m, &.{ .mode = mode, .rows = rows, .tokens = .of(toks), .slots = .of(slots), .positions = .of(positions), .ablation = abl }, &out);
    if (!v.isOk()) {
        std.debug.print("forward: {s}\n", .{v.why.slice()});
        return error.PluginForward;
    }
    const l: A = .{ .ctx = out.logits.ctx };
    defer _ = mlx.mlx_array_free(l);
    return dsv41.hostF32(l);
}

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var worst: f32 = 0;
    for (a, b) |x, y| worst = @max(worst, @abs(x - y));
    return worst;
}

/// `testing.expect` that names what it expected.
fn expectWhy(ok: bool, comptime what: []const u8, args: anytype) !void {
    if (ok) return;
    std.debug.print("expected " ++ what ++ "\n", args);
    return error.TestUnexpectedResult;
}

/// The largest magnitude in column `col`, and anywhere, of the rows a tap kept.
const Peaks = struct {
    col: f32 = 0,
    all: f32 = 0,

    fn add(p: *Peaks, rows: []const f32, dim: usize) void {
        for (rows, 0..) |v, i| {
            p.all = @max(p.all, @abs(v));
            if (i % dim == col) p.col = @max(p.col, @abs(v));
        }
    }
};

/// The calibration tap on a twin's model (what `MLX_SERVE_DSV41_TAP_DIR` turns on), writing into a directory of its
/// own. Detached before the twin closes: the model frees only a tap of its own allocator.
const Tapped = struct {
    tap: ablation.StreamTap,
    tmp: testing.TmpDir,
    path: [512]u8,
    model: *v41.Model,

    fn attach(self: *Tapped, t: *Twin) !void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.model = @ptrCast(@alignCast(t.m));
        self.tmp = testing.tmpDir(.{});
        errdefer self.tmp.cleanup();
        const dir = self.path[0..try self.tmp.dir.realPath(io, &self.path)];
        self.tap = try ablation.StreamTap.init(testing.allocator, self.model.mdl.n_layers, keep);
        self.model.tap_dir = dir;
        self.model.mdl.stream_tap = &self.tap;
    }

    fn detach(self: *Tapped) void {
        self.model.mdl.stream_tap = null;
        self.model.tap_dir = null;
        self.tap.deinit();
        self.tmp.cleanup();
    }

    /// The last prompt forward's file: `[layers + 1, rows, dim]` f32 past its header. Names come from the directory's
    /// own index, the way a restart continues it.
    fn lastDump(self: *Tapped, rows: usize) !Peaks {
        const io = std.Io.Threaded.global_single_threaded.io();
        const idx = try self.tmp.dir.readFileAlloc(io, "index.jsonl", testing.allocator, .limited(1 << 20));
        defer testing.allocator.free(idx);
        const trimmed = std.mem.trimEnd(u8, idx, "\r\n");
        const last = if (std.mem.lastIndexOfScalar(u8, trimmed, '\n')) |i| trimmed[i + 1 ..] else trimmed;
        const at = std.mem.indexOf(u8, last, "\"file\":\"") orelse return error.NoTapRecord;
        const rest = last[at + "\"file\":\"".len ..];
        const end = std.mem.indexOfScalar(u8, rest, '"') orelse return error.NoTapRecord;
        const bytes = try self.tmp.dir.readFileAlloc(io, rest[0..end], testing.allocator, .limited(1 << 20));
        defer testing.allocator.free(bytes);
        const dim: usize = self.model.cfg.hidden_size;
        const data = bytes[10 + @as(usize, std.mem.readInt(u16, bytes[8..10], .little)) ..];
        try testing.expectEqual((self.model.mdl.n_layers + 1) * rows * dim * 4, data.len);
        const v = try testing.allocator.alloc(f32, data.len / 4);
        defer testing.allocator.free(v);
        for (v, 0..) |*x, i| x.* = @bitCast(std.mem.readInt(u32, data[i * 4 ..][0..4], .little));
        var p: Peaks = .{};
        p.add(v, dim);
        return p;
    }

    /// Into `p`, the one pass every site recorded since the last `clear`: `rows` rows apiece.
    fn recorded(self: *Tapped, rows: usize, p: *Peaks) !void {
        const dim: usize = self.model.cfg.hidden_size;
        for (self.tap.sites) |list| {
            try testing.expectEqual(@as(usize, 1), list.items.len);
            const v = try dsv41.hostF32(list.items[0]);
            defer testing.allocator.free(v);
            try testing.expectEqual(rows * dim, v.len);
            p.add(v, dim);
        }
    }
};

/// The tap directory's records: every line of its index, 0 when there is none.
fn tapRecords(td: testing.TmpDir, io: std.Io) !usize {
    const idx = td.dir.readFileAlloc(io, "index.jsonl", testing.allocator, .limited(1 << 20)) catch |e| switch (e) {
        error.FileNotFound => return 0,
        else => return e,
    };
    defer testing.allocator.free(idx);
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, idx, '\n');
    while (it.next()) |line| if (line.len > 0) {
        n += 1;
    };
    return n;
}

const Run = struct { logits: []f32, peaks: Peaks };

/// A fresh twin's prompt forward of slot 1 at strength `s` (null: no table) with the tap on: its logits (the caller
/// frees) and the peaks of the tap's file. `region`: whether the prompt's rows take the compiled region path.
fn promptRun(files: []const sdk.fixture.File, prompt: []const u32, s: ?f32, region: bool) !Run {
    const t = try Twin.open(files, "");
    defer t.close();
    var tp: Tapped = undefined;
    try tp.attach(t);
    defer tp.detach();
    try testing.expect(dsv41.compiledAt(tp.model.mdl, prompt.len) == region and tp.model.mdl.streamed == null);
    const strengths = [_]f32{s orelse 0};
    const table = try Table.init(&strengths, &.{key}, tp.model.cfg.hidden_size);
    defer table.deinit();
    const logits = try forwardUnder(t, .prefill, &.{1}, prompt, &.{0}, if (s != null) &table.abl else null);
    errdefer testing.allocator.free(logits);
    return .{ .logits = logits, .peaks = try tp.lastDump(@min(prompt.len, keep)) };
}

test "deepseek_v41 ablation forward: a prompt at strength 1 leaves the stream nothing along r at any layer, on the region path and the plain path" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    try testing.expect(short_prompt.len > dkern.max_rows and short_prompt.len <= dsv41.region_rows and long_prompt.len > dsv41.region_rows);
    for ([_][]const u32{ &short_prompt, &long_prompt }, [_]bool{ true, false }) |prompt, region| {
        const plain = try promptRun(files, prompt, null, region);
        defer testing.allocator.free(plain.logits);
        const ablated = try promptRun(files, prompt, 1, region);
        defer testing.allocator.free(ablated.logits);
        // The tap read the stream at every site; only the ablated pass's column is empty.
        try testing.expect(plain.peaks.col > 0 and ablated.peaks.all > 0);
        try testing.expectEqual(@as(f32, 0), ablated.peaks.col);
        try testing.expect(maxAbsDiff(plain.logits, ablated.logits) > 0);
    }
}

test "deepseek_v41 ablation forward: a table at strength 0 runs the prompt and its decode bit for bit as none does" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const none = try Twin.open(files, "");
    defer none.close();
    const zero = try Twin.open(files, "");
    defer zero.close();
    const m: *v41.Model = @ptrCast(@alignCast(zero.m));
    const table = try Table.init(&.{0}, &.{key}, m.cfg.hidden_size);
    defer table.deinit();
    const want = try forwardUnder(none, .prefill, &.{1}, &short_prompt, &.{0}, null);
    defer testing.allocator.free(want);
    const got = try forwardUnder(zero, .prefill, &.{1}, &short_prompt, &.{0}, &table.abl);
    defer testing.allocator.free(got);
    try testing.expectEqualSlices(f32, want, got);
    const pos: u32 = short_prompt.len;
    const want_step = try forwardUnder(none, .decode, &.{1}, &.{44}, &.{pos}, null);
    defer testing.allocator.free(want_step);
    const got_step = try forwardUnder(zero, .decode, &.{1}, &.{44}, &.{pos}, &table.abl);
    defer testing.allocator.free(got_step);
    try testing.expectEqualSlices(f32, want_step, got_step);
}

test "deepseek_v41 ablation forward: a decode step at strength 1 leaves the stream nothing along r at any layer (the region path at one row)" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    var tp: Tapped = undefined;
    try tp.attach(t);
    defer tp.detach();
    const mdl = tp.model.mdl;
    // One row runs `dsv41.block`'s region path. The fused decode kernels are built on every GPU stream (`Dec.init`), but
    // `blockFused` hands the row back before any state moves: `dkern.in_proj.plan` declines each of the fixture's layers
    // (its hidden size, 128, is no multiple of 256). No bank streams.
    try expectWhy(dsv41.compiledAt(mdl, 1), "one row within the compiled regions (`dsv41.compiledAt`)", .{});
    try expectWhy(mdl.streamed == null, "no streamed bank", .{});
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    const shape = [_]c_int{ 1, @intCast(mdl.cfg.dsv4_hc_mult), @intCast(mdl.cfg.hidden_size) };
    try mlx.check(mlx.mlx_zeros(&row, &shape, shape.len, .bfloat16, mlx.gpuStream()));
    for (mdl.layers[0..mdl.n_layers], 0..) |*ly, li|
        try expectWhy(dkern.in_proj.plan(mdl, ly, row) == null, "the fused decode kernels to decline layer {d} (`in_proj.plan`), so `blockFused` falls back to the region path", .{li});
    const table = try Table.init(&.{1}, &.{key}, tp.model.cfg.hidden_size);
    defer table.deinit();
    var peaks: [2]Peaks = .{ .{}, .{} };
    for ([_]?*const abi.Ablation{ &table.abl, null }, &peaks) |abl, *pk| {
        // A request each: position 0 resets the slot's state, so the plain one is plain from its first row.
        testing.allocator.free(try forwardUnder(t, .prefill, &.{1}, &short_prompt, &.{0}, abl));
        for ([_]u32{ 44, 2 }, 0..) |tok, i| {
            // A decode forward leaves the tap as it is (`run`): it records while switched on here.
            tp.tap.on = true;
            const logits = forwardUnder(t, .decode, &.{1}, &.{tok}, &.{@intCast(short_prompt.len + i)}, abl);
            tp.tap.on = false;
            testing.allocator.free(try logits);
            try tp.recorded(1, pk);
            tp.tap.clear();
        }
    }
    try expectWhy(peaks[0].all > 0, "the ablated decode's tap to record the stream", .{});
    try expectWhy(peaks[1].col > 0, "the plain decode's stream to carry column {d}, so a missed write would show", .{col});
    try expectWhy(peaks[0].col == 0, "the ablated decode to leave column {d} of the stream 0 at every layer; its peak is {e}", .{ col, peaks[0].col });
}

test "deepseek_v41 ablation forward: a batched tick of an ablated slot and a plain one is each slot's own decode step, bit for bit" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const prompts = [2][]const u32{ &short_prompt, &.{ 5, 8, 13, 21, 34, 55, 89, 4, 6, 10, 12, 14, 16, 18 } };
    const toks = [2]u32{ 44, 9 };
    const pos = [2]u32{ @intCast(prompts[0].len), @intCast(prompts[1].len) };
    const solo = try Twin.open(files, "");
    defer solo.close();
    const batch = try Twin.open(files, "");
    defer batch.close();
    const m: *v41.Model = @ptrCast(@alignCast(solo.m));
    const one = try Table.init(&.{1}, &.{key}, m.cfg.hidden_size);
    defer one.deinit();
    // Slot 1 at strength 1, slot 2 at 0: the tick runs them one at a time, each under its own (`extendBatched`).
    const two = try Table.init(&.{ 1, 0 }, &.{ key, 0 }, m.cfg.hidden_size);
    defer two.deinit();
    for ([_]*Twin{ solo, batch }) |t| {
        testing.allocator.free(try forwardUnder(t, .prefill, &.{1}, prompts[0], &.{0}, &one.abl));
        testing.allocator.free(try forwardUnder(t, .prefill, &.{2}, prompts[1], &.{0}, null));
    }
    const want0 = try forwardUnder(solo, .decode, &.{1}, toks[0..1], pos[0..1], &one.abl);
    defer testing.allocator.free(want0);
    const want1 = try forwardUnder(solo, .decode, &.{2}, toks[1..2], pos[1..2], null);
    defer testing.allocator.free(want1);
    const got = try forwardUnder(batch, .batched_rows, &.{ 1, 2 }, &toks, &pos, &two.abl);
    defer testing.allocator.free(got);
    try testing.expectEqual(want0.len + want1.len, got.len);
    try testing.expectEqualSlices(f32, want0, got[0..want0.len]);
    try testing.expectEqualSlices(f32, want1, got[want0.len..]);
}

/// A draft's probe confidences, every position of the block computed: all 0 for an ablated state's draft, not all 0 for
/// a plain state's.
fn expectProbe(mdl: *const dsv41.Dsv41Model, d: dsv41.DsparkDraft, ablated: bool) !void {
    try testing.expectEqual(mdl.ds_block, d.confidence.len);
    var peak: f32 = 0;
    for (d.confidence) |c| peak = @max(peak, @abs(c));
    if (ablated) try testing.expectEqual(@as(f32, 0), peak) else try testing.expect(peak > 0);
}

test "deepseek_v41 ablation forward: a DSpark draft of an ablated state leaves its last stage's stream nothing along r, alone and batched" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41DsparkFiles(arena.allocator(), 7), "");
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    const mdl = m.mdl;
    try testing.expectEqual(@as(usize, 2), mdl.n_mtp);
    const table = try Table.init(&.{1}, &.{key}, m.cfg.hidden_size);
    defer table.deinit();
    // Slot 1's state ablated, slot 2's plain.
    testing.allocator.free(try forwardUnder(t, .prefill, &.{1}, &short_prompt, &.{0}, &table.abl));
    testing.allocator.free(try forwardUnder(t, .prefill, &.{2}, &short_prompt, &.{0}, null));
    var sts: [2]*dsv41.State = undefined;
    var gens: [2]u64 = undefined;
    try testing.expect(v41.slotStates(t.m, &.{ 1, 2 }, &sts, &gens));
    // The confidence head is linear in the draft's collapsed last-stage stream (and the Markov row): a probe head along
    // `col` reads that column, exactly 0 when the draft ablated every write. `DsparkW.conf` is f32 `[1, dim + rank]`.
    const ds = &mdl.dspark.?;
    try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(ds.conf));
    const host = try testing.allocator.alloc(f32, mlx.mlx_array_size(ds.conf));
    defer testing.allocator.free(host);
    @memset(host, 0);
    host[col] = 1;
    const shape = mlx.getShape(ds.conf);
    const probe = mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(probe);
    const head = ds.conf;
    ds.conf = probe;
    defer ds.conf = head;
    mdl.ds_want_conf = true;
    const t1s = [2]u32{ 5, 5 };
    // Each alone (`draftGraph`), both in one call (`dsparkDraftBatched`: each under its own), and the ablated one in a
    // batch of its own (one batch under its ablation).
    for (sts, 0..) |st, k| {
        var d = try dsv41.dsparkDraft(mdl, testing.allocator, st, t1s[k]);
        defer d.deinit(testing.allocator);
        try expectProbe(mdl, d, k == 0);
    }
    var both: [2]dsv41.DsparkDraft = undefined;
    try dsv41.dsparkDraftBatched(mdl, testing.allocator, &sts, &t1s, false, &both);
    defer for (&both) |*d| d.deinit(testing.allocator);
    for (both, 0..) |d, k| try expectProbe(mdl, d, k == 0);
    var own: [1]dsv41.DsparkDraft = undefined;
    try dsv41.dsparkDraftBatched(mdl, testing.allocator, sts[0..1], t1s[0..1], false, &own);
    defer own[0].deinit(testing.allocator);
    try expectProbe(mdl, own[0], true);
}
