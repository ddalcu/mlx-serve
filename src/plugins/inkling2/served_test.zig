//! The inkling2 pack served through the engine: the driver's load and forward of the tiny pack against mlx-lm's
//! logits, and a real pack (`INKLING2_MODEL`, references from `tests/dump_inkling2_fixtures.py --pack` in
//! `INKLING2_REF`) prefilled and teacher-forced through the plugin table on the GPU.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const env = @import("../../env.zig");
const model = @import("../../sdk/config/model.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const parity = @import("../../sdk/testing/parity.zig");
const conformance = @import("../../sdk/testing/conformance.zig");
const test_cache = @import("../../sdk/testing/cache_ops.zig");
const FsPackDir = @import("../../sdk/util/pack_dir.zig").FsPackDir;
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const loadWeightsOpt = @import("../../sdk/weights/load.zig").loadWeightsOpt;
const plugin = @import("plugin.zig");
const fixture = @import("../../sdk/testing/fixture.zig");
const device = @import("../../mlx/device.zig");

const testing = std.testing;

fn stage(io: std.Io, tmp: *testing.TmpDir, buf: []u8) ![]const u8 {
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = @embedFile("fixtures/tiny/config.json") });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = @embedFile("fixtures/tiny/model.safetensors") });
    return buf[0..try tmp.dir.realPath(io, buf)];
}

/// The plugin table's own prefill of `prompt` on the GPU (all rows), outside the engine.
fn tableRows(a: std.mem.Allocator, prompt: []const u32) ![]f32 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const fx: fixture.Fixture = .{ .files = &.{
        .{ .name = "config.json", .bytes = @embedFile("fixtures/tiny/config.json") },
        .{ .name = "model.safetensors", .bytes = @embedFile("fixtures/tiny/model.safetensors") },
    } };
    var p = try conformance.Loaded.openWith(testing.io, arena.allocator(), plugin.plugin.name, &plugin.table, fx, .{ .gpu = true });
    defer p.close();
    var cache = try test_cache.TestCache.init(a, p.parsed.shell.num_hidden_layers);
    defer cache.deinit();
    var h = [_]abi.Handle{cache.handle()};
    var out: abi.ForwardOut = .{};
    try testing.expect(p.table.forward(p.model, &.{ .mode = .prefill, .logits = .all_rows, .rows = @intCast(prompt.len), .tokens = .of(prompt), .slots = .of(&[_]u32{0}), .positions = .of(&[_]u32{0}), .caches = .of(&h), .cache_ops = &test_cache.ops }, &out).isOk());
    if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
    return conformance.consumeF32(a, out.logits);
}

test "inkling2 served: the driver's forward of the prompt matches mlx-lm's Inkling 2" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.gpuStream();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const dir = try stage(io, &tmp, &path_buf);
    var cfg = try model.parseConfig(io, a, dir);
    defer cfg.deinit(a);
    try testing.expect(cfg.plugin != null);
    var xfm = try driver.load(io, a, &cfg, dir, s);
    defer xfm.deinit();

    const tk = try std.json.parseFromSlice(struct { prompt: []const u32, decode: []const u32 }, a, @embedFile("fixtures/tiny/tokens.json"), .{});
    defer tk.deinit();
    const prompt = tk.value.prompt;
    const ids32 = try a.alloc(i32, prompt.len);
    defer a.free(ids32);
    for (ids32, prompt) |*d, t| d.* = @intCast(t);
    const ids = mlx.mlx_array_new_data(ids32.ptr, &[_]c_int{ 1, @intCast(prompt.len) }, 2, .int32);
    defer _ = mlx.mlx_array_free(ids);
    const logits = try xfm.forward(ids);
    defer _ = mlx.mlx_array_free(logits);
    const ours = try parity.qwen4ReadF32(a, logits, s);
    defer a.free(ours);
    const vocab = 120;
    if (device.tf32Matmul()) {
        // TF32 matmuls make this random f32 pack diverge from the f32 reference: the engine must serve the table's
        // own logits.
        const want = try tableRows(a, prompt);
        defer a.free(want);
        try testing.expectEqual(want.len, ours.len);
        var worst: f32 = 0;
        var peak: f32 = 0;
        for (ours, want) |x, y| {
            worst = @max(worst, @abs(x - y));
            peak = @max(peak, @abs(y));
        }
        if (!(worst <= 1e-4 * peak)) std.debug.print("[inkling2 served] the engine's logits differ from the table's by {e} of {e}\n", .{ worst, peak });
        return testing.expect(worst <= 1e-4 * peak);
    }
    const bytes = @embedFile("fixtures/tiny/prefill_logits.f32");
    const ref = @as([*]const f32, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / 4];
    const cmp = parity.qwen4CompareRows(ours, ref, prompt.len, vocab);
    const misses = parity.decidedMisses(ours, ref, prompt.len, vocab);
    if (cmp.min_cos <= 0.999 or misses > 0) std.debug.print("[inkling2 served] min cos {d:.6}, decided misses {d}\n", .{ cmp.min_cos, misses });
    // The bar of proof_test.zig.
    try testing.expect(cmp.min_cos > 0.999 and misses == 0);
}

fn readFile(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    var buf: [1024]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, name });
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30));
}

fn f32s(bytes: []u8) []const f32 {
    return std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(bytes)));
}

/// One forward of `toks` from `pos` on the slot's cache; the last row's logits.
fn lastRow(a: std.mem.Allocator, t: *const abi.ModelTable, m: *anyopaque, cache: *test_cache.TestCache, toks: []const u32, pos: u32) ![]f32 {
    var h = [_]abi.Handle{cache.handle()};
    var out: abi.ForwardOut = .{};
    const mode: abi.Mode = if (toks.len == 1) .decode else .prefill;
    const v = t.forward(m, &.{ .mode = mode, .logits = .last_row, .rows = @intCast(toks.len), .tokens = .of(toks), .slots = .of(&[_]u32{0}), .positions = .of(&[_]u32{pos}), .caches = .of(&h), .cache_ops = &test_cache.ops }, &out);
    if (!v.isOk()) std.debug.print("[inkling2 pack] forward refused: {s}\n", .{v.why.slice()});
    try testing.expect(v.isOk());
    if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
    return conformance.consumeF32(a, out.logits);
}

test "inkling2 pack: a real pack's prompt and teacher-forced decode match mlx-lm" {
    const pack_dir = std.mem.span(env.get(.INKLING2_MODEL) orelse return error.SkipZigTest);
    const ref_dir = std.mem.span(env.get(.INKLING2_REF) orelse return error.SkipZigTest);
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    const prompt_json = try readFile(a, io, ref_dir, "prompt.json");
    defer a.free(prompt_json);
    const steps_json = try readFile(a, io, ref_dir, "steps.json");
    defer a.free(steps_json);
    const last_bytes = try readFile(a, io, ref_dir, "prefill_last.f32");
    defer a.free(last_bytes);
    const dec_bytes = try readFile(a, io, ref_dir, "decode_logits.f32");
    defer a.free(dec_bytes);
    const prompt = try std.json.parseFromSlice([]const u32, a, prompt_json, .{});
    defer prompt.deinit();
    const steps = try std.json.parseFromSlice([]const u32, a, steps_json, .{});
    defer steps.deinit();
    const want_last = f32s(last_bytes);
    const want_dec = f32s(dec_bytes);
    const vocab = want_last.len;
    try testing.expectEqual(steps.value.len * vocab, want_dec.len);

    var fs = try FsPackDir.open(io, pack_dir);
    defer fs.close();
    const pd = fs.packDir();
    const cfg_json = try readFile(a, io, pack_dir, "config.json");
    defer a.free(cfg_json);
    const t = &plugin.table;
    var parsed: abi.Parsed = .{};
    const pv = t.parse(&pd, .of(cfg_json), .{}, null, false, &parsed);
    if (!pv.isOk()) std.debug.print("[inkling2 pack] parse refused: {s}\n", .{pv.why.slice()});
    try testing.expect(pv.isOk());
    defer t.free_parsed(&parsed);
    try testing.expectEqual(vocab, parsed.shell.vocab_size);
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var weights = Resident.adopt(try loadWeightsOpt(io, std.heap.c_allocator, pack_dir, .{}));
    defer weights.deinit();
    const src = weights.source();
    var why: abi.Verdict = .{};
    const m = t.load(&.{ .model_dir = .of(pack_dir), .dir = &pd, .parsed = &parsed, .stream = .{ .ctx = s.ctx }, .weights = &src }, &why) orelse {
        std.debug.print("[inkling2 pack] load refused: {s}\n", .{why.why.slice()});
        return error.LoadRefused;
    };
    defer {
        _ = mlx.mlx_synchronize(s);
        t.unload(m);
    }

    var cache = try test_cache.TestCache.init(a, parsed.shell.num_hidden_layers);
    defer cache.deinit();
    const n: u32 = @intCast(prompt.value.len);
    const ours = try a.alloc(f32, (1 + steps.value.len) * vocab);
    defer a.free(ours);
    const want = try a.alloc(f32, ours.len);
    defer a.free(want);
    @memcpy(want[0..vocab], want_last);
    @memcpy(want[vocab..], want_dec);
    const first = try lastRow(a, t, m, &cache, prompt.value, 0);
    defer a.free(first);
    @memcpy(ours[0..vocab], first);
    for (steps.value, 0..) |tok, i| {
        const row = try lastRow(a, t, m, &cache, &.{tok}, n + @as(u32, @intCast(i)));
        defer a.free(row);
        @memcpy(ours[(1 + i) * vocab ..][0..vocab], row);
    }
    const rows = 1 + steps.value.len;
    const cmp = parity.qwen4CompareRows(ours, want, rows, vocab);
    const misses = parity.decidedMisses(ours, want, rows, vocab);
    std.debug.print("[inkling2 pack] {s}: {d} rows, min cos {d:.6}, decided misses {d}\n", .{ pack_dir, rows, cmp.min_cos, misses });
    // 0.99: below the bf16 gap between two correct implementations.
    try testing.expect(cmp.min_cos > 0.99 and misses == 0);
}
