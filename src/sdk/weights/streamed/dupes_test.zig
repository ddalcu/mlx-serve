//! The real-pack oracle for the duplicated implementations (DeepSeek-V4.1, MiMo, GLM-5.3-Flash: `plugins/<model>/` vs
//! `plugins/<model>/stream/`): one pack loaded the way the engine loads it (`engines/mlx/load.zig`: the plugin's
//! table for a served type, else the legacy `Transformer`, which hands an mlx-stream pack to mlx-stream), its prompt
//! and a greedy continuation written as a receipt; a second test compares two receipts. A resident pack and a streamed
//! pack of one model never share a process: each receipt is its own run. Part of the main test module (it reaches the
//! engine's loader), kept beside the streamer it compares.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const env = @import("../../../env.zig");
const model = @import("../../config/model.zig");
const driver = @import("../../../engines/mlx/plugin/driver.zig");
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;
const Weights = @import("../resident.zig").Weights;

const testing = std.testing;
const top_k = 5;

/// One step of a receipt: the greedy id and the top-k (id, logit) of the row it was read from.
pub const Step = struct { id: u32, top: [top_k]struct { u32, f32 } };

/// A loaded pack: the engine's config and model, and which implementation serves it. The model owns the stream handle
/// `open` is given.
pub const Served = struct {
    cfg: model.ModelConfig,
    weights: Weights,
    xfm: Transformer,

    pub fn open(io: std.Io, a: std.mem.Allocator, dir: []const u8, s: mlx.mlx_stream) !*Served {
        const p = try a.create(Served);
        errdefer a.destroy(p);
        p.cfg = try model.parseConfig(io, a, dir);
        errdefer p.cfg.deinit(a);
        // The engine's two arms (`load.zig`): a served plugin reads its own tensors; the rest load the map first.
        if (p.cfg.plugin != null) {
            p.weights = Weights.init(a);
            p.xfm = try driver.loadWith(io, a, &p.cfg, dir, s, null);
        } else {
            p.weights = try model.loadModelWeights(io, a, dir, &p.cfg, false);
            errdefer p.weights.deinit();
            model.resolveWeightPrefix(&p.cfg, &p.weights);
            try model.narrowHadamardPackTables(&p.cfg, &p.weights, s);
            p.xfm = try Transformer.init(io, a, p.cfg);
        }
        return p;
    }

    pub fn close(p: *Served, a: std.mem.Allocator) void {
        p.xfm.deinit();
        p.weights.deinit();
        p.cfg.deinit(a);
        a.destroy(p);
    }

    pub fn implName(p: *const Served) []const u8 {
        if (p.cfg.mlxStream()) return "mlx-stream";
        if (p.cfg.plugin) |b| return b.table.name.slice();
        return "legacy";
    }
};

/// The last row of `logits` (`[1, L, V]` or `[1, V]`) as host f32.
fn lastRow(a: std.mem.Allocator, logits: mlx.mlx_array, s: mlx.mlx_stream) ![]f32 {
    const shape = mlx.getShape(logits);
    const v = shape[shape.len - 1];
    const rows = @divExact(@as(c_int, @intCast(mlx.mlx_array_size(logits))), v);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, logits, &[_]c_int{ rows, v }, 2, s));
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    try mlx.check(mlx.mlx_slice(&row, flat, &[_]c_int{ rows - 1, 0 }, 2, &[_]c_int{ rows, v }, 2, &[_]c_int{ 1, 1 }, 2, s));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, row, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const d = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return a.dupe(f32, d[0..@intCast(v)]);
}

/// The top-k of `row`, largest first, ties to the lower id.
pub fn topOf(row: []const f32) [top_k]struct { u32, f32 } {
    var out: [top_k]struct { u32, f32 } = @splat(.{ 0, -std.math.inf(f32) });
    for (row, 0..) |v, i| {
        var k: usize = top_k;
        while (k > 0 and v > out[k - 1][1]) k -= 1;
        if (k == top_k) continue;
        std.mem.copyBackwards(struct { u32, f32 }, out[k + 1 ..], out[k .. top_k - 1]);
        out[k] = .{ @intCast(i), v };
    }
    return out;
}

/// Ids separated by whitespace or commas.
pub fn parseIds(a: std.mem.Allocator, text: []const u8) ![]u32 {
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n,");
    while (it.next()) |t| try ids.append(a, try std.fmt.parseInt(u32, t, 10));
    if (ids.items.len == 0) return error.EmptyPrompt;
    return ids.toOwnedSlice(a);
}

/// The engine's forward over `ids` from the model's own cache; the last row's logits, host f32.
pub fn forwardIds(x: *Transformer, a: std.mem.Allocator, ids: []const u32, s: mlx.mlx_stream) ![]f32 {
    const ids32 = try a.alloc(i32, ids.len);
    defer a.free(ids32);
    for (ids, ids32) |v, *o| o.* = @intCast(v);
    const arr = mlx.mlx_array_new_data(ids32.ptr, &[_]c_int{ 1, @intCast(ids.len) }, 2, .int32);
    defer _ = mlx.mlx_array_free(arr);
    const logits = try x.forward(arr);
    defer _ = mlx.mlx_array_free(logits);
    return lastRow(a, logits, s);
}

/// Rows of logits from two implementations over the same ids: the worst row's cosine, the largest error against the
/// reference rows' peak, and how many rows' argmax differ.
pub const RowGap = struct { min_cos: f64, max_err: f64, argmax_misses: usize, finite: bool };

pub fn rowGap(got: []const f32, want: []const f32, width: usize) RowGap {
    var out: RowGap = .{ .min_cos = 1, .max_err = 0, .argmax_misses = 0, .finite = true };
    var peak: f64 = 0;
    for (want) |v| peak = @max(peak, @abs(v));
    for (0..want.len / width) |r| {
        const g = got[r * width ..][0..width];
        const w = want[r * width ..][0..width];
        var dot: f64 = 0;
        var gg: f64 = 0;
        var ww: f64 = 0;
        var ag: usize = 0;
        var aw: usize = 0;
        for (g, w, 0..) |x, y, k| {
            if (!std.math.isFinite(x) or !std.math.isFinite(y)) out.finite = false;
            dot += @as(f64, x) * y;
            gg += @as(f64, x) * x;
            ww += @as(f64, y) * y;
            out.max_err = @max(out.max_err, @abs(@as(f64, x) - y) / @max(peak, 1e-30));
            if (x > g[ag]) ag = k;
            if (y > w[aw]) aw = k;
        }
        out.min_cos = @min(out.min_cos, dot / @sqrt(@max(gg * ww, 1e-300)));
        if (ag != aw) out.argmax_misses += 1;
    }
    return out;
}

/// The receipt's JSON: `{"pack", "impl", "prompt", "steps": [{"id", "top": [[id, logit], ...]}]}`.
pub fn writeReceipt(w: *std.Io.Writer, pack: []const u8, impl: []const u8, prompt: usize, steps: []const Step) !void {
    try w.print("{{\"pack\":{f},\"impl\":\"{s}\",\"prompt\":{d},\"steps\":[", .{ std.json.fmt(pack, .{}), impl, prompt });
    for (steps, 0..) |st, i| {
        try w.print("{s}{{\"id\":{d},\"top\":[", .{ if (i == 0) "" else ",", st.id });
        for (st.top, 0..) |t, k| try w.print("{s}[{d},{e}]", .{ if (k == 0) "" else ",", t[0], t[1] });
        try w.writeAll("]}");
    }
    try w.writeAll("]}\n");
}

pub const Receipt = struct {
    impl: []const u8,
    prompt: usize,
    steps: []Step,
};

pub fn parseReceipt(a: std.mem.Allocator, text: []const u8) !Receipt {
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    const o = v.object;
    const raw = o.get("steps").?.array.items;
    const steps = try a.alloc(Step, raw.len);
    for (raw, steps) |r, *st| {
        st.id = @intCast(r.object.get("id").?.integer);
        for (r.object.get("top").?.array.items, 0..) |t, k| st.top[k] = .{ @intCast(t.array.items[0].integer), @floatCast(switch (t.array.items[1]) {
            .float => |f| f,
            .integer => |n| @as(f64, @floatFromInt(n)),
            else => return error.ReceiptLogit,
        }) };
    }
    return .{ .impl = o.get("impl").?.string, .prompt = @intCast(o.get("prompt").?.integer), .steps = steps };
}

/// Two receipts of one prompt: the first step whose greedy ids differ, how many steps agree before it, and over those
/// steps the top-k sets' smallest overlap and the largest logit gap on ids both top-k name.
pub const Agreement = struct { first_diff: ?usize, min_overlap: usize, max_gap: f32 };

pub fn compare(x: Receipt, y: Receipt) !Agreement {
    if (x.prompt != y.prompt) return error.ReceiptPromptsDiffer;
    var out: Agreement = .{ .first_diff = null, .min_overlap = top_k, .max_gap = 0 };
    for (x.steps[0..@min(x.steps.len, y.steps.len)], y.steps[0..@min(x.steps.len, y.steps.len)], 0..) |sx, sy, i| {
        var overlap: usize = 0;
        for (sx.top) |tx| for (sy.top) |ty| if (tx[0] == ty[0]) {
            overlap += 1;
            out.max_gap = @max(out.max_gap, @abs(tx[1] - ty[1]));
        };
        out.min_overlap = @min(out.min_overlap, overlap);
        if (sx.id != sy.id) {
            out.first_diff = i;
            break;
        }
    }
    return out;
}

// ── tests ──

test "dupes oracle: a pack's prompt and greedy continuation through the engine's own load, as a receipt (DUPES_PACK)" {
    const dir = env.span(.DUPES_PACK) orelse return error.SkipZigTest;
    const prompt_path = env.span(.DUPES_PROMPT) orelse return error.SkipZigTest;
    const out_path = env.span(.DUPES_OUT) orelse return error.SkipZigTest;
    const n_steps = env.int(.DUPES_STEPS, usize, 32);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = testing.io;
    const prompt = try parseIds(a, try std.Io.Dir.cwd().readFileAlloc(io, prompt_path, a, .limited(16 << 20)));
    const s = mlx.gpuStream();
    const p = try Served.open(io, testing.allocator, dir, s);
    defer p.close(testing.allocator);
    const steps = try a.alloc(Step, n_steps);
    var row = try forwardIds(&p.xfm, a, prompt, s);
    for (steps) |*st| {
        st.top = topOf(row);
        st.id = st.top[0][0];
        row = try forwardIds(&p.xfm, a, &.{st.id}, s);
    }
    var buf: [4096]u8 = undefined;
    const f = try std.Io.Dir.cwd().createFile(io, out_path, .{});
    defer f.close(io);
    var fw = f.writer(io, &buf);
    try writeReceipt(&fw.interface, dir, p.implName(), prompt.len, steps);
    try fw.interface.flush();
    std.debug.print("dupes oracle: {s} ({s}): {d} prompt ids, {d} greedy steps -> {s}\n", .{ dir, p.implName(), prompt.len, n_steps, out_path });
}

test "dupes oracle: two receipts of one prompt agree up to the first differing step (DUPES_RECEIPTS=a,b)" {
    const pair = env.span(.DUPES_RECEIPTS) orelse return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var it = std.mem.splitScalar(u8, pair, ',');
    var r: [2]Receipt = undefined;
    for (&r) |*x| x.* = try parseReceipt(a, try std.Io.Dir.cwd().readFileAlloc(testing.io, it.next() orelse return error.ReceiptPair, a, .limited(64 << 20)));
    const g = try compare(r[0], r[1]);
    std.debug.print("dupes oracle: {s} vs {s}: greedy ids {s}{d}; over the agreeing steps top-{d} overlap >= {d}, logit gap <= {d:.4}\n", .{
        r[0].impl,                                                      r[1].impl,
        if (g.first_diff == null) "identical over " else "first differ at step ", g.first_diff orelse @min(r[0].steps.len, r[1].steps.len),
        top_k,                                                          g.min_overlap,
        g.max_gap,
    });
}

test "dupes oracle: a receipt round-trips and two receipts compare to their first differing step" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const row_a = [_]f32{ 0.5, 3.0, -1.0, 2.0, 2.0, 0.0, 1.5 };
    const top = topOf(&row_a);
    try testing.expectEqual(@as(u32, 1), top[0][0]);
    try testing.expectEqual(@as(u32, 3), top[1][0]); // tie with id 4: the lower id first
    try testing.expectEqual(@as(u32, 4), top[2][0]);
    const steps_a = [_]Step{ .{ .id = 1, .top = top }, .{ .id = 1, .top = top } };
    var steps_b = steps_a;
    steps_b[1].id = 3;
    steps_b[0].top[1][1] = 2.25;
    var bufs: [2]std.Io.Writer.Allocating = .{ .init(a), .init(a) };
    try writeReceipt(&bufs[0].writer, "/x/pack", "mlx-stream", 7, &steps_a);
    try writeReceipt(&bufs[1].writer, "/y/pack", "deepseek_v41", 7, &steps_b);
    const ra = try parseReceipt(a, bufs[0].written());
    const rb = try parseReceipt(a, bufs[1].written());
    try testing.expectEqualStrings("mlx-stream", ra.impl);
    try testing.expectEqual(steps_a[0].top, ra.steps[0].top);
    const g = try compare(ra, rb);
    try testing.expectEqual(@as(?usize, 1), g.first_diff);
    try testing.expectEqual(@as(usize, top_k), g.min_overlap);
    try testing.expectApproxEqAbs(@as(f32, 0.25), g.max_gap, 1e-6);
    try testing.expectEqual(@as(u32, 7), (try parseIds(a, "7, 8\n9"))[0]);
}
