//! Logit shaping and logprobs on a host-side logits row (ds4's f32 copy, the ANE's f16 rows): what an engine whose
//! sampler takes only temperature, top-k, top-p and min-p adds on its plain decode path.

const std = @import("std");
const abi = @import("../../abi/engine.zig");

/// How often each generated id came out so far: what the penalties read.
pub const History = struct {
    counts: std.AutoHashMapUnmanaged(u32, u32) = .empty,

    pub fn add(h: *History, gpa: std.mem.Allocator, id: u32) !void {
        const e = try h.counts.getOrPut(gpa, id);
        e.value_ptr.* = if (e.found_existing) e.value_ptr.* + 1 else 1;
    }

    pub fn deinit(h: *History, gpa: std.mem.Allocator) void {
        h.counts.deinit(gpa);
    }
};

/// The request reshapes the logits it samples from: a penalty or a logit bias.
pub fn shapes(s: abi.Sampling) bool {
    return s.repeat_penalty != 1 or s.frequency_penalty != 0 or s.presence_penalty != 0 or s.logit_bias.len > 0;
}

/// The MLX path's penalties (`applyPenalties`): `repeat_penalty` divides a seen id's positive logit and multiplies
/// a negative one, then `presence_penalty` is subtracted once per seen id and `frequency_penalty` once per
/// occurrence. Each `logit_bias` is added last; an id past the row is ignored.
pub fn shape(comptime T: type, row: []T, s: abi.Sampling, h: *const History) void {
    var it = h.counts.iterator();
    while (it.next()) |e| {
        const id = e.key_ptr.*;
        if (id >= row.len) continue;
        var x: f32 = row[id];
        if (s.repeat_penalty != 1) x = if (x > 0) x * (1 / s.repeat_penalty) else x * s.repeat_penalty;
        x -= s.presence_penalty + s.frequency_penalty * @as(f32, @floatFromInt(e.value_ptr.*));
        row[id] = @floatCast(x);
    }
    for (abi.items(abi.Bias, s.logit_bias)) |b| if (b.token < row.len) {
        row[b.token] = @floatCast(@as(f32, row[b.token]) + b.bias);
    };
}

/// `token`'s log-probability under `row` (the model's distribution: no temperature, no shaping) and, in `top`,
/// the `top.len` most likely ids with theirs, descending.
pub fn logprob(comptime T: type, row: []const T, token: u32, top: []abi.Bias) abi.Logprob {
    var max: f32 = -std.math.inf(f32);
    for (row) |v| max = @max(max, @as(f32, v));
    var sum: f64 = 0;
    for (row) |v| sum += @exp(@as(f64, @as(f32, v) - max));
    const lse: f32 = max + @as(f32, @floatCast(@log(sum)));
    var n: usize = 0;
    for (row, 0..) |v, i| {
        const x: f32 = v;
        if (n == top.len and (top.len == 0 or x <= top[n - 1].bias)) continue;
        var j: usize = if (n < top.len) n else n - 1;
        if (n < top.len) n += 1;
        while (j > 0 and top[j - 1].bias < x) : (j -= 1) top[j] = top[j - 1];
        top[j] = .{ .token = @intCast(i), .bias = x };
    }
    for (top[0..n]) |*t| t.bias -= lse;
    return .{ .token = token, .logprob = @as(f32, row[token]) - lse, .top = abi.span(abi.Bias, top[0..n]) };
}

// ── tests ──

const testing = std.testing;

test "shape: penalties as the MLX path applies them, then the bias" {
    const gpa = testing.allocator;
    var h: History = .{};
    defer h.deinit(gpa);
    try h.add(gpa, 1);
    try h.add(gpa, 1);
    try h.add(gpa, 2);
    var row = [_]f32{ 1, 4, -2, 3 };
    shape(f32, &row, .{ .repeat_penalty = 2, .presence_penalty = 0.5, .frequency_penalty = 0.25 }, &h);
    try testing.expectEqualSlices(f32, &.{ 1, 4.0 / 2.0 - 0.5 - 0.5, -2 * 2 - 0.5 - 0.25, 3 }, &row);
    const bias = [_]abi.Bias{ .{ .token = 3, .bias = -10 }, .{ .token = 99, .bias = 1 } };
    var row16 = [_]f16{ 1, 4, -2, 3 };
    shape(f16, &row16, .{ .logit_bias = abi.span(abi.Bias, &bias) }, &History{});
    try testing.expectEqual(@as(f16, -7), row16[3]);
    try testing.expect(shapes(.{ .frequency_penalty = 0.1 }));
    try testing.expect(!shapes(.{}));
}

test "logprob: log-softmax of the row, the top ids descending" {
    const row = [_]f32{ 0, @log(3.0), 0, @log(4.0) }; // probabilities 1/9, 3/9, 1/9, 4/9
    var top: [2]abi.Bias = undefined;
    const lp = logprob(f32, &row, 1, &top);
    try testing.expectApproxEqAbs(@log(3.0 / 9.0), lp.logprob, 1e-6);
    try testing.expectEqual(@as(usize, 2), lp.top.len);
    try testing.expectEqual(@as(u32, 3), top[0].token);
    try testing.expectApproxEqAbs(@log(4.0 / 9.0), top[0].bias, 1e-6);
    try testing.expectEqual(@as(u32, 1), top[1].token);
    const none = logprob(f16, &[_]f16{ 1, 2 }, 0, &.{});
    try testing.expectEqual(@as(usize, 0), none.top.len);
}
