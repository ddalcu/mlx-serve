//! The `SpeechHooks` of LFM3-Audio and Omni: the depth decoder's frames (`depth.zig`), their input rows, and the
//! codec's streaming decode (`codec.zig`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const depth = @import("depth.zig");
const codec = @import("codec.zig");
const plugin = @import("plugin.zig");

const gpa = std.heap.c_allocator;

pub const hooks: abi.SpeechHooks = .{
    .frame = &frame,
    .frame_rows = &frameRows,
    .codec_open = &codecOpen,
    .codec_push = &codecPush,
    .codec_close = &codecClose,
};

fn loaded(m: *anyopaque) *plugin.Loaded {
    return @ptrCast(@alignCast(m));
}

fn failed(e: anyerror) abi.Verdict {
    return abi.Verdict.named(.failed, @errorName(e));
}

fn frame(m: *anyopaque, hidden: abi.Array, how: *const abi.SpeechSampling, history: abi.Span(u32), out: [*]u32) callconv(.c) abi.Verdict {
    const l = loaded(m);
    if (l.depth == null) return abi.Verdict.refuse(.unsupported, "this pack ships no depth decoder");
    const d = &l.depth.?;
    var order_buf: [depth.max_codebooks]u32 = undefined;
    const order = if (how.order.len == d.codebooks) how.order.slice() else depth.randomOrder(order_buf[0..d.codebooks], how.key);
    const codes = out[0..d.codebooks];
    d.frame(.{ .ctx = hidden.ctx }, .{
        .temperature = how.temperature,
        .top_k = how.top_k,
        .repetition_penalty = how.repetition_penalty,
        .order = order,
        .key = how.key,
    }, history.slice(), codes) catch |e| return failed(e);
    _ = depth.normalizeEos(codes, d.eos);
    return .{};
}

fn frameRows(m: *anyopaque, codes: abi.Span(u32), out: *abi.Array) callconv(.c) abi.Verdict {
    const l = loaded(m);
    if (l.depth == null) return abi.Verdict.refuse(.unsupported, "this pack ships no depth decoder");
    const d = &l.depth.?;
    const rows = d.frameRows(codes.slice()) catch |e| return failed(e);
    out.* = .{ .ctx = rows.ctx };
    return .{};
}

fn codecOpen(m: *anyopaque, out: *abi.Handle) callconv(.c) abi.Verdict {
    const l = loaded(m);
    if (l.codec == null) return abi.Verdict.refuse(.unsupported, "this pack ships no codec");
    const c = &l.codec.?;
    const st = gpa.create(codec.Stream) catch |e| return failed(e);
    st.* = codec.Stream.init(gpa, c) catch |e| {
        gpa.destroy(st);
        return failed(e);
    };
    out.* = st;
    return .{};
}

fn codecPush(m: *anyopaque, h: abi.Handle, codes: abi.Span(u32), finish: bool, out: [*]f32, cap: usize, len: *usize) callconv(.c) abi.Verdict {
    _ = m;
    const st: *codec.Stream = @ptrCast(@alignCast(h orelse return abi.Verdict.refuse(.failed, "no codec stream")));
    var samples: std.ArrayList(f32) = .empty;
    defer samples.deinit(gpa);
    st.push(codes.slice(), &samples) catch |e| return failed(e);
    if (finish) st.finish(&samples) catch |e| return failed(e);
    if (samples.items.len > cap) return abi.Verdict.refuse(.failed, "codec output past the caller's buffer");
    @memcpy(out[0..samples.items.len], samples.items);
    len.* = samples.items.len;
    return .{};
}

fn codecClose(m: *anyopaque, h: abi.Handle) callconv(.c) void {
    _ = m;
    const st: *codec.Stream = @ptrCast(@alignCast(h orelse return));
    st.deinit();
    gpa.destroy(st);
}
