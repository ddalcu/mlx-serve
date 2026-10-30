//! The streaming packs' glue in the module archs' unit (`src/modules_unit.zig`): `stream_glue.zig` behind the table the
//! engine calls (`stream_table.zig`), exported as `mlx_serve_stream_api`.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;
const glue = @import("stream_glue.zig");
const local = @import("../../../sdk/util/io_util.zig").local;
const host = @import("stream_table.zig");

const table: host.Table = .{
    .set_foreign_bytes = glue.setForeignBytes,
    .load_bytes = loadBytes,
    .context_length = glue.contextLength,
    .note_max_output = glue.noteMaxOutput,
    .open = open,
    .close = close,
    .begin = begin,
    .end = end,
    .forward = forward,
    .position = position,
    .block_size = blockSize,
    .lane_name = laneName,
    .arm = arm,
    .round = round,
    .lane_receipt = laneReceipt,
};

fn addr() callconv(.c) *const anyopaque {
    return &table;
}

comptime {
    @export(&addr, .{ .name = "mlx_serve_stream_api" });
}

fn of(m: *host.Model) *glue.Model {
    return @ptrCast(@alignCast(m));
}

fn ofConst(m: *const host.Model) *const glue.Model {
    return @ptrCast(@alignCast(m));
}

fn loadBytes(gpa: std.mem.Allocator, io: std.Io, config: *const ModelConfig, out: *u64) ?[]const u8 {
    out.* = glue.loadBytes(gpa, local(io), config) catch |e| return @errorName(e);
    return null;
}

fn open(gpa: std.mem.Allocator, io: std.Io, s: mlx.mlx_stream, config: *const ModelConfig, out: *?*host.Model) ?[]const u8 {
    out.* = @ptrCast(glue.open(gpa, local(io), s, config) catch |e| return @errorName(e));
    return null;
}

fn close(m: *host.Model) void {
    glue.close(of(m));
}

fn begin(m: *host.Model, prompt: []const u32, max_tokens: u32, context: u64, out: *u64) ?[]const u8 {
    out.* = glue.begin(of(m), prompt, max_tokens, context) catch |e| return @errorName(e);
    return null;
}

fn end(m: *host.Model) void {
    glue.end(of(m));
}

fn forward(m: *host.Model, ids: []const u32, s: mlx.mlx_stream, out: *mlx.mlx_array) ?[]const u8 {
    out.* = glue.forward(of(m), ids, s) catch |e| return @errorName(e);
    return null;
}

fn position(m: *const host.Model) u64 {
    return glue.position(ofConst(m));
}

fn blockSize(m: *const host.Model) u32 {
    return glue.blockSize(ofConst(m));
}

fn laneName(m: *const host.Model) []const u8 {
    return glue.laneName(ofConst(m));
}

fn arm(m: *host.Model, sampling: ?host.SamplingParams) bool {
    const sp = sampling orelse return glue.arm(of(m), null);
    return glue.arm(of(m), .{ .temperature = sp.temperature, .top_p = sp.top_p, .top_k = sp.top_k, .min_p = sp.min_p, .seed = sp.seed });
}

fn round(m: *host.Model, gpa: std.mem.Allocator, t1: u32, accepted_cap: u32, out: *host.Model.DraftRound) ?[]const u8 {
    const r = glue.round(of(m), gpa, t1, accepted_cap) catch |e| return @errorName(e);
    out.* = .{ .tokens = r.tokens, .accepted = r.accepted, .next_token = r.next_token };
    return null;
}

fn laneReceipt(m: *host.Model, buf: []u8) ?[]const u8 {
    return glue.laneReceipt(of(m), buf);
}
