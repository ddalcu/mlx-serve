//! The streaming packs' glue (`stream_glue.zig`) seen from the engine under `-Dunits`, where it compiles in the module
//! archs' unit (`src/modules_unit.zig`, exporting `mlx_serve_stream_api`): the stub's surface, called through one
//! table. Errors cross by name (`sdk/util/err_name.zig`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const err_name = @import("../../../sdk/util/err_name.zig");
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;

pub const built = true;

pub const Model = opaque {
    pub const DraftRound = struct {
        tokens: []u32,
        accepted: u32,
        next_token: u32,
    };
};

pub const SamplingParams = struct { temperature: f32 = 1.0, top_p: f32 = 1.0, top_k: u32 = 0, min_p: ?f32 = null, seed: u64 = 0 };

const Err = ?[]const u8;

pub const Table = struct {
    set_foreign_bytes: *const fn (u64) void,
    load_bytes: *const fn (std.mem.Allocator, std.Io, *const ModelConfig, *u64) Err,
    context_length: *const fn (*const ModelConfig) u32,
    note_max_output: *const fn ([]const u8, u64) void,
    open: *const fn (std.mem.Allocator, std.Io, mlx.mlx_stream, *const ModelConfig, *?*Model) Err,
    close: *const fn (*Model) void,
    begin: *const fn (*Model, []const u32, u32, u64, *u64) Err,
    end: *const fn (*Model) void,
    forward: *const fn (*Model, []const u32, mlx.mlx_stream, *mlx.mlx_array) Err,
    position: *const fn (*const Model) u64,
    block_size: *const fn (*const Model) u32,
    lane_name: *const fn (*const Model) []const u8,
    arm: *const fn (*Model, ?SamplingParams) bool,
    round: *const fn (*Model, std.mem.Allocator, u32, u32, *Model.DraftRound) Err,
    lane_receipt: *const fn (*Model, []u8) ?[]const u8,
};

extern fn mlx_serve_stream_api() callconv(.c) *const anyopaque;

fn api() *const Table {
    return @ptrCast(@alignCast(mlx_serve_stream_api()));
}

pub fn setForeignBytes(n: u64) void {
    api().set_foreign_bytes(n);
}

pub fn loadBytes(gpa: std.mem.Allocator, io: std.Io, config: *const ModelConfig) !u64 {
    var out: u64 = 0;
    try err_name.check(api().load_bytes(gpa, io, config, &out));
    return out;
}

pub fn contextLength(config: *const ModelConfig) u32 {
    return api().context_length(config);
}

pub fn noteMaxOutput(dir: []const u8, value: u64) void {
    api().note_max_output(dir, value);
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, s: mlx.mlx_stream, config: *const ModelConfig) !*Model {
    var out: ?*Model = null;
    try err_name.check(api().open(gpa, io, s, config, &out));
    return out.?;
}

pub fn close(m: *Model) void {
    api().close(m);
}

pub fn begin(m: *Model, prompt: []const u32, max_tokens: u32, context: u64) !u64 {
    var out: u64 = 0;
    try err_name.check(api().begin(m, prompt, max_tokens, context, &out));
    return out;
}

pub fn end(m: *Model) void {
    api().end(m);
}

pub fn forward(m: *Model, ids: []const u32, s: mlx.mlx_stream) !mlx.mlx_array {
    var out: mlx.mlx_array = .{ .ctx = null };
    try err_name.check(api().forward(m, ids, s, &out));
    return out;
}

pub fn position(m: *const Model) u64 {
    return api().position(m);
}

pub fn blockSize(m: *const Model) u32 {
    return api().block_size(m);
}

pub fn laneName(m: *const Model) []const u8 {
    return api().lane_name(m);
}

pub fn arm(m: *Model, sampling: ?SamplingParams) bool {
    return api().arm(m, sampling);
}

pub fn round(m: *Model, gpa: std.mem.Allocator, t1: u32, accepted_cap: u32) !Model.DraftRound {
    var out: Model.DraftRound = undefined;
    try err_name.check(api().round(m, gpa, t1, accepted_cap, &out));
    return out;
}

pub fn laneReceipt(m: *Model, buf: []u8) ?[]const u8 {
    return api().lane_receipt(m, buf);
}
