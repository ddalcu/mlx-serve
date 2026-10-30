//! engines/mlx/modules/mlx_stream.
//! Owns forwardMlxStream, initMlxStream.

const std = @import("std");
const mlx_stream_mod = @import("stream.zig");
const mlx = @import("../../../mlx/ffi.zig");
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const Transformer = @import("../transformer.zig").Transformer;
const moduleShell = @import("shell.zig").moduleShell;

/// An mlx-stream pack: the plugin runs the request's prompt pass, then decode (`mlx_stream.forward`).
pub fn forwardMlxStream(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array, m: *mlx_stream_mod.Model) !mlx.mlx_array {
    const ids = try hostIds(self, token_ids);
    defer self.allocator.free(ids);
    const logits = try mlx_stream_mod.forward(m, ids, self.s);
    ctx.cache.step += ids.len;
    return logits;
}

/// A pack mlx-stream serves (DeepSeek-V4.1's EXL3 repack, a GLM-5.3 or a GLM-5.3-Flash pack): the plugin's module.
pub fn initMlxStream(io: std.Io, allocator: std.mem.Allocator, config: ModelConfig, s: mlx.mlx_stream) !Transformer {
    const m = try mlx_stream_mod.open(allocator, io, s, &config);
    errdefer mlx_stream_mod.close(m);
    var t = try moduleShell(allocator, config, s);
    t.mlx_stream_ext = m;
    return t;
}

/// A module arch's token ids on the host.
fn hostIds(self: *Transformer, token_ids: mlx.mlx_array) ![]u32 {
    var ids32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids32);
    try mlx.check(mlx.mlx_astype(&ids32, token_ids, .int32, self.s));
    try mlx.check(mlx.mlx_array_eval(ids32));
    const n = mlx.mlx_array_size(ids32);
    const data = mlx.mlx_array_data_int32(ids32) orelse return error.NoData;
    const ids = try self.allocator.alloc(u32, n);
    for (ids, data[0..n]) |*o, id| o.* = @intCast(id);
    return ids;
}
