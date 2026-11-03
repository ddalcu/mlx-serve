//! Draft-tree verify inputs (GDN tree table and ancestor paths).
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const gdn_decode = @import("../../../sdk/blocks/gdn/decode.zig");
const MtpBatchedGraph = @import("mtp/batched_graph.zig").MtpBatchedGraph;
const SamplingParams = @import("../sampling/params.zig").SamplingParams;

/// The verify forward's inputs for a draft tree of `parents` (row 0 the root,
/// -1) at `depth`: the GDN tree table and each row's ancestor path.
pub fn specTreeFor(parents: []const i32, depth: []const i32, max_depth: i32) transformer_mod.SpecTree {
    const MAX_W = 16;
    const w = parents.len;
    const maxd: usize = @intCast(max_depth + 1);
    var path: [MAX_W * MAX_W]i32 = @splat(0);
    for (0..w) |r| {
        var cur: i32 = @intCast(r);
        var d = depth[r];
        while (cur >= 0) : (d -= 1) {
            path[r * maxd + @as(usize, @intCast(d))] = cur;
            cur = parents[@intCast(cur)];
        }
    }
    const wc: c_int = @intCast(w);
    var table: [6 * MAX_W]i32 = undefined;
    gdn_decode.treeTable(parents, table[0 .. 6 * w]);
    return .{
        .parents = mlx.mlx_array_new_data(&table, &[_]c_int{6 * wc}, 1, .int32),
        .attn = .{
            .depth = mlx.mlx_array_new_data(depth.ptr, &[_]c_int{wc}, 1, .int32),
            .path = mlx.mlx_array_new_data(&path, &[_]c_int{ wc, @intCast(maxd) }, 2, .int32),
            .max_depth = max_depth,
        },
    };
}

pub const MtpGraphFn = *const fn (mlx.mlx_array, []const mlx.mlx_array, ?[]const mlx.mlx_array, u32, f32, SamplingParams, mlx.mlx_stream) anyerror!MtpBatchedGraph;
