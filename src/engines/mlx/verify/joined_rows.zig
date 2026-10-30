//! engines/mlx/verify/joined_rows.
//! Owns AttnVerifyInputs, VerifyRowsKind, verifyRowsJoined, attnVerifyRows, GdnVerifyInputs, gdnVerifyRows.

const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const verify_projection = @import("../../../sdk/blocks/linear/verify_projection.zig");
const LinearAttnWeights = @import("../../../sdk/blocks/gdn/gated_delta_net.zig").LinearAttnWeights;
const verifyAttnProjection = @import("../../../sdk/blocks/linear/verify_projection.zig").verifyAttnProjection;
const verifyGdnProjection = @import("../../../sdk/blocks/linear/verify_projection.zig").verifyGdnProjection;
const MAX_BATCH_ROWS = @import("../batched/layered.zig").MAX_BATCH_ROWS;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const FullAttnWeights = @import("../layer_weights.zig").FullAttnWeights;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const Transformer = @import("../transformer.zig").Transformer;
const deinit = @import("../transformer.zig").Transformer.deinit;

pub const AttnVerifyInputs = struct {
    q: mlx.mlx_array = .{},
    k: mlx.mlx_array = .{},
    v: mlx.mlx_array = .{},

    fn deinit(self: *AttnVerifyInputs) void {
        inline for (.{ "q", "k", "v" }) |name| {
            if (@field(self, name).ctx != null) _ = mlx.mlx_array_free(@field(self, name));
        }
    }

    fn slice(self: *const AttnVerifyInputs, s: mlx.mlx_stream, start: c_int, width: c_int) !AttnVerifyInputs {
        var result: AttnVerifyInputs = .{};
        errdefer result.deinit();
        inline for (.{ "q", "k", "v" }) |name| {
            const value = @field(self, name);
            try mlx.check(mlx.mlx_slice(&@field(result, name), value, &.{ 0, start, 0 }, 3, &.{ 1, start + width, mlx.getShape(value)[2] }, 3, &.{ 1, 1, 1 }, 3, s));
        }
        return result;
    }
};

pub const VerifyRowsKind = enum { attention, gdn };

pub fn verifyRowsJoined(
    self: *Transformer,
    comptime kind: VerifyRowsKind,
    inputs: []const mlx.mlx_array,
    entries: []const *SSMCacheEntry,
    ctxs: []const *ForwardCtx,
    generations: []const u64,
    layer: usize,
    w: switch (kind) {
        .attention => *const FullAttnWeights,
        .gdn => *const LinearAttnWeights,
    },
) !?[]mlx.mlx_array {
    if (inputs.len < 2 or inputs.len > 8) return null;
    var widths: [MAX_BATCH_ROWS]c_int = undefined;
    var total: c_int = 0;
    for (inputs, 0..) |input, i| {
        const shape = mlx.getShape(input);
        if (shape.len != 3 or shape[0] != 1 or shape[2] != 2560 or shape[1] < 2 or shape[1] > 6) return null;
        widths[i] = shape[1];
        total += shape[1];
    }
    if (total > 12) return null;
    const vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var joined = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(joined);
    try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 1, self.s));
    var projections: switch (kind) {
        .attention => AttnVerifyInputs,
        .gdn => GdnVerifyInputs,
    } = .{};
    defer projections.deinit();
    inline for (switch (kind) {
        .attention => .{ "q", "k", "v" },
        .gdn => .{ "qkv", "z", "a", "b" },
    }) |name| {
        const part = switch (kind) {
            .attention => try verifyAttnProjection(self.s, joined, @field(w, name ++ "_w"), @field(w, name ++ "_s"), @field(w, name ++ "_b"), widths[0..inputs.len]),
            .gdn => try verifyGdnProjection(self.s, joined, @field(w, name ++ "_w"), @field(w, name ++ "_s"), @field(w, name ++ "_b"), widths[0..inputs.len]),
        };
        @field(projections, name) = part orelse return null;
    }
    var flat: [MAX_BATCH_ROWS]mlx.mlx_array = @splat(.{});
    defer for (flat[0..inputs.len]) |value| {
        if (value.ctx != null) _ = mlx.mlx_array_free(value);
    };
    var offset: c_int = 0;
    for (inputs, entries, generations, 0..) |input, entry, generation, i| {
        flat[i] = switch (kind) {
            .attention => blk: {
                var values = try projections.slice(self.s, offset, widths[i]);
                defer values.deinit();
                self.fwd_gen = generation;
                const ctx = ctxs[i];
                break :blk try self.qwen4AttnProjected(ctx, input, w, entry, @intCast(layer), @intCast(ctx.moe_seq_offset.*), 0, 1, widths[i], true, &values, true);
            },
            .gdn => blk: {
                const values = try projections.slice(self.s, offset, widths[i]);
                self.fwd_gen = generation;
                break :blk try self.gatedDeltaNetProjected(input, w, entry, layer, 1, widths[i], true, values, true);
            },
        };
        offset += widths[i];
    }
    const out_w, const out_s, const out_b = switch (kind) {
        .attention => .{ w.o_w, w.o_s, w.o_b },
        .gdn => .{ w.out_w, w.out_s, w.out_b },
    };
    const flat_vec = mlx.mlx_vector_array_new_data(&flat, inputs.len);
    defer _ = mlx.mlx_vector_array_free(flat_vec);
    var joined_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(joined_flat);
    try mlx.check(mlx.mlx_concatenate_axis(&joined_flat, flat_vec, 1, self.s));
    const projected = switch (kind) {
        .attention => try verifyAttnProjection(self.s, joined_flat, out_w, out_s, out_b, widths[0..inputs.len]),
        .gdn => try verifyGdnProjection(self.s, joined_flat, out_w, out_s, out_b, widths[0..inputs.len]),
    };
    defer if (projected) |value| {
        _ = mlx.mlx_array_free(value);
    };
    const result = try self.allocator.alloc(mlx.mlx_array, inputs.len);
    var built: usize = 0;
    errdefer {
        for (result[0..built]) |value| _ = mlx.mlx_array_free(value);
        self.allocator.free(result);
    }
    offset = 0;
    for (inputs, 0..) |_, i| {
        if (projected) |value| {
            result[i] = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(result[i]);
            try mlx.check(mlx.mlx_slice(&result[i], value, &.{ 0, offset, 0 }, 3, &.{ 1, offset + widths[i], mlx.getShape(value)[2] }, 3, &.{ 1, 1, 1 }, 3, self.s));
        } else result[i] = switch (kind) {
            .attention => try self.attnProj(flat[i], out_w, out_s, out_b, false, @intCast(layer)),
            .gdn => try self.qmatmul(flat[i], out_w, out_s, out_b),
        };
        built += 1;
        offset += widths[i];
    }
    switch (kind) {
        .attention => {
            verify_projection.mtp_verify_attn_rows_calls +%= 1;
            if (projected != null) verify_projection.mtp_verify_attn_output_calls +%= 1;
        },
        .gdn => {
            verify_projection.mtp_verify_gdn_rows_calls +%= 1;
            if (projected != null) verify_projection.mtp_verify_gdn_output_calls +%= 1;
        },
    }
    const Once = struct {
        const tag = kind;
        var logged = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[batched] " ++ switch (kind) {
            .attention => "attention",
            .gdn => "GDN",
        } ++ " verify projections engaged (slots={d})\n", .{inputs.len});
    }
    return result;
}

pub fn attnVerifyRows(self: *Transformer, inputs: []const mlx.mlx_array, entries: []const *SSMCacheEntry, ctxs: []const *ForwardCtx, generations: []const u64, layer: usize, fa: *const FullAttnWeights) !?[]mlx.mlx_array {
    return self.verifyRowsJoined(.attention, inputs, entries, ctxs, generations, layer, fa);
}

pub const GdnVerifyInputs = struct {
    qkv: mlx.mlx_array = .{},
    z: mlx.mlx_array = .{},
    a: mlx.mlx_array = .{},
    b: mlx.mlx_array = .{},

    fn deinit(self: *GdnVerifyInputs) void {
        inline for (.{ "qkv", "z", "a", "b" }) |name| {
            if (@field(self, name).ctx != null) _ = mlx.mlx_array_free(@field(self, name));
        }
    }

    fn slice(self: *const GdnVerifyInputs, s: mlx.mlx_stream, start: c_int, width: c_int) !GdnVerifyInputs {
        var result: GdnVerifyInputs = .{};
        errdefer result.deinit();
        inline for (.{ "qkv", "z", "a", "b" }) |name| {
            const value = @field(self, name);
            try mlx.check(mlx.mlx_slice(&@field(result, name), value, &.{ 0, start, 0 }, 3, &.{ 1, start + width, mlx.getShape(value)[2] }, 3, &.{ 1, 1, 1 }, 3, s));
        }
        return result;
    }
};

pub fn gdnVerifyRows(self: *Transformer, inputs: []const mlx.mlx_array, entries: []const *SSMCacheEntry, generations: []const u64, layer: usize, la: *const LinearAttnWeights) !?[]mlx.mlx_array {
    if (la.combined_proj or self.ane_prefill != null) return null;
    return self.verifyRowsJoined(.gdn, inputs, entries, &.{}, generations, layer, la);
}
