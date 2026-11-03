//! A grouped verify's joined projections: every slot's verify rows of one layer projected as one matrix (attention
//! q/k/v and out, GDN qkv/z/a/b and out), each slot's mixer run on its own slice.
//! The grouped verify itself (`forwardRowAxisVerify`) lives here too.
//! copied_from: engines/mlx/verify/joined_rows.zig, engines/mlx/batched/row_axis_verify.zig.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const layered = @import("../../../engines/mlx/batched/layered.zig");
const ple = @import("../ple.zig");
const VerifyRollback = @import("../../../engines/mlx/batched/row_axis_verify.zig").VerifyRollback;
const ssmFreeSpecCapture = @import("../../../engines/mlx/ssm/entry.zig").ssmFreeSpecCapture;
const log = @import("../../../sdk/util/log.zig");
const verify_projection = @import("../../../sdk/blocks/linear/verify_projection.zig");
const LinearAttnWeights = @import("../../../sdk/blocks/gdn/gated_delta_net.zig").LinearAttnWeights;
const verifyAttnProjection = verify_projection.verifyAttnProjection;
const verifyGdnProjection = verify_projection.verifyGdnProjection;
const MAX_BATCH_ROWS = @import("../../../engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const FullAttnWeights = @import("../../../engines/mlx/layer_weights.zig").FullAttnWeights;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const Trunk = @import("../trunk.zig").Trunk;

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
    self: *Trunk,
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

pub fn attnVerifyRows(self: *Trunk, inputs: []const mlx.mlx_array, entries: []const *SSMCacheEntry, ctxs: []const *ForwardCtx, generations: []const u64, layer: usize, fa: *const FullAttnWeights) !?[]mlx.mlx_array {
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

pub fn gdnVerifyRows(self: *Trunk, inputs: []const mlx.mlx_array, entries: []const *SSMCacheEntry, generations: []const u64, layer: usize, la: *const LinearAttnWeights) !?[]mlx.mlx_array {
    if (la.combined_proj) return null;
    return self.verifyRowsJoined(.gdn, inputs, entries, &.{}, generations, layer, la);
}

/// One verify block per slot (`[1, S]` rows): the grouped MoE rows where they fit (`qwen4VerifyRowsEligible`), else
/// each row's own forward, every recurrent row's state captured for its rollback and the deferred PLE settled after
/// every row's build. A group of two or more snapshots each slot first (`rollback_out` takes them, else an error
/// restores them).
/// copied_from: engines/mlx/batched/row_axis_verify.zig (`forwardRowAxisVerify`, the qwen4_exp arm).
pub fn forwardRowAxisVerify(
    self: *Trunk,
    token_rows: []const mlx.mlx_array,
    ctxs: []const *ForwardCtx,
    out_logits: []mlx.mlx_array,
    out_last: []mlx.mlx_array,
    out_all: []mlx.mlx_array,
    row_ns: ?[]u64,
    rollback_out: ?[]VerifyRollback,
) !void {
    std.debug.assert(token_rows.len == ctxs.len);
    std.debug.assert(out_logits.len >= token_rows.len and out_last.len >= token_rows.len and out_all.len >= token_rows.len);
    const outer_group_rows = verify_projection.verify_group_rows;
    verify_projection.verify_group_rows = token_rows.len;
    defer verify_projection.verify_group_rows = outer_group_rows;
    if (token_rows.len > 1 and !layered.row_axis_verify_logged) {
        layered.row_axis_verify_logged = true;
        log.info("[batched] row-axis mtp verify engaged (slots={d}, width={d})\n", .{ token_rows.len, mlx.getShape(token_rows[0])[1] });
    }
    if (row_ns) |ns| @memset(ns[0..token_rows.len], 0);
    self.verify_laps = .{};
    if (self.verify_joined_logits.ctx != null) {
        _ = mlx.mlx_array_free(self.verify_joined_logits);
        self.verify_joined_logits = .{ .ctx = null };
    }
    var rollback: [MAX_BATCH_ROWS]VerifyRollback = undefined;
    var n_rollback: usize = 0;
    defer for (rollback[0..n_rollback]) |*saved| saved.deinit();
    errdefer for (rollback[0..n_rollback]) |*saved| saved.restore();
    if (token_rows.len > 1) {
        if (token_rows.len > MAX_BATCH_ROWS) return error.MtpBatchTooWide;
        for (ctxs) |ctx| try self.ssmGroupRelease(ctx);
        for (ctxs, 0..) |ctx, i| {
            var snapshot_lap = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
            rollback[i] = try VerifyRollback.capture(self.allocator, &self.fwd_gen, ctx);
            n_rollback += 1;
            if (row_ns) |ns| ns[i] = snapshot_lap.read();
        }
    }
    var lap = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    var built: usize = 0;
    var touched: usize = 0;
    errdefer {
        for (0..built) |i| {
            _ = mlx.mlx_array_free(out_logits[i]);
            _ = mlx.mlx_array_free(out_last[i]);
            _ = mlx.mlx_array_free(out_all[i]);
        }
        for (0..touched) |i| {
            ple.discardDeferredPle(self, ctxs[i]);
            if (ctxs[i].ssm_entries) |entries| for (entries) |*e| ssmFreeSpecCapture(e);
        }
    }
    if (self.qwen4VerifyRowsEligible(token_rows, ctxs)) {
        const Once = struct {
            var logged = false;
        };
        if (!Once.logged) {
            Once.logged = true;
            log.info("[batched] row-axis mtp verify grouped MoE engaged (slots={d})\n", .{token_rows.len});
        }
        for (ctxs) |ctx| {
            ctx.capture_ssm_seq = true;
            ctx.ple_defer = true;
        }
        touched = token_rows.len;
        {
            defer for (ctxs) |ctx| {
                ctx.ple_defer = false;
                ctx.capture_ssm_seq = false;
            };
            try self.forwardQwen4VerifyRows(token_rows, ctxs, out_logits, out_last, out_all);
        }
        built = token_rows.len;
        const group_ns = lap.read();
        self.verify_laps.build_ns += group_ns;
        if (row_ns) |ns| {
            for (ns[0..token_rows.len]) |*value| value.* += group_ns;
        }
        lap.reset();
    } else {
        for (token_rows, ctxs, 0..) |row, ctx, i| {
            touched = i + 1;
            var last = mlx.mlx_array_new();
            var all = mlx.mlx_array_new();
            // Per-position SSM capture lets a partial accept roll back without a re-forward; the deferred PLE leaf
            // keeps the host gather out of the graph build.
            ctx.capture_ssm_seq = ctx.ssm_entries != null;
            ctx.ple_defer = true;
            const logits = self.forwardWithCaptureAll(ctx, row, &last, &all) catch |e| {
                ctx.ple_defer = false;
                ctx.capture_ssm_seq = false;
                _ = mlx.mlx_array_free(last);
                _ = mlx.mlx_array_free(all);
                return e;
            };
            ctx.ple_defer = false;
            ctx.capture_ssm_seq = false;
            out_logits[i] = logits;
            out_last[i] = last;
            out_all[i] = all;
            built = i + 1;
            const row_build_ns = lap.read();
            self.verify_laps.build_ns += row_build_ns;
            lap.reset();
            if (row_ns) |ns| ns[i] += row_build_ns;
        }
    }
    // Fills the deferred PLE leaves and advances every row's n-gram history: after every row's build, before the
    // first eval downstream of a leaf and before any rollback.
    try ple.flushDeferredPleGroup(self, ctxs[0..token_rows.len]);
    const ple_ns = lap.read();
    self.verify_laps.ple_ns = ple_ns;
    if (row_ns) |ns| ns[0] += ple_ns;
    if (rollback_out) |out| {
        @memcpy(out[0..n_rollback], rollback[0..n_rollback]);
        n_rollback = 0;
    }
}

/// The joined verify block of the last grouped verify, handed to the caller (null when it projected per row).
pub fn takeJoinedVerifyLogits(self: *Trunk) mlx.mlx_array {
    const out = self.verify_joined_logits;
    self.verify_joined_logits = .{ .ctx = null };
    return out;
}
