//! The streamed weight source: routed experts read from SSD into slot rows, for a trunk whose MoE reads them through
//! `Routed`. The streamer is mlx-stream's (`streamed/expert/`, `streamed/sdk_ext/expert/`, the read pool in
//! `streamed/csrc/`; module `mlx_stream`). Sources: the MXFP4 record bank (`experts.bin` + `expert-manifest.json`,
//! `streamed/bank_source.zig`; MiMo's and DeepSeek-V4.1's experts) and a standard MLX pack's affine experts as
//! downloaded (`openPack`, `streamed/mlx_pack.zig`). Without mlx-stream, or without a bank, opening refuses
//! by name.

const std = @import("std");
const build_options = @import("build_options");
const abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const plugin = if (build_options.mlx_stream) @import("mlx_stream") else struct {};

const routed = @import("routed.zig");
pub const Proj = routed.Proj;
pub const Banks = routed.Banks;
pub const Fetched = routed.Fetched;
pub const max_call_ids = routed.max_call_ids;
pub const Routed = routed.Routed;

/// The per-model `stream_experts` setting: on (a quarter of each routed layer's experts in slot rows), or the slot
/// rows per layer.
pub const Setting = union(enum) {
    on,
    rows: u32,

    /// `true` or a row count as JSON; null for anything else (`false` included).
    pub fn of(v: std.json.Value) ?Setting {
        return switch (v) {
            .bool => |b| if (b) .on else null,
            .integer => |n| if (n > 0 and n <= std.math.maxInt(u32)) .{ .rows = @intCast(n) } else null,
            else => null,
        };
    }

    /// The setting in a load's settings JSON (`stream_experts`); null when off or absent.
    pub fn parse(settings_json: []const u8) ?Setting {
        if (settings_json.len == 0) return null;
        const parsed = std.json.parseFromSlice(std.json.Value, std.heap.page_allocator, settings_json, .{}) catch return null;
        defer parsed.deinit();
        if (parsed.value != .object) return null;
        return of(parsed.value.object.get("stream_experts") orelse return null);
    }

    /// Slot rows per routed layer of `n_experts`.
    pub fn slotRows(s: Setting, n_experts: u32) u32 {
        return switch (s) {
            .on => @max(1, n_experts / 4),
            .rows => |n| @min(n, n_experts),
        };
    }
};

/// An EXL3 expert bank (`streamed/exl3_source.zig`) for a trunk with no EXL3 math: the source returns routed sums.
pub const Exl3 = if (build_options.mlx_stream) plugin.exl3_source else struct {};

/// `src`'s `Routed` (`source` math, its phases and read-ahead).
pub fn exl3Routed(src: *Exl3.Source) Routed {
    const C = struct {
        fn rows(ctx: *anyopaque, layer: u32, x: mlx.mlx_array, idx: mlx.mlx_array, wts: mlx.mlx_array) routed.Ret(mlx.mlx_array) {
            const s: *Exl3.Source = @ptrCast(@alignCast(ctx));
            return .of(s.rows(layer, x, idx, wts));
        }
        fn settle(ctx: *anyopaque) routed.Ret(void) {
            const s: *Exl3.Source = @ptrCast(@alignCast(ctx));
            return .of(s.settle());
        }
        fn phase(ctx: *anyopaque, prompt: bool) routed.Ret(void) {
            const s: *Exl3.Source = @ptrCast(@alignCast(ctx));
            return .of(s.setPhase(if (prompt) .prompt else .decode));
        }
        fn readAhead(ctx: *anyopaque, layer: u32, experts: []const u16) routed.Ret(void) {
            const s: *Exl3.Source = @ptrCast(@alignCast(ctx));
            return .of(s.readAhead(layer, experts));
        }
    };
    return .{ .ctx = src, .math = .{ .source = C.rows }, .settle_fn = C.settle, .phase_fn = C.phase, .read_ahead_fn = C.readAhead, .decode_ids = Exl3.decode_route_ids };
}

/// GLM-5.3-Flash's EXL3 bank (`streamed/glm5n_source.zig`): the source returns routed sums.
pub const Glm5n = if (build_options.mlx_stream) plugin.glm5n_source else struct {};

/// `src`'s `Routed` (`source` math, its phases, read-ahead and settle).
pub fn glm5nRouted(src: *Glm5n.Source) Routed {
    const C = struct {
        fn rows(ctx: *anyopaque, layer: u32, x: mlx.mlx_array, idx: mlx.mlx_array, wts: mlx.mlx_array) routed.Ret(mlx.mlx_array) {
            const s: *Glm5n.Source = @ptrCast(@alignCast(ctx));
            return .of(s.rows(layer, x, idx, wts));
        }
        fn settle(ctx: *anyopaque) routed.Ret(void) {
            const s: *Glm5n.Source = @ptrCast(@alignCast(ctx));
            return .of(s.settle());
        }
        fn phase(ctx: *anyopaque, prompt: bool) routed.Ret(void) {
            const s: *Glm5n.Source = @ptrCast(@alignCast(ctx));
            return .of(s.setPhase(prompt));
        }
        fn readAhead(ctx: *anyopaque, layer: u32, experts: []const u16) routed.Ret(void) {
            const s: *Glm5n.Source = @ptrCast(@alignCast(ctx));
            return .of(s.readAhead(layer, experts));
        }
    };
    return .{ .ctx = src, .math = .{ .source = C.rows }, .settle_fn = C.settle, .phase_fn = C.phase, .read_ahead_fn = C.readAhead };
}

pub const Streamed = struct {
    pub const refusal = abi.Verdict.refuse(.unsupported, "streamed weights: no expert bank, or a build without mlx-stream");

    pub const Options = struct {
        /// Slot rows per routed layer (null: every expert).
        rows: ?[]const u32 = null,
    };

    const Mxfp4 = if (build_options.mlx_stream) plugin.bank_source.Mxfp4 else void;
    const Pack = if (build_options.mlx_stream) plugin.mlx_pack.Source else void;

    src: union(enum) { mxfp4: *Mxfp4, pack: *Pack },

    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, s: mlx.mlx_stream, opt: Options) !Streamed {
        if (comptime !build_options.mlx_stream) return error.StreamedWeightsUnsupported;
        var d = std.Io.Dir.openDirAbsolute(io, dir, .{}) catch return error.StreamedWeightsUnsupported;
        defer d.close(io);
        _ = d.statFile(io, "expert-manifest.json", .{}) catch return error.StreamedWeightsUnsupported;
        return switch (try plugin.bank_source.Source.open(a, io, dir, s, .{ .rows = opt.rows })) {
            .mxfp4 => |x| .{ .src = .{ .mxfp4 = x } },
        };
    }

    /// The routed experts of the MLX pack in `dir` (`switch_mlp` stacked or one tensor per expert, affine), `rows`
    /// slot rows per layer (null: every expert), slot arrays on `s`.
    pub fn openPack(a: std.mem.Allocator, dir: []const u8, s: mlx.mlx_stream, rows: ?u32) !Streamed {
        if (comptime !build_options.mlx_stream) return error.StreamedWeightsUnsupported;
        return .{ .src = .{ .pack = try Pack.open(a, dir, s, rows) } };
    }

    /// What streaming the MLX pack in `dir` takes off its bill and puts on it: the routed experts' bytes in the pack,
    /// and the slot rows a stream with `rows` per layer (null: every expert) holds.
    pub fn packBytes(a: std.mem.Allocator, dir: []const u8, rows: ?u32) !struct { experts: u64, slots: u64 } {
        if (comptime !build_options.mlx_stream) return error.StreamedWeightsUnsupported;
        var b = try plugin.mlx_pack.Bank.open(a, dir);
        defer b.deinit();
        return .{ .experts = b.expert_bytes, .slots = b.slotBytes(@min(rows orelse b.n_experts, b.n_experts)) };
    }

    /// What a stream of the expert bank in `dir` holds: `rows` slot rows per layer (null: every expert) and the
    /// transient rows a wide call borrows.
    pub fn bankBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, rows: ?u32) !u64 {
        if (comptime !build_options.mlx_stream) return error.StreamedWeightsUnsupported;
        var b = try plugin.bank_source.Bank.open(a, io, dir);
        defer b.deinit();
        const g = b.geometry;
        const per_layer: u64 = @min(rows orelse g.num_experts, g.num_experts);
        return (per_layer * g.num_moe_layers + plugin.bank_source.transient_rows) * g.record_bytes;
    }

    pub fn close(self: *Streamed) void {
        if (comptime build_options.mlx_stream) switch (self.src) {
            inline else => |x| x.deinit(),
        };
    }

    /// An MLX pack's routed layer whose tensors sit under `prefix` (`<...>.layers.<i>.mlp`); null when it has none.
    pub fn layerOf(self: *const Streamed, prefix: []const u8) ?u32 {
        if (comptime !build_options.mlx_stream) return null;
        return switch (self.src) {
            .pack => |x| x.bank.layerOf(prefix),
            else => null,
        };
    }

    pub fn routedSource(self: *Streamed) Routed {
        if (comptime !build_options.mlx_stream) unreachable;
        const C = struct {
            fn fetch(comptime Src: type) fn (*anyopaque, u32, []const u16, []const f32) routed.Ret(Fetched) {
                return struct {
                    fn f(ctx: *anyopaque, layer: u32, ids: []const u16, scores: []const f32) routed.Ret(Fetched) {
                        return .of(g(ctx, layer, ids, scores));
                    }
                    fn g(ctx: *anyopaque, layer: u32, ids: []const u16, scores: []const f32) !Fetched {
                        const src: *Src = @ptrCast(@alignCast(ctx));
                        const got = if (comptime Src == Mxfp4) try src.fetch(layer, ids, scores) else try src.fetch(layer, ids);
                        var banks: [3]?Banks = @splat(null);
                        for (got.banks, &banks) |b, *o| if (b) |x| {
                            o.* = .{ .gate = proj(x.gate), .up = proj(x.up), .down = proj(x.down) };
                        };
                        return .{ .kind = got.kind, .row = got.row, .banks = banks };
                    }
                }.f;
            }
            fn proj(p: anytype) Proj {
                return .{ .w = p.w, .scales = p.scales, .biases = if (@hasField(@TypeOf(p), "biases")) p.biases else .{ .ctx = null } };
            }
            fn settle(comptime Src: type) fn (*anyopaque) routed.Ret(void) {
                return struct {
                    fn f(ctx: *anyopaque) routed.Ret(void) {
                        const src: *Src = @ptrCast(@alignCast(ctx));
                        return .of(src.settle());
                    }
                }.f;
            }
        };
        return switch (self.src) {
            .mxfp4 => |x| .{ .ctx = x, .math = .{ .trunk = C.fetch(Mxfp4) }, .settle_fn = C.settle(Mxfp4), .decode_ids = plugin.bank_source.max_route_ids },
            .pack => |x| .{ .ctx = x, .math = .{ .trunk = C.fetch(Pack) }, .settle_fn = C.settle(Pack), .decode_ids = plugin.mlx_pack.S.max_route_ids },
        };
    }
};

// ── tests ──

test "streamed: the stream_experts setting is true or a row count, a quarter of the experts when true" {
    try std.testing.expectEqual(@as(?Setting, .on), Setting.parse("{\"lm_head_mxfp8\":false,\"stream_experts\":true}"));
    try std.testing.expectEqual(@as(?Setting, .{ .rows = 24 }), Setting.parse("{\"stream_experts\":24}"));
    for ([_][]const u8{ "", "{}", "{\"stream_experts\":false}", "{\"stream_experts\":0}", "{\"stream_experts\":\"yes\"}", "[1]" }) |j|
        try std.testing.expectEqual(@as(?Setting, null), Setting.parse(j));
    try std.testing.expectEqual(@as(u32, 64), (Setting{ .on = {} }).slotRows(256));
    try std.testing.expectEqual(@as(u32, 1), (Setting{ .on = {} }).slotRows(2));
    try std.testing.expectEqual(@as(u32, 8), (Setting{ .rows = 300 }).slotRows(8));
}

test "streamed: a directory without a bank is refused by name" {
    try std.testing.expectError(error.StreamedWeightsUnsupported, Streamed.open(std.testing.allocator, std.testing.io, "/nowhere", .{ .ctx = null }, .{}));
    try std.testing.expectEqual(abi.Verdict.Code.unsupported, Streamed.refusal.code);
}
