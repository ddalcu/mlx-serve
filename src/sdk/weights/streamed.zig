//! The streamed weight source: routed experts read from SSD into slot rows, for a trunk whose MoE reads them through
//! `Routed`. The stream is `streamed/expert.zig` (slot pools, the residency policy, the read pool in `streamed/csrc/`);
//! the in-tree source is a standard MLX pack's affine experts as downloaded (`openPack`, `streamed/mlx_pack.zig`). A
//! plugin's own bank format streams over the same stream (the kit ships it). macOS only: elsewhere opening refuses by
//! name.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const plugin_abi = @import("../../abi/plugin.zig");

/// The stream reads into Metal buffers and waits on MTLSharedEvents: built on macOS only.
pub const built = @import("builtin").os.tag == .macos;
/// The expert stream every source builds on: its contract, policy, read pool and slot rows.
pub const expert = @import("streamed/expert.zig");
const mlx_pack = @import("streamed/mlx_pack.zig");

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

/// Whether pack tensor `key` is a routed expert's (a part `openPack` streams).
pub fn expertTensor(key: []const u8) bool {
    return mlx_pack.routedOf(key) != null;
}

pub const Streamed = struct {
    const Pack = if (built) mlx_pack.Source else void;

    src: *Pack,

    /// The routed experts of the MLX pack in `dir` (`switch_mlp` stacked or one tensor per expert, affine), `rows`
    /// slot rows per layer (null: every expert), slot arrays on GPU stream `s` (a CPU stream would read slot rows
    /// before their reads land).
    pub fn openPack(a: std.mem.Allocator, dir: []const u8, s: mlx.mlx_stream, rows: ?u32) !Streamed {
        if (comptime !built) return error.StreamedWeightsUnsupported;
        if (!mlx.streamIsGpu(s)) return error.StreamedNeedsGpuStream;
        return .{ .src = try Pack.open(a, dir, s, rows) };
    }

    /// The load's pack with its routed experts (`n_experts` a layer) streamed when its settings ask for it
    /// (`stream_experts`); null = they stay in memory. Each layer binds through `moe.bindStreamed`.
    pub fn forLoad(a: std.mem.Allocator, ctx: *const plugin_abi.LoadCtx, n_experts: u32, s: mlx.mlx_stream) !?Streamed {
        const st = Setting.parse(ctx.settings_json.slice()) orelse return null;
        return try openPack(a, ctx.model_dir.slice(), s, st.slotRows(n_experts));
    }

    /// What streaming the MLX pack in `dir` takes off its bill and puts on it: the routed experts' bytes in the pack,
    /// and the slot rows a stream with `rows` per layer (null: every expert) holds.
    /// The pack's routed expert bytes and what `s`'s slot rows hold, at the bank's own expert count (a plugin's
    /// adapted config may name none).
    pub fn packBytes(a: std.mem.Allocator, dir: []const u8, s: Setting) !struct { experts: u64, slots: u64 } {
        if (comptime !built) return error.StreamedWeightsUnsupported;
        var b = try mlx_pack.Bank.open(a, dir);
        defer b.deinit();
        return .{ .experts = b.expert_bytes, .slots = b.slotBytes(s.slotRows(b.n_experts)) };
    }

    pub fn close(self: *Streamed) void {
        if (comptime built) self.src.deinit();
    }

    /// The pack's routed layer whose tensors sit under `prefix` (`<...>.layers.<i>.mlp`); null when it has none.
    pub fn layerOf(self: *const Streamed, prefix: []const u8) ?u32 {
        if (comptime !built) return null;
        return self.src.bank.layerOf(prefix);
    }

    pub fn routedSource(self: *Streamed) Routed {
        if (comptime !built) unreachable;
        const C = struct {
            fn fetch(ctx: *anyopaque, layer: u32, ids: []const u16, _: []const f32) routed.Ret(Fetched) {
                return .of(fetchOf(ctx, layer, ids));
            }
            fn fetchOf(ctx: *anyopaque, layer: u32, ids: []const u16) !Fetched {
                const src: *Pack = @ptrCast(@alignCast(ctx));
                const got = try src.fetch(layer, ids);
                var banks: [3]?Banks = @splat(null);
                for (got.banks, &banks) |b, *o| if (b) |x| {
                    o.* = .{ .gate = proj(x.gate), .up = proj(x.up), .down = proj(x.down) };
                };
                return .{ .kind = got.kind, .row = got.row, .banks = banks };
            }
            fn proj(p: anytype) Proj {
                return .{ .w = p.w, .scales = p.scales, .biases = p.biases };
            }
            fn settle(ctx: *anyopaque) routed.Ret(void) {
                const src: *Pack = @ptrCast(@alignCast(ctx));
                return .of(src.settle());
            }
        };
        return .{ .ctx = self.src, .math = .{ .trunk = C.fetch }, .settle_fn = C.settle, .decode_ids = mlx_pack.S.max_route_ids };
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
