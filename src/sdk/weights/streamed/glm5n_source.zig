//! GLM-5.3-Flash's routed experts from its EXL3 bank (`experts.bin` + `expert-manifest-v2.json`) for the glm5_next
//! trunk: the streamed stack of record (`Exl3Source`: MiMo's stream and read schedule, GLM-shape EXL3 texts, the
//! in-place prompt combine, the decode hit pass) at its default routes, behind `Routed`. A call returns its experts'
//! weighted sum; the trunk adds the shared expert. Slot rows per phase: the prompt rows during a prompt pass, the
//! decode rows after `phase(decode)`; the pack's usage profile seeds the prompt rows at load.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const sdk_ext = @import("sdk_ext.zig");
const experts = @import("glm5n/experts.zig");
const settings = @import("glm5n/settings.zig");
const prompt_levers = @import("glm5n/prompt_levers.zig");
const gpu_warm = @import("mimo/gpu_warm.zig");
const expert_usage = sdk_ext.expert.usage;
const exl3_bank = @import("glm5n/exl3_bank.zig");
const mimo_stream = @import("mimo/stream.zig");

const T = mlx.mlx_array;
pub const Gate = experts.Gate;

/// The bank's records: one expert of one routed layer, `record_bytes` each.
pub const Geometry = struct { record_bytes: u64, first_moe_layer: u32, num_moe_layers: u32, num_experts: u32 };

/// A call of at most this many rows runs the decode lane; a wider one runs in the prompt phase.
pub const decode_lane_rows: u32 = experts.decode_lane_rows;

/// Rows a wide prompt call borrows past the slot rows (its in-flight windows).
pub const transient_rows: u32 = mimo_stream.transient_rows;

/// The bank in `dir` read from its manifests (no slot memory allocated).
pub fn geometryOf(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Geometry {
    var diag: sdk.Diag = .{};
    var b = exl3_bank.Bank.open(a, io, dir, exl3_bank.flash_first_moe_layer, exl3_bank.flash, &diag) catch |e| {
        if (diag.len > 0) std.log.err("glm5_next bank: {s}", .{diag.message()});
        return e;
    };
    defer b.deinit();
    const g = b.geometry;
    return .{ .record_bytes = g.record_bytes, .first_moe_layer = g.first_moe_layer, .num_moe_layers = g.num_moe_layers, .num_experts = g.num_experts };
}

pub const Source = struct {
    a: std.mem.Allocator,
    inner: *experts.Exl3Source,

    /// The bank in `dir`, streamed on `s` at `rows` slot rows per routed layer (the calibrated per-layer split of the
    /// decode rows). `gates`: every model layer's router as stored (bf16 `[experts, hidden]`) and its correction bias,
    /// the dense layers' empty (borrowed for the source's life).
    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, s: mlx.mlx_stream, gates: []const Gate, n_moe: u32, n_experts: u32, slot_rows: sdk.Rows) !*Source {
        const prompt_rows = try a.alloc(u32, n_moe);
        defer a.free(prompt_rows);
        const decode_rows = try a.alloc(u32, n_moe);
        defer a.free(decode_rows);
        try settings.layerRows(settings.defaultLayerRows(), slot_rows, n_experts, prompt_rows, decode_rows);
        // The stack of record's routes: the decode hit pass, the BF16 predictor, the keep-warm waits on a GPU stream, no
        // rate tally, the EXL3 forms, the prompt levers, direct demand reads, the prompt residents kept.
        const warm: gpu_warm.Route = if (gpu_warm.gpuStream(s)) .waits else .off;
        const inner = try experts.Exl3Source.init(a, io, dir, prompt_rows, decode_rows, gates, s, .hitpass, .bf16, warm, .{ .ratio = settings.default_tally_ratio }, .{}, prompt_levers.Settings{}, .direct, .keep);
        errdefer inner.deinit();
        try inner.ex.bindMergeGateWaits();
        _ = try inner.prebuildExperts();
        var d = try expert_usage.decide(a, io, &expert_usage.Settings{}, dir, "glm5_next", n_moe, n_experts);
        defer if (d.profile) |*p| p.deinit();
        if (d.profile) |*p| _ = try inner.seedResidents(p);
        const self = try a.create(Source);
        self.* = .{ .a = a, .inner = inner };
        return self;
    }

    pub fn deinit(self: *Source) void {
        self.inner.deinit();
        self.a.destroy(self);
    }

    /// Model layer `layer`'s routed sum `[n, hidden]` for the normed MoE input `x` `[n, hidden]`, the routes `idx`
    /// `[n, k]` and their weights `wts`.
    pub fn rows(self: *Source, layer: u32, x: T, idx: T, wts: T) !T {
        var g = &self.inner.g;
        const xb = if (g.dtypeOf(x) == .bfloat16) x else try g.astype(x, .bfloat16);
        const w = if (g.dtypeOf(wts) == .float32) wts else try g.astype(wts, .float32);
        return self.inner.forward(layer, xb, idx, w, &.{});
    }

    /// A prompt call's predicted routes of `layer` (every row's ids): its experts, hottest first, read while its
    /// attention runs.
    pub fn readAhead(self: *Source, layer: u32, ids: []const u16) !void {
        if (self.inner.decoding()) return;
        var counts: [512]u32 = @splat(0);
        const n = self.inner.bank.geometry.num_experts;
        for (ids) |e| counts[e] += 1;
        var ranked: [512]u16 = undefined;
        try self.inner.readAheadSeed(layer, sdk_ext.expert.policy.rankHottest(counts[0..n], ranked[0..n]));
    }

    /// A prompt pass starts the prompt phase (a new request's after a decode); a decode call grows the decode rows.
    pub fn setPhase(self: *Source, prompt: bool) !void {
        if (prompt) {
            if (self.inner.decoding()) try self.inner.beginPrefill();
        } else if (!self.inner.decoding()) try self.inner.beginDecode();
    }

    /// After the eval that consumed the forward's calls: the released routes settled.
    pub fn settle(self: *Source) !void {
        try self.inner.flush();
    }
};
