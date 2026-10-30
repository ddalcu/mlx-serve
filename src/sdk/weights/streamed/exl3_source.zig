//! A resident trunk's routed experts from an EXL3 expert bank (`experts.bin` + `expert-manifest-v2.json`): the EXL3
//! lane (`sdk/quant/exl3/stream/`) over the expert stream, the source returning each call's routed sum. The served
//! schedule: event-gated waves, the next layer's reads predicted from its router, the wide prompt calls' feed (seed,
//! hottest first, 5 groups ahead, the base call at the seed), and the decode forms, banked waves, hoist and device
//! route. Slot rows per phase: the prompt rows during a prompt pass, the decode rows after `phase(.decode)`.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("ops.zig");
const xp = @import("experts.zig");
const expert_bank = @import("expert/bank.zig");
const expert_stream = @import("expert/stream.zig");
const sdk_ext = @import("sdk_ext.zig");
const expert_event = sdk_ext.expert.event;
const expert_admission = @import("expert/admission.zig");
const xk = @import("../../quant/exl3/stream/kernels.zig");
const xq = @import("../../quant/exl3/stream/quant.zig");
const kernel_set = sdk_ext.kernels.KernelSet(xk);
const G = ops.MlxOps;
const T = mlx.mlx_array;

const Math = xp.QuantMath(G, xq.Accepted(G));
const Hook = xp.ExpertsWith(G, xp.StreamSource, Math, .{ .prefill = true, .lookahead = true, .gated = true });
pub const Gate = Hook.Gate;
pub const Phase = enum { prompt, decode };

/// The trunk's MoE geometry the bank must match.
pub const Geometry = struct { hidden_size: u32, moe_intermediate_size: u32, n_experts_per_tok: u32, n_routed_experts: u32, n_layers: u32, swiglu_limit: f64 };

/// Bytes the slot rows and the source's own buffers may hold in each phase.
pub const Budget = struct { prompt: u64, decode: u64 };

/// What the source holds, by term, at its rows.
pub const Bill = struct {
    /// One slot row in every layer (each layer's record).
    row_bytes: u64,
    /// The prompt's transient windows (`wide_depth` routes of `max_route_ids` rows, per record geometry).
    prompt_windows: u64,
    /// Decode's: window 0 (the phase change releases the rest).
    decode_windows: u64,
    /// The lookahead's staging rows.
    staging: u64,
    rows: struct { prompt: u32, decode: u32 },

    pub fn prompt(b: Bill) u64 {
        return b.row_bytes * b.rows.prompt + b.prompt_windows + b.staging;
    }

    pub fn decode(b: Bill) u64 {
        return b.row_bytes * b.rows.decode + b.decode_windows + b.staging;
    }
};

const wide_depth: u8 = 5;
const lookahead: expert_stream.Lookahead = .{ .k = 8, .tau = std.math.inf(f32), .budget = 2 };
const event_watchdog_ms = 2000;
const wide: xp.Wide = .{ .seed = true, .hot_first = true, .depth = wide_depth, .defer_base = true, .read_ahead = true, .base_at_seed = true, .seed_aligned = true };
const forms: xq.Forms = .{ .down_pair = true, .gu_one = true };

/// The bank geometry the EXL3 lane decodes for `geo`: mul1, K = 3 or 2 per layer.
fn implementedOf(geo: Geometry) expert_bank.Implemented {
    return .{ .codebooks = &.{"mul1"}, .k = &.{ 2, 3 }, .hidden = geo.hidden_size, .inter = geo.moe_intermediate_size, .n_experts = geo.n_routed_experts, .n_layers = geo.n_layers };
}

/// The rows `budget` holds over the bank in `dir`, by term (no slot memory allocated).
pub fn billOf(a: std.mem.Allocator, io: std.Io, dir: []const u8, geo: Geometry, budget: Budget) !Bill {
    var bank = try expert_bank.Bank.open(a, io, dir, implementedOf(geo), null);
    defer bank.deinit();
    return billBank(&bank, budget);
}

fn billBank(bank: *const expert_bank.Bank, budget: Budget) !Bill {
    var row: u64 = 0;
    var record: u64 = 0;
    var alt: u64 = 0;
    for (bank.layers) |l| {
        row += l.logical_bytes;
        record = @max(record, l.logical_bytes);
    }
    for (bank.layers) |l| if (l.logical_bytes != record) {
        alt = @max(alt, l.logical_bytes);
    };
    const window = sdk_ext.expert.policy.max_route_ids * (record + alt);
    const release = expert_stream.phase_change_releases_wide_windows and expert_stream.transient_release_default;
    var b: Bill = .{
        .row_bytes = row,
        .prompt_windows = wide_depth * window,
        .decode_windows = if (release) window else wide_depth * window,
        .staging = expert_admission.lookaheadCharge(record, 2 * lookahead.budget, std.heap.pageSize()),
        .rows = undefined,
    };
    const n: u64 = bank.n_experts;
    const p = (budget.prompt -| (b.prompt_windows + b.staging)) / row;
    const d = (budget.decode -| (b.decode_windows + b.staging)) / row;
    if (p == 0) return error.StreamedRowsDoNotFit;
    b.rows = .{ .prompt = @intCast(@min(p, n)), .decode = @intCast(@min(@max(d, p), n)) };
    return b;
}

/// Routed ids a decode-phase call takes (the stream's route width); a wider call is a prompt pass.
pub const decode_route_ids: usize = sdk_ext.expert.policy.max_route_ids;

pub const Source = struct {
    a: std.mem.Allocator,
    g: G,
    set: *kernel_set.Set,
    q: *xq.Accepted(G),
    bank: expert_bank.Bank,
    gates: []Gate,
    prompt_rows: []u32,
    decode_rows: []u32,
    stream: *expert_stream.Stream,
    source: xp.StreamSource,
    hook: Hook,
    bill: Bill,
    phase: Phase = .prompt,
    released: bool = false,

    /// The bank in `dir` at the rows `budget` holds, streamed on `s` (the trunk's stream). `gates`: every routed layer's
    /// router (its arrays borrowed for the source's life).
    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, s: mlx.mlx_stream, geo: Geometry, gates: []const Gate, budget: Budget) !*Source {
        var diag: xk.Diag = .{};
        errdefer if (diag.len > 0) std.log.err("streamed exl3: {s}", .{diag.message()});
        // One expert reader per process: a second streamed load is refused by name.
        if (comptime expert_stream.uses_reader) try sdk_ext.expert.takeReader();
        errdefer if (comptime expert_stream.uses_reader) sdk_ext.expert.giveReader();
        const self = try a.create(Source);
        errdefer a.destroy(self);
        self.a = a;
        self.phase = .prompt;
        self.released = false;
        self.bank = try expert_bank.Bank.open(a, io, dir, implementedOf(geo), null);
        errdefer self.bank.deinit();
        self.bill = try billBank(&self.bank, budget);
        self.gates = try a.dupe(Gate, gates);
        errdefer a.free(self.gates);
        self.prompt_rows = try a.alloc(u32, geo.n_layers);
        errdefer a.free(self.prompt_rows);
        @memset(self.prompt_rows, self.bill.rows.prompt);
        self.decode_rows = try a.alloc(u32, geo.n_layers);
        errdefer a.free(self.decode_rows);
        @memset(self.decode_rows, self.bill.rows.decode);
        self.g = try G.init(a, s);
        errdefer self.g.deinit();
        self.set = try kernel_set.Set.init(a, .{ .device = .{ .stream = s } }, &diag);
        errdefer self.set.deinit();
        self.set.install(G, &self.g);
        errdefer kernel_set.Set.uninstall(G, &self.g);
        self.q = try xq.accept(G, a, &self.g, .{ .kernels = self.set.ref() }, .{
            .hidden = geo.hidden_size,
            .inter = geo.moe_intermediate_size,
            .top_k = geo.n_experts_per_tok,
            .n_layers = geo.n_layers,
            .act = .{ .swiglu_clamped = geo.swiglu_limit },
            .input = .bfloat16,
        }, &diag);
        errdefer self.q.deinit(&self.g);
        try self.q.routeExpertBf16();
        try self.q.compileTexts(self.set, &xq.form_texts, &diag);
        try self.q.routeForms(&self.g, forms);
        try self.q.compileTexts(self.set, &xq.banked_texts, &diag);
        try self.q.routeBanked(&self.g);
        const event = try expert_event.createMetal();
        self.stream = try expert_stream.Stream.init(a, &self.bank, .{
            .rows = self.prompt_rows,
            .slot_memory = .{ .mlx = s },
            .lookahead = lookahead,
            .event = .{ .backend = .{ .metal = event.object }, .watchdog_ms = event_watchdog_ms },
            .wide_depth = wide_depth,
            .transient_rows = @as(u32, wide_depth) * sdk_ext.expert.policy.max_route_ids,
            .transient_release = expert_stream.phase_change_releases_wide_windows and expert_stream.transient_release_default,
        });
        errdefer self.stream.deinit();
        self.source = xp.StreamSource.init(self.stream);
        self.hook = try Hook.initWith(a, &self.g, &self.source, Math.init(self.q, geo), geo, .{ .gates = self.gates, .event = event, .wide = wide, .banked = true, .hoist_first = true, .devroute = true });
        return self;
    }

    pub fn deinit(self: *Source) void {
        _ = mlx.mlx_synchronize(self.g.s);
        self.hook.deinit();
        self.stream.deinit();
        self.q.deinit(&self.g);
        kernel_set.Set.uninstall(G, &self.g);
        self.set.deinit();
        self.g.deinit();
        self.bank.deinit();
        self.a.free(self.gates);
        self.a.free(self.prompt_rows);
        self.a.free(self.decode_rows);
        self.a.destroy(self);
        if (comptime expert_stream.uses_reader) sdk_ext.expert.giveReader();
    }

    /// Model layer `layer`'s routed sum `[n, hidden]` f32 for the normed MoE input `x` `[n, hidden]`, the routes `idx`
    /// `[n, k]` and their weights `wts`.
    pub fn rows(self: *Source, layer: u32, x: T, idx: T, wts: T) !T {
        const g = &self.g;
        defer g.reset();
        const ids = if (g.dtypeOf(idx) == .int32) idx else try g.astype(idx, .int32);
        const ro = try self.hook.at(layer).routed(g, x, ids);
        const w = if (g.dtypeOf(wts) == .float32) wts else try g.astype(wts, .float32);
        return g.keep(try g.sum(try g.mul(try g.astype(ro, .float32), try g.expandDims(w, -1)), -2, false));
    }

    /// A prompt pass's predicted routes of `layer` (every row's ids): its experts, hottest first, read into its rows
    /// while its attention runs.
    pub fn readAhead(self: *Source, layer: u32, ids: []const u16) !void {
        if (self.phase != .prompt) return;
        const n = self.bank.n_experts;
        const counts = try self.a.alloc(u32, n);
        defer self.a.free(counts);
        @memset(counts, 0);
        for (ids) |e| counts[e] += 1;
        const ranked = try self.a.alloc(u16, n);
        defer self.a.free(ranked);
        try self.hook.at(layer).readAheadSeed(sdk_ext.expert.policy.rankHottest(counts, ranked));
    }

    /// The rows of `p`: decode grows the slot banks once the prompt's windows are freed; a prompt shrinks them back,
    /// keeping the residents decode left.
    pub fn setPhase(self: *Source, p: Phase) !void {
        if (p == self.phase) return;
        try self.hook.flush();
        switch (p) {
            .decode => {
                self.released = (try self.hook.releaseTransient()) > 0;
                _ = mlx.mlx_synchronize(self.g.s);
                self.g.clearCache();
                try self.hook.grow(&self.g, self.decode_rows);
            },
            .prompt => {
                _ = try self.hook.shrink(self.prompt_rows, .keep);
                _ = mlx.mlx_synchronize(self.g.s);
                self.g.clearCache();
                _ = try self.hook.regrowTransient(&self.g, self.released);
            },
        }
        self.phase = p;
    }

    /// After the eval that consumed the forward's calls.
    pub fn settle(self: *Source) !void {
        try self.hook.flush();
    }

    pub fn stats(self: *Source) expert_stream.Stats {
        return self.source.stats();
    }
};

/// A synthetic EXL3 bank for fixtures (`expert/bank.zig` `writeSynth`).
pub const writeSynth = expert_bank.writeSynth;

const testing = std.testing;

test "exl3 source: the rows a budget holds, by term, floor-capped at the bank's experts and decode at least the prompt's" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const image = try expert_bank.writeSynth(a, &tmp, .{ .hidden = 5120, .inter = 2304, .n_experts = 8, .k = &.{ 3, 2, 3 }, .finite_scales = true });
    defer a.free(image);
    var rbuf: [512]u8 = undefined;
    const dir = try expert_bank.tmpRoot(&tmp, &rbuf);
    const geo: Geometry = .{ .hidden_size = 5120, .moe_intermediate_size = 2304, .n_experts_per_tok = 2, .n_routed_experts = 8, .n_layers = 3, .swiglu_limit = 10 };
    const z = try billOf(a, testing.io, dir, geo, .{ .prompt = std.math.maxInt(u64) / 2, .decode = std.math.maxInt(u64) / 2 });
    try testing.expectEqual(@as(u32, 8), z.rows.prompt);
    try testing.expectEqual(@as(u32, 8), z.rows.decode);
    const k3 = expert_bank.layerSegments(3, 5120, 2304).?.logical_bytes;
    const k2 = expert_bank.layerSegments(2, 5120, 2304).?.logical_bytes;
    try testing.expectEqual(2 * k3 + k2, z.row_bytes);
    try testing.expectEqual(@as(u64, wide_depth) * 48 * (k3 + k2), z.prompt_windows);
    const two = z.prompt_windows + z.staging + 2 * z.row_bytes;
    const b = try billOf(a, testing.io, dir, geo, .{ .prompt = two + z.row_bytes - 1, .decode = z.decode_windows + z.staging + 5 * z.row_bytes });
    try testing.expectEqual(@as(u32, 2), b.rows.prompt);
    try testing.expectEqual(@as(u32, 5), b.rows.decode);
    try testing.expect(b.prompt() <= two + z.row_bytes - 1);
    const lean = try billOf(a, testing.io, dir, geo, .{ .prompt = two, .decode = 0 });
    try testing.expectEqual(@as(u32, 2), lean.rows.decode);
    try testing.expectError(error.StreamedRowsDoNotFit, billOf(a, testing.io, dir, geo, .{ .prompt = two - z.row_bytes - 1, .decode = two }));
}
