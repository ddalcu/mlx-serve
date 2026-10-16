//! The streaming packs on mlx-stream (`lib/mlx-stream`): DeepSeek-V4.1's EXL3 repack (`ModelConfig.dsv41_stream`)
//! and MiMo-V2.6-Flash's prepared MXFP4 pack (`ModelConfig.mimo_stream`). The plugin owns each arch, its kernels and
//! the expert streamer, and loads its own weights; the host parses the config, renders the template, samples and
//! schedules. The plugin exports one glue surface per arch (`arch`, `default_context`, `generation_headroom`): the
//! root's for DeepSeek-V4.1, `mimo`'s for MiMo. An entry point an arch does not declare (prefix restore, decode
//! handover) is left out at comptime.

const std = @import("std");
const mlx = @import("../mlx.zig");
const log = @import("../log.zig");
const server = @import("../server.zig");
const status = @import("../status.zig");
const ModelConfig = @import("../model.zig").ModelConfig;
const plugin = @import("mlx_stream");
const sdk = plugin.sdk;

pub const built = true;

/// The bytes the host's other models hold in this process; the plugin's footprint bounds rise by it.
pub fn setForeignBytes(n: u64) void {
    sdk.memory.setForeignBytes(n);
}

/// The plugin's arches this host serves.
const Kind = enum { dsv41, mimo };

fn Side(comptime k: Kind) type {
    return switch (k) {
        .dsv41 => plugin,
        .mimo => plugin.mimo,
    };
}

fn kindOf(config: *const ModelConfig) Kind {
    return if (config.mimo_stream) .mimo else .dsv41;
}

fn Bound(comptime k: Kind) type {
    const A = Side(k).arch;
    return struct { cfg: *A.Config, module: *A.Module };
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    /// The arch's config and module (it keeps references into `weights`).
    arch: union(Kind) { dsv41: Bound(.dsv41), mimo: Bound(.mimo) },
    /// The trunk the module binds.
    weights: sdk.Weights,
    /// The request `begin` started: its prompt pass's shape, then its decode handover.
    request: ?sdk.RequestShape = null,
    /// The next forward is the request's prompt pass (the generator hands it the whole remaining prompt).
    prompt_due: bool = false,
    handover_due: bool = false,
    /// The draft lane is armed for this request (`arm`), under this sampling.
    drafting: bool = false,
    sampling: sdk.SamplingParams = .{},
};

/// The context the load bills: the model's, else the arch's standard request.
fn billedContext(comptime k: Kind, config: *const ModelConfig) u64 {
    const context = server.manualContext(config);
    return if (context > 0) context else Side(k).default_context;
}

/// The plugin's config of the pack, with the model's context and the load's facts (an arch with `applyLoadFacts`
/// gets the context stamped on the facts, which its init reads again).
fn parse(comptime k: Kind, gpa: std.mem.Allocator, io: std.Io, config: *const ModelConfig, f: *sdk.LoadFacts) !*Side(k).arch.Config {
    const A = Side(k).arch;
    const dir = config.pack_dir.?;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const text = try d.readFileAlloc(io, "config.json", arena.allocator(), .limited(16 << 20));
    const peek = try sdk.ConfigPeek.parse(arena.allocator(), dir, text);
    var diag: sdk.Diag = .{};
    const c = A.parse(gpa, &peek, &diag) catch |e| {
        log.err("mlx-stream: {s}\n", .{diag.message()});
        return e;
    };
    errdefer A.freeConfig(gpa, c);
    if (comptime @hasDecl(A, "applyLoadFacts")) {
        f.effective_context = billedContext(k, config);
        try A.applyLoadFacts(c, f);
    } else {
        const context = server.manualContext(config);
        if (context > 0) c.max_context_tokens = context;
        c.* = c.withFacts(f);
    }
    return c;
}

/// Sampled before the weights load: the memory already in use, which the plugin's rows fill on top of, and the
/// host's wired-limit margin.
fn facts() sdk.LoadFacts {
    return .{ .memory_baseline_bytes = status.getTotalMemBytes() -| status.getAvailableMemBytes(), .wired_margin_bytes = server.wired_limit_margin_bytes };
}

/// What a load holds under the GPU ceiling: the preflight's bill.
pub fn loadBytes(gpa: std.mem.Allocator, io: std.Io, config: *const ModelConfig) !u64 {
    return switch (kindOf(config)) {
        inline else => |k| {
            const A = Side(k).arch;
            var f = facts();
            const c = try parse(k, gpa, io, config, &f);
            defer A.freeConfig(gpa, c);
            return A.loadBytes(gpa, io, c, &f, server.staticGpuMemoryCeiling());
        },
    };
}

/// Positions a request may span: the prompts the load bills (the model's context, else the plugin's standard
/// request) and the generation past them.
pub fn contextLength(config: *const ModelConfig) u32 {
    return switch (kindOf(config)) {
        inline else => |k| @intCast(billedContext(k, config) + Side(k).generation_headroom),
    };
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, s: mlx.mlx_stream, config: *const ModelConfig) !*Model {
    return switch (kindOf(config)) {
        inline else => |k| openArch(k, gpa, io, s, config),
    };
}

fn openArch(comptime k: Kind, gpa: std.mem.Allocator, io: std.Io, s: mlx.mlx_stream, config: *const ModelConfig) !*Model {
    const A = Side(k).arch;
    if (@TypeOf(A.claimProcess) != void) try A.claimProcess();
    errdefer if (@TypeOf(A.releaseProcess) != void) A.releaseProcess();
    var f = facts();
    const c = try parse(k, gpa, io, config, &f);
    errdefer A.freeConfig(gpa, c);
    const dir = config.pack_dir.?;
    const m = try gpa.create(Model);
    errdefer gpa.destroy(m);
    m.* = .{ .gpa = gpa, .arch = @unionInit(@FieldType(Model, "arch"), @tagName(k), .{ .cfg = c, .module = undefined }), .weights = try sdk.loader.dir(io, gpa, dir, .{ .nocache = A.caps.residents_past_page_cache }) };
    errdefer m.weights.deinit();
    const load: sdk.LoadCtx = .{ .gpa = gpa, .io = io, .stream = s, .weights = &m.weights, .loader = &sdk.loader, .facts = f, .ceiling = server.staticGpuMemoryCeiling() };
    @field(m.arch, @tagName(k)).module = try A.init(&load, c);
    return m;
}

pub fn close(m: *Model) void {
    switch (m.arch) {
        inline else => |b, k| {
            const A = Side(k).arch;
            A.deinit(b.module);
            m.weights.deinit();
            A.freeConfig(m.gpa, b.cfg);
            if (@TypeOf(A.releaseProcess) != void) A.releaseProcess();
        },
    }
    m.gpa.destroy(m);
}

/// A request's start: how many leading positions of `prompt` the module's kept state already holds (the rest, at
/// least the last token, runs through `forward`), and the shape its prompt pass bills. A prompt past the context
/// the load billed is refused before any work (a 400).
pub fn begin(m: *Model, prompt: []const u32, max_tokens: u32, context: u64) !u64 {
    switch (m.arch) {
        inline else => |b, k| {
            const A = Side(k).arch;
            const bound: u64 = if (comptime @hasField(A.Module, "max_context")) b.module.max_context else b.module.context_bound;
            if (prompt.len > bound) {
                log.warn("[mlx-stream] a {d}-token prompt is over the {d} tokens this load billed (raise the model's ctx_size)\n", .{ prompt.len, bound });
                return error.PrefillDoesNotFit;
            }
            m.request = .{ .prompt_tokens = prompt.len, .max_tokens = max_tokens, .host_context = context };
            m.prompt_due = true;
            m.handover_due = true;
            m.drafting = false;
            return if (comptime @hasDecl(A, "restorePrefix")) A.restorePrefix(b.module, prompt[0..prompt.len -| 1]) else 0;
        },
    }
}

/// The request's prompt pass first, decode after it: the last row's logits, f32 `[1, 1, vocab]`. The decode
/// handover runs before the first decode forward.
pub fn forward(m: *Model, ids: []const u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const logits = switch (m.arch) {
        inline else => |b, k| blk: {
            const A = Side(k).arch;
            if (m.prompt_due) {
                m.prompt_due = false;
                break :blk try A.prefill(b.module, ids, m.request.?);
            }
            try handover(m);
            break :blk try A.step(b.module, ids);
        },
    };
    defer _ = mlx.mlx_array_free(logits);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, logits, .float32, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, f, &[_]c_int{ 1, 1, @intCast(mlx.mlx_array_size(f)) }, 3, s));
    return out;
}

fn handover(m: *Model) !void {
    if (!m.handover_due) return;
    m.handover_due = false;
    const req = m.request orelse return;
    switch (m.arch) {
        inline else => |b, k| {
            const A = Side(k).arch;
            if (comptime @hasDecl(A, "handover"))
                try A.handover(b.module, .{ .prompt_tokens = @intCast(req.prompt_tokens), .reserved_tokens = req.prompt_tokens + req.max_tokens, .native_draft = m.drafting });
        },
    }
}

pub fn position(m: *const Model) u64 {
    return switch (m.arch) {
        inline else => |b, k| Side(k).arch.position(b.module),
    };
}

/// The draft block, 0 when the pack ships no stages.
pub fn blockSize(m: *const Model) u32 {
    return switch (m.arch) {
        inline else => |b, k| Side(k).arch.draft_lane.blockSize(b.module),
    };
}

pub fn laneName(m: *const Model) []const u8 {
    return switch (m.arch) {
        inline else => |b, k| Side(k).arch.draft_lane.laneName(b.module),
    };
}

pub const SamplingParams = sdk.SamplingParams;

/// Arms the draft lane for a request with nothing that shapes its logits (null: serial). The plugin samples a sampled
/// request itself, from `sampling`, which every `round` of the request receives.
pub fn arm(m: *Model, sampling: ?SamplingParams) bool {
    const sp = sampling orelse {
        m.drafting = false;
        return false;
    };
    m.sampling = sp;
    m.drafting = switch (m.arch) {
        inline else => |b, k| blockSize(m) > 0 and Side(k).arch.draft_lane.arm(b.module, .{ .greedy = sp.greedy(), .clean = true, .sampling = sp }) != .off,
    };
    return m.drafting;
}

/// One draft round from `t1`: the committed tokens (`t1` first) and the next round's token.
pub fn round(m: *Model, gpa: std.mem.Allocator, t1: u32, accepted_cap: u32) !sdk.DraftRound {
    try handover(m);
    return switch (m.arch) {
        inline else => |b, k| Side(k).arch.draft_lane.round(b.module, gpa, t1, accepted_cap, m.sampling),
    };
}

/// The draft lane's receipt line for the request (rounds, accepted/round, misses/round, bytes/round), when the arch's
/// lane writes one; null otherwise.
pub fn laneReceipt(m: *Model, buf: []u8) ?[]const u8 {
    return switch (m.arch) {
        inline else => |b, k| if (comptime @hasDecl(Side(k).arch.draft_lane, "receipt"))
            Side(k).arch.draft_lane.receipt(b.module, buf)
        else
            null,
    };
}
