//! DeepSeek-V4.1's EXL3 streaming repack on mlx-stream (`lib/mlx-stream`): the plugin owns the arch, its kernels
//! and the expert streamer, and loads its own weights; the host parses the config, renders the template, samples and
//! schedules. A pack is the plugin's when its experts are an EXL3 bank (`ModelConfig.dsv41_stream`).

const std = @import("std");
const mlx = @import("../mlx.zig");
const log = @import("../log.zig");
const server = @import("../server.zig");
const status = @import("../status.zig");
const ModelConfig = @import("../model.zig").ModelConfig;
const plugin = @import("mlx_stream");
const sdk = plugin.sdk;
const arch = plugin.arch;

pub const built = true;

pub const Model = struct {
    gpa: std.mem.Allocator,
    cfg: *arch.Config,
    module: *arch.Module,
    /// The trunk the module binds (it keeps references into the map).
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

/// The plugin's config of the pack, with the model's context and the load's facts.
fn parse(gpa: std.mem.Allocator, io: std.Io, config: *const ModelConfig, f: sdk.LoadFacts) !*arch.Config {
    const dir = config.dsv41_dir.?;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const text = try d.readFileAlloc(io, "config.json", arena.allocator(), .limited(16 << 20));
    const peek = try sdk.ConfigPeek.parse(arena.allocator(), dir, text);
    var diag: sdk.Diag = .{};
    const c = arch.parse(gpa, &peek, &diag) catch |e| {
        log.err("mlx-stream: {s}\n", .{diag.message()});
        return e;
    };
    const context = server.manualContext(config);
    if (context > 0) c.max_context_tokens = context;
    c.* = c.withFacts(&f);
    return c;
}

/// Sampled before the weights load: the memory already in use, which the plugin's rows fill on top of, and the
/// host's wired-limit margin.
fn facts() sdk.LoadFacts {
    return .{ .memory_baseline_bytes = status.getTotalMemBytes() -| status.getAvailableMemBytes(), .wired_margin_bytes = server.wired_limit_margin_bytes };
}

/// What a load holds under the GPU ceiling: the preflight's bill.
pub fn loadBytes(gpa: std.mem.Allocator, io: std.Io, config: *const ModelConfig) !u64 {
    const f = facts();
    const c = try parse(gpa, io, config, f);
    defer arch.freeConfig(gpa, c);
    return arch.loadBytes(gpa, io, c, &f, server.staticGpuMemoryCeiling());
}

/// Positions a request may span: the prompts the load bills (the model's context, else the plugin's standard
/// request) and the generation past them.
pub fn contextLength(config: *const ModelConfig) u32 {
    const context = server.manualContext(config);
    return @intCast((if (context > 0) context else plugin.default_context) + plugin.generation_headroom);
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, s: mlx.mlx_stream, config: *const ModelConfig) !*Model {
    if (@TypeOf(arch.claimProcess) != void) try arch.claimProcess();
    errdefer if (@TypeOf(arch.releaseProcess) != void) arch.releaseProcess();
    const f = facts();
    const c = try parse(gpa, io, config, f);
    errdefer arch.freeConfig(gpa, c);
    const dir = config.dsv41_dir.?;
    const m = try gpa.create(Model);
    errdefer gpa.destroy(m);
    m.* = .{ .gpa = gpa, .cfg = c, .module = undefined, .weights = try sdk.loader.dir(io, gpa, dir, .{ .nocache = arch.caps.residents_past_page_cache }) };
    errdefer m.weights.deinit();
    const load: sdk.LoadCtx = .{ .gpa = gpa, .io = io, .stream = s, .weights = &m.weights, .loader = &sdk.loader, .facts = f, .ceiling = server.staticGpuMemoryCeiling() };
    m.module = try arch.init(&load, c);
    return m;
}

pub fn close(m: *Model) void {
    arch.deinit(m.module);
    m.weights.deinit();
    arch.freeConfig(m.gpa, m.cfg);
    if (@TypeOf(arch.releaseProcess) != void) arch.releaseProcess();
    m.gpa.destroy(m);
}

/// A request's start: how many leading positions of `prompt` the module's kept state already holds (the rest, at
/// least the last token, runs through `forward`), and the shape its prompt pass bills. A prompt past the context
/// the load billed is refused before any work (a 400).
pub fn begin(m: *Model, prompt: []const u32, max_tokens: u32, context: u64) !u64 {
    if (prompt.len > m.module.max_context) {
        log.warn("[mlx-stream] a {d}-token prompt is over the {d} tokens this load billed (raise the model's ctx_size)\n", .{ prompt.len, m.module.max_context });
        return error.PrefillDoesNotFit;
    }
    m.request = .{ .prompt_tokens = prompt.len, .max_tokens = max_tokens, .host_context = context };
    m.prompt_due = true;
    m.handover_due = true;
    m.drafting = false;
    return arch.restorePrefix(m.module, prompt[0 .. prompt.len -| 1]);
}

/// The request's prompt pass first, decode after it: the last row's logits, f32 `[1, 1, vocab]`. The decode
/// handover runs before the first decode forward.
pub fn forward(m: *Model, ids: []const u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const logits = if (m.prompt_due) blk: {
        m.prompt_due = false;
        break :blk try arch.prefill(m.module, ids, m.request.?);
    } else blk: {
        try handover(m);
        break :blk try arch.step(m.module, ids);
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
    try arch.handover(m.module, .{ .prompt_tokens = @intCast(req.prompt_tokens), .reserved_tokens = req.prompt_tokens + req.max_tokens, .native_draft = m.drafting });
}

pub fn position(m: *const Model) u64 {
    return arch.position(m.module);
}

/// The draft block, 0 when the pack ships no stages.
pub fn blockSize(m: *const Model) u32 {
    return arch.draft_lane.blockSize(m.module);
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
    m.drafting = blockSize(m) > 0 and arch.draft_lane.arm(m.module, .{ .greedy = sp.greedy(), .clean = true, .sampling = sp }) != .off;
    return m.drafting;
}

/// One draft round from `t1`: the committed tokens (`t1` first) and the next round's token.
pub fn round(m: *Model, gpa: std.mem.Allocator, t1: u32, accepted_cap: u32) !sdk.DraftRound {
    try handover(m);
    return arch.draft_lane.round(m.module, gpa, t1, accepted_cap, m.sampling);
}
