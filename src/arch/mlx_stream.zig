//! The streaming packs on mlx-stream (`lib/mlx-stream`): DeepSeek-V4.1's EXL3 repack (`ModelConfig.dsv41_stream`),
//! MiMo-V2.6-Flash's prepared MXFP4 pack (`ModelConfig.mimo_stream`), GLM-5.3's pack (`ModelConfig.glm_stream`),
//! GLM-5.3-Flash's NVFP4 pack (`ModelConfig.glm5_stream`) and Beam 1.0's NVFP4 pack (`ModelConfig.beam_stream`).
//! The plugin owns each arch, its kernels and the expert streamer, and loads its own weights; the host parses the
//! config, renders the template, samples and schedules. The plugin exports one glue surface per arch (`arch`,
//! `default_context`, `generation_headroom`): the root's for DeepSeek-V4.1, `mimo`'s for MiMo, `glm`'s for GLM-5.3,
//! `glm5_next`'s for GLM-5.3-Flash, `beam`'s for Beam.
//! An entry point an arch does not declare (prefix restore, decode handover, request end) is left out at comptime.
//! GLM-5.3 alone takes the model's model-settings.json entry, and generates past its billed context as far as its
//! arch's `maxOutput`.

const std = @import("std");
const mlx = @import("../mlx.zig");
const log = @import("../log.zig");
const server = @import("../server.zig");
const status = @import("../status.zig");
const model_settings = @import("../model_settings.zig");
const ModelConfig = @import("../model.zig").ModelConfig;
const plugin = @import("mlx_stream");
const sdk = plugin.sdk;

pub const built = true;

/// The bytes the host's other models hold in this process; the plugin's footprint bounds rise by it.
pub fn setForeignBytes(n: u64) void {
    sdk.memory.setForeignBytes(n);
}

/// The plugin's arches this host serves.
const Kind = enum { dsv41, mimo, glm, glm5, beam };

fn Side(comptime k: Kind) type {
    return switch (k) {
        .dsv41 => plugin,
        .mimo => plugin.mimo,
        .glm => plugin.glm,
        .glm5 => plugin.glm5_next,
        .beam => plugin.beam,
    };
}

fn kindOf(config: *const ModelConfig) Kind {
    if (config.glm_stream) return .glm;
    if (config.glm5_stream) return .glm5;
    if (config.beam_stream) return .beam;
    return if (config.mimo_stream) .mimo else .dsv41;
}

fn Bound(comptime k: Kind) type {
    const A = Side(k).arch;
    return struct { cfg: *A.Config, module: *A.Module };
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    /// The arch's config and module (it keeps references into `weights`).
    arch: union(Kind) { dsv41: Bound(.dsv41), mimo: Bound(.mimo), glm: Bound(.glm), glm5: Bound(.glm5), beam: Bound(.beam) },
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
    // Every arch that declares `applySettings` gets the model's model-settings.json entry (`.null` without one).
    if (comptime @hasDecl(A, "applySettings")) {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const settings = model_settings.load(arena.allocator(), io, model_settings.defaultPath(&buf));
        try applyEntry(A, c, settings.entry(dir) orelse .null);
    }
    if (comptime k == .glm) {
        // The arch takes the load's facts itself (its `loadBytes` and `init`). The manual context wins over the entry.
        try applyManualContext(A, arena.allocator(), c, config);
        noteMaxOutput(dir, A.maxOutput(c));
    } else if (comptime @hasDecl(A, "applyLoadFacts")) {
        f.effective_context = billedContext(k, config);
        try A.applyLoadFacts(c, f);
    } else {
        const context = server.manualContext(config);
        if (context > 0) c.max_context_tokens = context;
        c.* = c.withFacts(f);
    }
    return c;
}

/// The model's model-settings.json entry on an arch's config, when the arch declares `applySettings` (a `void` or an
/// error union result). An arch without it is not touched.
fn applyEntry(comptime A: type, c: *A.Config, entry: std.json.Value) !void {
    if (comptime !@hasDecl(A, "applySettings")) return;
    const R = @typeInfo(@TypeOf(A.applySettings)).@"fn".return_type.?;
    if (comptime @typeInfo(R) == .error_union) try A.applySettings(c, entry) else A.applySettings(c, entry);
}

/// GLM-5.3: the host's manual context after the entry, which it wins over (`server.manualContext`: the entry's
/// `ctx_size`, else `--ctx-size`).
fn applyManualContext(comptime A: type, arena: std.mem.Allocator, c: *A.Config, config: *const ModelConfig) !void {
    const context = server.manualContext(config);
    if (context > 0) try applyEntry(A, c, try std.json.parseFromSliceLeaky(std.json.Value, arena, try std.fmt.allocPrint(arena, "{{\"ctx_size\": {d}}}", .{context}), .{}));
}

/// GLM-5.3's `maxOutput` per pack, as the pack's last parse read it with the model's settings (the load's preflight and
/// its open), for `contextLength`. A pthread mutex, not `std.Io.Mutex`: `contextLength` runs on connection threads that
/// carry no `Io` (log.zig's sink mutex, the same reason).
const max_output_slots = 4;
var max_output_mu: std.c.pthread_mutex_t = .{};
var max_output_keys: [max_output_slots]u64 = @splat(0);
var max_output_values: [max_output_slots]u64 = @splat(0);
var max_output_next: usize = 0;

fn packKey(dir: []const u8) u64 {
    return std.hash.Wyhash.hash(0, dir) | 1;
}

/// Records GLM-5.3's `maxOutput` for the pack at `dir`.
pub fn noteMaxOutput(dir: []const u8, value: u64) void {
    const key = packKey(dir);
    _ = std.c.pthread_mutex_lock(&max_output_mu);
    defer _ = std.c.pthread_mutex_unlock(&max_output_mu);
    const i = std.mem.indexOfScalar(u64, &max_output_keys, key) orelse blk: {
        const at = max_output_next;
        max_output_next = (max_output_next + 1) % max_output_slots;
        break :blk at;
    };
    max_output_keys[i] = key;
    max_output_values[i] = value;
}

/// The generation past the billed context a GLM-5.3 request of the pack at `dir` may take: the `maxOutput` its last
/// parse recorded; the arch's default (`glm.generation_headroom`) for a pack not parsed yet.
fn maxOutputOf(dir: []const u8) u64 {
    const key = packKey(dir);
    _ = std.c.pthread_mutex_lock(&max_output_mu);
    defer _ = std.c.pthread_mutex_unlock(&max_output_mu);
    const i = std.mem.indexOfScalar(u64, &max_output_keys, key) orelse return plugin.glm.generation_headroom;
    return max_output_values[i];
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
/// request) and the generation past them (GLM-5.3: its arch's `maxOutput`).
pub fn contextLength(config: *const ModelConfig) u32 {
    return switch (kindOf(config)) {
        .glm => @intCast(billedContext(.glm, config) + maxOutputOf(config.pack_dir.?)),
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
            const bound: u64 = if (comptime @hasDecl(A.Module, "maxContext")) b.module.maxContext() else if (comptime @hasField(A.Module, "max_context")) b.module.max_context else b.module.context_bound;
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

/// The request's end (the scheduler's finish, on the inference thread): the arch's `requestEnd`, where it declares one.
pub fn end(m: *Model) void {
    switch (m.arch) {
        inline else => |b, k| if (comptime @hasDecl(Side(k).arch, "requestEnd")) Side(k).arch.requestEnd(b.module),
    }
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

const testing = std.testing;

test "mlx_stream: a GLM-5.3-Flash request spans the billed context (its generation inside it, as MiMo's)" {
    try testing.expectEqual(@as(u64, 0), plugin.glm5_next.generation_headroom);
    var config: ModelConfig = .{ .model_type = "glm5_next", .pack_dir = "/packs/glm-5.3-flash", .glm5_stream = true, .ctx_override = 32768 };
    try testing.expectEqual(Kind.glm5, kindOf(&config));
    try testing.expectEqual(@as(u32, 32768), contextLength(&config));
    config.ctx_override = 0;
    try testing.expectEqual(@as(u32, @intCast(plugin.glm5_next.default_context)), contextLength(&config));
}

test "mlx_stream: a Beam request spans the billed context (its generation inside it, as GLM-5.3-Flash's)" {
    try testing.expectEqual(@as(u64, 0), plugin.beam.generation_headroom);
    var config: ModelConfig = .{ .model_type = "beam", .pack_dir = "/packs/beam-1.0", .beam_stream = true, .ctx_override = 32768 };
    try testing.expectEqual(Kind.beam, kindOf(&config));
    try testing.expectEqual(@as(u32, 32768), contextLength(&config));
    config.ctx_override = 0;
    try testing.expectEqual(@as(u32, @intCast(plugin.beam.default_context)), contextLength(&config));
}

test "mlx_stream: Beam's arch receives the model's settings entry (rows set, a bad value marked, no entry untouched)" {
    const A = plugin.beam.arch;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const entry = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), "{\"expert_rows\": 6, \"expert_prefill_rows\": 0}", .{});
    var c: A.Config = .{};
    try applyEntry(A, &c, entry);
    try testing.expectEqual(@as(?u32, 6), c.expert_rows);
    try testing.expect(c.malformed.rows);
    var n: A.Config = .{};
    try applyEntry(A, &n, .null);
    try testing.expectEqual(@as(?u32, null), n.expert_rows);
    try testing.expect(!n.malformed.rows);
}

test "mlx_stream: a GLM-5.3 request spans the billed context and the arch's maxOutput (131,072 by default)" {
    try testing.expectEqual(@as(u64, 131072), plugin.glm.generation_headroom);
    var config: ModelConfig = .{ .model_type = "glm_moe_dsa", .pack_dir = "/packs/glm-5.3", .glm_stream = true, .ctx_override = 16448 };
    // A pack not parsed yet: the arch's default.
    try testing.expectEqual(@as(u32, 16448 + 131072), contextLength(&config));
    // The model's `max_output`, as its parse records it.
    noteMaxOutput("/packs/glm-5.3", 4096);
    try testing.expectEqual(@as(u32, 16448 + 4096), contextLength(&config));
    // The coding workload's 16,387-token prompt and max_tokens 131,072 fit whole at the default.
    noteMaxOutput("/packs/glm-5.3", 131072);
    try testing.expect(16387 + 131072 <= contextLength(&config));
}

test "mlx_stream: an arch that declares applySettings receives the model's entry; an arch without it is not touched" {
    const With = struct {
        pub const Config = struct { seen: i64 = 0 };
        pub fn applySettings(c: *Config, entry: std.json.Value) void {
            if (entry == .object) if (entry.object.get("probe")) |v| {
                c.seen = v.integer;
            };
        }
    };
    const WithError = struct {
        pub const Config = struct { seen: i64 = 0 };
        pub fn applySettings(c: *Config, entry: std.json.Value) !void {
            if (entry != .object) return error.NotAnObject;
            c.seen = entry.object.get("probe").?.integer;
        }
    };
    const Without = struct {
        pub const Config = struct { seen: i64 = 0 };
    };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const entry = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), "{\"probe\": 7}", .{});
    var a: With.Config = .{};
    try applyEntry(With, &a, entry);
    try std.testing.expectEqual(@as(i64, 7), a.seen);
    var b: WithError.Config = .{};
    try applyEntry(WithError, &b, entry);
    try std.testing.expectEqual(@as(i64, 7), b.seen);
    try std.testing.expectError(error.NotAnObject, applyEntry(WithError, &b, .null));
    var n: Without.Config = .{};
    try applyEntry(Without, &n, entry);
    try std.testing.expectEqual(@as(i64, 0), n.seen);
}
