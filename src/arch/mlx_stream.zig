//! The streaming packs on mlx-stream (`lib/mlx-stream`): DeepSeek-V4.1's EXL3 repack (`ModelConfig.dsv41_stream`),
//! MiMo-V2.6-Flash's prepared MXFP4 pack (`ModelConfig.mimo_stream`), GLM-5.3's pack (`ModelConfig.glm_stream`) and
//! GLM-5.3-Flash's NVFP4 pack (`ModelConfig.glm5_stream`).
//! The plugin owns each arch, its kernels and the expert streamer, and loads its own weights; the host parses the
//! config, renders the template, samples and schedules. The plugin exports one glue surface per arch (`arch`,
//! `default_context`, `generation_headroom`): the root's for DeepSeek-V4.1, `mimo`'s for MiMo, `glm`'s for GLM-5.3.
//! An entry point an arch does not declare (prefix restore, decode handover, request end) is left out at comptime.
//! GLM-5.3 alone takes the model's model-settings.json entry, and generates past its billed context as far as its
//! arch's `maxOutput`.

const std = @import("std");
const mlx = @import("../mlx.zig");
const log = @import("../log.zig");
const server = @import("../server.zig");
const status = @import("../status.zig");
const model_settings = @import("../model_settings.zig");
const scheduler = @import("../scheduler.zig");
const ModelConfig = @import("../model.zig").ModelConfig;
const plugin = @import("mlx_stream");
const sdk = plugin.sdk;

pub const built = true;

/// The bytes the host's other models hold in this process; the plugin's footprint bounds rise by it.
pub fn setForeignBytes(n: u64) void {
    sdk.memory.setForeignBytes(n);
}

/// The plugin's arches this host serves.
const Kind = enum { dsv41, mimo, glm, glm5 };

fn Side(comptime k: Kind) type {
    return switch (k) {
        .dsv41 => plugin,
        .mimo => plugin.mimo,
        .glm => plugin.glm,
        .glm5 => plugin.glm5_next,
    };
}

fn kindOf(config: *const ModelConfig) Kind {
    if (config.glm_stream) return .glm;
    if (config.glm5_stream) return .glm5;
    return if (config.mimo_stream) .mimo else .dsv41;
}

fn Bound(comptime k: Kind) type {
    const A = Side(k).arch;
    return struct { cfg: *A.Config, module: *A.Module };
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    /// The arch's config and module (it keeps references into `weights`).
    arch: union(Kind) { dsv41: Bound(.dsv41), mimo: Bound(.mimo), glm: Bound(.glm), glm5: Bound(.glm5) },
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

/// The context the load bills: the model's `ctx_size`, else the auto context this pack's preflight chose
/// (`resolveContext`), else the arch's standard request (a pack no load has billed yet).
fn billedContext(comptime k: Kind, config: *const ModelConfig) u64 {
    const context = server.manualContext(config);
    if (context > 0) return context;
    return auto_context.get(config.pack_dir.?) orelse Side(k).default_context;
}

/// The contexts an auto context tries, largest first (the model sets none): the first whose load bill at the fill's
/// floor rows fits the box wins, so the served context is what memory allows, not the plugin's 16K standard request.
const auto_context_rungs = [_]u64{ 262144, 131072, 65536, 32768 };

/// The largest rung whose bill fits `avail`; null when none does. `bills.need(rung)` is the bytes the load takes at it.
fn largestFitting(avail: u64, rungs: []const u64, bills: anytype) ?u64 {
    for (rungs) |rung| if (bills.need(rung) <= avail) return rung;
    return null;
}

/// The context a load of `config` bills, resolved once per pack: the model's `ctx_size`, else the auto rung
/// (`largestFitting` over the arch's own load bill against the memory available under the GPU ceiling), else the
/// arch's standard request. Recorded for `contextLength`, which runs on connection threads without an `Io`.
fn resolveContext(comptime k: Kind, gpa: std.mem.Allocator, io: std.Io, config: *const ModelConfig) !u64 {
    const manual = server.manualContext(config);
    if (manual > 0) return manual;
    const dir = config.pack_dir.?;
    if (auto_context.get(dir)) |c| return c;
    const ceiling = server.staticGpuMemoryCeiling();
    const avail = if (ceiling > 0) @min(status.getAvailableMemBytes(), ceiling) else status.getAvailableMemBytes();
    const Bills = struct {
        gpa: std.mem.Allocator,
        io: std.Io,
        config: *const ModelConfig,
        ceiling: u64,
        fn need(self: @This(), rung: u64) u64 {
            const A = Side(k).arch;
            var f = facts();
            const c = parse(k, self.gpa, self.io, self.config, &f, rung) catch return std.math.maxInt(u64);
            defer A.freeConfig(self.gpa, c);
            const bytes = A.loadBytes(self.gpa, self.io, c, &f, self.ceiling) catch return std.math.maxInt(u64);
            return scheduler.loadRequirementBytes(bytes);
        }
    };
    const chosen = largestFitting(avail, &auto_context_rungs, Bills{ .gpa = gpa, .io = io, .config = config, .ceiling = ceiling }) orelse Side(k).default_context;
    log.info("mlx-stream: context {d} tokens (auto: the largest rung whose load fits the {d:.1} GB available; set ctx_size to pin one)\n", .{ chosen, @as(f64, @floatFromInt(avail)) / 1_073_741_824.0 });
    auto_context.note(dir, chosen);
    return chosen;
}

/// The plugin's config of the pack, with the model's context and the load's facts (an arch with `applyLoadFacts`
/// gets the context stamped on the facts, which its init reads again).
fn parse(comptime k: Kind, gpa: std.mem.Allocator, io: std.Io, config: *const ModelConfig, f: *sdk.LoadFacts, context: u64) !*Side(k).arch.Config {
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
        // The arch takes the load's facts itself (its `loadBytes` and `init`). The resolved context wins over the entry.
        try applyContext(A, arena.allocator(), c, context);
        noteMaxOutput(dir, A.maxOutput(c));
    } else if (comptime @hasDecl(A, "applyLoadFacts")) {
        f.effective_context = context;
        try A.applyLoadFacts(c, f);
    } else {
        if (context > 0) c.max_context_tokens = @intCast(context);
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

/// GLM-5.3: the resolved context after the entry, which it wins over (the entry's `ctx_size`, `--ctx-size`, or the
/// auto rung), as the arch's own `ctx_size` key.
fn applyContext(comptime A: type, arena: std.mem.Allocator, c: *A.Config, context: u64) !void {
    if (context > 0) try applyEntry(A, c, try std.json.parseFromSliceLeaky(std.json.Value, arena, try std.fmt.allocPrint(arena, "{{\"ctx_size\": {d}}}", .{context}), .{}));
}

/// A value per pack as its last load read it (the preflight and the open), for `contextLength`: GLM-5.3's `maxOutput`
/// and the auto context. A pthread mutex, not `std.Io.Mutex`: `contextLength` runs on connection threads that carry
/// no `Io` (log.zig's sink mutex, the same reason).
const PackValues = struct {
    const slots = 4;
    mu: std.c.pthread_mutex_t = .{},
    keys: [slots]u64 = @splat(0),
    values: [slots]u64 = @splat(0),
    next: usize = 0,

    fn note(self: *PackValues, dir: []const u8, value: u64) void {
        const key = packKey(dir);
        _ = std.c.pthread_mutex_lock(&self.mu);
        defer _ = std.c.pthread_mutex_unlock(&self.mu);
        const i = std.mem.indexOfScalar(u64, &self.keys, key) orelse blk: {
            const at = self.next;
            self.next = (self.next + 1) % slots;
            break :blk at;
        };
        self.keys[i] = key;
        self.values[i] = value;
    }

    fn get(self: *PackValues, dir: []const u8) ?u64 {
        const key = packKey(dir);
        _ = std.c.pthread_mutex_lock(&self.mu);
        defer _ = std.c.pthread_mutex_unlock(&self.mu);
        const i = std.mem.indexOfScalar(u64, &self.keys, key) orelse return null;
        return self.values[i];
    }
};
var max_output: PackValues = .{};
var auto_context: PackValues = .{};

fn packKey(dir: []const u8) u64 {
    return std.hash.Wyhash.hash(0, dir) | 1;
}

/// Records GLM-5.3's `maxOutput` for the pack at `dir`.
pub fn noteMaxOutput(dir: []const u8, value: u64) void {
    max_output.note(dir, value);
}

/// The generation past the billed context a GLM-5.3 request of the pack at `dir` may take: the `maxOutput` its last
/// parse recorded; the arch's default (`glm.generation_headroom`) for a pack not parsed yet.
fn maxOutputOf(dir: []const u8) u64 {
    return max_output.get(dir) orelse plugin.glm.generation_headroom;
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
            const c = try parse(k, gpa, io, config, &f, try resolveContext(k, gpa, io, config));
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
    const c = try parse(k, gpa, io, config, &f, try resolveContext(k, gpa, io, config));
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

test "mlx_stream: a model that sets no context takes the largest rung whose load fits; none fitting keeps the standard request" {
    const Bills = struct {
        per_token: u64,
        fixed: u64,
        fn need(self: @This(), rung: u64) u64 {
            return self.fixed + rung * self.per_token;
        }
    };
    const GB: u64 = 1 << 30;
    // 200 GB free, 150 GB of residents, 256 KB of bill per token: 131,072 fits (182 GB), 262,144 does not (214 GB).
    try testing.expectEqual(@as(?u64, 131072), largestFitting(200 * GB, &auto_context_rungs, Bills{ .per_token = 256 << 10, .fixed = 150 * GB }));
    // Room for everything: the top rung.
    try testing.expectEqual(@as(?u64, 262144), largestFitting(240 * GB, &auto_context_rungs, Bills{ .per_token = 256 << 10, .fixed = 150 * GB }));
    // Not even 32,768 fits: the caller falls back to the arch's standard request (the fill refuses by name later).
    try testing.expectEqual(@as(?u64, null), largestFitting(100 * GB, &auto_context_rungs, Bills{ .per_token = 256 << 10, .fixed = 150 * GB }));
    // A rung's bill that cannot be computed never fits (the chooser skips it).
    const Broken = struct {
        fn need(_: @This(), rung: u64) u64 {
            return if (rung > 65536) std.math.maxInt(u64) else 1;
        }
    };
    try testing.expectEqual(@as(?u64, 65536), largestFitting(1, &auto_context_rungs, Broken{}));
    // The recorded auto context is what `contextLength` reads for a model without ctx_size.
    var config: ModelConfig = .{ .model_type = "deepseek_v41", .pack_dir = "/packs/dsv41-repack", .dsv41_stream = true, .ctx_override = 0 };
    try testing.expectEqual(@as(u32, @intCast(plugin.default_context + plugin.generation_headroom)), contextLength(&config));
    auto_context.note("/packs/dsv41-repack", 131072);
    try testing.expectEqual(@as(u32, 131072 + @as(u32, @intCast(plugin.generation_headroom))), contextLength(&config));
    config.ctx_override = 40000;
    try testing.expectEqual(@as(u32, 40000 + @as(u32, @intCast(plugin.generation_headroom))), contextLength(&config));
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
