//! An MLX pack served on the Neural Engine: programs built from the pack at load (`ane_qwen35`),
//! compiled once per Mac (`lib/ane`'s content-hashed cache), run through the private ANE runtime.
//! No MLX: requests run on the model's own worker threads beside the MLX inference thread.
//!
//! A `Model` holds the compiled programs (one per chunk of layers, plus the head) loaded as one
//! instance per slot. An `Engine` is one slot: its buffers, KV, DeltaNet states and requests on its
//! own instance of every program, so slots run at once on the ANE's clusters while sharing one copy
//! of the weights. A call runs every chunk then the head; chunk k's `y` buffer is chunk k+1's `x`.
//! DeltaNet states ping-pong between two buffer sets per chunk (a call reads set `par`, writes the
//! other). The host owns the KV history: accepted rows are copied out of each call's `k<j>_new` /
//! `v<j>_new` outputs.

const std = @import("std");
const log = @import("../log.zig");
const rt = @import("ane_rt.zig");
const q35 = @import("ane_qwen35.zig");
const pack_mod = @import("ane_pack.zig");
const prefix = @import("ane_prefix.zig");
const ane = @import("../ane.zig");
const mil = @import("ane_mil.zig");
const dflash = @import("ane_dflash.zig");
const load_progress = @import("../load_progress.zig");
const io_util = @import("../io_util.zig");

pub const P = q35.P;
const TP = q35.TP;
pub const Arch = q35.Arch;
const EntryKind = q35.EntryKind;

pub const Error = error{
    AneUnavailable,
    UnsupportedArch,
    LoadFailed,
    RunFailed,
    ContextFull,
    Cancelled,
    OutOfMemory,
};

/// Bytes a load of `arch` with `slots` engines holds at its top context: the int8 weights and each
/// slot's fp16 KV (the gate's and the preflight's bill; the KV grows to it rung by rung).
pub fn loadBytes(arch: Arch, slots: u32) u64 {
    var attn: u64 = 0;
    for (0..arch.layers) |l| attn += @intFromBool(!arch.isLinear(@intCast(l)));
    const top_ctx: u64 = std.mem.max(u32, (Options{}).ctxs);
    return q35.modelBytes(arch) + @as(u64, @max(slots, 1)) * attn * 2 * arch.nkv * arch.hd * 2 * top_ctx;
}

/// The most engines one load holds: `Model.open` pins each slot's programs per lane (`hb`).
pub const max_slots: u32 = 4;

/// `--ane-slots`: 1..`max_slots`.
pub fn parseSlots(raw: []const u8) error{InvalidSlots}!u32 {
    const n = std.fmt.parseInt(u32, raw, 10) catch return error.InvalidSlots;
    return if (n >= 1 and n <= max_slots) n else error.InvalidSlots;
}

pub const Options = struct {
    /// Engines on one load (`--ane-slots`): each its own instance of every program.
    slots: u32 = 1,
    /// The context ladder: one decode and one prefill entry per context; a sequence moves up as it grows.
    ctxs: []const u32 = &.{ 4096, 16384, 32768, 65536 },
    /// The DFlash2 drafter's dir (the pack's `drafter/` by default); null runs one token per call.
    drafter: ?[]const u8 = null,
    /// Tests: lanes per layer instead of one per Neural Engine, and tap layers without a drafter.
    lanes: u32 = 0,
    taps: []const u32 = &.{},
    /// Counts the programs as they are built (the server's load progress).
    progress: ?*load_progress.Progress = null,
    /// `Model.suppress_ids` / `defined_vocab`.
    suppress_ids: []const u32 = &.{},
    defined_vocab: u32 = 0,
};

/// The architecture of a dense qwen3_5 (`ModelConfig` spells every variant `qwen3_5_moe`) or Qwen3
/// pack, or `UnsupportedArch`: experts, Hadamard-rotated (Prism) packs, embedding packs (no
/// embeddings here) and other families.
pub fn archOf(cfg: anytype) Error!Arch {
    const qwen3 = std.mem.eql(u8, cfg.model_type, "qwen3");
    if (!qwen3 and !std.mem.eql(u8, cfg.model_type, "qwen3_5_moe")) return error.UnsupportedArch;
    if (cfg.num_experts > 0 or cfg.hadamard_block != 0 or cfg.hasEmbeddingCapability()) return error.UnsupportedArch;
    if (!qwen3 and (cfg.full_attention_interval == 0 or cfg.linear_num_value_heads == 0)) return error.UnsupportedArch;
    const a: Arch = .{
        .hidden = cfg.hidden_size,
        .ffn = cfg.intermediate_size,
        .layers = cfg.num_hidden_layers,
        .vocab = cfg.vocab_size,
        .nk = cfg.linear_num_key_heads,
        .nv = cfg.linear_num_value_heads,
        .dk = cfg.linear_key_head_dim,
        .dv = cfg.linear_value_head_dim,
        .nh = cfg.num_attention_heads,
        .nkv = cfg.num_key_value_heads,
        .hd = cfg.head_dim,
        .rot = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor),
        .rope_theta = cfg.rope_theta,
        .eps = cfg.rms_norm_eps,
        .full_interval = if (qwen3) 1 else cfg.full_attention_interval,
        .tie = cfg.tie_word_embeddings,
        .gate = !qwen3,
    };
    a.check() catch return error.UnsupportedArch;
    return a;
}

/// Serializes `Model.open`'s program builds (`rt.cacheLineage` is one per process).
var build_mu: std.Io.Mutex = .init;

/// The compile-cache group of the model at `dir`.
fn cacheGroup(buf: []u8, dir: []const u8) [:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "lm:{x:0>16}", .{std.hash.Wyhash.hash(0, dir)}, 0) catch unreachable;
}

/// The compile-cache variant of a configuration: what decides the programs' text. Not the engine
/// build: entries are keyed by the program itself, and a rebuild that still emits them restores them.
fn cacheVariant(buf: []u8, lanes: u32, ctxs: []const u32, taps: []const u32) [:0]const u8 {
    var h = std.hash.Wyhash.init(lanes);
    h.update(std.mem.sliceAsBytes(ctxs));
    h.update(std.mem.sliceAsBytes(taps));
    return std.fmt.bufPrintSentinel(buf, "{x:0>16}", .{h.final()}, 0) catch unreachable;
}

/// Lanes for a model on a Mac with `engines` Neural Engines: two split every layer when the shapes
/// split and the model is not small (`q35.smallModel`): halving each engine's read then outweighs the
/// wait between a layer's rounds.
pub fn lanesFor(arch: Arch, engines: u32) u32 {
    return if (engines >= 2 and arch.splits() and !q35.smallModel(arch)) 2 else 1;
}

/// Run requests in order on this thread.
fn runReqs(rs: []const rt.Request) !void {
    var err: rt.ErrBuf = .{};
    for (rs) |r| {
        // Two loads of one program evaluated at once for the first time can come back "Request
        // cancelled"; no request writes its own inputs, so running it again is exact.
        var tries: u32 = 0;
        while (true) : (tries += 1) {
            r.run(&err) catch {
                if (tries < 2) {
                    log.debug("[ane] run retried: {s}\n", .{err.text()});
                    continue;
                }
                log.err("[ane] run: {s}\n", .{err.text()});
                return error.RunFailed;
            };
            break;
        }
    }
}

var trace_env: ?bool = null;

/// `MLX_SERVE_ANE_TRACE` (set and not `0`, as `transformer.diagEnvValueOn`): per-call timings in the log.
fn trace() bool {
    if (trace_env == null) trace_env = if (std.c.getenv("MLX_SERVE_ANE_TRACE")) |v| v[0] != '0' else false;
    return trace_env.?;
}

// ── host math ──

/// RoPE inverse frequencies: 1 / theta^(i / rot) for i = 0, 2, ..., rot - 2.
pub fn ropeInv(out: []f64, rot: u32, theta: f64) void {
    for (out, 0..) |*v, i| v.* = 1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(rot)));
}

/// Row r's position: p0 + r, clamped to the last real token (padding rows repeat it).
pub fn rowPos(p0: u32, n: u32, r: u32) u32 {
    return @min(p0 + r, p0 + n - 1);
}

// ── sampling ──

pub const Sampling = struct {
    temperature: f32 = 0,
    top_k: u32 = 0,
    top_p: f32 = 1,
    min_p: f32 = 0,
};

/// First index of the largest logit (numpy argmax).
pub fn argmax(logits: []const f16) u32 {
    var best: usize = 0;
    var bv: f32 = logits[0];
    for (logits[1..], 1..) |v, i| {
        if (@as(f32, v) > bv) {
            bv = v;
            best = i;
        }
    }
    return @intCast(best);
}

pub const Cand = struct { logit: f32, id: u32 };

/// Rule out `ids` and every row from `defined_vocab` on (0 = none) before anything samples `row`.
pub fn suppress(row: []f16, ids: []const u32, defined_vocab: u32) void {
    for (ids) |id| if (id < row.len) {
        row[id] = -std.math.inf(f16);
    };
    if (defined_vocab > 0 and defined_vocab < row.len) @memset(row[defined_vocab..], -std.math.inf(f16));
}

/// Temperature / top-k / top-p / min-p sampling; temperature <= 0 is argmax. `scratch` holds vocab entries.
pub fn sample(logits: []const f16, s: Sampling, rng: std.Random, scratch: []Cand) u32 {
    return drawFrom(dist(logits, s, scratch) orelse return argmax(logits), rng);
}

/// Draw from candidates whose `logit` holds an unnormalized probability.
pub fn drawFrom(cands: []const Cand, rng: std.Random) u32 {
    var total: f32 = 0;
    for (cands) |c| total += c.logit;
    var u = rng.float(f32) * total;
    for (cands) |c| {
        if (u < c.logit) return c.id;
        u -= c.logit;
    }
    return cands[cands.len - 1].id;
}

/// One row's sampling distribution (top-k, temperature, min-p, top-p), descending, probabilities in
/// `logit`; null for greedy. Plain and speculative decoding draw from exactly this.
pub fn dist(logits: []const f16, s: Sampling, scratch: []Cand) ?[]Cand {
    if (s.temperature <= 0) return null;
    var cands = if (s.top_k == 0) nearTop(logits, 28 * s.temperature, scratch) else topK(logits, @min(s.top_k, logits.len), scratch);
    const inv_t = 1.0 / s.temperature;
    const top = cands[0].logit;
    var total: f32 = 0;
    for (cands) |*c| {
        c.logit = @exp((c.logit - top) * inv_t);
        total += c.logit;
    }
    var n: usize = cands.len;
    if (s.min_p > 0) {
        const floor = s.min_p * cands[0].logit;
        n = 1;
        while (n < cands.len and cands[n].logit >= floor) n += 1;
    }
    if (s.top_p < 1) {
        var acc: f32 = 0;
        var m: usize = 0;
        while (m < n) : (m += 1) {
            acc += cands[m].logit / total;
            if (acc >= s.top_p) {
                m += 1;
                break;
            }
        }
        n = @max(m, 1);
    }
    cands = cands[0..n];
    return cands;
}

/// Null when draft `dk` stands against the target's `row`: greedy, when it is the argmax; sampled,
/// with the target's probability of it. Otherwise the target's own token, drawn without the draft.
pub fn verifyDraft(row: []const f16, dk: u32, s: Sampling, rng: std.Random, scratch: []Cand) ?u32 {
    const dd = dist(row, s, scratch) orelse {
        const best = argmax(row);
        return if (best == dk) null else best;
    };
    const hit: ?usize = for (dd, 0..) |c, j| {
        if (c.id == dk) break j;
    } else null;
    var total: f32 = 0;
    for (dd) |c| total += c.logit;
    if (hit) |h| {
        if (rng.float(f32) * total < dd[h].logit) return null;
        dd[h].logit = 0;
    }
    return drawFrom(dd, rng);
}

/// The k largest logits, descending.
fn candDesc(_: void, a: Cand, b: Cand) bool {
    return a.logit > b.logit;
}

/// Every logit within `margin` of the largest, descending. At `margin` = 28 x temperature a token
/// left out is e^-28 as likely as the top one: no top-k, without sorting the whole vocabulary.
fn nearTop(logits: []const f16, margin: f32, scratch: []Cand) []Cand {
    var top: f32 = -std.math.inf(f32);
    for (logits) |v| top = @max(top, @as(f32, v));
    var n: usize = 0;
    for (logits, 0..) |v, i| if (@as(f32, v) >= top - margin) {
        scratch[n] = .{ .logit = v, .id = @intCast(i) };
        n += 1;
    };
    std.sort.pdq(Cand, scratch[0..n], {}, candDesc);
    return scratch[0..n];
}

pub fn topK(logits: []const f16, k: usize, scratch: []Cand) []Cand {
    if (k == logits.len) {
        for (logits, scratch[0..logits.len], 0..) |v, *c, i| c.* = .{ .logit = v, .id = @intCast(i) };
        std.sort.pdq(Cand, scratch[0..logits.len], {}, candDesc);
        return scratch[0..logits.len];
    }
    // Insertion into a sorted window of k: cheap for the small k sampling uses.
    var n: usize = 0;
    for (logits, 0..) |v, i| {
        const x: f32 = v;
        if (n == k and x <= scratch[k - 1].logit) continue;
        var j: usize = if (n < k) n else k - 1;
        if (n < k) n += 1;
        while (j > 0 and scratch[j - 1].logit < x) : (j -= 1) scratch[j] = scratch[j - 1];
        scratch[j] = .{ .logit = x, .id = @intCast(i) };
    }
    return scratch[0..n];
}

// ── the model (programs) ──

/// One lane's program over a chunk of layers, and its round procedures by layer index.
const UnitProg = struct {
    prog: rt.Program,
    info: q35.ChunkInfo,
    /// Two lanes: the decode mixer per context (one DeltaNet procedure serves every context) and
    /// the decode MLP. One lane: the whole unit's decode per context (`q35.emitDecode`).
    dm: [][]u32 = &.{},
    df: []u32 = &.{},
    dc: []u32 = &.{},
    /// The three prefill rounds.
    pm: []u32,
    po: []u32,
    pf: []u32,
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    arch: Arch,
    pack: *pack_mod.Pack,
    prefix: []const u8,
    embed: pack_mod.Linear,
    ctxs: []u32,
    /// Lanes splitting every layer: one per Neural Engine (two on an Ultra) when the shapes split.
    /// Each lane has a program per chunk of layers (`units[chunk * lanes + lane]`), its share of the
    /// head and its load of the cores, pinned to its ANE: two loads evaluated at once without a pin
    /// can land on one cluster.
    lanes: u32,
    chunks: usize,
    units: []UnitProg,
    heads: [2]rt.Program,
    head_parts: [2]u32,
    head_proc: [2]u32,
    head_raw_proc: [2]u32,
    /// The weight-free prefill mixers over one lane's heads: `gdn_p64`, and per context the attention
    /// cores over its history slices (`q35.ATT_SLICES`, shortest first); one load per (slot, lane).
    cores: rt.Program,
    gdn_core: u32,
    att_core: [][q35.ATT_SLICES]u32,
    instances: u32,
    /// The pack's DFlash2 drafter: its config, program, tensors (selector codebooks) and the target's
    /// embedding row of its mask token. Null decodes one token per call.
    dcfg: ?dflash.Config = null,
    dprog: ?rt.Program = null,
    dpack: ?*pack_mod.Pack = null,
    selector: dflash.Selector = undefined,
    mask_emb: []f32 = &.{},
    /// Bytes of compiled weights the ANE holds (one copy for every slot).
    weight_bytes: u64,
    /// Ids never sampled (reserved specials) and the first padding row past the tokenizer's last
    /// id (0 = none): `generate.installSuppressMask`'s rule, applied to every logits row.
    suppress_ids: []const u32,
    defined_vocab: u32,
    test_taps: []const u32 = &.{},

    /// Build (or restore from the compile cache) every program of the pack at `dir`, loaded `opts.slots` times.
    pub fn open(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, arch: Arch, opts: Options) !*Model {
        if (!rt.available()) return error.AneUnavailable;
        const self = try allocator.create(Model);
        errdefer allocator.destroy(self);
        self.* = undefined;
        self.allocator = allocator;
        self.arena = .init(allocator);
        errdefer self.arena.deinit();
        const a = self.arena.allocator();
        self.arch = arch;
        self.instances = @max(opts.slots, 1);
        self.weight_bytes = 0;
        self.suppress_ids = try a.dupe(u32, opts.suppress_ids);
        self.defined_vocab = opts.defined_vocab;
        self.dcfg = null;
        self.dprog = null;
        self.dpack = null;
        self.mask_emb = &.{};
        self.lanes = if (opts.lanes > 0) opts.lanes else lanesFor(arch, ane.aneInstanceCount(ane.chipBrand()));
        self.pack = try pack_mod.Pack.open(io, allocator, dir);
        errdefer self.pack.close();
        const pfx = self.pack.layerPrefix() orelse return error.LoadFailed;
        self.prefix = pfx;
        var nb: [256]u8 = undefined;
        self.embed = try pack_mod.Linear.find(self.pack, try std.fmt.bufPrint(&nb, "{s}.embed_tokens", .{pfx}), arch.vocab, arch.hidden);

        self.test_taps = try a.dupe(u32, opts.taps);
        self.ctxs = try a.dupe(u32, opts.ctxs);
        std.mem.sort(u32, self.ctxs, {}, std.sort.asc(u32));
        if (opts.drafter) |d| self.openDrafter(io, d) catch |err| {
            log.warn("[ane] drafter not loaded ({s}): one token per call\n", .{@errorName(err)});
            self.dcfg = null;
        };
        errdefer if (self.dpack) |d| d.close();

        // This build's cache entries: one set per model and configuration (`rt.cacheLineage`). One
        // build at a time: the cache's lineage is process-wide.
        build_mu.lockUncancelable(io);
        defer build_mu.unlock(io);
        var gb: [32]u8 = undefined;
        var vb: [32]u8 = undefined;
        rt.cacheLineage(cacheGroup(&gb, dir), cacheVariant(&vb, self.lanes, self.ctxs, self.taps()));
        defer rt.cacheLineage("", "");
        const starts = try q35.chunkStarts(a, arch, self.lanes, self.ctxs.len, self.taps());
        self.chunks = starts.len;
        self.units = try a.alloc(UnitProg, self.chunks * self.lanes);
        _ = trace();
        const t0 = std.Io.Timestamp.now(io, .awake);
        // Programs build and compile several at once (the ANE compiler runs them in parallel); each
        // one in flight holds its weight blob, so RAM bounds how many.
        const cpus = @max(1, std.Thread.getCpuCount() catch 4);
        const biggest = @max(q35.MAX_PROGRAM_BYTES, @as(u64, arch.vocab) * arch.hidden / self.lanes);
        const lanes: usize = @intCast(std.math.clamp(ane.totalMemBytes() / 8 / @max(biggest, 1), 1, @min(8, cpus)));
        const nunits = self.units.len;
        const B = Build{ .model = self, .io = io, .pfx = pfx, .starts = starts, .threads = @max(1, cpus / lanes), .n = nunits + self.lanes + @as(usize, if (self.dcfg != null) 2 else 1), .progress = opts.progress };
        if (opts.progress) |p| p.begin(.ane_programs, B.n, io_util.nowMsMonotonic(io));
        var build = B;
        const workers = try allocator.alloc(std.Thread, lanes - 1);
        defer allocator.free(workers);
        var spawned: usize = 0;
        for (workers) |*w| {
            w.* = std.Thread.spawn(.{}, Build.run, .{&build}) catch break;
            spawned += 1;
        }
        build.run();
        for (workers[0..spawned]) |w| w.join();
        if (build.failed) |err| {
            for (self.units, 0..) |u, i| if (build.done[i]) u.prog.deinit();
            for (0..self.lanes) |l| if (build.done[nunits + l]) self.heads[l].deinit();
            if (build.done[nunits + self.lanes]) self.cores.deinit();
            if (self.dcfg != null and build.done[nunits + self.lanes + 1]) self.dprog.?.deinit();
            return err;
        }
        errdefer {
            for (self.units) |u| u.prog.deinit();
            for (self.heads[0..self.lanes]) |h| h.deinit();
            self.cores.deinit();
            if (self.dprog) |d| d.deinit();
        }
        for (0..self.chunks) |ci| {
            const info = self.units[ci * self.lanes].info;
            self.weight_bytes += q35.unitBytes(arch, info.first, info.count, 1);
        }
        self.weight_bytes += @as(u64, arch.vocab) * (arch.hidden + 2);
        if (self.dcfg) |dc| self.weight_bytes += dflash.programBytes(dc);
        for (0..self.lanes) |l| {
            self.head_proc[l] = self.heads[l].function("head") catch return error.LoadFailed;
            self.head_raw_proc[l] = self.heads[l].function("head_raw") catch return error.LoadFailed;
        }
        self.gdn_core = if (arch.hasLinear()) self.cores.function("gdn_p64") catch return error.LoadFailed else 0;
        self.att_core = try a.alloc([q35.ATT_SLICES]u32, self.ctxs.len);
        for (self.ctxs, self.att_core) |c, *procs| for (procs, 1..) |*pi, i| {
            pi.* = self.cores.function(q35.attCoreName(&nb, c, @intCast(c * i / q35.ATT_SLICES)) catch unreachable) catch return error.LoadFailed;
        };
        // Entries of this model the build did not use are an older engine's.
        rt.cacheSweep();
        const dt = @divTrunc(t0.untilNow(io, .awake).nanoseconds, std.time.ns_per_ms);
        log.info("[ane] ready: {d} chunks x {d} lane(s) + head, {d} slot(s), contexts {any}, {d} MB of weights, in {d} ms ({d} at once)\n", .{ self.chunks, self.lanes, self.instances, self.ctxs, self.weight_bytes >> 20, dt, lanes });
        return self;
    }

    /// The ANE instance (1-based; 0 lets the system pick) slot `slot`'s load of lane `lane` is pinned
    /// to. Slots alternate, so two slots also use both engines.
    fn pin(self: *const Model, lane: usize, slot: usize) u32 {
        return if (self.lanes == 2) @intCast(1 + (lane + slot) % 2) else 0;
    }

    /// One pin per slot for lane `lane`'s programs.
    fn pins(self: *const Model, lane: usize, buf: []u32) []const u32 {
        for (buf[0..self.instances], 0..) |*h, s| h.* = self.pin(lane, s);
        return buf[0..self.instances];
    }

    /// The programs of a model built by several threads: each takes the next index (units, then
    /// each lane's head, the cores, the drafter).
    const Build = struct {
        model: *Model,
        pfx: []const u8,
        /// Each chunk's first layer.
        starts: []const u32,
        threads: usize,
        n: usize,
        next: std.atomic.Value(usize) = .init(0),
        io: std.Io,
        mu: std.Io.Mutex = .init,
        done: [512]bool = @splat(false),
        failed: ?anyerror = null,
        progress: ?*load_progress.Progress = null,

        /// One more program ready; a cache miss marks the load as compiling.
        fn built(b: *Build, p: rt.Program) void {
            const pr = b.progress orelse return;
            if (!p.cacheHit()) pr.compiling.store(true, .monotonic);
            pr.add(1);
        }

        fn run(b: *Build) void {
            while (true) {
                const i = b.next.fetchAdd(1, .monotonic);
                if (i >= b.n or i >= b.done.len) return;
                b.mu.lockUncancelable(b.io);
                const stop = b.failed != null;
                b.mu.unlock(b.io);
                if (stop) return;
                b.one(i) catch |err| {
                    b.mu.lockUncancelable(b.io);
                    defer b.mu.unlock(b.io);
                    if (b.failed == null) b.failed = err;
                    return;
                };
                b.mu.lockUncancelable(b.io);
                b.done[i] = true;
                b.mu.unlock(b.io);
            }
        }

        fn one(b: *Build, i: usize) !void {
            const m = b.model;
            const A = m.arch;
            const nunits = m.units.len;
            var hb: [16]u32 = undefined;
            if (i == nunits + m.lanes) {
                var blob = try mil.Blob.init(std.heap.c_allocator, 1 << 20);
                defer blob.deinit();
                var g = try mil.Builder.init(m.allocator, &blob);
                defer g.deinit();
                try q35.buildCores(&g, A.share(m.lanes), m.ctxs);
                const text = try g.finish();
                // One load per (slot, lane): a program object runs one call at a time.
                for (0..m.instances) |s| for (0..m.lanes) |l| {
                    hb[s * m.lanes + l] = m.pin(l, s);
                };
                var err: rt.ErrBuf = .{};
                m.cores = rt.Program.create("cores", text, try blob.bytes.toOwnedSlice(std.heap.c_allocator), hb[0 .. m.instances * m.lanes], &err) catch |e| {
                    log.err("[ane] cores: {s}\n", .{err.text()});
                    return e;
                };
                b.built(m.cores);
                log.info("[ane] prefill cores ready ({s}, {d:.1} s)\n", .{ if (m.cores.cacheHit()) "cached" else "compiled", m.cores.compileSeconds() });
                return;
            }
            if (i == nunits + m.lanes + 1) {
                var prog = try dflash.Program.init(m.allocator, m.dcfg.?, m.dpack.?);
                defer prog.deinit();
                try dflash.build(&prog);
                try prog.fill(b.threads);
                const out = try prog.take();
                @memset(hb[0..m.instances], 0);
                var err: rt.ErrBuf = .{};
                m.dprog = rt.Program.create("drafter", out.text, out.blob, hb[0..m.instances], &err) catch |e| {
                    log.err("[ane] drafter: {s}\n", .{err.text()});
                    return e;
                };
                b.built(m.dprog.?);
                log.info("[ane] drafter ready ({s}, {d:.1} s)\n", .{ if (m.dprog.?.cacheHit()) "cached" else "compiled", m.dprog.?.compileSeconds() });
                return;
            }
            if (i >= nunits) {
                const lane: u1 = @intCast(i - nunits);
                var prog = try q35.Program.init(m.allocator, A, m.pack, b.pfx, @as(usize, A.vocab) * (A.hidden + 2) / m.lanes + (16 << 20));
                defer prog.deinit();
                prog.lanes = m.lanes;
                try q35.buildHead(&prog, lane);
                var nb: [16]u8 = undefined;
                m.heads[lane] = try m.compile(&prog, std.fmt.bufPrintSentinel(&nb, "head{d}", .{lane}, 0) catch unreachable, b.threads, m.pins(lane, &hb));
                const half = q35.headHalf(A.vocab, lane, m.lanes);
                m.head_parts[lane] = q35.headParts(half.hi - half.lo, A.hidden);
                b.built(m.heads[lane]);
                log.info("[ane] head {d} ready ({s}, {d:.1} s)\n", .{ lane, if (m.heads[lane].cacheHit()) "cached" else "compiled", m.heads[lane].compileSeconds() });
                return;
            }
            const ci = i / m.lanes;
            const lane: u1 = @intCast(i % m.lanes);
            const first = b.starts[ci];
            const count = (if (ci + 1 < b.starts.len) b.starts[ci + 1] else A.layers) - first;
            var arena = std.heap.ArenaAllocator.init(m.allocator);
            defer arena.deinit();
            var prog = try q35.Program.init(m.allocator, A, m.pack, b.pfx, q35.unitBytes(A, first, count, m.lanes));
            defer prog.deinit();
            prog.lanes = m.lanes;
            const info = try q35.buildUnit(&prog, first, count, lane, m.ctxs, m.taps(), arena.allocator());
            var nb: [32]u8 = undefined;
            const name = std.fmt.bufPrintSentinel(&nb, "L{d:0>2}-{d:0>2}.{d}", .{ first, first + count - 1, lane }, 0) catch unreachable;
            const compiled = try m.compile(&prog, name, b.threads, m.pins(lane, &hb));
            errdefer compiled.deinit();
            b.mu.lockUncancelable(b.io);
            defer b.mu.unlock(b.io);
            const a = m.arena.allocator();
            const u = &m.units[i];
            u.prog = compiled;
            u.info = .{ .first = info.first, .count = info.count, .gdn_j = try a.dupe(u32, info.gdn_j), .att_j = try a.dupe(u32, info.att_j) };
            for ([_]*[]u32{ &u.pm, &u.po, &u.pf }) |list| list.* = try a.alloc(u32, count);
            if (m.fused()) {
                u.dc = try a.alloc(u32, m.ctxs.len);
                for (m.ctxs, u.dc) |c, *pi| pi.* = compiled.function(q35.decodeName(&nb, if (info.att_j.len > 0) c else 0)) catch return error.LoadFailed;
            } else {
                u.dm = try a.alloc([]u32, count);
                u.df = try a.alloc(u32, count);
            }
            for (0..count) |jj| {
                const j: u32 = @intCast(jj);
                const linear = std.mem.indexOfScalar(u32, info.gdn_j, j) != null;
                if (!m.fused()) {
                    u.dm[jj] = try a.alloc(u32, m.ctxs.len);
                    for (m.ctxs, u.dm[jj]) |c, *pi| pi.* = compiled.function(q35.roundName(&nb, .dm, j, if (linear) 0 else c)) catch return error.LoadFailed;
                    u.df[jj] = compiled.function(q35.roundName(&nb, .df, j, 0)) catch return error.LoadFailed;
                }
                for ([_]q35.Round{ .pm, .po, .pf }, [_][]u32{ u.pm, u.po, u.pf }) |r, list| list[jj] = compiled.function(q35.roundName(&nb, r, j, 0)) catch return error.LoadFailed;
            }
            b.built(compiled);
            log.info("[ane] chunk {d}/{d} lane {d} ready ({s}, {d:.1} s)\n", .{ ci + 1, m.chunks, lane, if (compiled.cacheHit()) "cached" else "compiled", compiled.compileSeconds() });
        }
    };

    fn compile(self: *Model, prog: *q35.Program, name: [:0]const u8, threads: usize, hints: []const u32) !rt.Program {
        _ = self;
        try prog.fill(threads);
        const out = try prog.take();
        var err: rt.ErrBuf = .{};
        return rt.Program.create(name, out.text, out.blob, hints, &err) catch |e| {
            log.err("[ane] {s}: {s}\n", .{ name, err.text() });
            return e;
        };
    }

    /// A DFlash2 drafter, checked against the target. Its taps read the residual a round hands on,
    /// which the last layer never does.
    fn openDrafter(self: *Model, io: std.Io, ddir: []const u8) !void {
        const a = self.arena.allocator();
        const json = std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ ddir, "config.json" }), a, .limited(1 << 20)) catch return error.NoDrafter;
        const dc = try dflash.parseConfig(a, json);
        if (dc.block != P or dc.hidden != self.arch.hidden or dc.mask_token >= self.arch.vocab) return error.DrafterMismatch;
        for (dc.taps) |t| if (t + 1 >= self.arch.layers) return error.DrafterMismatch;
        const dp = try pack_mod.Pack.open(io, self.allocator, ddir);
        errdefer dp.close();
        self.selector = try dflash.Selector.of(dp, self.arch.vocab, dc.rank);
        self.mask_emb = try a.alloc(f32, self.arch.hidden);
        self.embed.row(dc.mask_token, self.mask_emb);
        self.dpack = dp;
        self.dcfg = dc;
    }

    pub fn deinit(self: *Model) void {
        for (self.units) |u| u.prog.deinit();
        for (self.heads[0..self.lanes]) |h| h.deinit();
        self.cores.deinit();
        if (self.dprog) |d| d.deinit();
        if (self.dpack) |d| d.close();
        self.pack.close();
        self.arena.deinit();
        self.allocator.destroy(self);
    }

    /// The drafter's tap layers (none without one).
    fn taps(self: *const Model) []const u32 {
        return if (self.dcfg) |dc| dc.taps else self.test_taps;
    }

    /// One lane decodes a unit per round (`UnitProg.dc`); two run two rounds per layer.
    fn fused(self: *const Model) bool {
        return self.lanes == 1;
    }

    /// The head's round in a decode call (its parity picks the residual and partial it reads).
    fn headRound(self: *const Model) usize {
        return if (self.fused()) self.chunks else 2 * self.arch.layers;
    }

    pub fn maxPositions(self: *const Model) u32 {
        return self.ctxs[self.ctxs.len - 1];
    }

    fn ctxIndex(self: *const Model, ctx: u32) usize {
        return for (self.ctxs, 0..) |c, i| {
            if (c == ctx) break i;
        } else unreachable;
    }
};

// ── buffers ──

/// One named buffer: an IOSurface and its layout, read as a MIL tensor of `rank` axes.
const Buf = struct {
    name: []const u8,
    surf: rt.Surface,
    io: rt.Io,
    t: rt.Tensor,
};

fn rankOf(name: []const u8) u8 {
    const pairs = .{
        .{ "x", 4 },        .{ "y", 4 },     .{ "x0", 4 },           .{ "x1", 4 },       .{ "xo", 4 },          .{ "og", 4 },
        .{ "qkv", 4 },      .{ "z", 4 },     .{ "a", 4 },            .{ "b", 4 },        .{ "cos", 2 },         .{ "sin", 2 },
        .{ "mask", 2 },     .{ "oa", 2 },    .{ "gate", 2 },         .{ "conv_sel", 2 }, .{ "conv_sel_out", 2 }, .{ "commit", 3 },
        .{ "commit_last", 3 }, .{ "valid", 3 }, .{ "q", 3 },
    };
    inline for (pairs) |p| if (std.mem.eql(u8, name, p[0])) return p[1];
    if (std.mem.startsWith(u8, name, "tap")) return 4;
    if (std.mem.startsWith(u8, name, "cw") or std.mem.startsWith(u8, name, "normw")) return 2;
    if (std.mem.startsWith(u8, name, "conv")) return 2;
    if (std.mem.startsWith(u8, name, "logits")) return 4;
    if (Engine.isKv(name)) return 4; // (tile, head, ...), see `Hist`
    return 3; // rec, pend, k_new, v_new
}

fn ioDims(io: rt.Io, rank: u8, out: *[4]u32) []const u32 {
    const lanes = [4]u32{ io.batches, io.channels, io.height, io.width };
    @memcpy(out[0..rank], lanes[4 - rank ..]);
    return out[0..rank];
}

fn sameLayout(a: rt.Io, b: rt.Io) bool {
    return a.dtype == b.dtype and a.batches == b.batches and a.channels == b.channels and a.height == b.height and a.width == b.width and
        a.batch_stride == b.batch_stride and a.plane_stride == b.plane_stride and a.row_stride == b.row_stride;
}

const BufList = std.ArrayList(*Buf);

/// `b` as the binding of `io`: the surface, after checking the layouts agree.
fn use(b: *Buf, io: rt.Io) !rt.SurfaceRef {
    if (!sameLayout(b.io, io)) {
        log.err("[ane] {s} is bound where {s} is read, in another layout\n", .{ b.name, io.nameSlice() });
        return error.LoadFailed;
    }
    return b.surf.ref;
}

fn findBuf(list: []const *Buf, name: []const u8) ?*Buf {
    for (list) |b| if (std.mem.eql(u8, b.name, name)) return b;
    return null;
}

pub const Synced = struct { logits: []const f16, cached: usize };

/// One speculative cycle's output: the accepted drafts, then the target's next token.
pub const Cycle = struct { tokens: [P]u32, n: usize, accepted: usize };

/// A call's rounds: every lane's requests (owned by the engine's request cache), cut into rounds
/// by `ends` (round r runs requests [ends[r - 1], ends[r]) of each lane). Every lane finishes a
/// round before any starts the next: each round reads the partial sums all lanes returned from the
/// one before.
const Rounds = struct {
    reqs: [2]std.ArrayList(rt.Request) = .{ .empty, .empty },
    ends: [2]std.ArrayList(u32) = .{ .empty, .empty },

    fn add(self: *Rounds, a: std.mem.Allocator, lane: usize, r: rt.Request) !void {
        try self.reqs[lane].append(a, r);
    }

    /// End the round under construction.
    fn cut(self: *Rounds, a: std.mem.Allocator, lanes: u32) !void {
        for (0..lanes) |l| try self.ends[l].append(a, @intCast(self.reqs[l].items.len));
    }

    fn count(self: *const Rounds) usize {
        return self.ends[0].items.len;
    }

    fn slice(self: *const Rounds, lane: usize, r: usize) []const rt.Request {
        const lo = if (r == 0) 0 else self.ends[lane].items[r - 1];
        return self.reqs[lane].items[lo..self.ends[lane].items[r]];
    }

    fn deinit(self: *Rounds, a: std.mem.Allocator) void {
        for (&self.reqs, &self.ends) |*rs, *es| {
            rs.deinit(a);
            es.deinit(a);
        }
    }
};

/// What a request binds, given how `Engine.resolve` names buffers: the program and its load, the
/// procedure, and where in a call it runs.
const ReqKey = struct { prog: usize, inst: u32, proc: u32, unit: u32, round: u32, par: u1 };

/// A cached request; `ctx_bound` when it binds the KV history or the mask, which a context switch
/// replaces.
const Cached = struct { req: rt.Request, ctx_bound: bool };

/// Spin for up to 2 ms, then sleep, until `v` is no longer `old`; returns the new value. A round
/// takes about a millisecond, so the spin covers the waits of a call and a sleep's wake-up does not
/// land on every round.
fn waitChange(io: std.Io, v: *std.atomic.Value(u32), old: u32) u32 {
    const t0 = std.Io.Timestamp.now(io, .awake);
    var spins: u32 = 0;
    while (true) {
        const cur = v.load(.acquire);
        if (cur != old) return cur;
        spins +%= 1;
        if (spins % 1024 != 0 or t0.untilNow(io, .awake).nanoseconds < 2 * std.time.ns_per_ms) {
            std.atomic.spinLoopHint();
        } else io.futexWaitUncancelable(u32, &v.raw, old);
    }
}

/// The second lane's thread: runs lane 1's share of each round its engine posts.
const LaneThread = struct {
    io: std.Io,
    thread: std.Thread = undefined,
    seq: std.atomic.Value(u32) = .init(0),
    done: std.atomic.Value(u32) = .init(0),
    job: []const rt.Request = &.{},
    failed: bool = false,
    quit: bool = false,

    fn loop(self: *LaneThread) void {
        var seen: u32 = 0;
        while (true) {
            seen = waitChange(self.io, &self.seq, seen);
            if (self.quit) return;
            self.failed = false;
            runReqs(self.job) catch {
                self.failed = true;
            };
            self.done.store(seen, .release);
            self.io.futexWake(u32, &self.done.raw, 1);
        }
    }

    fn post(self: *LaneThread, job: []const rt.Request) u32 {
        self.job = job;
        const prev = self.done.load(.acquire);
        _ = self.seq.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.seq.raw, 1);
        return prev;
    }

    fn stop(self: *LaneThread) void {
        self.quit = true;
        _ = self.seq.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.seq.raw, 1);
        self.thread.join();
    }
};

/// Which call a round belongs to and where: the buffers its procedure's inputs and outputs name.
const Site = struct {
    unit: usize,
    lane: u1,
    kind: EntryKind,
    /// The round's index in its call (two per layer: the mixer, then the MLP).
    round: usize,
    /// The layer (absolute) and its index in the chunk.
    layer: u32,
    j: u32,
    par: u1,
    /// A whole unit's decode (`UnitProg.dc`): one round per unit, taps among its outputs.
    fused: bool = false,
};

/// The server's live prefill gauges across engines: prompt tokens still to feed and fed so far.
pub const PrefillGauges = struct {
    todo: std.atomic.Value(u64) = .init(0),
    done: std.atomic.Value(u64) = .init(0),
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    model: *Model,
    instance: u32,
    io: std.Io,
    inv: []f64,
    /// Per unit (chunk c, lane l at `c * lanes + l`): DeltaNet states by parity (a call reads set
    /// `par`, writes the other), the KV history at the current context, a call's K / V rows per
    /// kind, and each DeltaNet layer's parameters for the prefill core.
    st: [][2]BufList,
    kv: []BufList,
    outs: [][2]BufList,
    params: []BufList,
    /// Per kind (decode, prefill): each lane's residual and partial sum, ping-ponging by round, the
    /// zero partial of a call's first round, lane 0's residual after each drafter tap layer, the
    /// inputs every round shares (cos, sin, conv_sel, commit, commit_last, conv_sel_out, valid) and
    /// the history mask.
    res: [2][2][2]?*Buf = @splat(@splat(@splat(null))),
    part: [2][2][2]?*Buf = @splat(@splat(@splat(null))),
    zero: [2]?*Buf = .{ null, null },
    tapbufs: [2]BufList = .{ .empty, .empty },
    shared: [2]BufList = .{ .empty, .empty },
    mask: [2]?*Buf = .{ null, null },
    /// Prefill: each lane's projections and core outputs.
    pf: [2]BufList = .{ .empty, .empty },
    /// Every request the plans run, built once per binding (a plan is a list of them).
    reqs: std.AutoHashMapUnmanaged(ReqKey, Cached) = .empty,
    /// Requests outside the cache (the heads).
    owned: std.ArrayList(rt.Request) = .empty,
    /// The head over a decode call's last round, each lane's logits parts (lane 0's first).
    head_round: Rounds = .{},
    logit_parts: []*Buf = &.{},
    /// Decode rounds per (context, parity); prefill rounds per (parity, history slice), at the
    /// current context.
    plans: []?Rounds,
    pplans: [2][q35.ATT_SLICES]?Rounds = @splat(@splat(null)),
    lane1: ?*LaneThread = null,
    ctx: u32 = 0,
    par: u1 = 0,
    pos: u32 = 0,
    pending: u32 = 0,
    /// KV rows present in the cache.
    hi: u32 = 0,
    /// Tokens the model state holds, in order (prompt-prefix reuse across requests).
    fed: std.ArrayList(u32) = .empty,
    scratch: []Cand,
    /// The last head call's logits as P contiguous rows, gathered on first use.
    rows: []f16,
    rows_ready: bool = false,
    /// The server's live gauges (null: none), and what this engine counted into them.
    gauges: ?*PrefillGauges = null,
    gauged: [2]u64 = .{ 0, 0 },
    emb_scratch: []f32,
    cache: ?*prefix.Cache = null,
    node: ?*prefix.Node = null,
    stride: u32 = 0,
    next_cp: u32 = 0,
    /// The drafter (`Model.dprog` on this slot's instance), each lane's head over its hidden
    /// states, and per kind the residual buffer of each drafter tap; null decodes one token per call.
    drafter: ?dflash.Drafter = null,
    head_raw: Rounds = .{},
    taps: [2][]rt.Tensor = .{ &.{}, &.{} },
    feat: []f16 = &.{},

    /// Slot `instance` of `model`: its buffers at the smallest context, an empty state at position 0.
    pub fn open(model: *Model, io: std.Io, instance: u32) !*Engine {
        const allocator = model.allocator;
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .arena = .init(allocator),
            .model = model,
            .instance = instance,
            .io = io,
            .inv = undefined,
            .st = undefined,
            .kv = undefined,
            .outs = undefined,
            .params = undefined,
            .plans = undefined,
            .scratch = undefined,
            .rows = undefined,
            .emb_scratch = undefined,
        };
        errdefer self.release();
        const a = self.arena.allocator();
        const A = model.arch;
        self.inv = try a.alloc(f64, A.rot / 2);
        ropeInv(self.inv, A.rot, A.rope_theta);
        self.scratch = try a.alloc(Cand, A.vocab);
        self.rows = try a.alloc(f16, @as(usize, P) * A.vocab);
        self.emb_scratch = try a.alloc(f32, A.hidden);
        const nu = model.units.len;
        self.st = try a.alloc([2]BufList, nu);
        self.outs = try a.alloc([2]BufList, nu);
        self.kv = try a.alloc(BufList, nu);
        self.params = try a.alloc(BufList, nu);
        for (self.st, self.outs, self.kv, self.params) |*s, *o, *k, *pr| {
            s.* = .{ .empty, .empty };
            o.* = .{ .empty, .empty };
            k.* = .empty;
            pr.* = .empty;
        }
        self.plans = try a.alloc(?Rounds, model.ctxs.len * 2);
        @memset(self.plans, null);
        try self.bindKv(model.ctxs[0]);
        // Every buffer but the larger contexts' KV exists once these are built (the registry bills them).
        for (0..2) |par| {
            _ = try self.plan(0, @intCast(par));
            _ = try self.prefillPlan(@intCast(par), 0);
        }
        try self.bindHead();
        for (0..nu) |u| for (model.units[u].info.gdn_j) |j| try self.fillParams(u, j);
        if (model.dcfg) |dc| try self.openDrafter(dc);
        if (model.lanes == 2) {
            const lt = try a.create(LaneThread);
            lt.* = .{ .io = io };
            lt.thread = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, LaneThread.loop, .{lt});
            self.lane1 = lt;
        }
        return self;
    }

    fn openDrafter(self: *Engine, dc: dflash.Config) !void {
        const m = self.model;
        const a = self.arena.allocator();
        const dp = m.dprog.?;
        const draft = try dflash.bind(a, dp, self.instance, dp.function("draft") catch return error.LoadFailed);
        errdefer dflash.release(draft);
        const ctx64 = try dflash.bind(a, dp, self.instance, dp.function("ctx64") catch return error.LoadFailed);
        errdefer dflash.release(ctx64);
        var o: usize = 0;
        for (0..m.lanes) |l| {
            var err: rt.ErrBuf = .{};
            var outs: [32]rt.SurfaceRef = undefined;
            for (self.logit_parts[o..][0..m.head_parts[l]], 0..) |b, i| outs[i] = b.surf.ref;
            o += m.head_parts[l];
            try self.own(&self.head_raw, l, try m.heads[l].request(self.instance, m.head_raw_proc[l], &.{draft.surf("hidden", true)}, outs[0..m.head_parts[l]], &err));
        }
        try self.head_raw.cut(self.allocator, m.lanes);
        var nb: [16]u8 = undefined;
        for (&self.taps, 0..) |*list, k| {
            list.* = try a.alloc(rt.Tensor, dc.taps.len);
            for (dc.taps, list.*) |l, *t| t.* = (findBuf(self.tapbufs[k].items, std.fmt.bufPrint(&nb, "tap{d}", .{l}) catch unreachable) orelse return error.LoadFailed).t;
        }
        self.feat = try a.alloc(f16, TP * @as(usize, dc.taps.len) * dc.hidden);
        self.drafter = try dflash.Drafter.init(self.allocator, dc, draft, ctx64);
    }

    /// The KV history and both masks at context `ctx`, from the attention rounds' inputs.
    fn bindKv(self: *Engine, ctx: u32) !void {
        const m = self.model;
        self.dropCtxRequests();
        const ci = m.ctxIndex(ctx);
        for (m.units, 0..) |u, ui| for (u.info.att_j, 0..) |j, n| {
            // A fused unit's procedure reads every attention layer's history at once.
            if (m.fused() and n > 0) break;
            const proc = if (m.fused()) u.dc[ci] else u.dm[j][ci];
            for (0..u.prog.ioCount(proc, false)) |i| {
                const io = try u.prog.io(proc, false, @intCast(i));
                const name = io.nameSlice();
                if (isKv(name)) {
                    _ = try self.listBuf(&self.kv[ui], io, name);
                } else if (std.mem.eql(u8, name, "mask")) {
                    for (&self.mask) |*mk| _ = try self.slotBuf(mk, io, "mask");
                }
            }
        };
        self.ctx = ctx;
        self.dropPlans();
    }

    /// The head round: every lane's `head` over a decode call's last residual and partials.
    fn bindHead(self: *Engine) !void {
        const m = self.model;
        const a = self.arena.allocator();
        var total: usize = 0;
        for (m.head_parts[0..m.lanes]) |n| total += n;
        self.logit_parts = try a.alloc(*Buf, total);
        const site: Site = .{ .unit = 0, .lane = 0, .kind = .decode, .round = m.headRound(), .layer = m.arch.layers, .j = 0, .par = 0, .fused = m.fused() };
        var o: usize = 0;
        for (0..m.lanes) |l| {
            const h = m.heads[l];
            var s = site;
            s.lane = @intCast(l);
            var ins: [3]rt.SurfaceRef = undefined;
            const ni = h.ioCount(m.head_proc[l], false);
            for (0..ni) |i| {
                const io = try h.io(m.head_proc[l], false, @intCast(i));
                ins[i] = try use(try self.resolve(s, io, false), io);
            }
            var outs: [32]rt.SurfaceRef = undefined;
            for (0..m.head_parts[l]) |i| {
                const io = try h.io(m.head_proc[l], true, @intCast(i));
                self.logit_parts[o + i] = try self.newBuf(io, io.nameSlice());
                outs[i] = self.logit_parts[o + i].surf.ref;
            }
            o += m.head_parts[l];
            var err: rt.ErrBuf = .{};
            try self.own(&self.head_round, l, try h.request(self.instance, m.head_proc[l], ins[0..ni], outs[0..m.head_parts[l]], &err));
        }
        try self.head_round.cut(self.allocator, m.lanes);
    }

    /// The buffer `io` (an input, or an output when `out`) of a round at `site` binds: the residual
    /// and partials ping-pong by round, a tap layer's residual goes to its own buffer.
    fn resolve(self: *Engine, s: Site, io: rt.Io, out: bool) !*Buf {
        const name = io.nameSlice();
        const k = @intFromEnum(s.kind);
        const pp = s.round % 2;
        var nb: [24]u8 = undefined;
        if (out and std.mem.startsWith(u8, name, "tap")) return self.listBuf(&self.tapbufs[k], io, name);
        if (!out and std.mem.eql(u8, name, "x")) {
            // A mixer round's residual came from the previous MLP round; after a tap layer, lane 0's
            // came from its tap buffer (a fused unit hands the tap out beside it).
            if (!s.fused and s.lane == 0 and s.round % 2 == 1 and s.layer > 0 and self.isTap(s.layer - 1))
                return self.listBuf(&self.tapbufs[k], io, std.fmt.bufPrint(&nb, "tap{d}", .{s.layer - 1}) catch unreachable);
            return self.slotBuf(&self.res[k][s.lane][pp], io, "x");
        }
        if (out and std.mem.eql(u8, name, "xo")) {
            if (!s.fused and s.lane == 0 and s.round % 2 == 0 and s.layer > 0 and self.isTap(s.layer - 1))
                return self.listBuf(&self.tapbufs[k], io, std.fmt.bufPrint(&nb, "tap{d}", .{s.layer - 1}) catch unreachable);
            return self.slotBuf(&self.res[k][s.lane][1 - pp], io, "x");
        }
        if (!out and (std.mem.eql(u8, name, "p0") or std.mem.eql(u8, name, "p1"))) {
            if (s.round == 0) return self.slotBuf(&self.zero[k], io, "zero");
            return self.slotBuf(&self.part[k][name[1] - '0'][1 - pp], io, "p");
        }
        if (out and std.mem.eql(u8, name, "po")) return self.slotBuf(&self.part[k][s.lane][pp], io, "p");
        if (isState(name)) {
            if (out) return self.listBuf(&self.st[s.unit][1 - s.par], io, name[0 .. name.len - 4]);
            return self.listBuf(&self.st[s.unit][s.par], io, name);
        }
        if (isKv(name)) return findBuf(self.kv[s.unit].items, name) orelse error.LoadFailed;
        if (std.mem.eql(u8, name, "mask")) return self.mask[k] orelse error.LoadFailed;
        if (std.mem.endsWith(u8, name, "_new")) return self.listBuf(&self.outs[s.unit][k], io, name);
        if (std.mem.eql(u8, name, "gate_in")) return self.listBuf(&self.pf[s.lane], io, "gate");
        for ([_][]const u8{ "qkv", "z", "a", "b", "q", "gate", "og", "oa" }) |pn| if (std.mem.eql(u8, name, pn)) return self.listBuf(&self.pf[s.lane], io, name);
        return self.listBuf(&self.shared[k], io, name);
    }

    fn isTap(self: *const Engine, l: u32) bool {
        return std.mem.indexOfScalar(u32, self.model.taps(), l) != null;
    }

    /// `r` into `rounds` for lane `lane`, owned by the engine.
    fn own(self: *Engine, rounds: *Rounds, lane: usize, r: rt.Request) !void {
        self.owned.append(self.allocator, r) catch |err| {
            r.deinit();
            return err;
        };
        try rounds.add(self.allocator, lane, r);
    }

    fn keyOf(prog: rt.Program, inst: u32, proc: u32, s: Site) ReqKey {
        return .{ .prog = @intFromPtr(prog.handle), .inst = inst, .proc = proc, .unit = @intCast(s.unit), .round = @intCast(s.round), .par = s.par };
    }

    fn remember(self: *Engine, key: ReqKey, r: rt.Request, ctx_bound: bool) !rt.Request {
        self.reqs.put(self.allocator, key, .{ .req = r, .ctx_bound = ctx_bound }) catch |err| {
            r.deinit();
            return err;
        };
        return r;
    }

    /// Requests bound to the KV history or the mask go with a context switch.
    fn dropCtxRequests(self: *Engine) void {
        var it = self.reqs.iterator();
        var gone: std.ArrayList(ReqKey) = .empty;
        defer gone.deinit(self.allocator);
        while (it.next()) |e| if (e.value_ptr.ctx_bound) {
            e.value_ptr.req.deinit();
            gone.append(self.allocator, e.key_ptr.*) catch {};
        };
        for (gone.items) |k| _ = self.reqs.remove(k);
    }

    /// A request for procedure `proc` of `prog` at `site`, every input and output resolved.
    fn request(self: *Engine, prog: rt.Program, inst: u32, proc: u32, s: Site) !rt.Request {
        const key = keyOf(prog, inst, proc, s);
        if (self.reqs.get(key)) |c| return c.req;
        var bound = false;
        var ins: [64]rt.SurfaceRef = undefined;
        var outs: [64]rt.SurfaceRef = undefined;
        const ni = prog.ioCount(proc, false);
        const no = prog.ioCount(proc, true);
        if (ni > ins.len or no > outs.len) return error.LoadFailed;
        for (0..ni) |i| {
            const io = try prog.io(proc, false, @intCast(i));
            if (isKv(io.nameSlice()) or std.mem.eql(u8, io.nameSlice(), "mask")) bound = true;
            ins[i] = try use(try self.resolve(s, io, false), io);
        }
        for (0..no) |i| {
            const io = try prog.io(proc, true, @intCast(i));
            outs[i] = try use(try self.resolve(s, io, true), io);
        }
        var err: rt.ErrBuf = .{};
        const r = prog.request(inst, proc, ins[0..ni], outs[0..no], &err) catch |x| {
            log.err("[ane] bind: {s}\n", .{err.text()});
            return x;
        };
        return self.remember(key, r, bound);
    }

    /// The decode rounds at context index `ci` and parity `par`: per layer the mixer round, then the
    /// MLP round (one round per unit with one lane).
    fn plan(self: *Engine, ci: usize, par: u1) !*const Rounds {
        const key = ci * 2 + par;
        if (self.plans[key]) |*p| return p;
        const m = self.model;
        var rs: Rounds = .{};
        errdefer rs.deinit(self.allocator);
        for (0..m.chunks) |c| {
            const info = m.units[c * m.lanes].info;
            if (m.fused()) {
                const s: Site = .{ .unit = c, .lane = 0, .kind = .decode, .round = c, .layer = info.first, .j = 0, .par = par, .fused = true };
                try rs.add(self.allocator, 0, try self.request(m.units[c].prog, self.instance, m.units[c].dc[ci], s));
                try rs.cut(self.allocator, 1);
                continue;
            }
            for (0..info.count) |jj| {
                const j: u32 = @intCast(jj);
                const layer = info.first + j;
                for ([_]bool{ true, false }) |mixer| {
                    for (0..m.lanes) |l| {
                        const u = m.units[c * m.lanes + l];
                        const s: Site = .{ .unit = c * m.lanes + l, .lane = @intCast(l), .kind = .decode, .round = 2 * layer + @intFromBool(!mixer), .layer = layer, .j = j, .par = par };
                        try rs.add(self.allocator, l, try self.request(u.prog, self.instance, if (mixer) u.dm[jj][ci] else u.df[jj], s));
                    }
                    try rs.cut(self.allocator, m.lanes);
                }
            }
        }
        self.plans[key] = rs;
        return &self.plans[key].?;
    }

    /// The prefill rounds at parity `par` for a block starting at `p0`, at the current context:
    /// per layer the projections, the lane's mixer core and the out projection, then the MLP. The
    /// attention cores read the shortest history slice holding p0 positions.
    fn prefillPlan(self: *Engine, par: u1, p0: u32) !*const Rounds {
        const slice: usize = (@max(@as(u64, p0) * q35.ATT_SLICES, 1) + self.ctx - 1) / self.ctx - 1;
        if (self.pplans[par][slice]) |*p| return p;
        const m = self.model;
        const ai = m.ctxIndex(self.ctx);
        var rs: Rounds = .{};
        errdefer rs.deinit(self.allocator);
        for (0..m.chunks) |c| {
            const info = m.units[c * m.lanes].info;
            for (0..info.count) |jj| {
                const j: u32 = @intCast(jj);
                const layer = info.first + j;
                const linear = std.mem.indexOfScalar(u32, info.gdn_j, j) != null;
                for (0..m.lanes) |l| {
                    const ui = c * m.lanes + l;
                    const u = m.units[ui];
                    const s: Site = .{ .unit = ui, .lane = @intCast(l), .kind = .prefill, .round = 2 * layer, .layer = layer, .j = j, .par = par };
                    try rs.add(self.allocator, l, try self.request(u.prog, self.instance, u.pm[jj], s));
                    try rs.add(self.allocator, l, try self.coreRequest(s, if (linear) m.gdn_core else m.att_core[ai][slice], linear));
                    try rs.add(self.allocator, l, try self.request(u.prog, self.instance, u.po[jj], s));
                }
                try rs.cut(self.allocator, m.lanes);
                for (0..m.lanes) |l| {
                    const ui = c * m.lanes + l;
                    const u = m.units[ui];
                    const s: Site = .{ .unit = ui, .lane = @intCast(l), .kind = .prefill, .round = 2 * layer + 1, .layer = layer, .j = j, .par = par };
                    try rs.add(self.allocator, l, try self.request(u.prog, self.instance, u.pf[jj], s));
                }
                try rs.cut(self.allocator, m.lanes);
            }
        }
        self.pplans[par][slice] = rs;
        return &self.pplans[par][slice].?;
    }

    /// A lane's mixer core for layer j of its unit: the cores name a layer's states, parameters and
    /// KV without its index.
    fn coreRequest(self: *Engine, s: Site, proc: u32, linear: bool) !rt.Request {
        const m = self.model;
        const key = keyOf(m.cores, self.instance * m.lanes + s.lane, proc, s);
        if (self.reqs.get(key)) |c| return c.req;
        var ins: [24]rt.SurfaceRef = undefined;
        var outs: [8]rt.SurfaceRef = undefined;
        var nb: [24]u8 = undefined;
        const ni = m.cores.ioCount(proc, false);
        const no = m.cores.ioCount(proc, true);
        for (0..ni) |i| {
            const io = try m.cores.io(proc, false, @intCast(i));
            const name = io.nameSlice();
            const b: *Buf = if (linear) blk: {
                if (std.mem.eql(u8, name, "conv") or std.mem.eql(u8, name, "rec") or std.mem.eql(u8, name, "pend"))
                    break :blk try self.listBuf(&self.st[s.unit][s.par], io, std.fmt.bufPrint(&nb, "{s}{d}", .{ name, s.j }) catch unreachable);
                if (isParam(name)) break :blk try self.listBuf(&self.params[s.unit], io, std.fmt.bufPrint(&nb, "{s}{d}", .{ name, s.j }) catch unreachable);
                break :blk try self.resolve(s, io, false);
            } else blk: {
                if (std.mem.eql(u8, name, "k") or std.mem.eql(u8, name, "v"))
                    break :blk findBuf(self.kv[s.unit].items, std.fmt.bufPrint(&nb, "{s}{d}", .{ name, s.j }) catch unreachable) orelse return error.LoadFailed;
                if (std.mem.eql(u8, name, "k_new") or std.mem.eql(u8, name, "v_new"))
                    break :blk try self.listBuf(&self.outs[s.unit][1], io, std.fmt.bufPrint(&nb, "{c}{d}_new", .{ name[0], s.j }) catch unreachable);
                break :blk try self.resolve(s, io, false);
            };
            ins[i] = try use(b, io);
        }
        for (0..no) |i| {
            const io = try m.cores.io(proc, true, @intCast(i));
            const name = io.nameSlice();
            const b: *Buf = if (std.mem.eql(u8, name, "o"))
                try self.listBuf(&self.pf[s.lane], io, if (linear) "og" else "oa")
            else
                try self.listBuf(&self.st[s.unit][1 - s.par], io, std.fmt.bufPrint(&nb, "{s}{d}", .{ name[0 .. name.len - 4], s.j }) catch unreachable);
            outs[i] = try use(b, io);
        }
        var err: rt.ErrBuf = .{};
        const r = m.cores.request(self.instance * m.lanes + s.lane, proc, ins[0..ni], outs[0..no], &err) catch |x| {
            log.err("[ane] bind core: {s}\n", .{err.text()});
            return x;
        };
        return self.remember(key, r, !linear);
    }

    fn isParam(name: []const u8) bool {
        for ([_][]const u8{ "cw", "neg_a", "dt", "normw" }) |p| if (std.mem.eql(u8, name, p)) return true;
        return false;
    }

    /// Unit u's DeltaNet layer j: the conv taps, decay, dt bias and gated-norm weight of its lane's heads.
    fn fillParams(self: *Engine, u: usize, j: u32) !void {
        const m = self.model;
        const A = m.arch;
        const H = A.share(m.lanes);
        const lane: u32 = @intCast(u % m.lanes);
        const l = m.units[u].info.first + j;
        var b: [160]u8 = undefined;
        var nb: [24]u8 = undefined;
        const T = struct {
            fn get(pk: *const pack_mod.Pack, buf: []u8, pfx: []const u8, layer: u32, what: []const u8) !pack_mod.Tensor {
                return pk.get(std.fmt.bufPrint(buf, "{s}.layers.{d}.linear_attn.{s}", .{ pfx, layer, what }) catch return error.MissingWeight) orelse error.MissingWeight;
            }
        };
        const cw = try T.get(m.pack, &b, m.prefix, l, "conv1d.weight");
        const cwb = (findBuf(self.params[u].items, std.fmt.bufPrint(&nb, "cw{d}", .{j}) catch unreachable) orelse return error.LoadFailed).t;
        const ch = [_][2]u32{ .{ lane * H.kd(), H.kd() }, .{ A.kd() + lane * H.kd(), H.kd() }, .{ 2 * A.kd() + lane * H.vd(), H.vd() } };
        for (0..4) |t| {
            var o: u32 = 0;
            for (ch) |r| for (r[0]..r[0] + r[1]) |c| {
                cwb.f16At(&.{ @intCast(t), o }).* = @floatCast(if (cw.dim(1) == 4) cw.at(c * 4 + t) else cw.at(t * A.cdim() + c));
                o += 1;
            };
        }
        const alog = try T.get(m.pack, &b, m.prefix, l, "A_log");
        const dtb = try T.get(m.pack, &b, m.prefix, l, "dt_bias");
        const na = (findBuf(self.params[u].items, std.fmt.bufPrint(&nb, "neg_a{d}", .{j}) catch unreachable) orelse return error.LoadFailed).t;
        const dt = (findBuf(self.params[u].items, std.fmt.bufPrint(&nb, "dt{d}", .{j}) catch unreachable) orelse return error.LoadFailed).t;
        for (0..H.nv) |h| {
            na.f16At(&.{ @intCast(h), 0, 0 }).* = @floatCast(-@exp(alog.at(lane * H.nv + h)));
            dt.f16At(&.{ @intCast(h), 0, 0 }).* = @floatCast(dtb.at(lane * H.nv + h));
        }
        const nw = try T.get(m.pack, &b, m.prefix, l, "norm.weight");
        const nt = (findBuf(self.params[u].items, std.fmt.bufPrint(&nb, "normw{d}", .{j}) catch unreachable) orelse return error.LoadFailed).t;
        for (0..A.dv) |d| nt.f16At(&.{ 0, @intCast(d) }).* = @floatCast(nw.at(d));
    }

    /// The KV history inputs: `k<j>` and `v<j>` (not `valid`, not `k<j>_new`).
    fn isKv(name: []const u8) bool {
        if (name.len < 2 or (name[0] != 'k' and name[0] != 'v')) return false;
        for (name[1..]) |ch| if (!std.ascii.isDigit(ch)) return false;
        return true;
    }

    fn isState(name: []const u8) bool {
        const base = if (std.mem.endsWith(u8, name, "_out")) name[0 .. name.len - 4] else name;
        for ([_][]const u8{ "conv", "rec", "pend" }) |p| {
            if (std.mem.startsWith(u8, base, p) and base.len > p.len and std.ascii.isDigit(base[p.len])) return true;
        }
        return false;
    }

    fn newBuf(self: *Engine, io: rt.Io, name: []const u8) !*Buf {
        const a = self.arena.allocator();
        const b = try a.create(Buf);
        const surf = try rt.Surface.create(io.bytes());
        errdefer surf.release();
        var d: [4]u32 = undefined;
        b.* = .{ .name = try a.dupe(u8, name), .surf = surf, .io = io, .t = try rt.Tensor.of(surf.base, io, ioDims(io, rankOf(name), &d)) };
        return b;
    }

    fn slotBuf(self: *Engine, slot: *?*Buf, io: rt.Io, name: []const u8) !*Buf {
        if (slot.*) |b| {
            if (!sameLayout(b.io, io)) {
                log.err("[ane] {s}: two programs read it in different layouts\n", .{name});
                return error.LoadFailed;
            }
            return b;
        }
        slot.* = try self.newBuf(io, name);
        return slot.*.?;
    }

    fn listBuf(self: *Engine, list: *BufList, io: rt.Io, name: []const u8) !*Buf {
        if (findBuf(list.items, name)) |b| {
            if (!sameLayout(b.io, io)) {
                log.err("[ane] {s}: two programs read it in different layouts\n", .{name});
                return error.LoadFailed;
            }
            return b;
        }
        const b = try self.newBuf(io, name);
        try list.append(self.arena.allocator(), b);
        return b;
    }

    fn release(self: *Engine) void {
        if (self.lane1) |lt| lt.stop();
        self.lane1 = null;
        self.dropPlans();
        self.head_round.deinit(self.allocator);
        self.head_raw.deinit(self.allocator);
        var it = self.reqs.valueIterator();
        while (it.next()) |c| c.req.deinit();
        self.reqs.deinit(self.allocator);
        for (self.owned.items) |r| r.deinit();
        self.owned.deinit(self.allocator);
        const S = struct {
            fn list(l: []const *Buf) void {
                for (l) |b| b.surf.release();
            }
            fn slot(b: ?*Buf) void {
                if (b) |x| x.surf.release();
            }
        };
        for (self.st) |s| for (s) |l| S.list(l.items);
        for (self.outs) |o| for (o) |l| S.list(l.items);
        for (self.kv) |l| S.list(l.items);
        for (self.params) |l| S.list(l.items);
        for (self.res) |k| for (k) |l| for (l) |b| S.slot(b);
        for (self.part) |k| for (k) |l| for (l) |b| S.slot(b);
        for (self.zero) |b| S.slot(b);
        for (self.mask) |b| S.slot(b);
        for (self.tapbufs) |l| S.list(l.items);
        for (self.shared) |l| S.list(l.items);
        for (self.pf) |l| S.list(l.items);
        S.list(self.logit_parts);
        if (self.drafter) |*d| {
            dflash.release(d.draft);
            dflash.release(d.ctx64);
            d.deinit();
        }
        self.fed.deinit(self.allocator);
        self.arena.deinit();
    }

    pub fn deinit(self: *Engine) void {
        if (self.cache) |c| {
            c.mu.lockUncancelable(self.io);
            self.follow(null);
            c.mu.unlock(self.io);
        }
        self.release();
        self.allocator.destroy(self);
    }

    /// Bytes of this engine's own buffers.
    pub fn bufferBytes(self: *const Engine) u64 {
        var n: u64 = 0;
        const S = struct {
            fn list(l: []const *Buf) u64 {
                var s: u64 = 0;
                for (l) |b| s += b.surf.bytes;
                return s;
            }
            fn slot(b: ?*Buf) u64 {
                return if (b) |x| x.surf.bytes else 0;
            }
        };
        for (self.st) |s| for (s) |l| {
            n += S.list(l.items);
        };
        for (self.outs) |o| for (o) |l| {
            n += S.list(l.items);
        };
        for (self.kv) |l| n += S.list(l.items);
        for (self.params) |l| n += S.list(l.items);
        for (self.res) |k| for (k) |l| for (l) |b| {
            n += S.slot(b);
        };
        for (self.part) |k| for (k) |l| for (l) |b| {
            n += S.slot(b);
        };
        for (self.tapbufs) |l| n += S.list(l.items);
        for (self.shared) |l| n += S.list(l.items);
        for (self.pf) |l| n += S.list(l.items);
        n += S.list(self.logit_parts);
        if (self.drafter) |*d| n += d.bufferBytes();
        return n;
    }

    fn dropPlans(self: *Engine) void {
        for (self.plans) |*p| if (p.*) |*rs| {
            rs.deinit(self.allocator);
            p.* = null;
        };
        for (&self.pplans) |*slices| for (slices) |*p| if (p.*) |*rs| {
            rs.deinit(self.allocator);
            p.* = null;
        };
    }

    /// Run a call's rounds: lane 0 here, lane 1 on its thread, both finishing a round before the next.
    fn runRounds(self: *Engine, rs: *const Rounds) !void {
        const t0 = if (trace()) std.Io.Timestamp.now(self.io, .awake) else undefined;
        for (0..rs.count()) |r| {
            if (self.lane1) |lt| {
                const prev = lt.post(rs.slice(1, r));
                const mine = runReqs(rs.slice(0, r));
                _ = waitChange(self.io, &lt.done, prev);
                try mine;
                if (lt.failed) return error.RunFailed;
            } else try runReqs(rs.slice(0, r));
        }
        self.rows_ready = false;
        if (trace()) log.info("[ane-trace] {d} rounds {d:.2} ms\n", .{ rs.count(), @as(f64, @floatFromInt(t0.untilNow(self.io, .awake).nanoseconds)) / 1e6 });
    }

    // ── KV ──

    /// Switch to another context of the ladder: new KV and mask buffers, rows [0, keep) moved.
    fn resize(self: *Engine, ctx: u32) !void {
        const keep = @min(self.hi, ctx);
        const from = self.ctx;
        const old_kv = try self.allocator.dupe(BufList, self.kv);
        defer self.allocator.free(old_kv);
        const old_mask = self.mask;
        for (self.kv) |*l| l.* = .empty;
        self.mask = .{ null, null };
        self.bindKv(ctx) catch |err| {
            for (self.kv) |l| for (l.items) |b| b.surf.release();
            for (self.mask) |b| if (b) |x| x.surf.release();
            @memcpy(self.kv, old_kv);
            self.mask = old_mask;
            self.ctx = from;
            return err;
        };
        for (self.kv, old_kv) |new, old| for (old.items) |ob| {
            Hist.of(ob).copy(Hist.of(findBuf(new.items, ob.name).?), keep);
            ob.surf.release();
        };
        for (old_mask) |b| if (b) |x| x.surf.release();
        self.hi = keep;
        log.info("[ane] context {d} -> {d} at position {d}\n", .{ from, ctx, self.pos });
    }

    /// Grow to the smallest context that holds `need` positions; false when none does.
    fn fit(self: *Engine, need: u32) !bool {
        if (need <= self.ctx) return true;
        for (self.model.ctxs) |c| if (c >= need) {
            try self.resize(c);
            return true;
        };
        return false;
    }

    /// Fresh DeltaNet states at position 0; `shrink` returns to the smallest (fastest) context.
    pub fn reset(self: *Engine, shrink: bool) void {
        for (self.st) |s| for (s[self.par].items) |b| b.t.zero();
        self.pos = 0;
        self.pending = 0;
        self.hi = 0;
        self.fed.clearRetainingCapacity();
        if (self.drafter) |*d| d.reset();
        if (self.cache) |c| {
            c.mu.lockUncancelable(self.io);
            defer c.mu.unlock(self.io);
            self.follow(null);
        }
        self.next_cp = self.stride;
        if (shrink and self.ctx != self.model.ctxs[0]) self.resize(self.model.ctxs[0]) catch |err| log.warn("[ane] shrink failed: {s}\n", .{@errorName(err)});
    }

    // ── calls ──

    /// The embeddings of `ids` into every lane's first residual (padding rows repeat the last id).
    fn put(self: *Engine, kind: EntryKind, ids: []const u32, rows: u32) void {
        const H = self.model.arch.hidden;
        const k = @intFromEnum(kind);
        for (0..rows) |r| {
            const id = ids[@min(r, ids.len - 1)];
            self.model.embed.row(id, self.emb_scratch);
            for (0..self.model.lanes) |l| {
                const x = self.res[k][l][0].?.t;
                for (0..H) |h| x.f16At(&.{ 0, @intCast(h), 0, @intCast(r) }).* = @floatCast(self.emb_scratch[h]);
            }
        }
    }

    /// Positions, the DeltaNet commit of the last decode call's `k` accepted rows, and the history
    /// mask of a call of `kind`.
    fn common(self: *Engine, kind: EntryKind, p0: u32, n: u32, rows: u32, k: u32) void {
        const list = self.shared[@intFromEnum(kind)].items;
        if (findBuf(list, "cos")) |cb| {
            const sin = findBuf(list, "sin").?.t;
            const half = self.inv.len;
            for (0..rows) |r| {
                const p: f64 = @floatFromInt(rowPos(p0, n, @intCast(r)));
                for (0..2 * half) |c| {
                    const ang = p * self.inv[c % half];
                    cb.t.f16At(&.{ @intCast(r), @intCast(c) }).* = @floatCast(@cos(ang));
                    sin.f16At(&.{ @intCast(r), @intCast(c) }).* = @floatCast(@sin(ang));
                }
            }
        }
        if (findBuf(list, "conv_sel")) |sel| {
            sel.t.zero();
            for (0..3) |i| sel.t.f16At(&.{ @intCast(i), @intCast(k + i) }).* = 1;
            const commit = findBuf(list, "commit").?.t;
            commit.zero();
            for (0..k) |i| commit.f16At(&.{ 0, @intCast(i), 0 }).* = 1;
            const last = findBuf(list, "commit_last").?.t;
            last.zero();
            if (k > 0) last.f16At(&.{ 0, k - 1, 0 }).* = 1;
        }
        if (findBuf(list, "conv_sel_out")) |so| {
            so.t.zero();
            for (0..3) |i| so.t.f16At(&.{ @intCast(i), @intCast(n + i) }).* = 1;
            const valid = findBuf(list, "valid").?.t;
            valid.zero();
            for (0..n) |i| valid.f16At(&.{ 0, @intCast(i), 0 }).* = 1;
        }
        if (self.mask[@intFromEnum(kind)]) |mb| {
            const w = mb.t.dims[1];
            for (0..self.ctx) |i| mb.t.f16At(&.{ @intCast(i / w), @intCast(i % w) }).* = if (i < p0) 0 else -1e4;
        }
    }

    /// Up to P tokens at `pos` through the decode rounds; returns after the head ran (rows = ids).
    pub fn call(self: *Engine, ids: []const u32) !void {
        std.debug.assert(ids.len > 0 and ids.len <= P);
        const n: u32 = @intCast(ids.len);
        if (!try self.fit(self.pos + n)) return error.ContextFull;
        const t0 = if (trace()) std.Io.Timestamp.now(self.io, .awake) else undefined;
        self.put(.decode, ids, P);
        self.common(.decode, self.pos, n, P, self.pending);
        if (trace()) log.info("[ane-trace] prep {d:.2} ms\n", .{@as(f64, @floatFromInt(t0.untilNow(self.io, .awake).nanoseconds)) / 1e6});
        try self.runRounds(try self.plan(self.model.ctxIndex(self.ctx), self.par));
        try self.runRounds(&self.head_round);
        self.par ^= 1;
    }

    /// Commit the first k rows of the last decode call: their K / V rows go into the caches.
    pub fn accept(self: *Engine, k: u32) void {
        self.appendRows(.decode, self.pos, k);
        self.pending = k;
        self.pos += k;
        self.hi = @max(self.hi, self.pos);
    }

    /// Up to TP prompt tokens through the prefill rounds, all committed; the last token's logits
    /// when `logits` (only the feed's final block needs the head).
    fn prefillBlock(self: *Engine, ids: []const u32, logits: bool) ![]const f16 {
        const n: u32 = @intCast(ids.len);
        if (!try self.fit(self.pos + n)) return error.ContextFull;
        const p0 = self.pos;
        const pp = try self.prefillPlan(self.par, p0);
        self.put(.prefill, ids, TP);
        self.common(.prefill, p0, n, TP, self.pending);
        try self.runRounds(pp);
        self.par ^= 1;
        self.appendRows(.prefill, p0, n);
        self.pending = 0;
        self.pos = p0 + n;
        self.hi = @max(self.hi, self.pos);
        if (!logits) return &.{};
        // The head reads the decode rounds' last residual and partials: the last real row, in every row.
        const H = self.model.arch.hidden;
        const R = 2 * self.model.arch.layers;
        const HR = self.model.headRound();
        for (0..self.model.lanes) |l| {
            for ([_]*Buf{ self.res[1][l][R % 2].?, self.part[1][l][(R + 1) % 2].? }, [_]*Buf{ self.res[0][l][HR % 2].?, self.part[0][l][(HR + 1) % 2].? }) |src, dst| {
                for (0..H) |h| {
                    const v = src.t.f16At(&.{ 0, @intCast(h), 0, n - 1 }).*;
                    for (0..P) |r| dst.t.f16At(&.{ 0, @intCast(h), 0, @intCast(r) }).* = v;
                }
            }
        }
        try self.runRounds(&self.head_round);
        return self.logitsRow(0);
    }

    /// Row r of the last head call's logits. The head writes them channel-major (one 64-byte line of
    /// P values per vocabulary entry); the first read gathers every row in one pass.
    pub fn logitsRow(self: *Engine, r: usize) []const f16 {
        const V = self.model.arch.vocab;
        if (!self.rows_ready) {
            var o: usize = 0;
            for (self.logit_parts) |b| {
                const n = b.t.dims[1];
                const pitch = b.t.strides[1];
                for (0..n) |v| {
                    const line: [*]align(1) const f16 = @ptrCast(b.t.base + v * pitch);
                    for (0..P) |row| self.rows[row * V + o + v] = line[row];
                }
                o += n;
            }
            for (0..P) |row| suppress(self.rows[row * V ..][0..V], self.model.suppress_ids, self.model.defined_vocab);
            self.rows_ready = true;
        }
        return self.rows[r * V ..][0..V];
    }

    /// Accepted K / V rows of the last call of `kind` into the cache at [p0, p0 + count).
    fn appendRows(self: *Engine, kind: EntryKind, p0: u32, count: u32) void {
        if (count == 0) return;
        const k = @intFromEnum(kind);
        var key: [16]u8 = undefined;
        const hd = self.model.arch.hd;
        for (self.model.units, 0..) |u, ui| for (u.info.att_j) |j| {
            const kn = findBuf(self.outs[ui][k].items, std.fmt.bufPrint(&key, "k{d}_new", .{j}) catch unreachable).?.t;
            const vn = findBuf(self.outs[ui][k].items, std.fmt.bufPrint(&key, "v{d}_new", .{j}) catch unreachable).?.t;
            const kh: Hist = .{ .t = findBuf(self.kv[ui].items, std.fmt.bufPrint(&key, "k{d}", .{j}) catch unreachable).?.t, .keys = true };
            const vh: Hist = .{ .t = findBuf(self.kv[ui].items, std.fmt.bufPrint(&key, "v{d}", .{j}) catch unreachable).?.t, .keys = false };
            for (0..kh.t.dims[1]) |h| for (0..count) |r| {
                const t: u32 = p0 + @as(u32, @intCast(r));
                const hh: u32 = @intCast(h);
                const rr: u32 = @intCast(r);
                const krow = kn.row(f16, &.{ hh, rr, 0 });
                for (0..hd) |d| kh.at(hh, @intCast(d), t).* = krow[d];
                @memcpy(vh.row(hh, t), vn.row(f16, &.{ hh, rr, 0 }));
            };
        };
    }

    /// Prompt tokens from `pos`: prefill blocks while more than P remain, then decode calls.
    /// Returns the last token's logits row. Positions from `ingest_from` on also go into the
    /// drafter's context (its window only sees the last W + R rows before the prompt's end).
    pub fn feed(self: *Engine, ids: []const u32, stop: ?*const std.atomic.Value(bool), ingest_from: ?u32) ![]const f16 {
        var i: usize = 0;
        var out: []const f16 = &.{};
        while (i < ids.len) {
            if (stop) |s| if (s.load(.acquire)) return error.Cancelled;
            const r = ids.len - i;
            if (r > P) {
                const blk = @min(r, TP);
                const p0 = self.pos;
                out = try self.prefillBlock(ids[i .. i + blk], blk == r);
                if (self.ingesting(p0, @intCast(blk), ingest_from)) try self.ingest(self.features(.prefill, blk), p0, @intCast(blk), ingest_from.?);
                try self.fed.appendSlice(self.allocator, ids[i .. i + blk]);
                self.gauge(.done, blk);
                i += blk;
                if (self.cache != null and self.pos >= self.next_cp) self.checkpoint();
            } else {
                const n: u32 = @intCast(@min(r, P));
                try self.call(ids[i .. i + n]);
                out = self.logitsRow(n - 1);
                if (self.ingesting(self.pos, n, ingest_from)) try self.ingest(self.features(.decode, n), self.pos, n, ingest_from.?);
                self.accept(n);
                try self.fed.appendSlice(self.allocator, ids[i .. i + n]);
                self.gauge(.done, n);
                i += n;
            }
        }
        return out;
    }

    /// Count `n` prompt tokens into the live gauges: `.todo` once a sync knows what it will feed,
    /// `.done` per block fed.
    fn gauge(self: *Engine, comptime which: enum { todo, done }, n: u64) void {
        const g = self.gauges orelse return;
        _ = @field(g, @tagName(which)).fetchAdd(n, .monotonic);
        self.gauged[@intFromEnum(which)] += n;
    }

    /// Take back what this engine counted into the gauges: its sync is over.
    pub fn ungauge(self: *Engine) void {
        const g = self.gauges orelse return;
        _ = g.todo.fetchSub(self.gauged[0], .monotonic);
        _ = g.done.fetchSub(self.gauged[1], .monotonic);
        self.gauged = .{ 0, 0 };
    }

    /// Do positions [p0, p0 + n) reach the drafter's context (from `from` on)?
    fn ingesting(self: *const Engine, p0: u32, n: u32, from: ?u32) bool {
        const f = from orelse return false;
        return self.drafter != null and p0 + n > f;
    }

    /// Context rows (row, tap, hidden) of positions [p0, p0 + n) into the drafter, those before `from` skipped.
    fn ingest(self: *Engine, rows: []const f16, p0: u32, n: u32, from: u32) !void {
        const skip = if (from > p0) from - p0 else 0;
        const per = rows.len / n;
        try self.drafter.?.addContext(rows[skip * per ..], p0 + skip, n - skip);
    }

    /// The last call of `kind`'s first n tap rows, contiguous (row, tap, hidden).
    fn features(self: *Engine, kind: EntryKind, n: usize) []f16 {
        const H = self.model.arch.hidden;
        const list = self.taps[@intFromEnum(kind)];
        const per = list.len * H;
        for (list, 0..) |t, ti| for (0..n) |r| for (0..H) |h| {
            self.feat[r * per + ti * H + h] = t.f16At(&.{ 0, @intCast(h), 0, @intCast(r) }).*;
        };
        return self.feat[0 .. n * per];
    }

    /// One speculative cycle after `anchor` (sampled, not yet in the model): draft P - 1, verify
    /// [anchor + drafts] in one call, accept while each draft is what the target draws (greedy: its
    /// argmax; sampled: with the target's probability of it, the first rejection resampling without
    /// it), commit anchor + the accepted drafts. An accepted stop token ends the cycle uncommitted.
    /// A failure resets the state (a full context leaves it as it was).
    pub fn cycle(self: *Engine, anchor: u32, s: Sampling, rng: std.Random, eos: []const u32) !Cycle {
        return self.cycleOnce(anchor, s, rng, eos) catch |err| return if (err == error.ContextFull) err else self.dropState(err);
    }

    fn cycleOnce(self: *Engine, anchor: u32, s: Sampling, rng: std.Random, eos: []const u32) !Cycle {
        const d = &self.drafter.?;
        const p = self.pos;
        if (!try self.fit(p + P)) return error.ContextFull;
        var ids: [P]u32 = undefined;
        ids[0] = anchor;
        try self.propose(d, anchor, p, ids[1..P]);
        try self.call(ids[0..P]);
        var k: usize = 0;
        var nxt: ?u32 = null;
        while (k < P - 1) : (k += 1) {
            const dk = ids[k + 1];
            if (verifyDraft(self.logitsRow(k), dk, s, rng, self.scratch)) |t| {
                nxt = t;
                break;
            }
            if (std.mem.indexOfScalar(u32, eos, dk) != null) {
                nxt = dk;
                break;
            }
        }
        const next = nxt orelse sample(self.logitsRow(P - 1), s, rng, self.scratch);
        const committed: u32 = @intCast(k + 1);
        try d.addContext(self.features(.decode, committed), p, committed);
        self.accept(committed);
        try self.fed.appendSlice(self.allocator, ids[0..committed]);
        var out: Cycle = .{ .tokens = undefined, .n = k + 1, .accepted = k };
        @memcpy(out.tokens[0..k], ids[1 .. k + 1]);
        out.tokens[k] = next;
        return out;
    }

    /// Draft P - 1 tokens after `anchor` at position p: the drafter block, the head over its rows,
    /// each row's top-16 candidates walked through the selector.
    fn propose(self: *Engine, d: *dflash.Drafter, anchor: u32, p: u32, out: []u32) !void {
        const m = self.model;
        const t0 = if (trace()) std.Io.Timestamp.now(self.io, .awake) else undefined;
        m.embed.row(anchor, self.emb_scratch);
        try d.block(p, self.emb_scratch, m.mask_emb);
        const t1 = if (trace()) std.Io.Timestamp.now(self.io, .awake) else undefined;
        try self.runRounds(&self.head_raw);
        const t2 = if (trace()) std.Io.Timestamp.now(self.io, .awake) else undefined;
        defer if (trace()) log.info("[ane-trace] draft {d:.2} ms, head {d:.2} ms, select {d:.2} ms\n", .{
            @as(f64, @floatFromInt(t1.durationTo(t0).nanoseconds)) / -1e6,
            @as(f64, @floatFromInt(t2.durationTo(t1).nanoseconds)) / -1e6,
            @as(f64, @floatFromInt(t2.untilNow(self.io, .awake).nanoseconds)) / 1e6,
        });
        var cand: [P][dflash.TOP_K]u32 = undefined;
        var unary: [P][dflash.TOP_K]f32 = undefined;
        var hp_rows: [P][]f32 = undefined;
        const rank = d.cfg.rank;
        const hp_buf = try self.allocator.alloc(f32, rank * P);
        defer self.allocator.free(hp_buf);
        const steps = P - 1;
        for (0..steps) |i| {
            const top = topK(self.logitsRow(i + 1), dflash.TOP_K, self.scratch);
            for (top, 0..) |c, j| {
                cand[i][j] = c.id;
                unary[i][j] = c.logit;
            }
            hp_rows[i] = hp_buf[i * rank ..][0..rank];
            d.hp(i + 1, hp_rows[i]);
        }
        var hp_const: [P][]const f32 = undefined;
        for (0..steps) |i| hp_const[i] = hp_rows[i];
        dflash.walk(anchor, rank, cand[0..steps], unary[0..steps], hp_const[0..steps], m.selector.pred, m.selector.succ, out);
    }

    /// Feed one generated token and return its logits row.
    /// A failure resets the state (a full context leaves it as it was).
    pub fn step(self: *Engine, token: u32) ![]const f16 {
        self.call(&.{token}) catch |err| return if (err == error.ContextFull) err else self.dropState(err);
        self.accept(1);
        self.fed.append(self.allocator, token) catch |err| return self.dropState(err);
        return self.logitsRow(0);
    }

    /// Bring the state to `prompt`: continue the live state when the prompt extends it, or a cached
    /// checkpoint when that reaches further, else start over. Returns the last prompt token's logits
    /// and how many prompt tokens were already in the state. A cancelled sync leaves the state reset.
    pub fn sync(self: *Engine, prompt: []const u32, stop: ?*const std.atomic.Value(bool)) !Synced {
        const from: ?u32 = if (self.drafter) |d| @intCast(prompt.len -| (d.cfg.window + dflash.R)) else null;
        const n = self.fed.items.len;
        var start: usize = if (n > 0 and prompt.len > n and std.mem.eql(u32, self.fed.items, prompt[0..n])) n else 0;
        if (self.cache) |c| {
            c.mu.lockUncancelable(self.io);
            var failed: ?anyerror = null;
            if (c.lookup(prompt)) |hit| if (hit.end() > start) {
                if (self.restore(hit, prompt)) {
                    log.info("[ane] prompt: {d} of {d} tokens from a checkpoint\n", .{ hit.end(), prompt.len });
                    start = hit.end();
                } else |err| failed = err;
            };
            c.mu.unlock(self.io);
            if (failed) |err| return self.dropState(err);
        }
        if (start == 0) self.reset(true);
        const cached = start;
        self.gauge(.todo, prompt.len - start);
        // A checkpoint before the prompt's last P tokens: a chat's next turn renders this turn's
        // generation prompt differently, so the prefix it shares ends about there.
        const turn = prompt.len -| P;
        if (self.cache != null and turn > start) {
            _ = self.feed(prompt[start..turn], stop, from) catch |err| return self.dropState(err);
            self.checkpoint();
            start = turn;
        }
        return .{ .logits = self.feed(prompt[start..], stop, from) catch |err| return self.dropState(err), .cached = cached };
    }

    fn dropState(self: *Engine, err: anyerror) anyerror {
        self.reset(false);
        return err;
    }

    pub fn draw(self: *Engine, logits: []const f16, s: Sampling, rng: std.Random) u32 {
        return sample(logits, s, rng, self.scratch);
    }

    // ── prefix cache ──

    /// Keep checkpoints of earlier sequences in `c`: every `stride` positions, before each prompt's
    /// tail and after each request.
    pub fn useCache(self: *Engine, c: *prefix.Cache, stride: u32) void {
        std.debug.assert(stride > 0);
        self.cache = c;
        self.stride = stride;
        self.next_cp = stride;
    }

    /// Save the live state as a checkpoint under the one it continues. Without a cache or new
    /// tokens it does nothing; a failed save costs only the reuse.
    pub fn checkpoint(self: *Engine) void {
        const c = self.cache orelse return;
        std.debug.assert(self.fed.items.len == self.pos);
        self.next_cp = (self.pos / self.stride + 1) * self.stride;
        const start: u32 = if (self.node) |n| n.end() else 0;
        if (self.pos <= start) return;
        c.mu.lockUncancelable(self.io);
        defer c.mu.unlock(self.io);
        const toks = self.fed.items[start..self.pos];
        const n = c.child(self.node, toks) orelse blk: {
            const n = c.add(self.node, toks, self.kvLen(self.pos - start), self.stateLen()) catch |err| {
                log.warn("[ane] checkpoint at {d} not saved: {s}\n", .{ self.pos, @errorName(err) });
                return;
            };
            self.moveKv(start, self.pos, n.kv, true);
            self.saveState(n.state.?);
            break :blk n;
        };
        if (n.state == null) self.saveState(c.restate(n, self.stateLen()) catch |err| {
            log.warn("[ane] checkpoint at {d} not saved: {s}\n", .{ self.pos, @errorName(err) });
            return;
        });
        c.touch(n);
        self.follow(n);
        c.trim();
        log.debug("[ane] checkpoint at {d}; cache {d} states, {d} MB\n", .{ self.pos, c.states, c.bytes >> 20 });
    }

    /// The live state to checkpoint `n` of `prompt` (cache locked).
    fn restore(self: *Engine, n: *prefix.Node, prompt: []const u32) !void {
        const end = n.end();
        self.hi = 0; // every row comes from the checkpoint: a context switch moves none
        if (!try self.fit(end)) return error.ContextFull;
        var at: ?*prefix.Node = n;
        while (at) |x| : (at = x.parent) self.moveKv(x.start, x.end(), x.kv, false);
        try self.loadState(n.state.?);
        self.pos = end;
        self.hi = end;
        self.fed.clearRetainingCapacity();
        try self.fed.appendSlice(self.allocator, prompt[0..end]);
        self.next_cp = (end / self.stride + 1) * self.stride;
        self.cache.?.touch(n);
        self.follow(n);
    }

    /// The live state now continues `n` (cache locked).
    fn follow(self: *Engine, n: ?*prefix.Node) void {
        const c = self.cache.?;
        if (n) |x| c.pin(x);
        if (self.node) |old| c.unpin(old);
        self.node = n;
    }

    fn kvLen(self: *const Engine, rows: u32) usize {
        var len: usize = 0;
        for (self.kv) |l| for (l.items) |b| {
            len += Hist.of(b).len(rows);
        };
        return len;
    }

    /// KV rows [a, b) of every attention layer into `buf` (`out`) or back from it.
    fn moveKv(self: *Engine, a: u32, b: u32, buf: []u8, out: bool) void {
        var o: usize = 0;
        for (self.kv) |l| for (l.items) |kb| {
            const h = Hist.of(kb);
            const len = h.len(b - a);
            h.move(a, b, buf[o..][0..len], out);
            o += len;
        };
    }

    /// The pending count and the DeltaNet states (dense, in their surfaces' layout), then the
    /// drafter's context when it holds any.
    fn stateLen(self: *const Engine) usize {
        var len: usize = @sizeOf(u32);
        for (self.st) |s| for (s[self.par].items) |b| {
            len += b.surf.bytes;
        };
        if (self.drafter) |*d| if (d.hasContext()) {
            len += d.snapLen();
        };
        return len;
    }

    fn saveState(self: *const Engine, out: []u8) void {
        @memcpy(out[0..@sizeOf(u32)], std.mem.asBytes(&self.pending));
        var o: usize = @sizeOf(u32);
        for (self.st) |s| for (s[self.par].items) |b| {
            @memcpy(out[o..][0..b.surf.bytes], b.surf.base[0..b.surf.bytes]);
            o += b.surf.bytes;
        };
        if (self.drafter) |*d| if (o < out.len) d.save(out[o..]);
    }

    fn loadState(self: *Engine, in: []const u8) !void {
        @memcpy(std.mem.asBytes(&self.pending), in[0..@sizeOf(u32)]);
        var o: usize = @sizeOf(u32);
        for (self.st) |s| for (s[self.par].items) |b| {
            @memcpy(b.surf.base[0..b.surf.bytes], in[o..][0..b.surf.bytes]);
            o += b.surf.bytes;
        };
        if (self.drafter) |*d| if (o < in.len) try d.load(in[o..]) else d.reset();
    }
};

/// A KV history buffer: keys (tile, head, dim, position) or values (tile, head, position, dim);
/// history position p lives in tile p / width at p % width (`q35.histShape`).
const Hist = struct {
    t: rt.Tensor,
    keys: bool,

    fn of(b: *const Buf) Hist {
        return .{ .t = b.t, .keys = b.name[0] == 'k' };
    }

    fn width(h: Hist) u32 {
        return h.t.dims[if (h.keys) 3 else 2];
    }

    fn dim(h: Hist) u32 {
        return h.t.dims[if (h.keys) 2 else 3];
    }

    /// Bytes of n positions of every head.
    fn len(h: Hist, n: u32) usize {
        return @as(usize, n) * h.t.dims[1] * h.dim() * h.t.elemSize();
    }

    fn at(h: Hist, head: u32, d: u32, p: u32) *align(1) f16 {
        const w = h.width();
        return h.t.f16At(if (h.keys) &.{ p / w, head, d, p % w } else &.{ p / w, head, p % w, d });
    }

    /// A value row: position p's dims of one head.
    fn row(h: Hist, head: u32, p: u32) []align(1) f16 {
        std.debug.assert(!h.keys);
        return @as([*]align(1) f16, @ptrCast(h.at(head, 0, p)))[0..h.dim()];
    }

    /// Positions [a, b) into `buf` (`out`) or back from it, packed (head, dim, position) for keys
    /// and (head, position, dim) for values, whatever the tiling.
    fn move(h: Hist, a: u32, b: u32, buf: []u8, out: bool) void {
        const es = h.t.elemSize();
        var o: usize = 0;
        for (0..h.t.dims[1]) |head| {
            const hh: u32 = @intCast(head);
            if (h.keys) {
                for (0..h.dim()) |d| {
                    var p = a;
                    while (p < b) {
                        const run = @min(b - p, h.width() - p % h.width());
                        const mem = @as([*]u8, @ptrCast(h.at(hh, @intCast(d), p)))[0 .. run * es];
                        if (out) @memcpy(buf[o..][0..mem.len], mem) else @memcpy(mem, buf[o..][0..mem.len]);
                        o += mem.len;
                        p += run;
                    }
                }
            } else for (a..b) |p| {
                const mem = std.mem.sliceAsBytes(h.row(hh, @intCast(p)));
                if (out) @memcpy(buf[o..][0..mem.len], mem) else @memcpy(mem, buf[o..][0..mem.len]);
                o += mem.len;
            }
        }
    }

    /// Positions [0, keep) into `dst`, another tiling of the same heads and dims.
    fn copy(src: Hist, dst: Hist, keep: u32) void {
        const es = src.t.elemSize();
        for (0..src.t.dims[1]) |head| {
            const hh: u32 = @intCast(head);
            if (src.keys) {
                for (0..src.dim()) |d| {
                    var p: u32 = 0;
                    while (p < keep) {
                        const run = @min(keep - p, src.width() - p % src.width(), dst.width() - p % dst.width());
                        const dd: u32 = @intCast(d);
                        @memcpy(@as([*]u8, @ptrCast(dst.at(hh, dd, p)))[0 .. run * es], @as([*]const u8, @ptrCast(src.at(hh, dd, p)))[0 .. run * es]);
                        p += run;
                    }
                }
            } else for (0..keep) |p| @memcpy(dst.row(hh, @intCast(p)), src.row(hh, @intCast(p)));
        }
    }
};

pub const CacheLimits = struct { max_bytes: u64, max_states: u32, stride: u32 };

/// The model at `dir` built and loaded once with `opts.slots` engines on it behind one worker,
/// sharing one prefix cache (none without `limits`; `max_bytes` 0 is no byte cap).
pub fn openPool(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, arch: Arch, opts: Options, limits: ?CacheLimits, ctx: *anyopaque, runFn: *const fn (*anyopaque, *Engine, *anyopaque) void) !*Worker {
    const model = try Model.open(allocator, io, dir, arch, opts);
    errdefer model.deinit();
    // Before the engines: an engine's deinit still reads the cache it follows.
    const cache: ?*prefix.Cache = if (limits) |l| blk: {
        const c = try allocator.create(prefix.Cache);
        c.* = .init(allocator, if (l.max_bytes == 0) std.math.maxInt(u64) else l.max_bytes, l.max_states);
        break :blk c;
    } else null;
    errdefer if (cache) |c| {
        c.deinit();
        allocator.destroy(c);
    };
    const engines = try allocator.alloc(*Engine, model.instances);
    var opened: usize = 0;
    errdefer {
        var i = opened;
        while (i > 0) : (i -= 1) engines[i - 1].deinit();
        allocator.free(engines);
    }
    for (engines, 0..) |*e, i| {
        e.* = try Engine.open(model, io, @intCast(i));
        opened += 1;
        if (cache) |c| e.*.useCache(c, limits.?.stride);
    }
    return Worker.start(allocator, io, model, engines, ctx, runFn);
}

/// One thread per engine, all taking jobs from one queue, in order. The ANE never touches MLX, so
/// requests run here beside the MLX inference thread instead of taking turns with it.
pub const Worker = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    model: ?*Model,
    engines: []*Engine,
    threads: []std.Thread = &.{},
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, engine: *Engine, job: *anyopaque) void,
    mu: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    queue: std.ArrayList(*anyopaque) = .empty,
    stopping: bool = false,

    pub fn start(allocator: std.mem.Allocator, io: std.Io, model: ?*Model, engines: []*Engine, ctx: *anyopaque, runFn: *const fn (*anyopaque, *Engine, *anyopaque) void) !*Worker {
        const w = try allocator.create(Worker);
        errdefer allocator.destroy(w);
        w.* = .{ .allocator = allocator, .io = io, .model = model, .engines = engines, .ctx = ctx, .run = runFn };
        const threads = try allocator.alloc(std.Thread, engines.len);
        errdefer allocator.free(threads);
        for (threads, 0..) |*t, i| t.* = std.Thread.spawn(.{}, loop, .{ w, i }) catch |err| {
            w.threads = threads[0..i];
            w.stop();
            return err;
        };
        w.threads = threads;
        return w;
    }

    pub fn post(self: *Worker, job: *anyopaque) !void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        try self.queue.append(self.allocator, job);
        self.cond.signal(self.io);
    }

    /// Finish the posted jobs, join. The engines stay the owner's to close.
    pub fn deinit(self: *Worker) void {
        self.stop();
        self.allocator.free(self.threads);
        self.queue.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// What the pool holds beside the weights: every slot's buffers and the prefix cache's byte cap.
    pub fn stateBytes(self: *const Worker) u64 {
        var n: u64 = 0;
        for (self.engines) |e| n += e.bufferBytes();
        if (self.engines[0].cache) |c| if (c.max_bytes != std.math.maxInt(u64)) {
            n += c.max_bytes;
        };
        return n;
    }

    /// `deinit`, then the engines, their shared cache, the engine list and the model: an `openPool` worker.
    pub fn close(self: *Worker) void {
        const a = self.allocator;
        const engines = self.engines;
        const cache = engines[0].cache;
        const model = self.model;
        self.deinit();
        var i = engines.len;
        while (i > 0) : (i -= 1) engines[i - 1].deinit();
        if (cache) |c| {
            c.deinit();
            a.destroy(c);
        }
        a.free(engines);
        if (model) |m| m.deinit();
    }

    fn stop(self: *Worker) void {
        self.mu.lockUncancelable(self.io);
        self.stopping = true;
        self.cond.broadcast(self.io);
        self.mu.unlock(self.io);
        for (self.threads) |t| t.join();
    }

    fn loop(self: *Worker, i: usize) void {
        while (true) {
            self.mu.lockUncancelable(self.io);
            while (self.queue.items.len == 0 and !self.stopping) self.cond.waitUncancelable(self.io, &self.mu);
            // Stopping still runs what was posted: a dropped job would hold its slot's `in_pass`.
            if (self.queue.items.len == 0) {
                self.mu.unlock(self.io);
                return;
            }
            const job = self.queue.orderedRemove(0);
            self.mu.unlock(self.io);
            self.run(self.ctx, self.engines[i], job);
        }
    }
};

// ── tests ──

const testing = std.testing;

test "arch: qwen3_5 and qwen3 dense packs; experts and Hadamard packs are refused" {
    const Cfg = struct {
        model_type: []const u8 = "qwen3_5_moe",
        num_experts: u32 = 0,
        hadamard_block: u32 = 0,
        full_attention_interval: u32 = 4,
        linear_num_value_heads: u32 = 48,
        linear_num_key_heads: u32 = 16,
        linear_key_head_dim: u32 = 128,
        linear_value_head_dim: u32 = 128,
        hidden_size: u32 = 5120,
        intermediate_size: u32 = 17408,
        num_hidden_layers: u32 = 64,
        vocab_size: u32 = 248320,
        num_attention_heads: u32 = 24,
        num_key_value_heads: u32 = 4,
        head_dim: u32 = 256,
        partial_rotary_factor: f32 = 0.25,
        rope_theta: f32 = 1e7,
        rms_norm_eps: f32 = 1e-6,
        tie_word_embeddings: bool = false,
        embedding: bool = false,
        fn hasEmbeddingCapability(self: @This()) bool {
            return self.embedding;
        }
    };
    const a = try archOf(Cfg{});
    try testing.expectEqual(@as(u32, 64), a.rot);
    try testing.expect(a.isLinear(0) and !a.isLinear(3));
    try testing.expectError(error.UnsupportedArch, archOf(Cfg{ .num_experts = 256 }));
    try testing.expectError(error.UnsupportedArch, archOf(Cfg{ .hadamard_block = 1024 }));
    try testing.expectError(error.UnsupportedArch, archOf(Cfg{ .model_type = "llama" }));
    try testing.expect(a.gate);
    // Qwen3 dense: every layer attention, no output gate, full rotary.
    const q3 = try archOf(Cfg{ .model_type = "qwen3", .full_attention_interval = 0, .linear_num_value_heads = 0, .linear_num_key_heads = 0, .linear_key_head_dim = 0, .linear_value_head_dim = 0, .hidden_size = 1024, .intermediate_size = 3072, .num_hidden_layers = 28, .num_attention_heads = 16, .num_key_value_heads = 8, .head_dim = 128, .partial_rotary_factor = 1.0 });
    try testing.expect(!q3.gate and !q3.isLinear(0) and !q3.isLinear(5));
    try testing.expectEqual(@as(u32, 128), q3.rot);
    try testing.expect(q3.splits());
    // Qwen3-Embedding is a qwen3 pack too; embeddings stay on the GPU.
    try testing.expectError(error.UnsupportedArch, archOf(Cfg{ .model_type = "qwen3", .full_attention_interval = 0, .linear_num_value_heads = 0, .embedding = true }));
}

test "load bill: the int8 weights and every slot's KV at the top context" {
    const q27: Arch = .{ .hidden = 5120, .ffn = 17408, .layers = 64, .vocab = 248320, .nk = 16, .nv = 48, .dk = 128, .dv = 128, .nh = 24, .nkv = 4, .hd = 256, .rot = 64, .rope_theta = 1e7, .eps = 1e-6, .full_interval = 4, .tie = false };
    // 16 attention layers x K and V x 4 heads x 256 x fp16, over 64K positions: 4 GiB a slot.
    try testing.expectEqual(@as(u64, 4) << 30, loadBytes(q27, 2) - loadBytes(q27, 1));
    try testing.expectEqual(q35.modelBytes(q27) + (@as(u64, 4) << 30), loadBytes(q27, 1));
}

test "lanes: two Neural Engines split a layer only for a model big enough to gain from it" {
    const q27: Arch = .{ .hidden = 5120, .ffn = 17408, .layers = 64, .vocab = 248320, .nk = 16, .nv = 48, .dk = 128, .dv = 128, .nh = 24, .nkv = 4, .hd = 256, .rot = 64, .rope_theta = 1e7, .eps = 1e-6, .full_interval = 4, .tie = false };
    var small = q27;
    small.hidden = 1024;
    small.ffn = 3584;
    small.layers = 24;
    small.nv = 16;
    small.nh = 8;
    small.nkv = 2;
    try testing.expectEqual(@as(u32, 2), lanesFor(q27, 2));
    try testing.expectEqual(@as(u32, 1), lanesFor(q27, 1));
    try testing.expectEqual(@as(u32, 1), lanesFor(small, 2));
    var odd = q27;
    odd.nkv = 1;
    try testing.expectEqual(@as(u32, 1), lanesFor(odd, 2));
}

test "--ane-slots takes 1 to max_slots" {
    try std.testing.expectEqual(@as(u32, 1), try parseSlots("1"));
    try std.testing.expectEqual(max_slots, try parseSlots("4"));
    for ([_][]const u8{ "0", "5", "17", "-1", "x", "" }) |bad| try std.testing.expectError(error.InvalidSlots, parseSlots(bad));
}

test "sampling: without a top-k, only tokens near the top are candidates" {
    const logits = [_]f16{ 3, 10, 9, -40 };
    var scratch: [4]Cand = undefined;
    const wide = dist(&logits, .{ .temperature = 1 }, &scratch).?;
    try std.testing.expectEqual(@as(usize, 3), wide.len);
    try std.testing.expectEqual(@as(u32, 1), wide[0].id);
    try std.testing.expectEqual(@as(usize, 2), dist(&logits, .{ .temperature = 0.1 }, &scratch).?.len);
}

test "gauges: a sync counts what it will feed and what it fed, and takes both back" {
    var g: PrefillGauges = .{};
    var e: Engine = undefined;
    e.gauges = &g;
    e.gauged = .{ 0, 0 };
    e.gauge(.todo, 100);
    e.gauge(.done, 64);
    try testing.expectEqual(@as(u64, 100), g.todo.load(.monotonic));
    try testing.expectEqual(@as(u64, 64), g.done.load(.monotonic));
    e.ungauge();
    try testing.expectEqual(@as(u64, 0), g.todo.load(.monotonic) + g.done.load(.monotonic));
}

test "suppress: reserved ids and padding rows are never drawn" {
    var row = [_]f16{ 1, 9, 2, 8, 7 };
    suppress(&row, &.{ 1, 40 }, 3);
    try std.testing.expectEqual(@as(u32, 2), argmax(&row));
    var scratch: [5]Cand = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    for (0..32) |_| try std.testing.expect(sample(&row, .{ .temperature = 2 }, prng.random(), &scratch) != 1);
}

test "padding rows repeat the last real position" {
    try testing.expectEqual(@as(u32, 10), rowPos(10, 3, 0));
    try testing.expectEqual(@as(u32, 12), rowPos(10, 3, 2));
    try testing.expectEqual(@as(u32, 12), rowPos(10, 3, 7));
}

test "rope inverse frequencies match 1 / theta^(i / rot)" {
    var inv: [32]f64 = undefined;
    ropeInv(&inv, 64, 1e7);
    try testing.expectEqual(@as(f64, 1), inv[0]);
    try testing.expectApproxEqRel(1.0 / std.math.pow(f64, 1e7, 62.0 / 64.0), inv[31], 1e-15);
}

test "sampling: temperature 0 is the first argmax; top-k 1 always draws the top token" {
    const logits = [_]f16{ 0.5, 3.0, 3.0, -1 };
    var scratch: [4]Cand = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    try testing.expectEqual(@as(u32, 1), sample(&logits, .{}, prng.random(), &scratch));
    for (0..16) |_| try testing.expectEqual(@as(u32, 1), sample(&logits, .{ .temperature = 1, .top_k = 1 }, prng.random(), &scratch));
    const peaked = [_]f16{ 0, 20, 0, 0 };
    for (0..16) |_| try testing.expectEqual(@as(u32, 1), sample(&peaked, .{ .temperature = 0.7, .top_p = 0.9 }, prng.random(), &scratch));
}

test "dist: temperature 0 is greedy; min-p and top-p cut the tail" {
    const logits = [_]f16{ 0, 4, 3, 2 };
    var scratch: [4]Cand = undefined;
    try testing.expect(dist(&logits, .{ .temperature = 0 }, &scratch) == null);
    // Relative probabilities 1, e^-1, e^-2, e^-4: min-p 0.3 keeps the first two.
    const mp = dist(&logits, .{ .temperature = 1, .min_p = 0.3 }, &scratch).?;
    try testing.expectEqual(@as(usize, 2), mp.len);
    try testing.expectEqual(@as(u32, 1), mp[0].id);
    // The top token alone already holds 66% of the mass: top-p 0.5 keeps it alone.
    try testing.expectEqual(@as(usize, 1), dist(&logits, .{ .temperature = 1, .top_p = 0.5 }, &scratch).?.len);
    try testing.expectEqual(@as(usize, 4), dist(&logits, .{ .temperature = 1 }, &scratch).?.len);
}

test "verifyDraft: greedy keeps the argmax only; sampled keeps a likely draft, replaces an absent one" {
    const logits = [_]f16{ 0, 16, 1, 0 };
    var scratch: [4]Cand = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    try testing.expect(verifyDraft(&logits, 1, .{ .temperature = 0 }, rng, &scratch) == null);
    try testing.expectEqual(@as(?u32, 1), verifyDraft(&logits, 2, .{ .temperature = 0 }, rng, &scratch));
    for (0..20) |_| try testing.expect(verifyDraft(&logits, 1, .{ .temperature = 1 }, rng, &scratch) == null);
    // A draft outside the top-k is always replaced, and never by itself.
    for (0..20) |_| {
        const t = verifyDraft(&logits, 3, .{ .temperature = 1, .top_k = 2 }, rng, &scratch).?;
        try testing.expect(t == 1 or t == 2);
    }
}

test "isKv names the KV history inputs only" {
    try testing.expect(Engine.isKv("k3") and Engine.isKv("v12"));
    for ([_][]const u8{ "k3_new", "v0_new", "valid", "k", "kv" }) |n| try testing.expect(!Engine.isKv(n));
}

test "top-k keeps the k largest, descending" {
    const logits = [_]f16{ 1, 5, 3, 4, 2 };
    var scratch: [5]Cand = undefined;
    const got = topK(&logits, 3, &scratch);
    try testing.expectEqual(@as(usize, 3), got.len);
    try testing.expectEqual(@as(u32, 1), got[0].id);
    try testing.expectEqual(@as(u32, 3), got[1].id);
    try testing.expectEqual(@as(u32, 2), got[2].id);
}

/// A rank-4 fp16 tensor over `buf` with the last axis padded by `row_pad` bytes.
fn testTensor(buf: []u8, dims: [4]u32, row_pad: usize) rt.Tensor {
    const row = dims[3] * 2 + row_pad;
    return .{ .base = buf.ptr, .dtype = .fp16, .rank = 4, .dims = dims, .strides = .{ dims[1] * dims[2] * row, dims[2] * row, row, 2 }, .bytes = buf.len };
}

test "KV history: positions [a, b) out and back across tiles and padded rows, the rest untouched" {
    // Keys, 2 heads x 2 dims x 6 positions in 3 tiles of 2, rows padded to 8 bytes.
    var kb: [3 * 2 * 2 * 8]u8 align(2) = @splat(0);
    const k: Hist = .{ .t = testTensor(&kb, .{ 3, 2, 2, 2 }, 4), .keys = true };
    for (0..2) |h| for (0..2) |d| for (0..6) |p| {
        k.at(@intCast(h), @intCast(d), @intCast(p)).* = @floatFromInt(h * 100 + d * 10 + p);
    };
    try testing.expectEqual(@as(usize, 2 * 2 * 3 * 2), k.len(3));
    var buf: [24]u8 = undefined;
    k.move(1, 4, &buf, true);
    const got: []align(1) const f16 = std.mem.bytesAsSlice(f16, &buf);
    try testing.expectEqualSlices(f16, &.{ 1, 2, 3, 11, 12, 13, 101, 102, 103, 111, 112, 113 }, @as([]const f16, @alignCast(got)));
    // Back into one tile of 6: the packing does not depend on the tiling.
    var ob: [2 * 2 * 16]u8 align(2) = @splat(0);
    const one: Hist = .{ .t = testTensor(&ob, .{ 1, 2, 2, 6 }, 4), .keys = true };
    one.move(1, 4, &buf, false);
    try testing.expectEqual(@as(f16, 0), one.at(1, 1, 0).*);
    try testing.expectEqual(@as(f16, 113), one.at(1, 1, 3).*);
    try testing.expectEqual(@as(f16, 0), one.at(1, 1, 4).*);
    // Values (1 head, 3 positions in one tile, 2 dims) carried into 2 tiles of 2.
    var vb: [3 * 8]u8 align(2) = @splat(0);
    const v: Hist = .{ .t = testTensor(&vb, .{ 1, 1, 3, 2 }, 4), .keys = false };
    for (0..3) |p| for (0..2) |d| {
        v.at(0, @intCast(d), @intCast(p)).* = @floatFromInt(p * 10 + d);
    };
    var wb: [2 * 2 * 8]u8 align(2) = @splat(0);
    const w: Hist = .{ .t = testTensor(&wb, .{ 2, 1, 2, 2 }, 4), .keys = false };
    v.copy(w, 3);
    try testing.expectEqual(@as(f16, 11), w.at(0, 1, 1).*);
    try testing.expectEqual(@as(f16, 21), w.at(0, 1, 2).*);
    try testing.expectEqual(@as(f16, 0), w.at(0, 0, 3).*);
    // Keys across both tilings at once.
    var cb: [2 * 2 * 2 * 8]u8 align(2) = @splat(0);
    const c: Hist = .{ .t = testTensor(&cb, .{ 2, 2, 2, 2 }, 4), .keys = true };
    one.copy(c, 4);
    try testing.expectEqual(@as(f16, 102), c.at(1, 0, 2).*);
    try testing.expectEqual(@as(f16, 0), c.at(1, 0, 0).*);
}

// ── gated: a real pack on the Neural Engine (ANE_TEST_PACK=<MLX pack dir>) ──

const model_mod = @import("../model.zig");

fn testModel(io: std.Io, slots: u32) !?*Model {
    const dir = std.c.getenv("ANE_TEST_PACK") orelse return null;
    const drafter = try std.fs.path.join(testing.allocator, &.{ std.mem.span(dir), "drafter" });
    defer testing.allocator.free(drafter);
    // ANE_TEST_LANES=1 runs the one-engine path on an Ultra.
    const lanes: u32 = if (std.c.getenv("ANE_TEST_LANES")) |l| std.fmt.parseInt(u32, std.mem.span(l), 10) catch 0 else 0;
    return testModelWith(io, .{ .slots = slots, .drafter = drafter, .lanes = lanes });
}

fn testModelWith(io: std.Io, opts: Options) !?*Model {
    const dir = std.c.getenv("ANE_TEST_PACK") orelse return null;
    var cfg = try model_mod.parseConfig(io, testing.allocator, std.mem.span(dir));
    defer cfg.deinit(testing.allocator);
    return try Model.open(testing.allocator, io, std.mem.span(dir), try archOf(&cfg), opts);
}

fn cosine(x: []const f32, y: []const f32) f64 {
    var xy: f64 = 0;
    var xx: f64 = 0;
    var yy: f64 = 0;
    for (x, y) |p, q| {
        xy += p * q;
        xx += p * p;
        yy += q * q;
    }
    return xy / @sqrt(xx * yy);
}

/// A long deterministic prompt (no tokenizer here: plausible ids, repeated phrases the drafter can follow).
fn testPrompt(a: std.mem.Allocator, len: usize) ![]u32 {
    const phrase = [_]u32{ 785, 6722, 315, 9625, 374, 12366, 13, 576, 6722, 315, 9856, 374, 19846, 13, 198 };
    const out = try a.alloc(u32, len);
    for (out, 0..) |*t, i| t.* = phrase[i % phrase.len] + @as(u32, @intCast((i / phrase.len) % 3));
    return out;
}

fn greedy(e: *Engine, logits0: []const f16, n: usize, out: []u32) !void {
    var logits = logits0;
    for (out[0..n]) |*t| {
        t.* = argmax(logits);
        logits = try e.step(t.*);
    }
}

test "pack on the ANE: speculative greedy equals plain greedy (ANE_TEST_PACK)" {
    const io = testing.io;
    const m = (try testModel(io, 1)) orelse return error.SkipZigTest;
    defer m.deinit();
    const e = try Engine.open(m, io, 0);
    defer e.deinit();
    const prompt = try testPrompt(testing.allocator, 200);
    defer testing.allocator.free(prompt);
    var want: [48]u32 = undefined;
    try greedy(e, (try e.sync(prompt, null)).logits, want.len, &want);
    if (e.drafter == null) return;
    e.reset(true);
    const s = try e.sync(prompt, null);
    var got: std.ArrayList(u32) = .empty;
    defer got.deinit(testing.allocator);
    try got.append(testing.allocator, argmax(s.logits));
    var prng = std.Random.DefaultPrng.init(1);
    var cycles: usize = 0;
    while (got.items.len < want.len) : (cycles += 1) {
        const c = try e.cycle(got.items[got.items.len - 1], .{}, prng.random(), &.{});
        try got.appendSlice(testing.allocator, c.tokens[0..c.n]);
    }
    std.debug.print("speculative: {d} tokens in {d} cycles\n", .{ want.len, cycles });
    try testing.expectEqualSlices(u32, &want, got.items[0..want.len]);
}

test "pack on the ANE: a checkpoint taken mid-prefill restores its position's state (ANE_TEST_PACK)" {
    const io = testing.io;
    const a = testing.allocator;
    const m = (try testModel(io, 1)) orelse return error.SkipZigTest;
    defer m.deinit();
    // The reference has a cache of its own, so it splits the prompt at the same turn boundary.
    var ref = prefix.Cache.init(a, std.math.maxInt(u64), 64);
    defer ref.deinit();
    var cache = prefix.Cache.init(a, std.math.maxInt(u64), 64);
    defer cache.deinit();
    const e = try Engine.open(m, io, 0);
    defer e.deinit();
    const prompt = try testPrompt(a, 700);
    defer a.free(prompt);
    // The same first 300 tokens, then others: it shares the checkpoint at 256.
    const other = try a.dupe(u32, prompt[0..400]);
    defer a.free(other);
    for (other[300..]) |*t| t.* += 7;
    e.useCache(&ref, 256);
    const want = try a.dupe(f16, (try e.sync(other, null)).logits);
    defer a.free(want);
    e.reset(true);
    e.useCache(&cache, 256);
    _ = try e.sync(prompt, null);
    e.reset(false);
    const got = try e.sync(other, null);
    try testing.expectEqual(@as(usize, 256), got.cached);
    try testing.expectEqualSlices(f16, want, got.logits);
}

test "pack on the ANE: a prompt past the first context decodes like a short one (ANE_TEST_PACK)" {
    const io = testing.io;
    const a = testing.allocator;
    const m = (try testModel(io, 1)) orelse return error.SkipZigTest;
    defer m.deinit();
    const e = try Engine.open(m, io, 0);
    defer e.deinit();
    const prompt = try testPrompt(a, m.ctxs[0] + 200);
    defer a.free(prompt);
    var logits = (try e.sync(prompt, null)).logits;
    try testing.expect(e.ctx > m.ctxs[0]);
    for (0..24) |_| logits = try e.step(argmax(logits));
    try testing.expect(std.math.isFinite(@as(f32, logits[0])));
}

test "pack on the ANE: a restored checkpoint continues like the live state (ANE_TEST_PACK)" {
    const io = testing.io;
    const a = testing.allocator;
    const m = (try testModel(io, 1)) orelse return error.SkipZigTest;
    defer m.deinit();
    // The cache outlives the engine (whose deinit unpins its node), as in `Worker.close`.
    var cache = prefix.Cache.init(a, std.math.maxInt(u64), 64);
    defer cache.deinit();
    const e = try Engine.open(m, io, 0);
    defer e.deinit();
    e.useCache(&cache, 256);
    const prompt = try testPrompt(a, 600);
    defer a.free(prompt);
    // Live: 24 greedy tokens, with a checkpoint after 16 as a finished request leaves one.
    var live: [24]u32 = undefined;
    var logits = (try e.sync(prompt, null)).logits;
    for (&live, 0..) |*t, i| {
        if (i == 16) e.checkpoint();
        t.* = argmax(logits);
        logits = try e.step(t.*);
    }
    // The same prompt again: from the checkpoint before its last P tokens.
    e.reset(false);
    const s2 = try e.sync(prompt, null);
    try testing.expectEqual(prompt.len - P, s2.cached);
    logits = s2.logits;
    for (live) |want| {
        try testing.expectEqual(want, argmax(logits));
        logits = try e.step(want);
    }
    // The prompt and 17 generated tokens: from the checkpoint at +16.
    const p3 = try std.mem.concat(a, u32, &.{ prompt, live[0..17] });
    defer a.free(p3);
    e.reset(false);
    const s3 = try e.sync(p3, null);
    try testing.expectEqual(prompt.len + 16, s3.cached);
    logits = s3.logits;
    for (live[17..]) |want| {
        try testing.expectEqual(want, argmax(logits));
        logits = try e.step(want);
    }
}

test "pack on the ANE: one lane decodes like two, taps included (ANE_TEST_PACK)" {
    const io = testing.io;
    const a = testing.allocator;
    if (ane.aneInstanceCount(ane.chipBrand()) < 2) return error.SkipZigTest;
    const taps = [_]u32{ 1, 5 };
    const prompt = try testPrompt(a, 100);
    defer a.free(prompt);
    var got: [2][2]u32 = undefined;
    var tap_rows: [2][taps.len][]f32 = @splat(@splat(&.{}));
    defer for (tap_rows) |rows| for (rows) |r| a.free(r);
    // One model at a time: two big ones do not both fit on the ANE.
    for ([_]u32{ 1, 2 }, 0..) |lanes, k| {
        const m = (try testModelWith(io, .{ .lanes = lanes, .taps = &taps })) orelse return error.SkipZigTest;
        defer m.deinit();
        if (!m.arch.splits()) return error.SkipZigTest;
        const e = try Engine.open(m, io, 0);
        defer e.deinit();
        got[k][0] = argmax((try e.sync(prompt, null)).logits);
        got[k][1] = argmax(try e.step(got[k][0]));
        for (taps, 0..) |l, i| {
            var nb: [16]u8 = undefined;
            const t = findBuf(e.tapbufs[0].items, try std.fmt.bufPrint(&nb, "tap{d}", .{l})).?.t;
            tap_rows[k][i] = try a.alloc(f32, m.arch.hidden);
            for (tap_rows[k][i], 0..) |*v, h| v.* = t.f16At(&.{ 0, @intCast(h), 0, 0 }).*;
        }
    }
    try testing.expectEqual(got[1], got[0]);
    for (tap_rows[0], tap_rows[1]) |x, y| try testing.expect(cosine(x, y) > 0.999);
}

test "pack on the ANE: two slots interleaved token by token decode like each alone (ANE_TEST_PACK)" {
    const io = testing.io;
    const a = testing.allocator;
    const m = (try testModel(io, 2)) orelse return error.SkipZigTest;
    defer m.deinit();
    const e0 = try Engine.open(m, io, 0);
    defer e0.deinit();
    const e1 = try Engine.open(m, io, 1);
    defer e1.deinit();
    const p1 = try testPrompt(a, 90);
    defer a.free(p1);
    const p2 = try testPrompt(a, 130);
    defer a.free(p2);
    var w1: [16]u32 = undefined;
    var w2: [16]u32 = undefined;
    try greedy(e0, (try e0.sync(p1, null)).logits, w1.len, &w1);
    // From a fresh state as slot 1's will be: p1 and its greedy tokens can be a prefix of p2.
    e0.reset(true);
    try greedy(e0, (try e0.sync(p2, null)).logits, w2.len, &w2);
    e0.reset(true);
    var l1 = (try e0.sync(p1, null)).logits;
    var l2 = (try e1.sync(p2, null)).logits;
    for (w1, w2) |t1, t2| {
        try testing.expectEqual(t1, argmax(l1));
        l1 = try e0.step(t1);
        try testing.expectEqual(t2, argmax(l2));
        l2 = try e1.step(t2);
    }
}
