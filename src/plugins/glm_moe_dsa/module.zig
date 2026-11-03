//! GLM-5.3's module: the arch's decode state and its phases, over a bank module `Bk` and a routed-expert quant `Q`
//! bound at comptime (`ModuleOf`; the served module binds the affine bank and MLX's `gather_qmm`). Construction checks
//! the pack (the residents against the spec from the shard headers, the bank's manifest, the quant's claim), bills
//! both phases for the longest request the load takes (the billed context and `max_output` generated tokens) and
//! fills the slot rows under the host's ceiling less its wired margin before any slot bank is allocated, binds the
//! residents and reads them into their buffers, starts the stream (prompt rows, the read pool sized from the bank's
//! widest span, the decode lookahead, the event gates), allocates nothing else until a prompt, and checks its footprint against the bill's construction
//! terms (`ConstructionOverBill`). A request: the reverse phase change if the previous one decoded, the KV lanes at
//! the prompt's positions, the prompt pass (layer by layer in calls of `glm_moe_dsa.promptCallRows`, or chunk by
//! chunk), the decode handover (the transient scratch freed, the KV lanes at the request's own prompt and `max_tokens`,
//! MLX's counters read once the host side is back at its reading before the frees, the slot rows grown to what the bill
//! at the request admits and the reading leaves under the target), serial steps. A later prompt keeps the KV of the prefix it shares with the state
//! (`restorePrefix`). One log line at the prompt pass's end, one at the handover and one at the request's end report
//! the phase's reads from the stream's counters and its memory from MLX's and the kernel's (`PromptLine`,
//! `HandoverLine`, `DecodeLine`). With `GLM53_ROUTES` set at construction, the route recorder (`expert.routes`) writes
//! each request's decode routes at its end.
//!
//! With `mtp_depth` set, the MTP draft lane (`glm_moe_dsa_mtp`, its bank kind bound at comptime: the served module's is
//! EXL3): the prompt pass keeps every row's final-normed hidden and appends the MTP layer's keys of each pair whose next
//! token it knows; a round drafts `mtp_depth` tokens, verifies `[t1, drafts]` in one forward of the target
//! (`graph.Want.verify`), decides under the lane's acceptance and the request's sampling, and truncates the target's
//! lanes to the accepted rows; the MTP layer appends nothing for a draft, so a rejection leaves its cache as it was.

const std = @import("std");
const env = @import("mlx_host").env;
const host_caps = @import("../../sdk/weights/streamed/host_caps.zig");
const mlx = @import("sdk").mlx;
const sdk = @import("sdk");
const log = @import("sdk").log;
const sdk_ext = @import("../../sdk/weights/streamed/sdk_ext.zig");
const builtin = @import("builtin");
const quant = sdk_ext.quant;
const expert_event = sdk_ext.expert.event;
const expert_routes = sdk_ext.expert.routes;
/// The route recorder's builds (profile and test): `expert.stream.recording`.
const recording = sdk_ext.expert.stream.recording;
const expert_tally = sdk_ext.expert.tally;
const glm = @import("glm_moe_dsa.zig");
const settings = @import("settings.zig");
const bill_mod = @import("bill.zig");
const split_mod = @import("split.zig");
const graph = @import("graph.zig");
const topk_mod = @import("topk.zig");
const trunk = @import("trunk_kernels.zig");
const bank_mod = @import("bank.zig");
const experts_mod = @import("experts.zig");
const mtp_mod = @import("mtp.zig");
const exl3_bank = @import("exl3_bank.zig");
const exl3_quant = @import("exl3_quant.zig");
const glm53_quant = @import("glm53_exl3/quant.zig");
/// PROFILE builds (`-Dplugin-profile=true`): each draft round's verify on the GPU timeline, its summary at the
/// request's end (`VERIFY_GPU_TIMELINE`); every other build compiles it out.
const timeline = @import("../../sdk/weights/streamed/profile/verify_timeline.zig");
const prefill_timers = @import("prefill_timers.zig");
const phase_mem = @import("../../sdk/weights/streamed/phase_memory.zig");

const G = graph.G;
/// The plugin's draw for every row handed to the host (`sdk_ext.sampler`): bound per request by `draft_lane.arm`.
pub const Sampler = @import("../../sdk/weights/streamed/sdk_ext.zig").sampler.SamplerOf(G);
const Stats = sdk_ext.expert.Stats;
const LayerCounts = sdk_ext.expert.LayerCounts;

/// The decode lookahead: the next routed layer's top-8 candidates (its records per call: `expert_lookahead_budget`).
pub const lookahead: struct { k: u32 = 8 } = .{};
/// A gate whose bytes never land is forced after this (and fails the stream).
pub const event_watchdog_ms: u32 = 2000;
/// The chunk-major prompt pass's chunk (`layer_major_prefill` off).
pub const prefill_chunk_tokens: u32 = 2048;
/// How far the constructed footprint may sit above the bill's construction terms (the ledger's page rounding and the
/// host side's own movement; `phase_memory`'s, shared with DeepSeek-V4.1's module).
pub const construction_tolerance_bytes = phase_mem.construction_tolerance_bytes;
/// A phase change's settle (`settle`).
pub const settle_poll_ms = phase_mem.settle_poll_ms;
pub const settle_max_ms = phase_mem.settle_max_ms;
/// The route recorder's diagnostic switch (`expert.routes`), read once at construction.
pub const routes_env = "GLM53_ROUTES";
/// The wave probe's diagnostic switch (`graph.WaveProbe`), read once at construction.
pub const wave_probe_env = "GLM53_WAVE_PROBE";

/// What the host hands the construction: the GPU ceiling and the wired margin (`LoadCtx`), never read elsewhere.
pub const Host = struct { ceiling: u64, wired_margin: u64 };

/// A harness's construction routes (the served path passes none).
pub const Overrides = struct {
    /// The chunk-major pass's chunk (null: `prefill_chunk_tokens`).
    prefill_chunk: ?u32 = null,
};

/// The served modules, the MTP layer's experts from its EXL3 records in both: the affine bank through MLX's
/// `gather_qmm`; the EXL3 bank (its K3 and K4 bank layers per routed layer) through GLM-5.3's own EXL3 kernels
/// (`glm53_exl3_quant`). The pack's manifest picks one at load (`Served`). `Exl3SushiModule`: the EXL3 bank through
/// sushi's EXL3 MoE, the comparison's.
pub const Module = ModuleOf(bank_mod, quant.FromGatherMatmul(quant.GatherQmm), .exl3);
pub const Exl3Module = ModuleOf(exl3_bank, glm53_quant, .exl3);
pub const Exl3SushiModule = ModuleOf(exl3_bank, exl3_quant, .exl3);

/// The module the pack's bank serves: the EXL3 one when the pack carries the EXL3 manifest, else the affine one.
pub const Served = union(enum) {
    affine: *Module,
    exl3: *Exl3Module,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, weights: *sdk.Weights, s: mlx.mlx_stream, host: Host) !*Served {
        const self = try gpa.create(Served);
        errdefer gpa.destroy(self);
        const dir = cfg.model_dir orelse return error.GlmPackDir;
        self.* = if (exl3_bank.present(dir)) .{ .exl3 = try Exl3Module.init(gpa, io, cfg, weights, s, host) } else .{ .affine = try Module.init(gpa, io, cfg, weights, s, host) };
        return self;
    }

    /// The longest prompt the construction billed: the host's glue refuses a longer one before any work.
    pub fn maxContext(self: *const Served) u64 {
        return switch (self.*) {
            inline else => |m| m.max_context,
        };
    }

    pub fn deinit(self: *Served) void {
        const gpa = switch (self.*) {
            inline else => |m| m.gpa,
        };
        switch (self.*) {
            inline else => |m| m.deinit(),
        }
        gpa.destroy(self);
    }
};

pub fn ModuleOf(comptime Bk: type, comptime Q: type, comptime mtp_kind: mtp_mod.BankKind) type {
    comptime quant.checkAccepted(Q, G);
    return struct {
        const Self = @This();
        const Stream = Bk.Stream.Stream;
        const Math = Q.Accepted(G);
        /// A quant with pinned texts gets the load's kernel set over its registry, built on the module's stream.
        const has_kernels = @hasDecl(Q, "kernel_registry");
        const KSet = if (has_kernels) sdk_ext.kernels.Set(Q.kernel_registry) else void;
        pub const Experts = experts_mod.Experts(G, Bk, Math);
        pub const Lane = mtp_mod.Lane(mtp_kind);

        /// `Stream.preload`'s order: each bank layer's experts in id order (with several bank layers per routed layer,
        /// each holds only its own).
        const Preload = struct {
            bank: *const Bk.Bank,
            buf: []u16,

            fn order(p: *Preload, layer: u32) []const u16 {
                var n: usize = 0;
                for (0..p.buf.len) |e| {
                    if (Experts.bpl > 1 and p.bank.streamLayer(layer / Experts.bpl, @intCast(e)) != layer) continue;
                    p.buf[n] = @intCast(e);
                    n += 1;
                }
                return p.buf[0..n];
            }
        };

        gpa: std.mem.Allocator,
        io: std.Io,
        g: G,
        /// The request's sampler (mlx-stream's one stream): the first token and every serial step's, a clean request's.
        sampler: Sampler,
        model: *const glm.Config,
        cfg: settings.Config,
        w: graph.Weights,
        bank: Bk.Bank,
        stream: *Stream,
        math: *Math,
        kset: if (has_kernels) *KSet else void,
        ex: Experts,
        cache: graph.Cache,
        /// The tokens whose KV the lanes hold, in order (the prefix a later prompt may keep).
        history: std.ArrayList(u32) = .empty,
        /// The longest prompt the construction billed (the host refuses a longer one before any work).
        max_context: u64,
        /// The most positions decode's KV lanes take: the billed context, `max_output` and a round's verify.
        max_positions: u64,
        /// Each stream layer's rows at the prompt's units (`bill.rowsAt`) and at the units the handover grows to
        /// (`decodeRowsAt`; the construction's fill until the first handover, then its request's).
        prompt_rows: []u32,
        decode_rows: []u32,
        prompt_units: u32,
        decode_units: u32,
        /// The bytes the handover's grow adds past the prompt rows (`bill.grownOf`), in the decode records' slot rows.
        grown_bytes: u64 = 0,
        /// The rows the draft lane held pending at the handover: the decode records bill them.
        decode_pending_rows: u64 = 0,
        /// The harness forced the rows (`expert_rows`): the handover grows to them, no live fill.
        forced_rows: bool,
        /// The allocator's size-class switch the construction bound (`transientClasses`).
        size_classes: bill_mod.SizeClassSwitch,
        /// The decode rows are the split's (`glm_moe_dsa_split`), not the even split's.
        split_rows: bool = false,
        decoding: bool = false,
        /// The bill's inputs (the handover bills its request's KV from them), the load's baseline, MLX's target (the
        /// host's ceiling less its wired margin) and the box's RAM (the handover's target, `bill.targetOf`).
        inputs: bill_mod.Inputs,
        /// The construction's bill, rows-free (`bill.Bill`): the phase records' billed terms, the cache limits the module
        /// installs, the settle's host-side bound.
        billed: bill_mod.Bill,
        baseline: u64,
        gpu_target: u64,
        ram: u64,
        /// Decode's window rows (`bill.decodeWindowRows`): a decode call over more routed ids is refused.
        window_ids: u32,
        routes: graph.Routes,
        /// The indexer's top-k route over the trunk set where the construction takes it (`graph.Routes.topk`).
        topk_kernels: ?topk_mod.Kernels = null,
        /// The trunk's kernels (the indexer's top-k, the routed combine): the pinned set (`trunk_kernels`) built at
        /// construction where a route takes one of them, each route's kernels run once there at the model's
        /// instantiation; null on a CPU stream or where neither route is the kernels'.
        trunk_set: ?*trunk.Set = null,
        overrides: Overrides,
        event: ?expert_event.Event = null,
        prev_cache_limit: usize = 0,
        bill: sdk.MemoryBill,
        /// The request's decode since its handover, for its request-end line; null before the handover and after the line.
        decode_mark: ?DecodeMark = null,
        /// Each routed layer's counts at the handover and at the request's end, and the line's hit rates.
        layer_counts0: []LayerCounts,
        layer_counts1: []LayerCounts,
        layer_rates: []f64,
        /// The MTP draft lane (null: `mtp_depth` 0).
        mtp: ?*Lane = null,
        /// The route recorder (`routes_env`; null: off) and the directory its files go to.
        recorder: ?*expert_routes.Recorder = null,
        routes_dir: []const u8 = "",
        /// The box's file-backed pages when the construction started: each phase record's page cache is counted from them.
        file_backed_start: u64 = 0,
        /// The wave probe (`wave_probe_env`; null: off).
        probe: ?*graph.WaveProbe = null,
        /// The positions the request's KV lanes hold since its handover (its decode terms' KV).
        decode_positions: u64 = 0,
        /// The rate tally's file (`expert.tally.pathOf`; null: none, as in test builds) and the rates each request's end
        /// writes to it (every stream layer's `n_experts`).
        tally_path: ?[]u8 = null,
        tally_rates: []f32 = &.{},

        const DecodeMark = struct { s0: Stats, mem0: Mem, by0: Bk.Stream.ByRank, steps: u64 = 0, tokens: u64 = 0, wall_ns: u64 = 0 };

        pub fn init(gpa: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, weights: *sdk.Weights, s: mlx.mlx_stream, host: Host) !*Self {
            return initWith(gpa, io, cfg, weights, s, host, .{});
        }

        pub fn initWith(gpa: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, weights: *sdk.Weights, s: mlx.mlx_stream, host: Host, ov: Overrides) !*Self {
            try cfg.checkRetired();
            try cfg.checkCtxSize();
            try cfg.checkMaxOutput();
            try cfg.checkUnitsCap();
            try cfg.tally.check("glm_moe_dsa");
            try cfg.checkMtp();
            // The handover bills the rest of the box from the kernel's page counts: a box that gives none is refused
            // here, once, never billed as an empty box.
            if (sdk.memory.boxUsedBytes() == null) {
                log.err("glm_moe_dsa: load refused: the kernel gives no reading of the box's used memory (host_statistics64), and the decode rows are billed beside it (BoxReadingUnavailable)\n", .{});
                return error.BoxReadingUnavailable;
            }
            const file_backed_start = sdk.memory.vmBytes().external;
            const dir = cfg.model_dir orelse return error.GlmPackDir;
            const model: *const glm.Config = if (cfg.model) |*m| m else return error.GlmPackDir;
            var diag: glm.Diag = .{};
            errdefer if (diag.len > 0) log.err("glm_moe_dsa: load refused: {s}\n", .{diag.message()});
            // The pack, host-side, before any slot bank: the residents against the spec, the bank's manifest.
            // The allocator's size classes when the process's MLX has the switch (the bill charges them exactly then).
            const size_classes = bill_mod.sizeClassSwitch();
            const resident_bytes = try glm.residentBufferBytes(gpa, io, dir, model, std.heap.pageSize(), size_classes != null, &diag);
            var bank = try Bk.Bank.open(gpa, io, dir, model, &diag);
            errdefer bank.deinit();
            // The admission: the prompt phase billed at the context, decode at the longest request (the context and
            // `max_output`), the rows filled up to the target (or forced).
            const gpu_target = host.ceiling -| host.wired_margin;
            const ram = sdk.memory.totalMemBytes();
            const max_context = bill_mod.servedContext(cfg);
            // The prompt's attention form this device runs (the bill charges its arrays), and the indexer's top-k route: the
            // kernel on a GPU stream where it serves the model's top-k, else argpartition (the bill charges its ids).
            const dsa = try graph.dsaKernelsRun(gpa, s, model);
            // The draft lane's bill (its pack refused by name here, before the stream is read).
            const mtp_inputs = try bill_mod.mtpOf(gpa, io, cfg, mtp_kind, &diag);
            const select: bill_mod.Select = if (topk_mod.serves(model.index_topk) and mlx.streamIsGpu(s)) .kernel else .argpartition;
            const inputs: bill_mod.Inputs = .{ .model = model, .bank = bank.geometryOf(), .resident_bytes = resident_bytes, .stream = bill_mod.streamShape(cfg), .prompt_tokens = max_context, .decode_positions = bill_mod.maxPositions(cfg), .mtp = mtp_inputs, .dsa = dsa, .gemm = comptime bill_mod.promptGemmOf(Q), .select = select, .prompt_cache = bill_mod.prefillCacheBytes(ram), .size_classes = size_classes != null };
            const billed = bill_mod.billOf(inputs);
            const mb = try bill_mod.memoryBill(gpa, billed, null);
            errdefer mb.free(gpa);
            const baseline = bill_mod.restOfBox(cfg.memory_baseline_bytes orelse 0, cfg.rest_of_box_floor_bytes);
            const target = bill_mod.targetOf(gpu_target, ram, baseline);
            // The fill's unit (`bill.unitBytes`): every stream layer's share of its experts (`bill.rowsAt`).
            const max_units = bill_mod.maxUnits(inputs.bank);
            const rows: sdk.Rows = if (cfg.expert_rows) |forced| .{ .prompt = @min(cfg.expert_prefill_rows orelse forced, forced), .decode = forced } else bill_mod.capRows(sdk.fill(mb, baseline, target, max_units, bill_mod.min_fill_rows) catch |e| {
                log.err("glm_moe_dsa: admission refused: {s} (baseline {d} B, target {d} B, {d} B a row)\n", .{ @errorName(e), baseline, target, mb.per_row });
                return e;
            }, cfg.expert_units_cap);
            if (rows.prompt > rows.decode or rows.decode > max_units) return error.InvalidRows;
            sdk.admit(mb, baseline, rows, target) catch |e| {
                log.err("glm_moe_dsa: admission refused before construction: {s} (prompt {d} B, decode {d} B, target {d} B)\n", .{ @errorName(e), mb.total(.prompt, baseline, rows.prompt), mb.total(.decode, baseline, rows.decode), target });
                return e;
            };
            log.info("glm_moe_dsa: admission {d} prompt / {d} decode units of {d} B ({d}-token context, {d} generated, baseline {d} B, target {d} B, the GPU's {d} B), units cap {d} (0: off), rest-of-box floor {d} B (0: off; the baseline read {d} B)\n", .{ rows.prompt, rows.decode, mb.per_row, max_context, cfg.maxOutput(), baseline, target, gpu_target, if (cfg.expert_rows == null) cfg.expert_units_cap orelse 0 else 0, cfg.rest_of_box_floor_bytes orelse 0, cfg.memory_baseline_bytes orelse 0 });
            log.info("{f}\n", .{bill_mod.BillLine{ .mb = &mb, .rows = rows, .baseline = baseline, .target = target, .context = max_context, .decode_positions = inputs.decode_positions }});

            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);
            self.* = .{ .gpa = gpa, .io = io, .g = try G.init(gpa, s), .model = model, .cfg = cfg.*, .w = undefined, .bank = bank, .stream = undefined, .math = undefined, .kset = undefined, .ex = undefined, .cache = undefined, .max_context = max_context, .max_positions = inputs.decode_positions, .prompt_rows = &.{}, .decode_rows = &.{}, .prompt_units = rows.prompt, .decode_units = rows.decode, .forced_rows = cfg.expert_rows != null, .size_classes = size_classes orelse &noSizeClasses, .inputs = inputs, .baseline = baseline, .gpu_target = gpu_target, .ram = ram, .window_ids = bill_mod.decodeWindowRows(cfg), .routes = .{}, .overrides = ov, .bill = mb, .billed = billed, .layer_counts0 = &.{}, .layer_counts1 = &.{}, .layer_rates = &.{}, .sampler = undefined };
            errdefer self.g.deinit();
            self.sampler = try Sampler.init(gpa, s);
            errdefer self.sampler.deinit();
            // The module owns its config's copy (the host's config keeps the parsed model's storage).
            self.model = if (self.cfg.model) |*m| m else unreachable;
            self.inputs.model = self.model;
            // The quant the bank's description is claimed by, accepted on this backend.
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const peek = try self.bank.peek(arena.allocator());
            if (Q.claims(&peek, &diag) == null) return error.QuantNotClaimed;
            var kref: ?sdk_ext.kernels.SetRef = null;
            if (comptime has_kernels) {
                self.kset = try KSet.init(gpa, .{ .device = .{ .stream = s } }, &diag);
                // the profile hook's launch probe sees the set's launches (profile builds; a no-op otherwise)
                self.kset.bound.observe(sdk_ext.profile.of(G).launch);
                kref = self.kset.ref();
            }
            errdefer if (comptime has_kernels) self.kset.deinit();
            self.math = try Q.accept(G, gpa, &self.g, .{ .kernels = kref, .peek = &peek }, .{ .hidden = model.hidden_size, .inter = model.moe_intermediate_size, .top_k = model.n_experts_per_tok, .n_layers = model.nSparse(), .act = .swiglu, .input = .bfloat16 }, &diag);
            errdefer self.math.deinit(&self.g);
            // The quant's self-check freed its arrays into MLX's cache: emptied, so the stream's arrays and the draft lane's
            // are allocated at their own sizes (a cached buffer serves a request up to two pages smaller than itself).
            self.g.clearCache();
            const n_bank: usize = self.inputs.bank.layers.len;
            self.prompt_rows = try gpa.alloc(u32, n_bank);
            errdefer gpa.free(self.prompt_rows);
            self.decode_rows = try gpa.alloc(u32, n_bank);
            errdefer gpa.free(self.decode_rows);
            _ = bill_mod.rowsAt(self.inputs.bank, rows.prompt, self.prompt_rows);
            if (self.forced_rows) _ = bill_mod.rowsAt(self.inputs.bank, rows.decode, self.decode_rows) else self.decodeRowsAt(rows.decode, bill_mod.grownBytes(self.inputs.bank, rows.prompt, rows.decode));
            if (n_bank > 1 and self.inputs.bank.layers.len > self.model.nSparse()) log.info("glm_moe_dsa: {d} bank layers, rows per bank layer prompt {d} / {d}, decode {d} / {d} (the first routed layer's two)\n", .{ n_bank, self.prompt_rows[0], self.prompt_rows[1], self.decode_rows[0], self.decode_rows[1] });
            self.layer_counts0 = try gpa.alloc(LayerCounts, n_bank);
            errdefer gpa.free(self.layer_counts0);
            self.layer_counts1 = try gpa.alloc(LayerCounts, n_bank);
            errdefer gpa.free(self.layer_counts1);
            self.layer_rates = try gpa.alloc(f64, n_bank);
            errdefer gpa.free(self.layer_rates);
            // The stream: slot rows in MLX arrays on this stream, the event gates on its device.
            const gated = cfg.eventGates();
            const gpu = mlx.streamIsGpu(s);
            if (gated and gpu) self.event = try expert_event.createMetal();
            const shape = bill_mod.streamShape(cfg);
            // The rate tally's geometry: each stream layer's experts (a routed layer's bank layers split its experts).
            const layer_experts = try gpa.alloc(u32, n_bank);
            defer gpa.free(layer_experts);
            for (layer_experts, 0..) |*n, sl| {
                n.* = 0;
                for (0..self.bank.n_experts) |e| n.* += @intFromBool(Experts.bpl == 1 or self.bank.streamLayer(@intCast(sl / Experts.bpl), @intCast(e)) == sl);
            }
            self.stream = try Stream.init(gpa, &self.bank, .{
                .rows = self.prompt_rows,
                .max_route_ids = shape.max_route_ids,
                .transient_rows = @as(u32, shape.wide_depth) * shape.max_route_ids * self.inputs.bank.rows_per_id,
                .wide_depth = shape.wide_depth,
                .transient_release = true,
                .decode_route_ids = shape.decode_window_rows * self.inputs.bank.rows_per_id,
                .slot_memory = .{ .mlx = s },
                .staging_from_bank = true,
                .records_per_part = @min(3, sdk_ext.expert.io.max_items / Bk.Stream.maxMinis(&self.bank)),
                .pool = .{ .workers = shape.workers, .tickets = bill_mod.pool_tickets_per_mini * Bk.Stream.maxMinis(&self.bank) },
                .lookahead = .{ .k = lookahead.k, .budget = cfg.lookaheadBudget() },
                .event = if (!gated) null else if (gpu) .{ .backend = .{ .metal = self.event.?.object }, .watchdog_ms = event_watchdog_ms } else .{ .backend = .host, .watchdog_ms = event_watchdog_ms },
                .tally = if (cfg.tally.enabled()) .{ .ratio = cfg.tally.ratio, .window = cfg.tally.window, .layer_experts = layer_experts } else null,
            });
            errdefer self.stream.deinit();
            try self.openTally(dir);
            errdefer self.closeTally();
            // The recorder exists in profile builds only (`expert.stream.recording`): a served build reads no switch.
            if (comptime recording) if (expert_routes.dirOf(env.named(routes_env))) |d| {
                self.recorder = try expert_routes.Recorder.init(gpa, @intCast(n_bank), self.bank.n_experts, model.n_experts_per_tok, Experts.bpl);
                self.routes_dir = d;
                self.stream.recorder = self.recorder;
                log.info("glm_moe_dsa: {s} is set: each request's decode routes are written under {s}\n", .{ routes_env, d });
            };
            errdefer if (self.recorder) |rec| rec.deinit();
            if (gated and !gpu) self.event = try expert_event.createHost(@constCast(self.stream.eventWord().?), @as(i64, event_watchdog_ms) * std.time.ns_per_ms);
            self.w = try graph.Weights.bind(gpa, weights, model, &diag);
            errdefer self.w.deinit(gpa);
            try self.residentsNow(weights);
            // The trunk's kernel routes (the indexer's top-k, the routed combine) on a GPU stream at instantiations the
            // registry serves: its pinned set built here and each route's kernels run once at the model's instantiation
            // (`trunk_selfcheck.runRoutes`), never inside a request; elsewhere argpartition and the op chain's combine.
            const combine_route = gpu and trunk.registersCombine(model.n_experts_per_tok, model.hidden_size);
            if (select == .kernel or combine_route) self.trunk_set = try trunk.Set.init(gpa, .{ .device = .{ .stream = s } }, &diag);
            errdefer if (self.trunk_set) |t| t.deinit();
            if (self.trunk_set) |t| {
                // the profile hook's launch probe sees the set's launches (profile builds; a no-op otherwise)
                t.bound.observe(sdk_ext.profile.of(G).launch);
                var routes_at: [trunk.topk_kernels.len + 1]trunk.selfcheck.Route = undefined;
                var n_routes: usize = 0;
                if (select == .kernel) for (trunk.topk_kernels) |k| {
                    routes_at[n_routes] = .{ .kernel = k, .at = .{ .k = model.index_topk } };
                    n_routes += 1;
                };
                if (combine_route) {
                    routes_at[n_routes] = .{ .kernel = .glm_moe_dsa_combine, .at = .{ .k = model.n_experts_per_tok, .hidden = model.hidden_size } };
                    n_routes += 1;
                }
                var report: trunk.selfcheck.Report = .{};
                defer report.deinit(gpa);
                try trunk.selfcheck.runRoutes(gpa, &t.reg, &t.bound, routes_at[0..n_routes], &report, &diag);
            }
            self.ex = try Experts.init(gpa, &self.g, self.stream, self.math, model.hidden_size, .{ .gated = gated, .event = self.event, .wide_depth = shape.wide_depth, .combine = if (combine_route) &self.trunk_set.?.bound else null });
            errdefer self.ex.deinit(&self.g);
            // The slot arrays are the quant's signature (every routed layer's base bank).
            for (self.ex.banks) |b| if (b[@backingInt(experts_mod.BankKind.base)]) |arr| try self.math.checkBank(&self.g, arr, &diag);
            // The lanes allocate at their first append; each prompt and each handover sets their positions (`resizeKv`).
            self.cache = try graph.Cache.init(gpa, model, @intCast(max_context));
            errdefer self.cache.deinit(&self.g);
            if (cfg.mtpDepth() > 0) self.mtp = try Lane.open(gpa, io, &self.g, dir, model, cfg.mtpDepth(), cfg.mtpAcceptance(), @intCast(max_context), cfg.nocache_weights orelse true, &diag);
            errdefer if (self.mtp) |ln| ln.deinit(&self.g);
            // The lane's open measured what it holds (the DeepSeek-V4.1 way: a measured install against its billed term).
            if (self.mtp) |ln| {
                const term = self.billed.phases[@intFromEnum(bill_mod.Phase.prompt)].mtp_residents;
                log.info("glm_moe_dsa: MTP lane: {d} B measured at its open, {d} B billed, residual {d} B\n", .{ ln.measured_bytes, term, @as(i64, @intCast(term)) - @as(i64, @intCast(ln.measured_bytes)) });
                if (ln.measured_bytes > term) {
                    log.err("glm_moe_dsa: load refused: the MTP lane holds {d} B, over its billed {d} B (ConstructionOverBill)\n", .{ ln.measured_bytes, term });
                    return error.ConstructionOverBill;
                }
            }
            // The prompt rows filled before the first request, each bank layer's in expert order (a long prompt routes
            // nearly every expert, so any filled row is a hit); a later request finds its predecessor's residents (the
            // reverse change keeps them, `reverse`).
            var pre: Preload = .{ .bank = &self.bank, .buf = try gpa.alloc(u16, self.bank.n_experts) };
            defer gpa.free(pre.buf);
            const pre0 = self.stream.stats();
            const t_pre = std.Io.Timestamp.now(io, .awake);
            const preloaded = try self.stream.preload(&pre, Preload.order);
            const pre_ns: u64 = @intCast(t_pre.untilNow(io, .awake).nanoseconds);
            log.info("glm_moe_dsa: preloaded {d} experts ({d:.2} GB) into the prompt rows in {d:.2} s\n", .{ preloaded, gigabytes(Reads.of(pre0, self.stream.stats()).ssd_bytes), seconds(pre_ns) });
            // The top-k route over the trunk set (its kernels run once at construction above), never built in a prompt.
            if (select == .kernel) self.topk_kernels = .{ .bound = &self.trunk_set.?.bound };
            self.routes = .{ .read_ahead = true, .lookahead = true, .max_route_ids = shape.max_route_ids, .dsa = dsa, .topk = if (self.topk_kernels) |*k| k else null };
            if (self.mtp) |ln| ln.topk = self.routes.topk;
            self.file_backed_start = file_backed_start;
            if (env.named(wave_probe_env) != null) {
                self.probe = try gpa.create(graph.WaveProbe);
                self.probe.?.* = .{};
                self.routes.wave_probe = self.probe;
                log.info("glm_moe_dsa: {s} is set: each prompt layer's MLX peak by stage is logged at the pass's end\n", .{wave_probe_env});
            }
            errdefer if (self.probe) |p| gpa.destroy(p);
            self.g.clearCache();
            _ = mlx.mlx_set_cache_limit(&self.prev_cache_limit, self.cacheLimit(.prompt));
            log.info("glm_moe_dsa: constructed ({d} routed layers, {d} experts, record {d} B, expert reads {s}, prompt {s}, prompt attention {s}, decode window {d} rows)\n", .{ n_bank, self.bank.n_experts, self.bank.geometryOf().widest_record, if (gated) "event gates" else "host waits", if (cfg.layerMajor()) "layer by layer" else "chunk by chunk", if (self.routes.dsa) "on the DSA kernels" else "in the reference's form", shape.decode_window_rows });
            try self.checkConstruction(rows.prompt);
            self.phaseRecord("module constructed", self.billed.terms(.construction, self.prompt_units, self.prompt_units), null);
            return self;
        }

        /// The trunk's residents read into their buffers at construction, as the draft lane's are at its open: the
        /// loader's arrays are lazy, and each layer's would be read inside the first prompt, its qmms waiting on the
        /// SSD beside the expert stream's reads. Their bytes as MLX counts them, beside the bill's term (the
        /// construction check holds the footprint to it).
        fn residentsNow(self: *Self, weights: *const sdk.Weights) !void {
            self.g.clearCache();
            var active0: usize = 0;
            _ = mlx.mlx_get_active_memory(&active0);
            const t0 = std.Io.Timestamp.now(self.io, .awake);
            try weights.materialize();
            const ns: u64 = @intCast(t0.untilNow(self.io, .awake).nanoseconds);
            var active1: usize = 0;
            _ = mlx.mlx_get_active_memory(&active1);
            const term = self.billed.phases[@intFromEnum(bill_mod.Phase.construction)].residents;
            log.info("glm_moe_dsa: {d} residents read into their buffers in {d:.2} s: {d} B measured, {d} B billed, residual {d} B\n", .{ weights.count(), seconds(ns), active1 -| active0, term, @as(i64, @intCast(term)) - @as(i64, @intCast(active1 -| active0)) });
        }

        /// Once, at construction (every command retired, MLX's cache cleared): the footprint within
        /// `construction_tolerance_bytes` of the bill's construction terms at the prompt rows, and the host side past the
        /// read pool's staging within its measured bound; refused by name (`ConstructionOverBill`). The arithmetic is
        /// `phase_memory.Construction`'s (DeepSeek-V4.1's module's too).
        fn checkConstruction(self: *Self, prompt_rows: u32) !void {
            _ = mlx.mlx_synchronize(self.g.s);
            self.g.clearCache();
            const m = Mem.now();
            const box = try Box.now(m);
            var cc = phase_mem.Construction.of(self.bill, prompt_rows, m.footprint, m.active, m.cache);
            cc.host_known = self.billed.phases[@intFromEnum(bill_mod.Phase.construction)].pool_staging;
            // Another model in the process (the host's reading, #8): the footprint holds its bytes, the bill does not.
            cc.foreign = phase_mem.foreignBytes();
            log.info("glm_moe_dsa: construction check: footprint {d} B (MLX active {d} B, cache {d} B, host side {d} B), billed construction terms {d} B, other models in the process {d} B, residual {d} B (tolerance {d} B); the box {d} B used, the rest of it {d} B (the load's baseline {d} B, billed at {d} B)\n", .{ m.footprint, m.active, m.cache, m.hostSide(), cc.billed, cc.foreign, cc.residual(), construction_tolerance_bytes, box.used, box.others, self.cfg.memory_baseline_bytes orelse 0, self.baseline });
            cc.checkFootprint(construction_tolerance_bytes) catch |e| {
                log.err("glm_moe_dsa: load refused: the constructed footprint {d} B exceeds the billed construction terms {d} B and the other models' {d} B by more than {d} B ({s})\n", .{ m.footprint, cc.billed, cc.foreign, construction_tolerance_bytes, @errorName(e) });
                return e;
            };
            if (cc.overMeasured(self.bill)) |t| {
                log.err("glm_moe_dsa: load refused: the {s} {d} B exceeds its billed bound {d} B (ConstructionOverBill)\n", .{ t.name, cc.measuredHost(), t.atConstruction() });
                return error.ConstructionOverBill;
            }
        }

        /// The request's decode terms at its grown units: the bill at its own KV (`bill.atRequest`).
        fn requestTerms(self: *const Self) bill_mod.PhaseTerms {
            return bill_mod.atRequest(self.inputs, self.decode_positions, self.decode_pending_rows).termsAt(.decode, bill_mod.slotBytes(self.inputs.bank, self.prompt_units) + self.grown_bytes);
        }

        /// The MLX cache limit the module installs for `phase` (the bill's own term: one value for both).
        fn cacheLimit(self: *const Self, phase: bill_mod.Phase) usize {
            return @intCast(self.billed.phases[@intFromEnum(phase)].mlx_cache);
        }

        /// One phase boundary's record (`phase_memory.Record` over the bill's terms `billed`), logged as one JSON line;
        /// then both high-water marks restart for the next phase. `peaks`: MLX's peak and the footprint's over the phase
        /// when the wave probe restarted them within it (the probe's own).
        fn phaseRecord(self: *Self, phase: []const u8, billed: bill_mod.PhaseTerms, peaks: ?Peaks) void {
            var r = phase_mem.Reading.now();
            if (peaks) |p| {
                r.peak = @max(r.peak, p.mlx);
                r.process.footprint_interval_peak = @max(r.process.footprint_interval_peak, p.footprint);
            }
            phase_mem.restart();
            const rec = phase_mem.recordAt(bill_mod.PhaseTerms, phase, billed, billed.sum(), billed.device(), r, self.file_backed_start);
            var buf: [4096]u8 = undefined;
            var fba = std.heap.FixedBufferAllocator.init(&buf);
            const json = phase_mem.jsonOf(fba.allocator(), rec) catch return;
            log.info("glm_moe_dsa: phase memory {s}\n", .{json});
        }

        /// Every KV lane (the target's and the draft lane's) bounded at `cap` positions (`Cache.resize`).
        fn resizeKv(self: *Self, cap: u32) !void {
            try self.cache.resize(&self.g, cap);
            if (self.mtp) |ln| try ln.cache.resize(&self.g, cap);
        }

        /// The bytes every KV lane's buffer holds now.
        fn kvAllocated(self: *const Self) u64 {
            const page = std.heap.pageSize();
            return self.cache.allocatedBytes(page) + if (self.mtp) |ln| ln.cache.allocatedBytes(page) else 0;
        }

        /// The rate tally's file read (the load seeds the stream's tally from it: the preload's order and the pins) and the
        /// rates its writes start from. Nothing at ratio 0 (no tally); no file under `expert_tally_file: off` or in test
        /// builds (their HOME is the box user's).
        fn openTally(self: *Self, dir: []const u8) !void {
            if (!self.cfg.tally.enabled()) return;
            const n_layers: u32 = @intCast(self.bank.layers.len);
            const n_experts: u32 = self.bank.n_experts;
            self.tally_rates = try self.gpa.alloc(f32, @as(usize, n_layers) * n_experts);
            @memset(self.tally_rates, 0);
            errdefer self.closeTally();
            if (builtin.is_test or !self.cfg.tally.readsFile()) return;
            const key = try expert_tally.keyOf(self.gpa, "glm_moe_dsa", dir);
            defer self.gpa.free(key);
            self.tally_path = try expert_tally.pathOf(self.gpa, key);
            const path = self.tally_path orelse {
                log.info("glm_moe_dsa: expert tally: no HOME, no tally file: every request after the first pins from the last one's rates\n", .{});
                return;
            };
            switch (try expert_tally.read(self.gpa, self.io, path, n_layers, n_experts)) {
                .rates => |r| {
                    defer self.gpa.free(r);
                    @memcpy(self.tally_rates, r);
                    self.stream.seedTally(r);
                    log.info("glm_moe_dsa: expert tally: read {s}: the preload fills the prompt rows hottest first, and the top {d} of each layer's experts by rate are pinned (window {d} routes)\n", .{ path, self.cfg.tally.ratio, self.cfg.tally.window });
                },
                .missing => log.info("glm_moe_dsa: expert tally: no file at {s}: this load plans as a first run (ratio {d}, window {d} routes){s}\n", .{ path, self.cfg.tally.ratio, self.cfg.tally.window, if (self.cfg.tally.writesFile()) ", its requests' ends write it" else " (pinned: never written)" }),
                .mismatched => log.warn("glm_moe_dsa: expert tally: {s} is not this stream's ({d} layers of {d} experts, version {d}): ignored, this load plans as a first run\n", .{ path, n_layers, n_experts, expert_tally.version }),
            }
        }

        fn closeTally(self: *Self) void {
            if (self.tally_path) |p| self.gpa.free(p);
            self.tally_path = null;
            if (self.tally_rates.len > 0) self.gpa.free(self.tally_rates);
            self.tally_rates = &.{};
        }

        /// The request's end for the tally: each stream layer's decode rates folded in (the tally pins from now on), the
        /// file written. A failure is logged; the request is not failed.
        fn endTally(self: *Self) void {
            if (self.tally_rates.len == 0) return;
            const routed = self.stream.endTally(self.tally_rates);
            const path = self.tally_path orelse return;
            if (routed == 0 or !self.cfg.tally.writesFile()) return;
            expert_tally.write(self.gpa, self.io, path, self.tally_rates, @intCast(self.bank.layers.len), self.bank.n_experts, self.cfg.tally.window) catch |e| {
                log.warn("glm_moe_dsa: expert tally: not written to {s}: {s}\n", .{ path, @errorName(e) });
                return;
            };
            log.info("glm_moe_dsa: expert tally: written to {s} ({d} of {d} stream layers routed in the decode)\n", .{ path, routed, self.bank.layers.len });
        }

        pub fn deinit(self: *Self) void {
            _ = mlx.mlx_synchronize(self.g.s);
            self.closeTally();
            if (self.mtp) |ln| ln.deinit(&self.g);
            self.cache.deinit(&self.g);
            self.ex.deinit(&self.g);
            self.stream.deinit();
            if (self.recorder) |rec| rec.deinit();
            self.math.deinit(&self.g);
            if (comptime has_kernels) self.kset.deinit();
            if (self.trunk_set) |t| t.deinit();
            self.w.deinit(self.gpa);
            self.bank.deinit();
            self.history.deinit(self.gpa);
            self.gpa.free(self.prompt_rows);
            self.gpa.free(self.decode_rows);
            self.gpa.free(self.layer_counts0);
            self.gpa.free(self.layer_counts1);
            self.gpa.free(self.layer_rates);
            self.bill.free(self.gpa);
            if (self.probe) |p| self.gpa.destroy(p);
            self.sampler.deinit();
            self.g.deinit();
            var prev: usize = 0;
            _ = mlx.mlx_set_cache_limit(&prev, self.prev_cache_limit);
            self.gpa.destroy(self);
        }

        pub fn position(self: *const Self) u64 {
            return self.cache.len;
        }

        fn nowNs(self: *const Self) u64 {
            return @intCast(std.Io.Timestamp.now(self.io, .awake).nanoseconds);
        }

        /// The host matched `prefix` against its prefix cache: the state keeps the positions it shares with it (the
        /// lanes truncated after them) and returns how many. With the draft lane, at most the positions it tracks
        /// whose pairs stay valid (`Lane.keepFor`).
        pub fn restorePrefix(self: *Self, prefix: []const u32) u64 {
            var n = std.mem.indexOfDiff(u32, self.history.items, prefix) orelse @min(self.history.items.len, prefix.len);
            if (self.mtp) |ln| n = @intCast(ln.keepFor(&self.g, n));
            self.cache.truncateTo(@intCast(n));
            self.history.shrinkRetainingCapacity(n);
            return n;
        }

        /// The prompt pass of `ids` at position `start` (the positions `restorePrefix` kept): the last row's logits,
        /// an owned handle.
        pub fn prefillAt(self: *Self, start: u64, ids: []const u32) !mlx.mlx_array {
            if (ids.len == 0) return error.EmptyPrompt;
            // A previous request whose end the host did not report gets its line here.
            self.requestEnd();
            if (start != self.cache.len) return error.PrefixNotKept;
            if (start + ids.len > self.max_context) {
                log.warn("glm_moe_dsa: a {d}-token prompt is over the {d} tokens this load billed (ContextOverBill)\n", .{ start + ids.len, self.max_context });
                return error.ContextOverBill;
            }
            try self.reverse(@intCast(start + ids.len));
            // The lanes the pass appends to are allocated before the allocator's size classes go on for its transients.
            try self.cache.preallocate(&self.g, self.model);
            self.stream.resetPromptCounts();
            if (comptime recording) if (self.recorder) |rec| rec.begin(@intCast(ids.len), self.mtpDepth());
            const s0 = self.stream.stats();
            startPhase();
            const mem0 = Mem.now();
            const t0 = self.nowNs();
            var logits: ?mlx.mlx_array = null;
            errdefer if (logits) |x| self.g.release(x);
            // Layer by layer in calls of `glm.promptCallRows`, or chunk by chunk.
            const chunk: usize = self.overrides.prefill_chunk orelse prefill_chunk_tokens;
            var at: usize = 0;
            while (at < ids.len) {
                const end = at + if (self.cfg.layerMajor()) glm.promptCallRows(ids.len, at) else @min(chunk, ids.len - at);
                if (logits) |x| self.g.release(x);
                logits = null;
                if (self.mtp) |ln| {
                    const out = blk: {
                        self.transientClasses(true);
                        defer self.transientClasses(false);
                        break :blk try graph.forwardRows(&self.g, self.gpa, self.model, &self.w, ids[at..end], @intCast(self.cache.len), &self.cache, &self.ex, self.routes, .{ .hidden = true });
                    };
                    logits = out.logits;
                    defer self.g.release(out.hidden.?);
                    try self.history.appendSlice(self.gpa, ids[at..end]);
                    // The rows' pairs whose next token the prompt gives: their keys in the MTP layer's cache.
                    try ln.pend(&self.g, self.cache.len, out.hidden.?, @intCast(end - at), false);
                    if (ln.tracked() == self.cache.len) try ln.append(&self.g, &self.w, self.history.items[ln.cache.len + 1 ..]);
                } else {
                    logits = blk: {
                        self.transientClasses(true);
                        defer self.transientClasses(false);
                        break :blk try graph.forward(&self.g, self.gpa, self.model, &self.w, ids[at..end], @intCast(self.cache.len), &self.cache, &self.ex, self.routes);
                    };
                    try self.history.appendSlice(self.gpa, ids[at..end]);
                }
                at = end;
            }
            try self.g.evalAll(&.{logits.?});
            try self.ex.flush();
            _ = mlx.mlx_synchronize(self.g.s);
            // The wave probe restarted both high-water marks at each of its marks: the pass's are its own.
            const peaks: ?Peaks = if (self.probe) |p| p.passPeaks() else null;
            var mem1 = Mem.now();
            if (peaks) |p| {
                mem1.peak = @max(mem1.peak, p.mlx);
                mem1.footprint_peak = @max(mem1.footprint_peak, p.footprint);
            }
            log.info("{f}\n", .{PromptLine{ .tokens = ids.len, .wall_ns = self.nowNs() - t0, .r = .of(s0, self.stream.stats()), .mem = .{ .start = mem0, .end = mem1 } }});
            prefill_timers.report(ids.len);
            if (self.probe) |p| log.info("glm_moe_dsa: wave probe {f}\n", .{p.*});
            self.phaseRecord("prompt pass", self.billed.terms(.prompt, self.prompt_units, self.prompt_units), peaks);
            return logits.?;
        }

        /// The allocator's size classes (a request of 1 MB or more rounded up to one of 8 classes per power of two, so
        /// a nearly equal request reuses its cached buffer) on for the forward layers only: their transients are
        /// rounded, never a lane (`Cache.preallocate`), a resident, the scratch or the MTP lane's rows. The switch the
        /// construction bound (`size_classes`): the patched MLX's, else `noSizeClasses`.
        fn transientClasses(self: *const Self, on: bool) void {
            var prev: bool = false;
            _ = self.size_classes(&prev, on);
        }

        /// A stock MLX has no size classes: the module leaves the allocator as it is, and the bill charged no classes.
        fn noSizeClasses(prev: *bool, on: bool) callconv(.c) c_int {
            _ = on;
            prev.* = false;
            return 0;
        }

        /// A serial step: `ids` (decode width) after the committed positions; the last row's logits, an owned handle.
        pub fn extend(self: *Self, ids: []const u32) !mlx.mlx_array {
            if (ids.len == 0 or ids.len * self.model.n_experts_per_tok > self.routes.max_route_ids) return error.StepWiderThanRoute;
            if (self.decoding and ids.len * self.model.n_experts_per_tok > self.window_ids) return error.StepWiderThanDecodeWindow;
            if (self.cache.len + ids.len > self.cache.cap) return error.ContextOverBill;
            const t0 = self.nowNs();
            const out = try graph.forwardRows(&self.g, self.gpa, self.model, &self.w, ids, @intCast(self.cache.len), &self.cache, &self.ex, self.routes, .{ .hidden = self.mtp != null });
            const logits = out.logits;
            errdefer self.g.release(logits);
            defer if (out.hidden) |x| self.g.release(x);
            try self.history.appendSlice(self.gpa, ids);
            // A request the lane does not draft: its positions pending for a later round, up to `mtp.max_pending`.
            if (self.mtp) |ln| try ln.pend(&self.g, self.cache.len, out.hidden.?, @intCast(ids.len), true);
            try self.g.evalAll(&.{logits});
            try self.ex.flush();
            if (self.decode_mark) |*d| {
                d.steps += 1;
                d.tokens += ids.len;
                d.wall_ns += self.nowNs() - t0;
            }
            if (comptime recording) if (self.recorder) |rec| rec.endStep(@intCast(ids.len), 0, 0, @intCast(ids.len), self.nowNs() - t0);
            return logits;
        }

        /// The phase change, once per request before its first decode step: every prompt command retired, the transient
        /// scratch freed, the MLX cache cleared and its decode limit set; the KV lanes at the request's decode positions
        /// (its prompt and `max_tokens`, `bill.decodeCap`); MLX's counters read once the host side is back (`settle`);
        /// then the slot rows grown within the fewer grown bytes of what the bill at this request's KV admits
        /// (`bill.requestBudget`) and what MLX's active bytes leave under the target for decode's own terms
        /// (`bill.Live.budget`), never below the prompt rows (forced rows: grown to them). Refused by name when even the
        /// prompt rows are over the target (the scratch comes back; the request fails, the module stays in its prompt
        /// configuration).
        pub fn decodeHandover(self: *Self, h: sdk.DecodeHandover) !void {
            if (self.decoding) return;
            _ = mlx.mlx_synchronize(self.g.s);
            const before = Mem.now();
            _ = try self.ex.releaseTransient();
            var prev: usize = 0;
            errdefer {
                _ = self.ex.regrowTransient() catch {};
                _ = mlx.mlx_set_cache_limit(&prev, self.cacheLimit(.prompt));
            }
            self.g.clearCache();
            _ = mlx.mlx_set_cache_limit(&prev, self.cacheLimit(.decode));
            const cap: u32 = @intCast(@min(bill_mod.decodeCap(&self.cfg, @max(h.reserved_tokens, self.cache.len)), self.max_positions));
            try self.resizeKv(cap);
            _ = mlx.mlx_synchronize(self.g.s);
            self.g.clearCache();
            const kv1 = self.kvAllocated();
            const st = settle(LiveReader{ .io = self.io }, before.hostSide());
            const box = try Box.now(st.after);
            // The box's target beside the rest of the box as it stands now (not the load's baseline, an older reading of
            // it), at least its floor, for the live reading and the bill at the request alike.
            const rest = bill_mod.restOfBox(box.others, self.cfg.rest_of_box_floor_bytes);
            const target = bill_mod.targetOf(self.gpu_target, self.ram, rest);
            var line: HandoverLine = .{ .before = before, .after = st.after, .settle_ms = st.waited_ms, .settled = st.settled, .positions = cap, .kv_bytes = kv1, .prompt_rows = self.prompt_units, .rows = self.decode_units, .target = target, .gpu_target = self.gpu_target, .baseline = self.cfg.memory_baseline_bytes orelse 0, .box = box, .rest = rest };
            self.decode_pending_rows = if (self.mtp) |ln| ln.n_pending else 0;
            const after = bill_mod.decodeAfter(self.inputs, cap);
            // The host terms are the bill's bound, or the reading where it sits above them (then the bound under-bills and
            // the live reading must not).
            const live: bill_mod.Live = .{ .target = target, .gpu_target = self.gpu_target, .baseline = rest, .mlx_bytes = st.after.active, .device_after = after.device, .host = @max(after.host, st.after.hostSide()) };
            if (!self.forced_rows) {
                line.live_bytes = live.budget() catch |e| {
                    line.refused = @errorName(e);
                    log.err("{f}\n", .{line});
                    return e;
                };
                line.bill_bytes = bill_mod.requestBudget(self.inputs, cap, self.decode_pending_rows, self.prompt_units, rest, target);
                // The bill at the request's inputs in bytes (the receipts rebuild its budget from them).
                log.info("glm_moe_dsa: bill at the request: {d} positions, {d} MTP pending rows, beside {d} B of the rest of the box, target {d} B: {d} B grown\n", .{ cap, self.decode_pending_rows, rest, target, line.bill_bytes.? });
                line.cap = self.cfg.expert_units_cap;
                if (line.cap) |c| line.cap_bytes = bill_mod.grownBytes(self.inputs.bank, self.prompt_units, c);
                const budget = @min(@min(line.live_bytes.?, line.bill_bytes.?), if (line.cap != null) line.cap_bytes else std.math.maxInt(u64));
                self.decodeRowsAt(bill_mod.unitsIn(self.inputs.bank, self.prompt_units, budget), budget);
                line.rows = self.decode_units;
            }
            self.grown_bytes = bill_mod.grownOf(self.inputs.bank, self.prompt_rows, self.decode_rows);
            line.grown_bytes = self.grown_bytes;
            line.bound = live.total(self.grown_bytes);
            line.mlx_bound = live.mlxAt(self.grown_bytes);
            try self.ex.grow(self.decode_rows);
            line.grown = Mem.now();
            log.info("{f}\n", .{line});
            self.decode_positions = cap;
            self.phaseRecord("handover", self.requestTerms(), null);
            self.decoding = true;
            if (comptime timeline.enabled) if (self.mtp != null) {
                // The timeline's storage holds DeepSeek-V4.1's 64 layers: GLM-5.3's first 64 routed layers are stamped (the
                // verify's GPU busy and idle cover every layer).
                timeline.install(self.g.s, @intCast(@min(h.reserved_tokens -| h.prompt_tokens, timeline.max_cycles + 1)), @intCast(@min(self.bank.layers.len / Experts.bpl, timeline.max_layers))) catch |e|
                    log.warn("glm_moe_dsa: the verify timeline is off ({s})\n", .{@errorName(e)});
            };
            self.decode_mark = .{ .s0 = self.stream.stats(), .mem0 = Mem.now(), .by0 = self.stream.lookaheadByRank() };
            self.ex.host = .{};
            if (self.mtp) |ln| ln.counts = .{};
            for (self.layer_counts0, 0..) |*c, l| c.* = self.stream.layerCounts(@intCast(l));
        }

        /// Each stream layer's decode rows grown from the prompt's within `budget` grown bytes, `units` the whole units in
        /// it (`bill.unitsIn`): the split's (`glm_moe_dsa_split`) on its geometry, filled to the budget; else the even
        /// split's at the units (`bill.rowsAt`), with a line saying so.
        fn decodeRowsAt(self: *Self, units: u32, budget: u64) void {
            self.decode_units = units;
            self.split_rows = split_mod.decodeRows(self.inputs.bank, self.prompt_units, units, budget, self.decode_rows) != null;
            if (!self.split_rows) {
                log.info("glm_moe_dsa: decode rows: no split table for this pack (K mask 0x{x}, {d} stream layers): the even split\n", .{ self.inputs.bank.k_mask, self.inputs.bank.layers.len });
                _ = bill_mod.rowsAt(self.inputs.bank, units, self.decode_rows);
            }
        }

        /// The request's end (the host's finish): its decode line, once; nothing when it never decoded.
        pub fn requestEnd(self: *Self) void {
            if (comptime timeline.enabled) if (timeline.active) {
                const pending = timeline.settle(self.g.s, 2000);
                timeline.uninstall();
                var tb: [16384]u8 = undefined;
                log.info("glm_moe_dsa: {s}\n", .{timeline.line(&tb, pending)});
            };
            if (self.mtp) |ln| if (ln.counts.rounds > 0) {
                log.info("{f}\n", .{ln.line()});
                if (comptime mtp_mod.histograms) log.info("{f}\n", .{mtp_mod.ProbLine{ .c = ln.counts }});
                ln.counts = .{};
            };
            const d = self.decode_mark orelse return;
            self.decode_mark = null;
            if (d.steps == 0) return;
            for (self.layer_counts1, 0..) |*c, l| c.* = self.stream.layerCounts(@intCast(l));
            _ = mlx.mlx_synchronize(self.g.s);
            const end = Mem.now();
            log.info("{f}\n", .{DecodeLine{
                .steps = d.steps,
                .tokens = d.tokens,
                .wall_ns = d.wall_ns,
                .r = .of(d.s0, self.stream.stats()),
                .hit_rate = hitSpread(self.layer_counts0, self.layer_counts1, self.layer_rates),
                .rows = self.decode_rows[0],
                .rows_alt = if (Experts.bpl > 1) self.decode_rows[1] else null,
                .spread = if (self.split_rows) split_mod.spread(self.decode_rows) else null,
                .mem = .{ .start = d.mem0, .end = end },
                .box = Box.now(end) catch null,
            }});
            self.phaseRecord("decode", self.requestTerms(), null);
            log.info("{f}\n", .{HostLine{ .h = self.ex.host, .wall_ns = d.wall_ns }});
            const by1 = self.stream.lookaheadByRank();
            log.info("{f}\n", .{PrecisionLine{ .by0 = &d.by0, .by1 = &by1 }});
            self.endTally();
            if (comptime recording) if (self.recorder) |rec| self.writeRoutes(rec);
        }

        /// The route recorder's file of the request that ended. A failure is logged; the request is not failed.
        fn writeRoutes(self: *Self, rec: *expert_routes.Recorder) void {
            if (!rec.any()) return;
            var buf: [1024]u8 = undefined;
            const path = std.fmt.bufPrint(&buf, "{s}/glm53-routes-{d}-{d}.bin", .{ self.routes_dir, std.c.getpid(), expert_routes.nextFile() }) catch return;
            rec.write(self.gpa, self.io, path) catch |e| {
                log.warn("glm_moe_dsa: the routes were not written to {s}: {s}\n", .{ path, @errorName(e) });
                return;
            };
            log.info("glm_moe_dsa: routes written to {s} ({d} routes, {d} steps, {d} dropped)\n", .{ path, rec.routes.items.len, rec.steps.items.len, rec.dropped });
        }

        /// The reverse phase change, before a prompt of `positions` positions (its kept prefix included) after a decode:
        /// every route settled, the grown rows and window 0 freed, the KV lanes at the prompt's positions, the MLX cache
        /// cleared, MLX's counters read once the host side is back (`settle`), the prompt's scratch re-created, the prompt
        /// cache limit. Without a decode before it: the KV lanes at the prompt's positions.
        fn reverse(self: *Self, positions: u32) !void {
            if (!self.decoding) {
                try self.resizeKv(positions);
                self.g.clearCache();
                return;
            }
            _ = mlx.mlx_synchronize(self.g.s);
            const before = Mem.now();
            try self.stream.settleRoutes();
            // The reverse change's residents, per request (`Stream.shrink`): GLM-5.3 keeps the prompt rows' residents for
            // every later prompt (a long prompt routes nearly every expert, and the preload and the tally chose them).
            _ = try self.ex.shrink(self.prompt_rows, .keep);
            try self.resizeKv(positions);
            _ = mlx.mlx_synchronize(self.g.s);
            self.g.clearCache();
            const kv1 = self.kvAllocated();
            const st = settle(LiveReader{ .io = self.io }, before.hostSide());
            _ = try self.ex.regrowTransient();
            var prev: usize = 0;
            _ = mlx.mlx_set_cache_limit(&prev, self.cacheLimit(.prompt));
            self.decoding = false;
            log.info("glm_moe_dsa: reverse: footprint {d:.3} GB before the frees, {d:.3} GB after ({s} in {d} ms), KV at {d} positions {d:.3} GB, {d} rows per layer\n", .{ gigabytes(before.footprint), gigabytes(st.after.footprint), if (st.settled) "settled" else "not settled", st.waited_ms, positions, gigabytes(kv1), self.prompt_units });
        }

        /// The routed experts' counters (the stream's).
        pub fn stats(self: *Self) sdk_ext.expert.Stats {
            return self.stream.stats();
        }

        /// The draft lane's drafts per round (0: no lane).
        pub fn mtpDepth(self: *const Self) u32 {
            return if (self.mtp) |ln| ln.depth else 0;
        }

        /// The lane's counters over the request.
        pub fn draftStats(self: *const Self) sdk.DraftStats {
            const ln = self.mtp orelse return .{};
            return .{ .rounds = ln.counts.rounds, .drafted = ln.counts.drafted, .accepted = ln.counts.accepted, .generated = ln.counts.generated };
        }

        /// One draft round from `t1` (`sdk.DraftLane.round`): the drafts, the verify of `[t1, drafts]`, the decision,
        /// the target truncated to the accepted rows. Without room for a draft (the request's token budget, the
        /// lanes' end) or for a request the lane does not track, a round of `t1` alone.
        pub fn mtpRound(self: *Self, a: std.mem.Allocator, t1: u32, accepted_cap: u32, sampling: sdk.SamplingParams) !sdk.DraftRound {
            return self.roundWith(a, t1, accepted_cap, sampling, null);
        }

        /// A test's view into a round: its first drafts forced, the drafts and each draft step's logits read back.
        pub const Probe = struct {
            force: []const u32 = &.{},
            depth: u32 = 0,
            drafts: [mtp_mod.max_depth]u32 = undefined,
            /// `mtp_depth` x vocab.
            logits: ?[]f32 = null,
            /// The host's waits on the GPU (`G.waits`) from the verify's eval to the lane's pend: the decision's read
            /// and the commit.
            decision_waits: u64 = 0,
        };

        /// `mtpRound` with a test's `probe`.
        pub fn roundWith(self: *Self, a: std.mem.Allocator, t1: u32, accepted_cap: u32, sampling: sdk.SamplingParams, probe: ?*Probe) !sdk.DraftRound {
            const ln = self.mtp orelse return error.NoDraftLane;
            if (comptime timeline.enabled) timeline.cycleBegin();
            const t0 = self.nowNs();
            const peak0 = self.g.peakFrom();
            const len = self.cache.len;
            if (len >= self.cache.cap) return error.ContextOverBill;
            const tracked = ln.tracked() == len and ln.n_pending >= 1;
            const depth: u32 = if (tracked) @min(ln.depth, accepted_cap, self.cache.cap - len - 1) else 0;
            var ids: [mtp_mod.max_depth + 1]u32 = undefined;
            ids[0] = t1;
            const drafts = ids[1..][0..depth];
            if (depth > 0) try ln.draft(&self.g, a, &self.w, self.history.items[ln.cache.len + 1 .. len], t1, depth, drafts, if (probe) |p| p.force else &.{}, if (probe) |p| p.logits else null);
            const t_draft = self.nowNs();
            if (probe) |p| {
                p.depth = depth;
                @memcpy(p.drafts[0..depth], drafts);
            }
            const m0 = self.g.mark();
            defer self.g.resetTo(m0);
            self.ex.row_bytes = @splat(0);
            if (comptime timeline.enabled) timeline.verifyBegin();
            const out = try graph.forwardRows(&self.g, self.gpa, self.model, &self.w, ids[0 .. depth + 1], len, &self.cache, &self.ex, self.routes, .{ .verify = true, .hidden = true });
            defer self.g.release(out.logits);
            defer self.g.release(out.hidden.?);
            // The decision's arrays in the verify's eval: the GPU goes on from the logits to the decision with no
            // round trip to the host between them.
            const dp = try mtp_mod.prepare(&self.g, out.logits, drafts, ln.mode, sampling, len);
            try self.g.evalAll(&.{ out.logits, out.hidden.?, dp.tok, dp.flags orelse dp.tok });
            const waits_verify = self.g.waits;
            if (comptime timeline.enabled) timeline.verifyEnd();
            try self.ex.flush();
            const t_verify = self.nowNs();
            const d = try dp.read(&self.g);
            self.cache.truncateTo(len + 1 + d.accepted);
            try self.history.appendSlice(self.gpa, ids[0 .. 1 + d.accepted]);
            const rows = d.accepted + 1;
            if (probe) |p| p.decision_waits = self.g.waits - waits_verify;
            try ln.pend(&self.g, self.cache.len, try graph.rowSlice(&self.g, out.hidden.?, 0, @intCast(rows)), rows, depth == 0);
            const tokens = try a.dupe(u32, ids[0..rows]);
            const ns = self.nowNs() - t0;
            ln.counts.rounds += 1;
            ln.counts.drafted += depth;
            ln.counts.accepted += d.accepted;
            ln.counts.generated += rows;
            ln.counts.serial += @intFromBool(depth == 0);
            ln.counts.wall_ns += ns;
            ln.counts.draft_ns += t_draft - t0;
            ln.counts.verify_ns += t_verify - t_draft;
            ln.counts.peak_rise = @max(ln.counts.peak_rise, self.g.peakAbove(peak0));
            for (self.ex.row_bytes[0 .. depth + 1], 0..) |b, i| {
                ln.counts.verify_bytes += b;
                if (i > d.accepted) ln.counts.rejected_bytes += b;
            }
            // The steps the decision reached: the accepted drafts and the first rejected one.
            if (comptime mtp_mod.histograms) for (ln.top_p[0..@min(depth, d.accepted + 1)], 0..) |tp, i| {
                ln.counts.decided[mtp_mod.tenth(tp)] += 1;
                if (i < d.accepted) ln.counts.accepted_by_p[mtp_mod.tenth(tp)] += 1;
            };
            if (comptime recording) if (self.recorder) |rec| rec.endStep(depth + 1, depth, d.accepted, rows, ns);
            if (self.decode_mark) |*dm| {
                dm.steps += 1;
                dm.tokens += rows;
                dm.wall_ns += ns;
            }
            return .{ .tokens = tokens, .accepted = d.accepted, .next_token = d.next };
        }
    };
}

/// MLX's allocator and this process's footprint at a boundary (MLX's counters, the kernel's ledgers:
/// `phase_memory.Ledgers`, the shared reading).
pub const Mem = phase_mem.Ledgers;
/// The wave probe's pass high-water marks.
const Peaks = graph.WaveProbe.Peaks;

/// The box's used memory beside this process's (`sdk.memory.boxUsedBytes`, read right after `mem`): what the rest of
/// the box holds is the used memory less this process's footprint and the page tables of its MLX bytes. The
/// construction refused a box without the reading (`BoxReadingUnavailable`); a reading the kernel refuses later fails
/// the boundary that needs it, never reads as an empty box.
pub const Box = struct {
    used: u64,
    others: u64,

    pub fn now(mem: Mem) error{BoxReadingUnavailable}!Box {
        const used = sdk.memory.boxUsedBytes() orelse return error.BoxReadingUnavailable;
        return .{ .used = used, .others = used -| mem.footprint -| bill_mod.wireTables(mem.active + mem.cache) };
    }
};

/// A phase's start: MLX's peak and the footprint's interval peak restarted (`phase_memory.restart`).
pub const startPhase = phase_mem.restart;

/// The live boundary reader (`settle`): MLX's counters and the footprint; waits on the host's io
/// (`phase_memory.LiveReader`, shared with DeepSeek-V4.1's module).
const LiveReader = phase_mem.LiveReader(Mem);

/// After a phase change's frees (all MLX's: the scratch or the grown rows, the cache, the KV lanes' resize), `reader`
/// read until the host side is back at `host_before` (`phase_memory.HostSideBack`): the grow and the phase record then
/// read MLX's bytes and the footprint without freed pages the driver has not retired yet. The last reading either way;
/// `settled` says which (the handover logs "not settled" and goes on).
pub fn settle(reader: anytype, host_before: u64) phase_mem.Settled(Mem) {
    return phase_mem.settle(reader, settle_poll_ms, settle_max_ms, phase_mem.HostSideBack(Mem){ .host_before = host_before });
}

/// A phase's memory between its start and its end readings (a prompt pass, a request's decode).
pub const PhaseMem = struct {
    start: Mem,
    end: Mem,

    pub fn format(p: PhaseMem, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("MLX active {d:.3} -> {d:.3} GB (peak {d:.3}), MLX cache {d:.3} GB, footprint {d:.3} -> {d:.3} GB (peak {d:.3}), host side {d:.3} -> {d:.3} GB", .{
            gigabytes(p.start.active), gigabytes(p.end.active), gigabytes(p.end.peak), gigabytes(p.end.cache), gigabytes(p.start.footprint), gigabytes(p.end.footprint), gigabytes(p.end.footprint_peak), gigabytes(p.start.hostSide()), gigabytes(p.end.hostSide()),
        });
    }
};

/// The decode handover's line: the readings before the frees and after them, the KV lanes' positions, the rows the
/// bill at the request and the live reading allow, the rows grown, decode's bound at them and the reading after the grow.
pub const HandoverLine = struct {
    before: Mem,
    after: Mem,
    settle_ms: u32,
    settled: bool,
    positions: u64,
    kv_bytes: u64,
    prompt_rows: u32,
    /// The whole units grown (forced rows: the harness's) and the grow's bytes past the prompt rows (`bill.grownOf`).
    rows: u32,
    grown_bytes: u64 = 0,
    /// The grown bytes the bill at the request's KV and the live reading admit (null: forced rows).
    bill_bytes: ?u64 = null,
    live_bytes: ?u64 = null,
    /// The live total and MLX's bytes at the grown bytes (`bill.Live.total`, `bill.Live.mlxAt`) and their targets.
    bound: ?u64 = null,
    mlx_bound: ?u64 = null,
    target: u64,
    gpu_target: u64 = 0,
    baseline: u64,
    /// The box then (`Box.now` after the settle).
    box: Box = .{ .used = 0, .others = 0 },
    grown: ?Mem = null,
    refused: ?[]const u8 = null,
    /// The units cap (`expert_units_cap`; null: off) and its grown bytes (`bill.grownBytes` from the prompt units).
    cap: ?u32 = null,
    cap_bytes: u64 = 0,
    /// The rest of the box the grow sat beside (`bill.restOfBox`: the reading, at least its floor; 0: the reading). The
    /// load's baseline (`baseline`) is the reading.
    rest: u64 = 0,

    pub fn format(p: HandoverLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("glm_moe_dsa: handover: footprint {d:.3} GB before the frees, {d:.3} GB after ({s} in {d} ms; MLX active {d:.3} GB, host side {d:.3} GB; the box {d:.3} GB used, the rest of it {d:.3} GB), KV at {d} positions {d:.3} GB", .{
            gigabytes(p.before.footprint), gigabytes(p.after.footprint), if (p.settled) "settled" else "not settled", p.settle_ms, gigabytes(p.after.active), gigabytes(p.after.hostSide()), gigabytes(p.box.used), gigabytes(p.box.others), p.positions, gigabytes(p.kv_bytes),
        });
        if (p.refused) |e| return w.print(", refused: {s} (the prompt's {d} rows per layer leave decode over its target)", .{ e, p.prompt_rows });
        try w.print(", rows {d} -> {d} per layer, {d} B grown", .{ p.prompt_rows, p.rows, p.grown_bytes });
        if (p.bill_bytes) |b| try w.print(" (the bill at the request {d} B, the live reading {d} B)", .{ b, p.live_bytes.? }) else try w.print(" (forced)", .{});
        // A capped arm's rows: at the cap, or under it when the bill or the live reading admits fewer (a run to flag).
        if (p.cap) |c| if (p.bill_bytes) |b| try w.print("{s} {d} units", .{ if (@min(b, p.live_bytes.?) >= p.cap_bytes) ", capped at" else ", under the units cap of", c });
        if (p.bound) |b| try w.print(", decode bound {d:.3} GB of the {d:.3} GB target (the load's baseline {d:.3} GB)", .{ gigabytes(b), gigabytes(p.target), gigabytes(p.baseline) });
        if (p.mlx_bound) |b| try w.print(", MLX {d:.3} GB of the GPU's {d:.3} GB", .{ gigabytes(b), gigabytes(p.gpu_target) });
        if (p.grown) |g| try w.print(", footprint {d:.3} GB after the grow", .{gigabytes(g.footprint)});
        if (p.rest > p.box.others) try w.print(", the rest of the box billed at its floor {d:.3} GB", .{gigabytes(p.rest)});
    }
};

/// One phase's reads between two snapshots of the stream's counters.
pub const Reads = struct {
    /// Bytes off the SSD: the reads that ran a preadv (a copy out of a speculative record excluded) and the
    /// speculative records' own.
    ssd_bytes: u64 = 0,
    /// Records the routes read (a load whose slot still held its record reads nothing) / the read-ahead posted.
    demand_records: u64 = 0,
    ahead_records: u64 = 0,
    /// Read-ahead records the layer's call then routed.
    ahead_hits: u64 = 0,
    /// Unique experts per route: resident / to load.
    hits: u64 = 0,
    misses: u64 = 0,
    /// The decode lookahead's speculative records: issued / claimed by a demand read; of the rest, landed, dropped
    /// while queued, abandoned while read, cancelled by a demand read of the same bytes, read and never served.
    spec_issued: u64 = 0,
    spec_used: u64 = 0,
    spec_landed: u64 = 0,
    spec_expired: u64 = 0,
    spec_abandoned: u64 = 0,
    spec_cancelled: u64 = 0,
    spec_discarded: u64 = 0,
    /// Of the used, those still in flight at the claim (late); the bytes the lookahead read, of them the bytes it
    /// served to a demand read.
    spec_late: u64 = 0,
    spec_bytes: u64 = 0,
    spec_served_bytes: u64 = 0,
    /// Loads whose slot still held the record (no read); demand ranges read straight into their rows.
    loads_skipped: u64 = 0,
    direct: u64 = 0,
    /// Prompt records copied from a transient row into a resident's row (`Stream.promote`), and their bytes.
    promoted: u64 = 0,
    promoted_bytes: u64 = 0,
    /// Host time blocked in the read waits; wall time with any read in flight.
    wait_ns: u64 = 0,
    in_flight_ns: u64 = 0,

    pub fn of(s0: Stats, s1: Stats) Reads {
        const d = s1.since(s0);
        return .{
            .ssd_bytes = (d.expert_bytes_read -| d.adopt_bytes) + d.spec_bytes,
            .demand_records = (d.persistent_loads + d.transient_loads) -| d.loads_skipped,
            .ahead_records = d.ahead_posted,
            .ahead_hits = d.ahead_hits,
            .hits = d.expert_cache_hits,
            .misses = d.expert_cache_misses,
            .spec_issued = d.spec_issued,
            .spec_used = d.claimed,
            .spec_landed = d.spec_landed,
            .spec_expired = d.spec_expired,
            .spec_abandoned = d.spec_abandoned,
            .spec_cancelled = d.spec_cancelled,
            .spec_discarded = d.spec_discarded,
            .spec_late = d.spec_claimed_inflight,
            .spec_bytes = d.spec_bytes,
            .spec_served_bytes = d.adopt_bytes,
            .loads_skipped = d.loads_skipped,
            .direct = d.direct_ranges,
            .promoted = d.promoted,
            .promoted_bytes = d.promoted_bytes,
            .wait_ns = d.read_wait_ns,
            .in_flight_ns = d.read_wall_ns,
        };
    }
};

/// Min / median / max over layers, in percent.
pub const Spread = struct { min: f64, median: f64, max: f64 };

/// Each layer's hit rate between two snapshots (`rates`: scratch, one per layer); null when no layer routed.
pub fn hitSpread(c0: []const LayerCounts, c1: []const LayerCounts, rates: []f64) ?Spread {
    var n: usize = 0;
    for (c0, c1) |a, b| {
        const hits = b.hits -| a.hits;
        const routed = hits + (b.misses -| a.misses);
        if (routed == 0) continue;
        rates[n] = 100 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(routed));
        n += 1;
    }
    if (n == 0) return null;
    std.sort.pdq(f64, rates[0..n], {}, std.sort.asc(f64));
    return .{ .min = rates[0], .median = rates[(n - 1) / 2], .max = rates[n - 1] };
}

fn seconds(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

fn perSecond(n: u64, ns: u64) f64 {
    return if (ns == 0) 0 else @as(f64, @floatFromInt(n)) / seconds(ns);
}

/// The share of `wall` with a read in flight, in percent (0 for no wall time).
fn busyPct(in_flight: u64, wall: u64) f64 {
    if (wall == 0) return 0;
    return 100 * @min(@as(f64, @floatFromInt(in_flight)) / @as(f64, @floatFromInt(wall)), 1);
}

fn gigabytes(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / 1e9;
}

/// The prompt pass's line, at its end.
pub const PromptLine = struct {
    tokens: u64,
    wall_ns: u64,
    r: Reads,
    mem: ?PhaseMem = null,

    pub fn format(p: PromptLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("glm_moe_dsa: prompt {d} tokens in {d:.2} s ({d:.1} tok/s): {d:.2} GB from the SSD, {d} records on demand, {d} read ahead ({d} routed), host wait {d:.2} s", .{
            p.tokens, seconds(p.wall_ns), perSecond(p.tokens, p.wall_ns), gigabytes(p.r.ssd_bytes), p.r.demand_records, p.r.ahead_records, p.r.ahead_hits, seconds(p.r.wait_ns),
        });
        if (p.r.promoted > 0) try w.print(", {d} records made resident from transient rows ({d:.2} GB copied)", .{ p.r.promoted, gigabytes(p.r.promoted_bytes) });
        if (p.mem) |m| try w.print("; {f}", .{m});
    }
};

/// The decode's line, at the request's end. `wall_ns`: the time inside the steps.
pub const DecodeLine = struct {
    steps: u64,
    tokens: u64,
    wall_ns: u64,
    r: Reads,
    hit_rate: ?Spread,
    rows: u32,
    /// A bank of two bank layers per routed layer: the second's rows (`rows` the first's).
    rows_alt: ?u32 = null,
    /// Uneven rows (`glm_moe_dsa_split`): per K the min, median and max over the routed layers (printed in place of
    /// `rows`, `rows_alt`: the medians, then the spread).
    spread: ?[2][3]u32 = null,
    mem: ?PhaseMem = null,
    /// The box at the request's end (the rest of it beside the handover's reading).
    box: ?Box = null,

    pub fn format(p: DecodeLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("glm_moe_dsa: decode {d} steps, {d} tokens in {d:.2} s ({d:.1} tok/s): {d} routed records, {d} hits, {d} misses", .{
            p.steps, p.tokens, seconds(p.wall_ns), perSecond(p.tokens, p.wall_ns), p.r.hits + p.r.misses, p.r.hits, p.r.misses,
        });
        if (p.hit_rate) |h| try w.print(" (hit rate per layer min {d:.0}% median {d:.0}% max {d:.0}%)", .{ h.min, h.median, h.max });
        try w.print(", {d:.2} GB from the SSD ({d:.3} GB per emitted token), lookahead {d} issued / {d} used (landed {d}, expired {d}, abandoned {d}, cancelled {d}, discarded {d}, used while in flight {d}), lookahead {d:.2} GB read / {d:.2} GB served, {d} loads skipped, {d} direct reads, host wait {d:.2} s and reads in flight {d:.2} s of {d:.2} s ({d:.1}% busy, {d:.1}% with no read in flight), {d}", .{
            gigabytes(p.r.ssd_bytes), if (p.tokens == 0) 0 else gigabytes(p.r.ssd_bytes) / @as(f64, @floatFromInt(p.tokens)), p.r.spec_issued, p.r.spec_used, p.r.spec_landed, p.r.spec_expired, p.r.spec_abandoned, p.r.spec_cancelled, p.r.spec_discarded, p.r.spec_late, gigabytes(p.r.spec_bytes), gigabytes(p.r.spec_served_bytes), p.r.loads_skipped, p.r.direct, seconds(p.r.wait_ns), seconds(p.r.in_flight_ns), seconds(p.wall_ns), busyPct(p.r.in_flight_ns, p.wall_ns), 100 - busyPct(p.r.in_flight_ns, p.wall_ns), if (p.spread) |sp| sp[0][1] else p.rows,
        });
        if (p.spread) |sp| {
            try w.print(" / {d} rows per bank layer (its two Ks, the medians; split: min / median / max {d} / {d} / {d} and {d} / {d} / {d})", .{ sp[1][1], sp[0][0], sp[0][1], sp[0][2], sp[1][0], sp[1][1], sp[1][2] });
        } else if (p.rows_alt) |r2| try w.print(" / {d} rows per bank layer (its two Ks)", .{r2}) else try w.writeAll(" rows per layer");
        if (p.mem) |m| try w.print("; {f}", .{m});
        if (p.box) |b| try w.print("; the box {d:.3} GB used, the rest of it {d:.3} GB", .{ gigabytes(b.used), gigabytes(b.others) });
    }
};

/// The decode lane's host split at the request's end.
pub const HostLine = struct {
    h: experts_mod.HostSplit,
    wall_ns: u64,

    pub fn format(p: HostLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const other = p.wall_ns -| (p.h.barrier_ns + p.h.route_ns + p.h.wave_ns);
        try w.print("glm_moe_dsa: decode host split over {d} routed calls: barrier waits {d:.2} s, routes {d:.2} s, wave encode {d:.2} s, the rest {d:.2} s of {d:.2} s; {d} calls joined before the combine", .{
            p.h.calls, seconds(p.h.barrier_ns), seconds(p.h.route_ns), seconds(p.h.wave_ns), seconds(other), seconds(p.wall_ns), p.h.joined,
        });
    }
};

/// The decode lookahead's precision at the request's end: per bank layer of a routed layer (K ascending; one for the
/// affine bank) and top-k position of the candidate's best row, the records read ahead and of them those the target
/// call routed, between the stream's counts at the handover (`by0`) and at the end (`by1`).
pub const PrecisionLine = struct {
    by0: []const [sdk_ext.expert.lookahead.max_k][2]u64,
    by1: []const [sdk_ext.expert.lookahead.max_k][2]u64,

    pub fn format(p: PrecisionLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.writeAll("glm_moe_dsa: lookahead precision per top-k position (read ahead / of them routed by the target call)");
        for (p.by0, p.by1, 0..) |side0, side1, si| {
            if (p.by1.len > 1) try w.print("{s} bank layer {d}:", .{ if (si == 0) ":" else ";", si }) else try w.writeAll(":");
            var all: [2]u64 = .{ 0, 0 };
            for (side0, side1, 0..) |x0, x1, r| if (x1[0] > x0[0]) {
                const x = [2]u64{ x1[0] - x0[0], x1[1] - x0[1] };
                try w.print(" #{d} {d}/{d} ({d:.0}%),", .{ r, x[0], x[1], pct(x[1], x[0]) });
                all[0] += x[0];
                all[1] += x[1];
            };
            try w.print(" all {d}/{d} ({d:.0}%)", .{ all[0], all[1], pct(all[1], all[0]) });
        }
    }

    fn pct(n: u64, d: u64) f64 {
        return if (d == 0) 0 else 100 * @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(d));
    }
};

const testing = std.testing;

test "glm stats lines: the lookahead's precision per top-k position, per bank layer, from fixed counters" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const a = testing.allocator;
    // The counts at the handover (an earlier request's) and at the end.
    var two0: [2][sdk_ext.expert.lookahead.max_k][2]u64 = @splat(@splat(.{ 0, 0 }));
    two0[0][0] = .{ 5, 4 };
    two0[1][3] = .{ 2, 1 };
    var two: [2][sdk_ext.expert.lookahead.max_k][2]u64 = two0;
    two[0][0] = .{ 45, 34 };
    two[0][2] = .{ 10, 2 };
    two[1][1] = .{ 8, 6 };
    const l2 = try std.fmt.allocPrint(a, "{f}", .{PrecisionLine{ .by0 = &two0, .by1 = &two }});
    defer a.free(l2);
    try testing.expectEqualStrings("glm_moe_dsa: lookahead precision per top-k position (read ahead / of them routed by the target call): bank layer 0: #0 40/30 (75%), #2 10/2 (20%), all 50/32 (64%); bank layer 1: #1 8/6 (75%), all 8/6 (75%)", l2);
    const l1 = try std.fmt.allocPrint(a, "{f}", .{PrecisionLine{ .by0 = two0[1..], .by1 = two[1..] }});
    defer a.free(l1);
    try testing.expectEqualStrings("glm_moe_dsa: lookahead precision per top-k position (read ahead / of them routed by the target call): #1 8/6 (75%), all 8/6 (75%)", l1);
    const l0 = try std.fmt.allocPrint(a, "{f}", .{PrecisionLine{ .by0 = two0[1..], .by1 = two0[1..] }});
    defer a.free(l0);
    try testing.expectEqualStrings("glm_moe_dsa: lookahead precision per top-k position (read ahead / of them routed by the target call): all 0/0 (0%)", l0);
}

test "glm stats lines: the prompt and decode lines and the per-layer hit spread, from fixed counters" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const s0: Stats = .{ .expert_bytes_read = 100, .adopt_bytes = 10, .spec_bytes = 5, .persistent_loads = 3, .transient_loads = 1, .loads_skipped = 1, .expert_cache_hits = 7, .expert_cache_misses = 4 };
    const s1: Stats = .{ .expert_bytes_read = 8_000_000_100, .adopt_bytes = 1_000_000_010, .spec_bytes = 2_000_000_005, .persistent_loads = 303, .transient_loads = 101, .loads_skipped = 21, .expert_cache_hits = 607, .expert_cache_misses = 404, .ahead_posted = 50, .ahead_hits = 40, .spec_issued = 90, .claimed = 60, .read_wait_ns = 1_500_000_000, .read_wall_ns = 2_250_000_000, .spec_landed = 70, .spec_expired = 12, .spec_abandoned = 5, .spec_cancelled = 3, .spec_discarded = 7, .direct_ranges = 300, .spec_claimed_inflight = 4 };
    const r = Reads.of(s0, s1);
    try testing.expectEqual(Reads{ .ssd_bytes = 9_000_000_000, .demand_records = 380, .ahead_records = 50, .ahead_hits = 40, .hits = 600, .misses = 400, .spec_issued = 90, .spec_used = 60, .spec_landed = 70, .spec_expired = 12, .spec_abandoned = 5, .spec_cancelled = 3, .spec_discarded = 7, .spec_late = 4, .spec_bytes = 2_000_000_000, .spec_served_bytes = 1_000_000_000, .loads_skipped = 20, .direct = 300, .wait_ns = 1_500_000_000, .in_flight_ns = 2_250_000_000 }, r);
    // Three layers: 1 of 4 hit, 3 of 4, none routed (skipped), 2 of 4.
    const c0 = [_]LayerCounts{ .{}, .{ .hits = 5 }, .{ .hits = 9, .misses = 9 }, .{} };
    const c1 = [_]LayerCounts{ .{ .hits = 1, .misses = 3 }, .{ .hits = 8, .misses = 1 }, .{ .hits = 9, .misses = 9 }, .{ .hits = 2, .misses = 2 } };
    var rates: [4]f64 = undefined;
    try testing.expectEqual(Spread{ .min = 25, .median = 50, .max = 75 }, hitSpread(&c0, &c1, &rates).?);
    try testing.expectEqual(@as(?Spread, null), hitSpread(&c0, &c0, &rates));
    const a = testing.allocator;
    const p = try std.fmt.allocPrint(a, "{f}", .{PromptLine{ .tokens = 1008, .wall_ns = 31_500_000_000, .r = r }});
    defer a.free(p);
    try testing.expectEqualStrings("glm_moe_dsa: prompt 1008 tokens in 31.50 s (32.0 tok/s): 9.00 GB from the SSD, 380 records on demand, 50 read ahead (40 routed), host wait 1.50 s", p);
    const d = try std.fmt.allocPrint(a, "{f}", .{DecodeLine{ .steps = 40, .tokens = 128, .wall_ns = 25_000_000_000, .r = r, .hit_rate = hitSpread(&c0, &c1, &rates), .rows = 136 }});
    defer a.free(d);
    const hl = try std.fmt.allocPrint(a, "{f}", .{HostLine{ .h = .{ .calls = 300, .barrier_ns = 9_000_000_000, .route_ns = 1_500_000_000, .wave_ns = 2_250_000_000 }, .wall_ns = 25_000_000_000 }});
    defer a.free(hl);
    try testing.expectEqualStrings("glm_moe_dsa: decode host split over 300 routed calls: barrier waits 9.00 s, routes 1.50 s, wave encode 2.25 s, the rest 12.25 s of 25.00 s; 0 calls joined before the combine", hl);
    try testing.expectEqualStrings("glm_moe_dsa: decode 40 steps, 128 tokens in 25.00 s (5.1 tok/s): 1000 routed records, 600 hits, 400 misses (hit rate per layer min 25% median 50% max 75%), 9.00 GB from the SSD (0.070 GB per emitted token), lookahead 90 issued / 60 used (landed 70, expired 12, abandoned 5, cancelled 3, discarded 7, used while in flight 4), lookahead 2.00 GB read / 1.00 GB served, 20 loads skipped, 300 direct reads, host wait 1.50 s and reads in flight 2.25 s of 25.00 s (9.0% busy, 91.0% with no read in flight), 136 rows per layer", d);
}

test "glm stats lines: a prompt pass with read-ahead and a decode with the lookahead on the synthetic bank read back exactly" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var sb = try bank_mod.SynthBank.open();
    defer sb.close();
    const b = &sb.bank;
    const s = try bank_mod.Stream.Stream.init(testing.allocator, b, .{ .rows = &.{ 4, 6, 3, 5 }, .max_route_ids = 16, .transient_rows = 16, .staging_from_bank = true, .lookahead = .{ .budget = 2 }, .pool = .{ .workers = 2, .tickets = 256 } });
    defer s.deinit();
    var rng = std.Random.DefaultPrng.init(53);
    const pick = struct {
        fn f(rnd: std.Random, out: []u16, n: usize) []u16 {
            var k: usize = 0;
            while (k < n) {
                const e = rnd.uintLessThan(u16, 16);
                if (std.mem.indexOfScalar(u16, out[0..k], e) == null) {
                    out[k] = e;
                    k += 1;
                }
            }
            return out[0..n];
        }
    }.f;
    var ids_buf: [16]u16 = undefined;
    var seed_buf: [16]u16 = undefined;

    // The prompt pass: each layer's predicted seed read ahead, its barrier, then one route of its ids.
    var want: Reads = .{};
    var p0 = s.stats();
    var t0 = std.Io.Timestamp.now(testing.io, .awake);
    for (0..12) |step| {
        const l: u32 = @intCast(step % 4);
        const ids = pick(rng.random(), &ids_buf, 1 + rng.random().uintLessThan(usize, 12));
        try s.readAheadSeed(l, pick(rng.random(), &seed_buf, 4));
        for (s.ahead.loads[0..s.ahead.n], s.ahead.reads[0..s.ahead.n]) |ld, rd| {
            want.ahead_hits += @intFromBool(std.mem.indexOfScalar(u16, ids, ld.expert) != null);
            want.ahead_records += @intFromBool(rd);
            if (rd) want.ssd_bytes += b.layers[l].logical_bytes;
        }
        try s.seedPrefill(l, ids);
        const r = try s.route(l, ids, &.{});
        want.hits += r.plan.n_hits;
        want.misses += r.plan.n_misses;
        for (r.reads[0..r.plan.n_loads]) |rd| if (rd) {
            want.demand_records += 1;
            want.ssd_bytes += b.layers[l].logical_bytes;
        };
        for (0..r.n_parts) |p| {
            try s.waitGu(r, @intCast(p));
            try s.waitDown(r, @intCast(p));
        }
        s.release(r);
    }
    try s.flush();
    const prompt = Reads.of(p0, s.stats());
    var wall: u64 = @intCast(t0.untilNow(testing.io, .awake).nanoseconds);
    try testing.expect(prompt.wait_ns > 0 and prompt.wait_ns <= wall and want.ahead_records > 0 and want.demand_records > 0);
    want.wait_ns = prompt.wait_ns;
    want.in_flight_ns = prompt.in_flight_ns;
    try testing.expectEqual(want, prompt);
    // The bank has no tokens: its 12 layer calls stand in for them.
    std.debug.print("{f}\n", .{PromptLine{ .tokens = 12, .wall_ns = wall, .r = prompt }});

    // The decode: top-8 routes with the next layer's scores (the lookahead), each layer's hit rate from its own routes.
    try s.grow(&.{ 6, 8, 5, 7 });
    p0 = s.stats();
    const refreshed0 = s.pool.counter(.refreshed);
    const noslot0 = s.pool.counter(.noslot);
    var c0: [4]LayerCounts = undefined;
    for (&c0, 0..) |*c, l| c.* = s.layerCounts(@intCast(l));
    var per_layer: [4]LayerCounts = @splat(.{});
    want = .{};
    var demand_bytes: u64 = 0;
    var scores: [16]f32 = undefined;
    t0 = std.Io.Timestamp.now(testing.io, .awake);
    for (0..24) |step| {
        const l: u32 = @intCast(step % 4);
        const ids = pick(rng.random(), &ids_buf, glm.routed_top_k);
        for (&scores) |*x| x.* = rng.random().float(f32);
        const r = try s.route(l, ids, &scores);
        want.hits += r.plan.n_hits;
        want.misses += r.plan.n_misses;
        per_layer[l].hits += r.plan.n_hits;
        per_layer[l].misses += r.plan.n_misses;
        for (r.reads[0..r.plan.n_loads]) |rd| if (rd) {
            want.demand_records += 1;
            demand_bytes += b.layers[l].logical_bytes;
        };
        for (0..r.n_parts) |p| {
            try s.waitGu(r, @intCast(p));
            try s.waitDown(r, @intCast(p));
        }
        s.release(r);
    }
    try s.flush();
    const s1 = s.stats();
    const decode = Reads.of(p0, s1);
    wall = @intCast(t0.untilNow(testing.io, .awake).nanoseconds);
    try testing.expect(decode.wait_ns <= wall);
    try testing.expectEqual(want.hits, decode.hits);
    try testing.expectEqual(want.misses, decode.misses);
    try testing.expectEqual(@as(u64, 24 * glm.routed_top_k), decode.hits + decode.misses);
    try testing.expectEqual(want.demand_records, decode.demand_records);
    // Every demand record landed whole (read, pre-read or copied out of a speculative record); the SSD bytes are the
    // ones a preadv read: the demand's less its copies, plus the speculative records'.
    try testing.expectEqual(demand_bytes, s1.expert_bytes_read - p0.expert_bytes_read);
    try testing.expectEqual(demand_bytes - (s1.adopt_bytes - p0.adopt_bytes) + (s1.spec_bytes - p0.spec_bytes), decode.ssd_bytes);
    try testing.expect(decode.spec_issued > 0 and decode.spec_used <= decode.spec_issued and decode.ahead_records == 0);
    // The lookahead's precision: every record the selector chose for the next routed layer counted at its top-k
    // position (the prompt pass chose none), its target call routing at least the ones it claimed. The pool issued each
    // chosen record, refreshed it (its record of an earlier step still live) or skipped it for want of a slot: which one
    // depends on how far the reads got, the sum does not.
    const by1 = s.lookaheadByRank();
    var la: [2]u64 = .{ 0, 0 };
    for (by1[0]) |x| {
        la[0] += x[0];
        la[1] += x[1];
    }
    const refreshed: u64 = @intCast(s.pool.counter(.refreshed) - refreshed0);
    const noslot: u64 = @intCast(s.pool.counter(.noslot) - noslot0);
    try testing.expectEqual(decode.spec_issued + refreshed + noslot, la[0]);
    try testing.expect(la[1] >= decode.spec_used and la[1] <= la[0]);
    var c1: [4]LayerCounts = undefined;
    for (&c1, 0..) |*c, l| c.* = s.layerCounts(@intCast(l));
    var rates: [4]f64 = undefined;
    var want_rates: [4]f64 = undefined;
    const spread = hitSpread(&c0, &c1, &rates).?;
    try testing.expectEqual(hitSpread(&@as([4]LayerCounts, @splat(.{})), &per_layer, &want_rates).?, spread);
    // 24 layer calls: 6 steps of the bank's 4 routed layers.
    std.debug.print("{f}\n", .{DecodeLine{ .steps = 6, .tokens = 6, .wall_ns = wall, .r = decode, .hit_rate = spread, .rows = 6 }});
    const by0: @TypeOf(by1) = @splat(@splat(.{ 0, 0 }));
    std.debug.print("{f}\n", .{PrecisionLine{ .by0 = &by0, .by1 = &by1 }});
}

test "glm handover: the settle reads until the cache is empty and the host side is back at its reading before the frees, at most its wait" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const Fake = phase_mem.FakeReader(Mem);
    var i: usize = 0;
    var slept: u32 = 0;
    var cleared: u32 = 0;
    // Before the frees: the host side 0.92 GB. A late buffer in the cache, then the footprint trailing the frees, then
    // settled.
    const host_before: u64 = 920_000_000;
    const lagging = [_]Mem{ .{ .cache = 4096, .active = 89_000_000_000, .footprint = 100_000_000_000 }, .{ .active = 89_000_000_000, .footprint = 100_000_000_000 }, .{ .active = 89_000_000_000, .footprint = 89_930_000_000 } };
    const st = settle(Fake{ .readings = &lagging, .i = &i, .slept_ms = &slept, .clears = &cleared }, host_before);
    try testing.expect(st.settled);
    try testing.expectEqual(@as(u32, 2 * settle_poll_ms), st.waited_ms);
    try testing.expectEqual(@as(u32, 1), cleared);
    try testing.expectEqual(@as(u64, 89_930_000_000), st.after.footprint);
    // The footprint 0.21 GB short of its frees (the host side 1.13 GB): the settle waits for them.
    i = 0;
    slept = 0;
    const trailing = [_]Mem{ .{ .active = 80_000_000_000, .footprint = 81_130_000_000 }, .{ .active = 80_000_000_000, .footprint = 80_960_000_000 } };
    const st3 = settle(Fake{ .readings = &trailing, .i = &i, .slept_ms = &slept, .clears = &cleared }, host_before);
    try testing.expect(st3.settled);
    try testing.expectEqual(@as(u32, settle_poll_ms), st3.waited_ms);
    try testing.expectEqual(@as(u64, 80_960_000_000), st3.after.footprint);
    // Never down to it: the wait ends at its bound with the last reading, not settled.
    i = 0;
    slept = 0;
    const never = [_]Mem{.{ .active = 80_000_000_000, .footprint = 81_000_000_000 }};
    const st2 = settle(Fake{ .readings = &never, .i = &i, .slept_ms = &slept, .clears = &cleared }, host_before);
    try testing.expect(!st2.settled);
    try testing.expectEqual(settle_max_ms, st2.waited_ms);
    try testing.expectEqual(settle_max_ms, slept);
}

test "glm handover: the settle waits past the stack f2be1ba handover reading that the footprint test took 0.226 GB early" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    // Before the frees: the footprint 225.033 GB, MLX active 206.944 GB and cache 17.166 GB (the host side 0.923 GB). The
    // handover freed the 8.004 GB scratch and resized the KV lanes from 1.586 to 1.685 GB.
    const before: Mem = .{ .active = 206_944_070_968, .cache = 17_165_538_394, .footprint = 225_033_000_000 };
    const scratch: u64 = 8_003_911_680;
    const kv0: u64 = 1_586_298_880;
    const kv1: u64 = 1_685_258_240;
    // The readings after them: the footprint 12.8 GB behind the frees, then 0.226 GB behind (the reading the run took),
    // then back.
    const readings = [_]Mem{
        .{ .active = 199_038_725_432, .footprint = 211_801_191_784 },
        .{ .active = 199_039_000_000, .footprint = 200_188_000_000 },
        .{ .active = 199_039_000_000, .footprint = 199_969_000_000 },
    };
    // The footprint test the module had: within 0.25 GB of the reading before less the frees plus the resize, and the
    // host side within 0.25 GB of the prompt phase's host terms (1.039 GB). It takes the second reading.
    const footprint_test = struct {
        fn ok(m: Mem, b: Mem) bool {
            const expected = (b.footprint + kv1) -| (b.cache + scratch + kv0);
            return m.cache == 0 and m.footprint <= expected + 250_000_000 and m.hostSide() <= 1_039_362_432 + 250_000_000;
        }
    }.ok;
    try testing.expect(!footprint_test(readings[0], before) and footprint_test(readings[1], before));
    var i: usize = 0;
    var slept: u32 = 0;
    const st = settle(phase_mem.FakeReader(Mem){ .readings = &readings, .i = &i, .slept_ms = &slept }, before.hostSide());
    try testing.expect(st.settled);
    try testing.expectEqual(readings[2], st.after);
    try testing.expectEqual(@as(u32, 2 * settle_poll_ms), st.waited_ms);
}

test "glm handover: the handover line reports the readings, the rows each bound allows and decode's bound" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const a = testing.allocator;
    var l: HandoverLine = .{ .before = .{ .footprint = 225_500_000_000 }, .after = .{ .footprint = 221_000_000_000, .active = 220_400_000_000 }, .settle_ms = 15, .settled = true, .positions = 17_415, .kv_bytes = 1_660_000_000, .prompt_rows = 126, .rows = 136, .grown_bytes = 24_459_264_000, .bill_bytes = 24_500_000_000, .live_bytes = 26_900_000_000, .bound = 255_000_000_000, .mlx_bound = 238_000_000_000, .target = 255_550_554_112, .gpu_target = 255_550_554_112, .baseline = 12_402_409_472, .box = .{ .used = 233_900_000_000, .others = 12_600_000_000 }, .grown = .{ .footprint = 237_100_000_000 } };
    const s = try std.fmt.allocPrint(a, "{f}", .{l});
    defer a.free(s);
    try testing.expectEqualStrings("glm_moe_dsa: handover: footprint 225.500 GB before the frees, 221.000 GB after (settled in 15 ms; MLX active 220.400 GB, host side 0.600 GB; the box 233.900 GB used, the rest of it 12.600 GB), KV at 17415 positions 1.660 GB, rows 126 -> 136 per layer, 24459264000 B grown (the bill at the request 24500000000 B, the live reading 26900000000 B), decode bound 255.000 GB of the 255.551 GB target (the load's baseline 12.402 GB), MLX 238.000 GB of the GPU's 255.551 GB, footprint 237.100 GB after the grow", s);
    // A capped arm: at the cap (its grown bytes within both budgets), and under it.
    l.cap = 136;
    l.cap_bytes = 24_500_000_000;
    const at = try std.fmt.allocPrint(a, "{f}", .{l});
    defer a.free(at);
    try testing.expect(std.mem.indexOf(u8, at, "(the bill at the request 24500000000 B, the live reading 26900000000 B), capped at 136 units, decode bound") != null);
    l.cap_bytes = 24_500_000_001;
    const under = try std.fmt.allocPrint(a, "{f}", .{l});
    defer a.free(under);
    try testing.expect(std.mem.indexOf(u8, under, "(the bill at the request 24500000000 B, the live reading 26900000000 B), under the units cap of 136 units, decode bound") != null);
    l.cap = null;
    // The rest of the box under its floor: billed at the floor.
    l.rest = 21_000_000_000;
    const floored = try std.fmt.allocPrint(a, "{f}", .{l});
    defer a.free(floored);
    try testing.expect(std.mem.endsWith(u8, floored, ", footprint 237.100 GB after the grow, the rest of the box billed at its floor 21.000 GB"));
    l.rest = 0;
    l.refused = "DecodeOverTarget";
    const r = try std.fmt.allocPrint(a, "{f}", .{l});
    defer a.free(r);
    try testing.expect(std.mem.endsWith(u8, r, ", refused: DecodeOverTarget (the prompt's 126 rows per layer leave decode over its target)"));
    const pm: PhaseMem = .{ .start = .{ .active = 200_000_000_000, .footprint = 201_000_000_000 }, .end = .{ .active = 201_000_000_000, .peak = 205_000_000_000, .cache = 500_000_000, .footprint = 202_100_000_000, .footprint_peak = 206_000_000_000 } };
    const p = try std.fmt.allocPrint(a, "{f}", .{pm});
    defer a.free(p);
    try testing.expectEqualStrings("MLX active 200.000 -> 201.000 GB (peak 205.000), MLX cache 0.500 GB, footprint 201.000 -> 202.100 GB (peak 206.000), host side 1.000 -> 0.600 GB", p);
}

test "glm box: the box's used memory holds this process's footprint, and the rest of the box is what is left" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    if (comptime !@import("builtin").os.tag.isDarwin()) return error.SkipZigTest;
    const block = try std.heap.page_allocator.alloc(u8, 64 << 20);
    defer std.heap.page_allocator.free(block);
    @memset(block, 0x5a);
    const m: Mem = .{ .footprint = sdk.memory.processMemory().footprint };
    const box = try Box.now(m);
    try testing.expect(box.used >= m.footprint and box.used <= sdk.memory.totalMemBytes());
    try testing.expectEqual(box.used - m.footprint - bill_mod.wireTables(0), box.others);
}

test "glm exl3 module: on a synthetic EXL3 pack the served union builds the EXL3 module, each bank layer's rows its share of the units, and a prompt and decode steps run on the GPU" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const a = testing.allocator;
    var model = try exl3_bank.tinyConfigInter(a, 512);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const img = try exl3_bank.writeSynth(a, testing.io, tmp.dir, &model, .{ .signs = true });
    defer a.free(img);
    try glm.writeResidents(a, testing.io, tmp.dir, &model);
    var rbuf: [512]u8 = undefined;
    const dir = try exl3_bank.tmpRoot(&tmp, &rbuf);
    var cfg: settings.Config = .{ .model_dir = dir, .model = model, .max_context_tokens = 64, .expert_rows = 5, .expert_prefill_rows = 3, .expert_lookahead_budget = 3 };
    defer cfg.deinit(a);
    var weights = try sdk.loader.dir(testing.io, a, dir, .{});
    defer weights.deinit();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const sv = try Served.init(a, testing.io, &cfg, &weights, s, .{ .ceiling = 64 << 30, .wired_margin = 0 });
    defer sv.deinit();
    const m = sv.exl3;
    // The prompt pass's cache limit the module set is the bill's prompt cache term: this box's sixteenth of its RAM.
    var limit: usize = 0;
    _ = mlx.mlx_set_cache_limit(&limit, m.cacheLimit(.prompt));
    try testing.expectEqual(m.billed.phases[@intFromEnum(bill_mod.Phase.prompt)].mlx_cache, @as(u64, limit));
    try testing.expectEqual(bill_mod.prefillCacheBytes(sdk.memory.totalMemBytes()), m.inputs.prompt_cache);
    try testing.expectEqual(m.inputs.prompt_cache, bill_mod.billOf(m.inputs).phases[@intFromEnum(bill_mod.Phase.prompt)].mlx_cache);
    // The setting's lookahead budget is the stream's.
    try testing.expectEqual(@as(u32, 3), m.stream.selector.?.budget);
    try testing.expectEqual(@as(usize, 8), m.decode_rows.len);
    for (m.prompt_rows, m.decode_rows) |p, d| {
        try testing.expectEqual(@as(u32, 3), p);
        try testing.expectEqual(@as(u32, 5), d);
    }
    const prompt = [_]u32{ 3, 17, 9, 101, 44, 250, 7, 63, 12, 5, 99, 31 };
    const logits = try m.prefillAt(0, &prompt);
    var t = try m.g.hostArgmax(logits);
    _ = mlx.mlx_array_free(logits);
    try m.decodeHandover(.{ .prompt_tokens = prompt.len, .reserved_tokens = prompt.len + 4, .native_draft = false });
    var row: [256]f32 = undefined;
    for (0..4) |_| {
        const lg = try m.extend(&.{t});
        defer _ = mlx.mlx_array_free(lg);
        _ = try m.g.hostF32(try m.g.astype(lg, .float32), &row);
        for (row) |v| try testing.expect(std.math.isFinite(v));
        t = try m.g.hostArgmax(lg);
    }
    const st = m.stats();
    try testing.expect(st.expert_cache_misses > 0 and st.route_calls > 0);
    // Four serial steps over the 4 routed layers: 16 decode-lane calls in the host split.
    try testing.expectEqual(@as(u64, 16), m.ex.host.calls);
    try testing.expect(m.ex.host.barrier_ns > 0 and m.ex.host.route_ns > 0 and m.ex.host.wave_ns > 0);
    m.requestEnd();
}

test "glm residents: the constructed module holds every resident in its own MLX buffer, before any forward" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const a = testing.allocator;
    var model = try exl3_bank.tinyConfigInter(a, 512);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const img = try exl3_bank.writeSynth(a, testing.io, tmp.dir, &model, .{ .signs = true });
    defer a.free(img);
    try glm.writeResidents(a, testing.io, tmp.dir, &model);
    var rbuf: [512]u8 = undefined;
    const dir = try exl3_bank.tmpRoot(&tmp, &rbuf);
    var cfg: settings.Config = .{ .model_dir = dir, .model = model, .max_context_tokens = 64, .expert_rows = 5, .expert_prefill_rows = 3 };
    defer cfg.deinit(a);
    // The host's loader as the served path calls it (`residents_past_page_cache`): lazy arrays, none read yet.
    var weights = try sdk.loader.dir(testing.io, a, dir, .{ .nocache = true });
    defer weights.deinit();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const sv = try Served.init(a, testing.io, &cfg, &weights, s, .{ .ceiling = 64 << 30, .wired_margin = 0 });
    defer sv.deinit();
    var it = weights.map.iterator();
    var n: u32 = 0;
    while (it.next()) |e| {
        errdefer std.debug.print("resident {s} not in its buffer at construction\n", .{e.key_ptr.*});
        var ok = false;
        try mlx.check(mlx._mlx_array_is_available(&ok, e.value_ptr.*));
        try testing.expect(ok);
        n += 1;
    }
    try testing.expect(n > 0);
}
