//! GLM-5.3-Flash's routed experts from its EXL3 K=3 mul1 bank over MiMo's stream (`mimo/stream_source.For`: the read
//! pool, the event gates, the lookahead, the read schedule) and the shared scheduler (`experts.zig` `ExpertsWith`: the
//! decode lane, the wide prompt lane, the wide feed and the read-ahead), the GLM-shape EXL3 texts (`exl3_math.zig`) and
//! MiMo's in-place prompt combine (`mimo/prefill_combine.zig`). A routed call's output is the weighted sum of its experts
//! [nt, H] bf16; the trunk adds the shared expert.

const std = @import("std");
const env = @import("mlx_host").env;
const sdk = @import("sdk");
const mlx = sdk.mlx;
const sdk_ext = @import("../sdk_ext.zig");
const expert_event = sdk_ext.expert.event;
const expert_tally = sdk_ext.expert.tally;
const expert_usage = sdk_ext.expert.usage;
const ops = @import("../ops.zig");
const shared_experts = @import("../experts.zig");
const mimo_stream = @import("../mimo/stream.zig");
const bridge = @import("../mimo/stream_source.zig");
const mimo_math = @import("../mimo/predictor.zig");
const prefill_combine = @import("../mimo/prefill_combine.zig");
const exl3_bank = @import("exl3_bank.zig");
const exl3_math = @import("exl3_math.zig");
const exl3_kernels = @import("exl3_kernels.zig");
const settings = @import("settings.zig");
const prompt_levers = @import("prompt_levers.zig");
const profile = @import("../mimo/profile.zig");
const gpu_warm = @import("../mimo/gpu_warm.zig");
const ptl = @import("../mimo/prefill_timeline.zig");
const route_trace = @import("../mimo/route_trace.zig");
const log = sdk.log;

const T = mlx.mlx_array;
const Module = @This();

pub const top_k: u32 = 8;
/// Profile builds: the route trace over the bank's 288 experts (u16 ids, `GLM5_NEXT_ROUTE_TRACE`).
pub const Trace = route_trace.TraceOf(u16, "glm5_next-route-trace-v2");
/// The bridge over the NVFP4 bank: its {w, scales, global_scale} slot bindings, its read pool's 10 MiB staging.
pub const B = bridge.For(exl3_bank, .{ .Arrays = exl3_bank.Arrays, .staging_bytes = settings.reader_staging_bytes, .Trace = Trace });
pub const Stream = B.Stream;
pub const StreamSource = B.StreamSource;
pub const Math = exl3_math.Math;
pub const Experts = shared_experts.ExpertsWith(ops.MlxOps, StreamSource, Math, .{ .prefill = true, .gated = true, .lookahead = true });
pub const Gate = Experts.Gate;

/// The routed experts' quant family: the EXL3 K=3 mul1 bank-v2 pack (exl3_bank.zig, exl3_math.zig).
pub const Family = enum { exl3 };

/// The family's bank module, stream bridge, math and scheduler.
pub fn Types(comptime _: Family) type {
    return struct {
        pub const bank = exl3_bank;
        pub const Bridge = Module.B;
        pub const Math = Module.Math;
        pub const Experts = Module.Experts;
    };
}


/// `decode_predict: bf16`: the next routed layer's scores from its router's BF16 weight as stored ([experts, hidden], MLX's
/// gemv over x [n, hidden]), then the sigmoid and the correction as MiMo's F32 route: f32 [n, experts]. Half the F32
/// copy's bytes and no input cast; the scores pick the lookahead's reads only.
pub fn scoresBf16(_: *const mimo_math.Predictor, g: *ops.MlxOps, x: T, router: T, correction: T) anyerror!T {
    const logits = try g.matmul(x, try g.transpose(router));
    return g.add(try g.sigmoid(try g.astype(logits, .float32)), correction);
}

pub const Residents = B.S.Residents;

/// MiMo's construction routes, fixed here (its stack of record; each exact against its control).
pub const read_schedule: mimo_stream.ReadSchedule = .early_whole_trim;
pub const decode_window: mimo_stream.DecodeWindow = .verify;
/// Prompt lever 1 (glm5next/prefill-levers): the overlap schedule's relocation (`Options.prompt_relocate`). A read-ahead
/// record the call's plans evict before any group hits it (a misprediction: an earlier group's seed miss takes its row)
/// and load again later in the call into a transient row no earlier group reads is copied at the plans from its
/// read-ahead row into that row, so it is read once, not twice (the profile's 33.8 double reads per layer, read again
/// inside the drive-bound MoE stage). The same plans, slots and kernels, the record's own bytes in its row before any
/// kernel reads it: exact by construction, installed unconditionally.
pub const prompt_relocate = true;
/// P7 `pooled`: MLX's cache limit through the prompt pass, purged at a prompt layer's end once half full; decode keeps
/// its own limit (`decode_cache_bytes`).
pub const prompt_cache_bytes: usize = 2 << 30;
pub const decode_cache_bytes: usize = 256 << 20;
/// The prompt pass's routed calls take the wide lane; the decode lane takes a decode-phase call of at most
/// `decode_lane_rows` rows (a serial token; the MTP verify's 4 rows in phase 4).
pub const decode_lane_rows: u32 = Math.decode_forward_rows;
pub const max_experts: u32 = 288;

comptime {
    if (Math.prefill_min_tokens != decode_lane_rows + 1) @compileError("glm5_next: the scheduler's wide threshold is not the decode lane's bound");
    if (mimo_stream.decode_lane_route_ids != decode_lane_rows * top_k) @compileError("glm5_next: decode's window 0 is not the decode lane's route");
}

/// The stream's construction options over a family's bridge `Bp` (pure: the CPU tests read them; the same options for
/// every family): MiMo's event gates and lookahead over `rows`, the phase rows' transient release, the read schedule's
/// overlap route with prompt lever 1's relocation, decode's window 0, and the rate tally (`tally`: every layer pins at
/// most floor(ratio x 288) experts; none at ratio 0), with the demand reads' route
/// (`settings.MissReads`): `direct` sets the pool's direct route and turns the decode pre-read off (a pre-range stages);
/// the stream proves the geometry at its construction (`checkDirectReads`, `checkDirectRows`).
pub fn streamOptionsOf(comptime Bp: type, rows: []const u32, memory: Bp.SlotMemory, event: expert_event.Event, tally: expert_tally.Settings, miss_reads: settings.MissReads) !Bp.Options {
    var opts = Bp.eventOptions(rows, memory, event);
    opts.transient_release = true;
    try Bp.applyReadSchedule(&opts, read_schedule);
    opts.prompt_relocate = prompt_relocate;
    opts.decode_route_ids = mimo_stream.decodeRouteIds(decode_window);
    opts.tally = if (tally.enabled()) .{ .ratio = tally.ratio, .window = tally.window } else null;
    switch (miss_reads) {
        .staged => {},
        .direct => {
            opts.pool.direct = true;
            opts.lookahead.?.preread = false;
        },
    }
    return opts;
}

/// The construction's engagement line of the demand reads' route (`miss_reads`): the installed block (0 = staged), whether
/// the decode pre-read is installed, the geometry the stream checked, and the decode forms installed beside it (`forms`:
/// null on the NVFP4 family), so one line names the experts' whole decode install.
pub fn missReadsLine(buf: []u8, miss_reads: settings.MissReads, block: u64, preread: bool, layers: u32, experts: u32, record_bytes: u64, forms: ?settings.Exl3Forms) []const u8 {
    var fb: [96]u8 = undefined;
    const f = if (forms) |x| std.fmt.bufPrint(&fb, "gateup={s} down={s}", .{ @tagName(x.gateup), @tagName(x.down) }) catch "" else "-";
    return std.fmt.bufPrint(buf, "glm5_next experts: miss_reads={s} block={d} preread={s} records={d}x{d}x{d}B forms {s} ({s})", .{
        @tagName(miss_reads), block, if (preread) "on" else "off", layers, experts, record_bytes, f,
        if (miss_reads == .direct) "demand ranges preadv straight into the slot rows, no staging copy; ranges and rows checked on the block" else "demand ranges through the staging buffer and the copy",
    }) catch buf[0..0];
}

/// `prompt_residents`'s install line (settings.zig `PromptResidents`), printed once at construction.
pub fn promptResidentsLine(residents: settings.PromptResidents) []const u8 {
    return switch (residents) {
        .keep => "glm5_next prompt residents: keep (decode's residents stay for the next prompt)",
        .forget => "glm5_next prompt residents: forget (every request starts from empty rows)",
    };
}

/// What came before a request's start (`requestStart`): the construction (the stream still in its prompt phase, its
/// warm-up's residents), the usage seed (`Source.seedResidents`: the residents the load read), or a reverse change.
pub const Start = enum { construction, seeded, reversed };

/// A request's start on the stream (or its source), after `Source.resumePrefill` (nothing live). After the construction:
/// every resident and the prompt state forgotten (`forgetResidents`: the warm-up's residents belong to no prompt). After
/// the usage seed: the seeded residents stay (the seed left them as `shrink(.keep)` leaves a kept one), whatever
/// `prompt_residents` says. After a reverse change: `forget` (the control) forgets every resident; `keep` leaves the
/// residents `shrink(.keep)` left in the prompt rows (that shrink already cleared their prompt state). A request that
/// stops at its prompt still enters decode (`ttft_tail: deferred` runs the phase change in its tail, `finishTail` ->
/// `enterDecode`), so the next request starts after a reverse change and finds that prompt's residents kept, as after
/// any decode. Then the prompt counts are zeroed and the pins refreshed (`resetPromptCounts`).
pub fn requestStart(stream: anytype, residents: settings.PromptResidents, start: Start) !void {
    const forget = switch (start) {
        .construction => true,
        .seeded => false,
        .reversed => residents == .forget,
    };
    if (forget) _ = try stream.forgetResidents();
    stream.resetPromptCounts();
}


/// The phase route of a routed call (MiMo's `decodeLane`): one row in either phase, up to `decode_lane_rows` in decode.
pub fn decodeLane(nt: u32, decoding: bool) bool {
    return nt == 1 or (decoding and nt <= decode_lane_rows);
}

/// The decode lane's combine (MiMo's stock): `routed` [n, 8, H] bf16 in router order, f32 weights [n, 8]; f32 products,
/// the sum over the 8 routes, bf16 once.
fn combineStock(g: *ops.MlxOps, routed: T, route_weights: T) !T {
    const n = g.shapeOf(routed).d[0];
    const expanded = try g.reshape(route_weights, &.{ n, top_k, 1 });
    return g.astype(try g.sum(try g.mul(try g.astype(routed, .float32), expanded), 1, false), .bfloat16);
}

/// The host's monotonic clock (ns): the phase change's one timing (`beginDecodeTimed`), never per call.
fn monotonicNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// The prompt combine the host's wired policy allows (`MLX_SERVE_WIRED`, read once at construction): `inplace` under
/// `max` (the host's default), `fused` under any other.
pub fn combineRoute() prefill_combine.Route {
    const wired = env.get(.MLX_SERVE_WIRED) orelse return .inplace;
    return if (std.mem.eql(u8, std.mem.span(wired), "max")) .inplace else .fused;
}


pub fn SourceOf(comptime fam: Family) type {
    return struct {
        const Self = @This();
        const Fam = Types(fam);
        const PartsCombineFn = fn (self: *const Self, g: *ops.MlxOps, outs: []const T, loc: T, route_weights: T, nt: u32) anyerror!T;

        /// The prompt combine over the wide call's sources read in place (P7 `inplace`, the host's wired policy `max`).
        fn partsCombineInplace(self: *const Self, g: *ops.MlxOps, outs: []const T, loc: T, route_weights: T, nt: u32) !T {
            if (comptime ptl.enabled) ptl.tile(.reduce, nt, 0, &.{});
            const encode_start = profile.now();
            const output = try self.combine_kernel.callInplace(g, outs, loc, route_weights, nt);
            profile.charge(.encode, encode_start);
            const drain_start = profile.now();
            try g.evalAll(&.{output});
            profile.charge(.drain, drain_start);
            return g.keep(output);
        }

        /// The same text over at most `prefill_combine.sources` bound sources (the scheduler merges the rest): under any other
        /// wired policy (the in-place form needs every source in MLX's residency sets).
        fn partsCombineFused(self: *const Self, g: *ops.MlxOps, outs: []const T, loc: T, route_weights: T, nt: u32) !T {
            if (comptime ptl.enabled) ptl.tile(.reduce, nt, 0, &.{});
            const encode_start = profile.now();
            const output = try self.combine_kernel.call(g, outs, loc, route_weights, nt);
            profile.charge(.encode, encode_start);
            const drain_start = profile.now();
            try g.evalAll(&.{output});
            profile.charge(.drain, drain_start);
            return g.keep(output);
        }

        allocator: std.mem.Allocator,
        bank: Fam.bank.Bank,
        sdk_bank: Fam.bank.SDKBank,
        sdk_stream: *Fam.Bridge.Stream,
        stream_source: Fam.Bridge.StreamSource,
        g: ops.MlxOps,
        /// The prompt phase's rows per routed layer and the decode phase's (each at least the prompt's).
        rows: []const u32,
        decode_rows: []const u32,
        ex: Fam.Experts,
        /// exl3: the routers' gates as the family's scheduler types them (the module's table, copied once at construction).
        own_gates: if (fam == .exl3) []Fam.Experts.Gate else void = if (fam == .exl3) &.{} else {},
        parts_combine: *const PartsCombineFn = partsCombineInplace,
        combine_kernel: prefill_combine.Kernel = .{},
        combine_route: prefill_combine.Route = .inplace,
        /// `prompt_residents` (settings.zig), installed at construction: the residents the reverse change keeps.
        prompt_residents: settings.PromptResidents = .keep,
        /// The usage seed ran and no request started since (`beginPrefill` keeps the seeded residents once).
        seeded_start: bool = false,
        /// MiMo's X12 (`gpu_keep_warm: waits`): the warm engine (its own queue, kernel, buffer and parked thread); the
        /// stream source holds its arm (`warm_arm`: armed by `gate` and `awaitReadAhead`). Null on the control.
        warm: ?*gpu_warm.Warm = null,
        /// P1's host scratch (reused): the call's predicted counts (read back from the GPU, prompt lever 5), the ranking.
        ahead_counts: [max_experts]u32 = undefined,
        ahead_ranked: [max_experts]u16 = undefined,

        /// Construction (the GPU stream, under the caller's lock): the bank opened, the stream at `rows`, the scheduler
        /// bound. `gates` holds every model layer's router (the dense layers' empty): the lookahead's predictor reads the next
        /// routed layer's. `prompt`: the prompt levers' routes (prompt_levers.zig; the scheduler's route cut on either
        /// family, the DIG-X wave size and the rows' counting sort on exl3).
        pub fn init(a: std.mem.Allocator, io: std.Io, model_dir: []const u8, rows: []const u32, decode_rows: []const u32, gates: []const Gate, stream: mlx.mlx_stream, decode_moe: settings.DecodeMoe, predict: settings.DecodePredict, keep_warm: gpu_warm.Route, tally: expert_tally.Settings, exl3_forms: settings.Exl3Forms, prompt: prompt_levers.Settings, miss_reads: settings.MissReads, prompt_residents: settings.PromptResidents) !*Self {
            const Bm = Fam.Bridge;
            const bm = Fam.bank;
            var open_diag: sdk.Diag = .{};
            var bank = (if (comptime fam == .exl3) bm.Bank.open(a, io, model_dir, bm.flash_first_moe_layer, bm.flash, &open_diag) else bm.Bank.open(a, io, model_dir)) catch |e| {
                if (open_diag.len > 0) log.err("{s}\n", .{open_diag.message()});
                return e;
            };
            errdefer bank.deinit();
            const geo = bank.geometry;
            try geo.requireFlash();
            if (rows.len != geo.num_moe_layers or decode_rows.len != rows.len) return error.Glm5NextExpertRows;
            var widest: u32 = mimo_stream.transient_rows;
            for (rows, decode_rows) |p, d| {
                if (p == 0 or p > d or d > geo.num_experts) return error.Glm5NextExpertRows;
                widest = @max(widest, d);
            }
            // Every slot array within the device's largest buffer (once, before any allocation).
            const buffer_limit = mlx.maxBufferLength();
            if (buffer_limit == 0) return error.Glm5NextBufferLimitUnknown;
            for (geo.segments) |seg| if (@as(u64, widest) * seg.length > buffer_limit) return error.Glm5NextBufferLimit;
            if (gates.len != geo.first_moe_layer + geo.num_moe_layers) return error.Glm5NextGateTable;

            const self = try a.create(Self);
            errdefer a.destroy(self);
            self.* = .{ .allocator = a, .bank = bank, .sdk_bank = undefined, .sdk_stream = undefined, .stream_source = undefined, .g = undefined, .rows = undefined, .decode_rows = undefined, .ex = undefined };
            self.prompt_residents = prompt_residents;
            self.rows = try a.dupe(u32, rows);
            errdefer a.free(self.rows);
            self.decode_rows = try a.dupe(u32, decode_rows);
            errdefer a.free(self.decode_rows);
            self.g = try ops.MlxOps.init(a, stream);
            errdefer self.g.deinit();
            self.sdk_bank = try bm.SDKBank.init(a, &self.bank);
            errdefer self.sdk_bank.deinit();
            // The reader's staging holds the bank's widest span (Flash's gate/up span plus its page alignment and a page):
            // refused here, once, if a bank's spans outgrow it.
            if (Bm.S.stagingBytes(&self.sdk_bank) > settings.reader_staging_bytes) return error.Glm5NextReaderStaging;
            const event = try expert_event.createMetal();
            self.sdk_stream = try Bm.Stream.init(a, &self.sdk_bank, try streamOptionsOf(Bm, self.rows, .{ .mlx = stream }, event, tally, miss_reads));
            errdefer self.sdk_stream.deinit();
            self.stream_source = Bm.StreamSource.init(self.sdk_stream);
            try self.stream_source.installReadSchedule(read_schedule, mimo_stream.default_read_schedule_threshold);
            // MiMo's X12 (`gpu_keep_warm`, the route the module resolved: `settings.Config.warmRoute`): `.waits` makes the warm
            // queue, kernel and buffer on MLX's device (no kernel runs) and starts the parked warm thread, once, and binds its
            // arm into the stream source (mimo_v2/source.zig's install); `.off` installs nothing; refused by name on a CPU
            // stream or without the stream's Metal event.
            self.warm = try gpu_warm.install(a, keep_warm, stream, event.object);
            errdefer if (self.warm) |w| w.destroy();
            if (self.warm) |w| self.stream_source.warm_arm = gpu_warm.Arm.of(gpu_warm.Warm, w);
            // The slot arrays are the format's, every routed layer's (base and transient banks), checked once.
            for (0..geo.num_moe_layers) |i| {
                const layer: u32 = geo.first_moe_layer + @as(u32, @intCast(i));
                for ([_]Bm.BankKind{ .base, .transient }) |kind| if (try self.stream_source.bankArrays(&self.g, layer, kind)) |arrays| {
                    try bm.checkArrays(&self.g, "gate", arrays.gate, false);
                    try bm.checkArrays(&self.g, "up", arrays.up, false);
                    try bm.checkArrays(&self.g, "down", arrays.down, true);
                };
            }
            var math = blk: {
                // The exl3 family's math: the GLM-shape EXL3 texts installed (their pins, kernels, decode launches per wave
                // width and compile check), refused by name; MiMo's predictor as the NVFP4 family binds it.
                var diag: sdk.Diag = .{};
                var m = Fam.Math.init(a, stream, exl3_math.max_call_rows, exl3_forms.forms(), prompt.wave_rows, if (prompt.route_cut == .on) .count else .sort, &diag) catch |e| {
                    log.err("glm5_next: the exl3 expert texts refused: {s} ({s})\n", .{ diag.message(), @errorName(e) });
                    return e;
                };
                errdefer m.deinit(&self.g);
                m.predictor = switch (predict) {
                    .f32 => try mimo_math.Predictor.install(.f32, geo),
                    .bf16 => .{ .scores = scoresBf16 },
                };
                break :blk m;
            };
            errdefer math.deinit(&self.g);
            self.combine_route = combineRoute();
            switch (self.combine_route) {
                .inplace => {
                    self.combine_kernel = try prefill_combine.Kernel.initInplace(geo);
                    self.parts_combine = partsCombineInplace;
                },
                else => {
                    self.combine_kernel = try prefill_combine.Kernel.init(geo);
                    self.parts_combine = partsCombineFused;
                },
            }
            errdefer self.combine_kernel.deinit();
            const config: struct { hidden_size: u32, moe_intermediate_size: u32, n_layers: u32, n_routed_experts: u32, n_experts_per_tok: u32 } = .{
                .hidden_size = geo.hidden_size,
                .moe_intermediate_size = geo.intermediate_size,
                .n_layers = geo.first_moe_layer + geo.num_moe_layers,
                .n_routed_experts = geo.num_experts,
                .n_experts_per_tok = top_k,
            };
            if (comptime fam == .exl3) {
                self.own_gates = try a.alloc(Fam.Experts.Gate, gates.len);
                for (gates, self.own_gates) |gt, *o| o.* = .{ .w = gt.w, .bias = gt.bias };
            }
            errdefer if (comptime fam == .exl3) a.free(self.own_gates);
            const fam_gates = if (comptime fam == .exl3) self.own_gates else gates;
            self.ex = try Fam.Experts.initWith(a, &self.g, &self.stream_source, math, &config, .{ .gates = fam_gates, .event = event, .wide = .{ .seed = true, .hot_first = true, .depth = mimo_stream.wide_depth, .read_ahead = true, .route_cut = prompt.route_cut == .on }, .banked = decode_moe != .stock, .devroute = decode_moe == .hitpass });
            errdefer self.ex.deinit();
            if (self.combine_route == .inplace) self.ex.setJoinlessBound(prefill_combine.inplace_sources);
            if (comptime fam == .exl3) {
                log.info("glm5_next experts: exl3 k3 mul1 routes (decode_moe={s}: DeepSeek-V4.1's banked PREP=rin chain at hidden {d} / inter {d}, in_rin + gate/up mul1h GEMVs + gu_epi + din_rin + down mul1h GEMV + dpost(bf16), one launch each per wave over the three banks{s}; prompt: DIG-X waves of <= {d} rows / {d} experts, take2 + gate|up GEMM (m128) + onepass + down GEMM (m128) + rot_widen1(bf16); bank {d} x {d} records of {d} B, codebook mul1 0x{X}, no global-scale word: scale-line isolation not applicable; the NVFP4 draft bank unchanged)\n", .{ @tagName(decode_moe), exl3_kernels.hidden, exl3_kernels.inter, if (decode_moe == .hitpass) ", each call's hit wave from the layer's resident slot LUT" else if (decode_moe == .stock) ", unbanked: one group per bank" else "", prompt.wave_rows, @import("../../../quant/exl3/stream/quant.zig").wave_max, geo.num_moe_layers, geo.num_experts, geo.record_bytes, exl3_kernels.codebook_multiplier });
                // The decode forms bound at Math.init (`exl3_gateup_form` / `exl3_down_form`; seam = lane B's chain above).
                log.info("glm5_next experts: exl3 k3 forms gateup={s} down={s}\n", .{ @tagName(exl3_forms.gateup), @tagName(exl3_forms.down) });
            }
            // The demand reads' route as the stream installed it (`miss_reads`): its block, read back from the stream.
            {
                var lb: [320]u8 = undefined;
                log.info("{s}\n", .{missReadsLine(&lb, miss_reads, self.sdk_stream.direct_block, self.sdk_stream.preread, geo.num_moe_layers, geo.num_experts, geo.record_bytes, exl3_forms)});
            }
            log.info("{s}\n", .{promptResidentsLine(prompt_residents)});
            try self.beginPrefill();
            return self;
        }

        pub fn deinit(self: *Self) void {
            _ = mlx.mlx_synchronize(self.g.s);
            // X12: the warm thread joined (its chunks in flight retired) before anything else goes (MiMo's order).
            if (self.warm) |w| w.destroy();
            self.ex.releaseParts(&self.g);
            self.ex.math.deinit(&self.g);
            self.combine_kernel.deinit();
            self.ex.deinit();
            if (comptime fam == .exl3) self.allocator.free(self.own_gates);
            self.g.deinit();
            // Stops and joins every read before the slot arrays go.
            self.sdk_stream.deinit();
            self.sdk_bank.deinit();
            self.bank.deinit();
            self.allocator.free(self.rows);
            self.allocator.free(self.decode_rows);
            self.allocator.destroy(self);
        }

        fn settle(self: *Self) !void {
            try mlx.check(mlx.mlx_synchronize(self.g.s));
            self.ex.releaseParts(&self.g);
            try self.sdk_stream.settleRoutes();
        }

        fn setCacheLimit(bytes: usize) void {
            var previous: usize = 0;
            _ = mlx.mlx_set_cache_limit(&previous, bytes);
        }

        /// A prompt layer's end (its output evaluated): MLX's cache purged once half full (P7 `pooled`).
        pub fn promptLayerEnd(_: *const Self) void {
            var cached: usize = 0;
            _ = mlx.mlx_get_cache_memory(&cached);
            if (cached >= prompt_cache_bytes / 2) _ = mlx.mlx_clear_cache();
        }

        /// A new request: the prompt phase, the previous request's seed and prompt counts forgotten; its residents
        /// forgotten (`forget`, MiMo's) or kept after a decode (`keep`; `requestStart`).
        pub fn beginPrefill(self: *Self) !void {
            const start: Start = if (self.sdk_stream.phase == .decode) .reversed else if (self.seeded_start) .seeded else .construction;
            self.seeded_start = false;
            try self.resumePrefill();
            try requestStart(&self.stream_source, self.prompt_residents, start);
        }

        /// The usage seed at the load's end (after the warm-up and the pre-builds, before the first request): the prompt
        /// phase, the warm-up's residents forgotten, then each layer's empty prompt rows filled with its most used experts
        /// of `profile` (`Stream.seedResidents`: the read-ahead's reads, landed before this returns). The first request
        /// keeps them (`Start.seeded`).
        pub fn seedResidents(self: *Self, usage_profile: *expert_usage.Profile) !expert_usage.Seeded {
            try self.resumePrefill();
            _ = try self.sdk_stream.forgetResidents();
            const records = try self.sdk_stream.seedResidents(expert_usage.Order{ .p = usage_profile, .rows = self.rows }, expert_usage.Order.of);
            self.seeded_start = true;
            return expert_usage.Seeded.of(self.sdk_stream.layers, self.rows, records);
        }

        /// The usage profile's record on (`expert_usage_record`), once at construction: decode routes counted.
        pub fn countRoutes(self: *Self) void {
            self.sdk_stream.countRoutes();
        }

        /// The request's usage into `counts` (`tallyShape`: layers x experts), at its end, before `endTally`.
        pub fn addUsage(self: *Self, counts: []u64) void {
            self.sdk_stream.addUsage(counts);
        }

        /// The request's read schedule from its prompt length, once, after `beginPrefill`.
        pub fn chooseReadSchedule(self: *Self, prompt_tokens: u64) !mimo_stream.ReadSchedule {
            return self.stream_source.chooseReadSchedule(prompt_tokens);
        }

        /// The prompt phase: after a decode (every route settled: `settle`), the grown rows and window 0 freed, the
        /// prompt rows' residents kept or forgotten (`prompt_residents`, the stream's `shrink`), MLX's cache cleared, the
        /// prompt's scratch re-created; the prompt pass's cache limit.
        pub fn resumePrefill(self: *Self) !void {
            try self.settle();
            defer setCacheLimit(prompt_cache_bytes);
            if (self.sdk_stream.phase == .decode) {
                _ = try self.ex.shrink(self.rows, switch (self.prompt_residents) {
                    .keep => .keep,
                    .forget => .forget,
                });
                self.g.clearCache();
                _ = try self.ex.regrowTransient(&self.g, true);
            }
        }

        /// The phase change after the prompt pass: decode's cache limit, the prompt's scratch freed, MLX's cache cleared,
        /// every layer grown to its decode rows (the added rows empty) and decode's window 0 allocated.
        pub fn beginDecode(self: *Self) !void {
            _ = try self.beginDecodeTimed();
        }

        /// `beginDecode`, its grow's wall returned (ns; 0 when the stream is past its prompt phase): the module's tail line.
        pub fn beginDecodeTimed(self: *Self) !u64 {
            try self.settle();
            if (self.sdk_stream.phase != .prefill) return 0;
            setCacheLimit(decode_cache_bytes);
            self.g.clearCache();
            _ = try self.ex.releaseTransient();
            self.g.clearCache();
            const t0 = monotonicNs();
            try self.ex.grow(&self.g, self.decode_rows);
            return monotonicNs() - t0;
        }

        /// After the forward's last eval (the logits): the released routes settled and unpinned.
        pub fn flush(self: *Self) !void {
            try self.ex.flush();
        }

        pub fn decoding(self: *const Self) bool {
            return self.sdk_stream.phase == .decode;
        }

        pub fn decodeLaneAt(self: *const Self, nt: u32) bool {
            return decodeLane(nt, self.decoding());
        }

        /// One routed call of absolute layer `layer`: x [nt, H] bf16, ids [nt, 8], f32 weights [nt, 8] -> the weighted sum
        /// [nt, H] bf16, a KEPT handle the caller adopts. The decode lane's output is lazy (the next routing barrier or the
        /// logits evaluate it), with `hoist` (the shared expert) committed during its read wait; a wide call's is evaluated.
        pub fn forward(self: *Self, layer: u32, x: T, expert_ids: T, route_weights: T, hoist: []const T) !T {
            const g = &self.g;
            defer g.reset();
            const nt: u32 = @intCast(g.shapeOf(x).d[0]);
            profile.call();
            if (self.decodeLaneAt(nt)) {
                const routed = try self.ex.at(layer).routedHoist(g, x, try g.astype(expert_ids, .int32), hoist);
                const encode_start = profile.now();
                const output = try combineStock(g, routed, route_weights);
                profile.charge(.encode, encode_start);
                return g.keep(output);
            }
            // Decode's window 0 holds one decode-lane route: no decode-phase call is wide (the module resumes the prompt
            // phase for any wider forward). An invariant of the callers.
            std.debug.assert(!self.decoding());
            const hook = self.ex.at(layer);
            const parts = try hook.routedParts(g, x, try g.astype(expert_ids, .int32));
            defer hook.releaseParts(g);
            if (comptime ptl.enabled) ptl.stage(.join);
            const join_start = profile.now();
            // The tiles are evaluated; this evaluates the merged source, if any.
            try g.evalAll(parts.outs);
            try mlx.check(mlx.mlx_synchronize(g.s));
            try self.ex.flush();
            profile.charge(.join, join_start);
            return self.parts_combine(self, g, parts.outs, parts.loc, route_weights, nt);
        }

        /// Prompt lever 3 (warm.zig): the installed prompt combine's text built by its first dispatch (MLX compiles a custom
        /// kernel's library then) over one token of zero rows, evaluated and dropped. No route, read or slot is touched.
        pub fn warmCombine(self: *Self) !void {
            const g = &self.g;
            const m = g.mark();
            defer g.resetTo(m);
            const h: c_int = @intCast(self.bank.geometry.hidden_size);
            const rows = try g.zeros(&.{ @intCast(top_k), h }, .bfloat16);
            try g.evalAll(&.{rows});
            // Token 0's routed position k reads row k of the one source.
            var pairs: [2 * top_k]i32 = undefined;
            for (0..top_k) |k| {
                pairs[2 * k] = 0;
                pairs[2 * k + 1] = @intCast(k);
            }
            const loc = try g.hostArray(std.mem.sliceAsBytes(&pairs), &.{ 1, @intCast(top_k), 2 }, .int32);
            const weights = try g.zeros(&.{ 1, @intCast(top_k) }, .float32);
            const out = switch (self.combine_route) {
                .inplace => try self.combine_kernel.callInplace(g, &.{rows}, loc, weights, 1),
                else => try self.combine_kernel.call(g, &.{rows}, loc, weights, 1),
            };
            try g.evalAll(&.{out});
        }

        /// The pipeline pre-build of the expert texts (prebuild.zig's caller, construction): the exl3 family's prompt waves
        /// and decode chain over the first routed layer's base bank (exl3_math.zig `Math.prebuild`: zero rows, slot row 0 read
        /// only; no route, read or slot of the stream). The NVFP4 family pre-builds nothing here. Returns the calls made.
        pub fn prebuildExperts(self: *Self) !u32 {
            const g = &self.g;
            const arrays = (try self.stream_source.bankArrays(g, self.bank.geometry.first_moe_layer, .base)) orelse return error.Glm5NextPrebuildBank;
            const n = try self.ex.math.prebuild(g, arrays);
            try mlx.check(mlx.mlx_synchronize(g.s));
            g.clearCache();
            return n;
        }








        /// P1: layer `layer`'s predicted experts, hottest first, read into its persistent rows (the empty ones, then the
        /// rows of the residents it leaves out) ahead of its call.
        pub fn readAheadSeed(self: *Self, layer: u32, ranked: []const u16) !void {
            try self.ex.at(layer).readAheadSeed(ranked);
        }

        pub fn stats(self: *Self) sdk_ext.expert.Stats {
            return self.stream_source.stats();
        }

        /// The rate tally's geometry: the stream's layers (the routed layers) and each one's experts.
        pub fn tallyShape(self: *const Self) struct { layers: u32, experts: u32 } {
            return .{ .layers = @intCast(self.sdk_stream.layers.len), .experts = self.bank.geometry.num_experts };
        }

        /// Every layer's tally seeded from a tally file's rates (`tallyShape`: layers x experts): the pins of the next request.
        pub fn seedTally(self: *Self, rates: []const f32) void {
            self.sdk_stream.seedTally(rates);
        }

        /// The request's end: each layer's decode rates folded in and written to `out` (layers x experts); the layers that
        /// routed in the decode.
        pub fn endTally(self: *Self, out: []f32) u32 {
            return self.sdk_stream.endTally(out);
        }
    };
}

/// The NVFP4 family's source (the stack of record; its code is this file's as before the exl3 family).
/// The exl3 family's source (an EXL3 K=3 mul1 bank-v2 pack).
pub const Exl3Source = SourceOf(.exl3);

test "glm5_next experts: the stream's options install the overlap schedule early_whole_trim with prompt lever 1's relocation (exact), the phase rows' release and decode's window" {
    const testing = std.testing;
    const rows = [_]u32{ 124, 124 };
    const opts = try streamOptionsOf(B, &rows, .host, .{ .id = 7, .object = 123 }, .{ .ratio = 0 }, .staged);
    try testing.expect(opts.tally == null);
    const tallied = try streamOptionsOf(B, &rows, .host, .{ .id = 7, .object = 123 }, .{ .ratio = 0.25, .window = 16 }, .staged);
    try testing.expectEqual(@as(f64, 0.25), tallied.tally.?.ratio);
    try testing.expectEqual(@as(u32, 16), tallied.tally.?.window);
    try testing.expect(tallied.tally.?.layer_experts == null);
    try testing.expectEqual(B.S.PromptReads.overlap, opts.prompt_reads);
    try testing.expectEqual(B.S.AheadPost.whole_trim, opts.prompt_ahead);
    try testing.expect(opts.prompt_relocate);
    try testing.expect(opts.transient_release);
    try testing.expectEqual(mimo_stream.decodeRouteIds(decode_window), opts.decode_route_ids);
    try testing.expectEqual(@as(u64, 123), opts.event.?.backend.metal);
    try testing.expectEqualSlices(u32, &rows, opts.rows);
}

test "glm5_next experts: miss_reads: staged keeps the pool's staging route and the decode pre-read; direct sets the direct route with the pre-read off; the engagement line names the installed route" {
    const testing = std.testing;
    const rows = [_]u32{ 190, 190 };
    const Bx = Types(.exl3).Bridge;
    const staged = try streamOptionsOf(Bx, &rows, .host, .{ .id = 7, .object = 123 }, .{ .ratio = 0 }, .staged);
    try testing.expect(!staged.pool.direct);
    try testing.expect(staged.lookahead.?.preread);
    const direct = try streamOptionsOf(Bx, &rows, .host, .{ .id = 7, .object = 123 }, .{ .ratio = 0 }, .direct);
    try testing.expect(direct.pool.direct);
    try testing.expect(!direct.lookahead.?.preread);
    // Everything else is the control's.
    var same = direct;
    same.pool.direct = false;
    same.lookahead.?.preread = true;
    try testing.expectEqualDeep(staged, same);
    var buf: [320]u8 = undefined;
    const forms: settings.Exl3Forms = .{ .gateup = .guone, .down = .pair };
    try testing.expectEqualStrings("glm5_next experts: miss_reads=direct block=4096 preread=off records=42x288x9474048B forms gateup=guone down=pair (demand ranges preadv straight into the slot rows, no staging copy; ranges and rows checked on the block)", missReadsLine(&buf, .direct, 4096, false, 42, 288, 9_474_048, forms));
    try testing.expectEqualStrings("glm5_next experts: miss_reads=staged block=0 preread=on records=42x288x9474048B forms gateup=guone down=pair (demand ranges through the staging buffer and the copy)", missReadsLine(&buf, .staged, 0, true, 42, 288, 9_474_048, forms));
    try testing.expectEqualStrings("glm5_next experts: miss_reads=staged block=0 preread=on records=42x288x9474048B forms - (demand ranges through the staging buffer and the copy)", missReadsLine(&buf, .staged, 0, true, 42, 288, 9_474_048, null));
}

test "glm5_next experts: prompt_residents: the install line names the installed route" {
    const testing = std.testing;
    try testing.expectEqualStrings("glm5_next prompt residents: keep (decode's residents stay for the next prompt)", promptResidentsLine(.keep));
    try testing.expectEqualStrings("glm5_next prompt residents: forget (every request starts from empty rows)", promptResidentsLine(.forget));
}

test "glm5_next experts: the rate tally's pins on Flash's geometry: the top floor(ratio x 288) experts by rate, at most the layer's rows, ties by the lower id; none unseeded" {
    const testing = std.testing;
    const a = testing.allocator;
    const policy_mod = sdk_ext.expert.policy;
    // A synthetic rate table: expert e's rate falls with e, two ties (experts 10 and 11 share a rate) and four experts
    // never routed (rate 0: never pinned).
    var rates: [max_experts]f32 = undefined;
    for (&rates, 0..) |*r, e| r.* = if (e >= max_experts - 4) 0 else 1.0 / @as(f32, @floatFromInt(e + 1));
    rates[11] = rates[10];
    for ([_]struct { ratio: f64, rows: u32, pins: u32 }{
        .{ .ratio = 0.25, .rows = 143, .pins = 72 },
        .{ .ratio = 0.5, .rows = 143, .pins = 143 }, // floor(0.5 x 288) = 144 > 143 rows: every row pinned
        .{ .ratio = 0.5, .rows = 160, .pins = 144 },
        .{ .ratio = 1.0, .rows = 288, .pins = 284 }, // only experts of positive rate
    }) |case| {
        var p = try policy_mod.LayerPolicy.init(a, max_experts, case.rows);
        defer p.deinit(a);
        p.setTally(expert_tally.pinLimit(case.ratio, max_experts), expert_tally.alphaOf(expert_tally.default_window));
        for (0..max_experts) |e| try testing.expect(!p.isPinned(@intCast(e)));
        p.seedTally(&rates);
        var n: u32 = 0;
        for (0..max_experts) |e| n += @intFromBool(p.isPinned(@intCast(e)));
        try testing.expectEqual(case.pins, n);
        // The hottest are pinned, in rate order (ties by the lower id: 10 before 11).
        for (0..case.pins) |e| try testing.expect(p.isPinned(@intCast(e)));
        if (case.pins < max_experts - 4) try testing.expect(!p.isPinned(@intCast(case.pins)));
    }
}

test "glm5_next experts: the rate tally's file round-trips at Flash's geometry (42 layers x 288 experts); another geometry is ignored" {
    const testing = std.testing;
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &dir_buf);
    const root = dir_buf[0..root_len];
    const key = try expert_tally.keyOf(a, "glm5_next", "/models/GLM-5.3-Flash-NVFP4-fp8res-mlx-stream/");
    defer a.free(key);
    try testing.expectEqualStrings("glm5_next-GLM-5.3-Flash-NVFP4-fp8res-mlx-stream", key);
    const path = try std.fmt.allocPrint(a, "{s}/{s}/{s}", .{ root, key, expert_tally.file_name });
    defer a.free(path);
    const n_layers: u32 = 42;
    const rates = try a.alloc(f32, n_layers * max_experts);
    defer a.free(rates);
    for (rates, 0..) |*r, i| r.* = @as(f32, @floatFromInt(i % 97)) / 97.0;
    try testing.expect((try expert_tally.read(a, io, path, n_layers, max_experts)) == .missing);
    try expert_tally.write(a, io, path, rates, n_layers, max_experts, expert_tally.default_window);
    switch (try expert_tally.read(a, io, path, n_layers, max_experts)) {
        .rates => |back| {
            defer a.free(back);
            try testing.expectEqualSlices(f32, rates, back);
        },
        else => return error.TestUnexpectedResult,
    }
    try testing.expect((try expert_tally.read(a, io, path, 43, max_experts)) == .mismatched);
    try testing.expect((try expert_tally.read(a, io, path, n_layers, 256)) == .mismatched);
}
