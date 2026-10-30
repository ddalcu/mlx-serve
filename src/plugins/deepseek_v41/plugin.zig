//! DeepSeek-V4.1-Flash: the module-owned native arch (`deepseek_v41.zig`) and its vision tower behind the model table.
//! One state per slot: the module runs the active one (`dec_state`), the others wait here, and plain decode tokens of
//! several slots run as one batch (`extendBatched`). Prompt captures transfer to the engine's prefix cache.
//! A streamed pack's routed experts come from its EXL3 bank (`sdk/weights/streamed.zig` `Exl3`).

const std = @import("std");
const build_options = @import("build_options");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const dsv41 = @import("deepseek_v41.zig");
const vision = @import("vision.zig");
const media = @import("media.zig");
const bill_mod = @import("bill.zig");
const fixture = @import("../deepseek_v4/fixture.zig");
const v4plugin = @import("../deepseek_v4/plugin.zig");
const quant_abi = @import("../../abi/quant.zig");
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const Weights = @import("../../sdk/weights/resident.zig").Weights;
const Layout = @import("../../sdk/quant/native.zig").Layout;
const streamed = @import("../../sdk/weights/streamed.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill_mod.bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .release_slot = &releaseSlot,
    .activate_slot = &activateSlot,
    .cache = &.{ .layout = &layout, .resume_prompt = &resumePrompt, .state_bytes = &stateBytes, .aux = &aux_hooks },
    .media = &.{ .tower = &media.tower, .prepare = &media.prepare, .placeholder_rows = &media.placeholderRows },
    .spec = &.{ .trunk_weight = &trunkWeight },
    .weights = &.{ .resident_bytes = &residentBytes },
};

const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;

/// A KV source grows its compressed rows; a compressed layer that is not one reads the last source at or above it;
/// a ratio-0 layer keeps its window. All of it lives in the module's state, never in the engine's cache.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    const c = config.of(p);
    const first: u32 = if (span) |s| s.first else 0;
    const n = @min(if (span) |s| s.count else c.num_hidden_layers, cap);
    var source: i16 = -1;
    for (0..first) |li| if ((c.dsv41_kv_sources >> @intCast(li)) & 1 != 0) {
        source = @intCast(li);
    };
    for (out[0..n], first..) |*l, li| {
        const ratio = c.dsv4_compress_ratios[li];
        const own = (c.dsv41_kv_sources >> @intCast(li)) & 1 != 0;
        if (own) source = @intCast(li);
        l.* = if (ratio == 0)
            .{ .kind = .sliding, .window = c.sliding_window, .kv_heads = 1, .k_dim = c.head_dim }
        else
            .{ .kind = .full, .kv_source = if (own) -1 else source, .kv_heads = 1, .k_dim = c.head_dim / ratio };
    }
    return n;
}

pub const Module = dsv41.Dsv41Model;

/// The module a loaded model runs on (`engines/mlx/plugin/intree.zig`).
pub fn moduleOf(mp: *anyopaque) *Module {
    const m: *Model = @ptrCast(@alignCast(mp));
    return m.mdl;
}

pub const Model = struct {
    cfg: config.Config,
    weights: Weights,
    mdl: *dsv41.Dsv41Model,
    tower: ?vision.Dsv41Vision,
    caps: abi.Caps,
    s: mlx.mlx_stream,
    /// A streamed pack's expert bank, and the prompt positions its slot rows were billed for.
    bank: ?Bank = null,
    bank_context: u64 = 0,
    /// The slot whose state is `mdl.dec_state` (0: none claimed it yet, a resumed state or the module's own); every
    /// other live slot's waits in `parked`. Excluded configurations keep the last finished state in `idle`.
    owner: u32 = 0,
    parked: std.AutoHashMapUnmanaged(u32, Kept) = .empty,
    idle: ?Kept = null,
};

/// A state off the module, with its identity (`Dsv41Model.state_gen`).
const Kept = struct {
    st: dsv41.State,
    gen: u64,

    fn deinit(k: *Kept) void {
        k.st.deinit();
    }
};

/// Parks a live slot; excluded configurations retain an unclaimed state for legacy resume.
fn park(m: *Model) !void {
    const st = m.mdl.dec_state orelse return;
    const k: Kept = .{ .st = st, .gen = m.mdl.state_gen };
    if (m.owner != 0) try m.parked.put(gpa, m.owner, k) else if (!auxEnabled(m)) keepIdle(m, k) else {
        var orphan = k;
        orphan.deinit();
    }
    m.mdl.dec_state = null;
    m.owner = 0;
}

fn keepIdle(m: *Model, k: Kept) void {
    if (m.idle) |*old| old.deinit();
    m.idle = k;
}

/// Makes `slot`'s state the module's. A slot new here claims an unclaimed state (the prompt `resumePrompt` just
/// readied), else starts with none: its prompt at position 0 builds one. Slot 0 parks a live state only for aux.
fn activate(m: *Model, slot: u32) !void {
    if (slot == m.owner) return;
    if (slot == 0) {
        if (auxEnabled(m)) try park(m);
        return;
    }
    const mine = m.parked.fetchRemove(slot);
    if (mine == null and m.owner == 0) {
        m.owner = slot;
        return;
    }
    park(m) catch |e| {
        if (mine) |kv| m.parked.putAssumeCapacity(slot, kv.value);
        return e;
    };
    if (mine) |kv| {
        m.mdl.dec_state = kv.value.st;
        m.mdl.state_gen = kv.value.gen;
    }
    m.owner = slot;
}

fn releaseSlot(mp: *anyopaque, slot: u32) callconv(.c) void {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (!auxEnabled(mp)) {
        if (slot == m.owner) {
            m.owner = 0;
            park(m) catch unreachable; // An unclaimed state never allocates a parked entry.
        } else if (m.parked.fetchRemove(slot)) |kv| keepIdle(m, kv.value);
        return;
    }
    if (slot == m.owner) {
        m.owner = 0;
        if (m.mdl.dec_state) |*st| st.deinit();
        m.mdl.dec_state = null;
    } else if (m.parked.fetchRemove(slot)) |kv| {
        var k = kv.value;
        k.deinit();
    }
}

fn activateSlot(mp: *anyopaque, slot: u32) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    activate(m, slot) catch return Verdict.refuse(.failed, "deepseek_v41: out of memory parking a slot's state");
    return .ok;
}

const Bank = if (build_options.mlx_stream) union(enum) { exl3: *streamed.Exl3.Source, pack: streamed.Streamed } else void;

/// The trunk's embedding table or LM head as loaded (borrowed), for the DSpark drafter.
fn trunkWeight(mp: *anyopaque, which: abi.TrunkWeight, out: *quant_abi.Weight) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    const l = switch (which) {
        .embed => &m.mdl.embed,
        .head => &m.mdl.head,
    };
    out.* = .{ .w = .{ .ctx = l.w.ctx }, .scales = .{ .ctx = l.s.ctx }, .biases = .{ .ctx = l.b.ctx }, .bits = @intCast(l.qp.bits), .group_size = l.qp.group_size };
    if (l.s.ctx != null) out.layout = @backingInt(switch (l.qp.mode) {
        .nvfp4 => Layout.nvfp4,
        .mxfp4 => Layout.mxfp4,
        .mxfp8 => Layout.mxfp8,
        else => Layout.affine,
    });
    return .ok;
}

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const s = gpuOf(.{ .ctx = ctx.stream.ctx }) orelse {
        out.* = Verdict.refuse(.unsupported, "deepseek_v41: its kernels run on Metal only");
        return null;
    };
    const c = config.of(ctx.parsed);
    // The EXL3 prompt GEMMs are built on MLX's NAX tiles: below the M5 they compute wrong (f64 self-check, M4).
    if (c.dsv41_exl3_bank and !@import("../../mlx/device.zig").naxAvailable()) {
        out.* = Verdict.refuse(.unsupported, "deepseek_v41: the streamed EXL3 repack needs an M5 or newer GPU (its prompt kernels run on the M5's matrix units); stream the MLX pack with the stream_experts setting instead");
        return null;
    }
    return loadModel(ctx, s) catch |e| {
        // Too little free memory is a refusal the host answers by name, not a fault.
        if (e == error.StreamedRowsDoNotFit) std.log.warn("deepseek_v41: load refused: {t}", .{e}) else std.log.err("deepseek_v41: load failed: {t}", .{e});
        out.* = switch (@as(anyerror, e)) {
            error.OutOfMemory => Verdict.refuse(.failed, "deepseek_v41: out of memory"),
            error.MissingWeight => Verdict.refuse(.failed, "deepseek_v41: a weight is missing"),
            error.StreamedRowsDoNotFit => Verdict.refuse(.too_large, "deepseek_v41: the streamed bank's rows do not fit the free memory"),
            error.StreamedWeightsUnsupported => Verdict.refuse(.unsupported, "deepseek_v41: a streamed (experts.bin) pack needs the macOS build's expert streamer"),
            else => Verdict.refuse(.failed, "deepseek_v41: the pack does not load"),
        };
        return null;
    };
}

/// The host's stream when it is a GPU one, else this thread's GPU stream; null off Metal: the module's kernels are
/// Metal-only.
fn gpuOf(s: mlx.mlx_stream) ?mlx.mlx_stream {
    if (!mlx.metalKernelsAvailable()) return null;
    if (mlx.streamIsGpu(s)) return s;
    return mlx.gpuStream();
}

/// The per-model `mtp` setting (`{"mtp": false}` keeps the DSpark stages off).
fn mtpSetting(settings_json: []const u8) ?bool {
    if (settings_json.len == 0) return null;
    const S = struct { mtp: ?bool = null };
    const parsed = std.json.parseFromSlice(S, gpa, settings_json, .{ .ignore_unknown_fields = true }) catch return null;
    defer parsed.deinit();
    return parsed.value.mtp;
}

fn loadModel(ctx: *const abi.LoadCtx, s: mlx.mlx_stream) !*Model {
    const io = std.Io.Threaded.global_single_threaded.io();
    const c = config.of(ctx.parsed);
    const m = try gpa.create(Model);
    errdefer gpa.destroy(m);
    m.cfg = c.*;
    // The driver's map never holds the vision tower (the engine's encoder reads it and hands its rows to `forward`).
    const taken = if (ctx.weights) |w| Resident.take(w) else null;
    m.weights = taken orelse try dsv41.loadPack(io, gpa, ctx.model_dir.slice(), c.dsv41_vision);
    errdefer m.weights.deinit();
    // The worker's layers when the driver's load planned a Thunderbolt pair.
    const pair: ?dsv41.Pair = if (ctx.pair_remote.count > 0) .{ .lo = ctx.pair_remote.first, .hi = ctx.pair_remote.first + ctx.pair_remote.count, .worker = false } else null;
    m.s = s;
    m.caps = ctx.parsed.caps;
    m.mdl = try dsv41.init(gpa, &m.cfg, &m.weights, s, .{ .dspark = dsv41.dsparkWanted(&m.cfg, pair, mtpSetting(ctx.settings_json.slice())), .pair = pair, .head_8bit = dsv41.head8Wanted(), .streamed_experts = c.dsv41_streamed, .pack_dir = ctx.model_dir.slice() });
    errdefer m.mdl.deinit();
    m.bank = null;
    m.owner = 0;
    m.parked = .empty;
    m.idle = null;
    if (c.dsv41_streamed) {
        const dir = ctx.model_dir.slice();
        if (c.dsv41_exl3_bank) try openBank(m, io, dir, ctxSetting(ctx.settings_json.slice()), ctx.wired_margin_bytes) else try openPackBank(m, dir, ctxSetting(ctx.settings_json.slice()), ctx.settings_json.slice());
    }
    errdefer closeBank(m);
    m.tower = if (c.dsv41_vision and taken == null) try vision.Dsv41Vision.init(gpa, m.cfg, &m.weights) else null;
    return m;
}

/// The pack's EXL3 bank at the slot rows the box holds beside the residents and a request of the model's `ctx_size`,
/// else of the largest `bill.stream_context_rungs` rung that keeps `bill.stream_min_rows` rows, else of
/// `bill.stream_context`.
fn openBank(m: *Model, io: std.Io, dir: []const u8, ctx_size: ?u64, wired_margin: u64) !void {
    if (comptime !build_options.mlx_stream) return error.StreamedWeightsUnsupported;
    const mdl = m.mdl;
    const gates = try gpa.alloc(streamed.Exl3.Gate, mdl.n_layers);
    defer gpa.free(gates);
    for (gates, mdl.layers[0..mdl.n_layers]) |*g, *ly| g.* = .{ .w = ly.gate_w, .bias = ly.gate_bias };
    _ = mlx.mlx_synchronize(m.s);
    const c = &m.cfg;
    const geo: streamed.Exl3.Geometry = .{ .hidden_size = c.hidden_size, .moe_intermediate_size = c.moe_intermediate_size, .n_experts_per_tok = c.num_experts_per_tok, .n_routed_experts = c.num_experts, .n_layers = @intCast(mdl.n_layers), .swiglu_limit = c.dsv4_swiglu_limit };
    const context = ctx_size orelse for (bill_mod.stream_context_rungs) |rung| {
        const at = bill_mod.streamBudget(c, rung, wired_margin);
        const fit = streamed.Exl3.billOf(gpa, io, dir, geo, .{ .prompt = at.prompt, .decode = at.decode }) catch continue;
        if (fit.rows.prompt >= bill_mod.stream_min_rows) break rung;
    } else bill_mod.stream_context;
    const budget = bill_mod.streamBudget(c, context, wired_margin);
    const b = try streamed.Exl3.Source.open(gpa, io, dir, m.s, geo, gates, .{ .prompt = budget.prompt, .decode = budget.decode });
    std.log.info("deepseek_v41: streamed experts, {d} prompt / {d} decode slot rows per layer for a {d}-token context ({d} B a row)", .{ b.bill.rows.prompt, b.bill.rows.decode, context, b.bill.row_bytes });
    m.bank = .{ .exl3 = b };
    m.bank_context = context;
    mdl.streamed = streamed.exl3Routed(b);
}

/// A standard MLX pack's routed experts streamed as downloaded, at the slot rows the `stream_experts` setting names.
fn openPackBank(m: *Model, dir: []const u8, ctx_size: ?u64, settings_json: []const u8) !void {
    if (comptime !build_options.mlx_stream) return error.StreamedWeightsUnsupported;
    const rows = (streamed.Setting.parse(settings_json) orelse return error.StreamedWeightsUnsupported).slotRows(m.cfg.num_experts);
    _ = mlx.mlx_synchronize(m.s);
    var st = try streamed.Streamed.openPack(gpa, dir, m.s, rows);
    errdefer st.close();
    m.bank = .{ .pack = st };
    m.bank_context = ctx_size orelse bill_mod.stream_context_rungs[0];
    m.mdl.streamed = m.bank.?.pack.routedSource();
    std.log.info("deepseek_v41: streamed MLX pack experts, {d} slot rows per layer", .{rows});
}

fn closeBank(m: *Model) void {
    if (comptime !build_options.mlx_stream) return;
    var b = m.bank orelse return;
    m.mdl.streamed = null;
    switch (b) {
        .exl3 => |x| x.deinit(),
        .pack => |*x| x.close(),
    }
    m.bank = null;
}

/// The per-model `ctx_size` setting: the prompt positions a streamed bank bills.
fn ctxSetting(settings_json: []const u8) ?u64 {
    if (settings_json.len == 0) return null;
    const S = struct { ctx_size: ?u64 = null };
    const parsed = std.json.parseFromSlice(S, gpa, settings_json, .{ .ignore_unknown_fields = true }) catch return null;
    defer parsed.deinit();
    return parsed.value.ctx_size;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Model = @ptrCast(@alignCast(mp));
    closeBank(m);
    if (m.tower) |*t| t.deinit();
    var it = m.parked.valueIterator();
    while (it.next()) |k| k.deinit();
    m.parked.deinit(gpa);
    if (m.idle) |*k| k.deinit();
    m.mdl.deinit();
    m.weights.deinit();
    gpa.destroy(m);
}

pub fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (!abi.declaresMode(m.caps, f.mode)) return Verdict.refuse(.unsupported, "deepseek_v41: forward mode not declared");
    if (f.capture_layers.len > 0) return Verdict.refuse(.unsupported, "deepseek_v41: no layer capture");
    // Before any forward: a position-0 run would reset the live state for rows nobody reads.
    if (f.want_hidden) return Verdict.refuse(.unsupported, "deepseek_v41: no hidden-state output (embeddings)");
    if (f.mode == .batched_rows) return forwardBatched(m, f, out);
    if (f.slots.len > 1 or f.rows == 0) return Verdict.refuse(.unsupported, "deepseek_v41: one slot's rows at a time");
    if (f.slots.len == 1) activate(m, f.slots.slice()[0]) catch return Verdict.refuse(.failed, "deepseek_v41: out of memory parking a slot's state");
    const in = v4plugin.Ids.of(f) catch return Verdict.refuse(.failed, "deepseek_v41: the token ids do not read back");
    defer in.deinit();
    const pos: usize = if (f.positions.len > 0) f.positions.slice()[0] else 0;
    if (pos != 0 and (m.mdl.dec_state == null or m.mdl.dec_state.?.n != pos)) return Verdict.refuse(.failed, "deepseek_v41: the position does not follow the model's state");
    for (in.host) |t| if (t >= m.cfg.vocab_size) return Verdict.refuse(.failed, "deepseek_v41: token id past the vocabulary");
    if (m.bank != null and in.host.len > 1 and f.logits != .all_rows and pos + in.host.len > m.bank_context)
        return Verdict.refuse(.failed, "deepseek_v41: the prompt is longer than the streamed bank's billed context (ctx_size)");
    const rows: ?dsv41.VisionRows = if (f.media.ctx != null) .{ .rows = .{ .ctx = f.media.ctx }, .offset = f.media_offset } else null;
    const logits = run(m, in.host, pos, f.logits == .all_rows, rows, in.dev) catch return Verdict.refuse(.failed, "deepseek_v41: the forward failed");
    if (f.logits == .none) _ = mlx.mlx_array_free(logits) else out.logits = .{ .ctx = logits.ctx };
    return .ok;
}

/// One decode token per slot, each at its state's position: one batch of rows (`extendBatched`), or each slot alone
/// where the module runs no batch (a Thunderbolt pair, a streamed bank, an open verify, a slot starting at 0).
fn forwardBatched(m: *Model, f: *const abi.Forward, out: *abi.ForwardOut) Verdict {
    const n = f.slots.len;
    if (n == 0 or n > max_batch or f.rows != 1 or f.tokens.len != n or f.logits == .all_rows or f.want_hidden)
        return Verdict.refuse(.unsupported, "deepseek_v41: a batch is one host token per slot");
    var states: [max_batch]*dsv41.State = undefined;
    var gens: [max_batch]u64 = undefined;
    var alone = m.mdl.pair != null or m.mdl.streamed != null;
    for (f.slots.slice(), 0..) |slot, k| {
        if (f.positions.len == n and f.positions.slice()[k] == 0) {
            alone = true;
            continue;
        }
        if (slot == m.owner and m.mdl.dec_state != null) {
            states[k] = &m.mdl.dec_state.?;
            gens[k] = m.mdl.state_gen;
        } else if (m.parked.getPtr(slot)) |kept| {
            states[k] = &kept.st;
            gens[k] = kept.gen;
        } else return Verdict.refuse(.failed, "deepseek_v41: a batched slot has no state");
        if (f.positions.len == n and f.positions.slice()[k] != states[k].n) return Verdict.refuse(.failed, "deepseek_v41: the position does not follow the slot's state");
        if (states[k].verify != null) alone = true;
    }
    for (f.tokens.slice()) |t| if (t >= m.cfg.vocab_size) return Verdict.refuse(.failed, "deepseek_v41: token id past the vocabulary");
    const logits = if (alone) eachAlone(m, f) else dsv41.extendBatched(m.mdl, gpa, states[0..n], gens[0..n], f.tokens.slice());
    out.logits = .{ .ctx = (logits catch return Verdict.refuse(.failed, "deepseek_v41: the batched forward failed")).ctx };
    return .ok;
}

const max_batch = 16;

/// `forwardBatched`'s rows one slot at a time, as each slot's own decode step.
fn eachAlone(m: *Model, f: *const abi.Forward) !mlx.mlx_array {
    var rows: [max_batch]mlx.mlx_array = undefined;
    var done: usize = 0;
    defer for (rows[0..done]) |r| {
        _ = mlx.mlx_array_free(r);
    };
    for (f.slots.slice(), f.tokens.slice(), 0..) |slot, t, k| {
        try activate(m, slot);
        const pos = if (f.positions.len == f.slots.len) f.positions.slice()[k] else if (m.mdl.dec_state) |st| st.n else 0;
        rows[done] = try run(m, &.{t}, pos, false, null, .{ .ctx = null });
        done += 1;
    }
    var o = mlx.mlx_array_new();
    const v = mlx.mlx_vector_array_new_data(&rows, done);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_concatenate_axis(&o, v, 0, m.s));
    return o;
}

/// The legacy seam's arms: position 0 starts a request (a resumed one arrives past 0); a decode token stays on the
/// GPU (`dev`, or the host id uploaded); everything else extends the state. `all` returns every row's logits.
fn run(m: *Model, host_ids: []const u32, pos: usize, all: bool, rows: ?dsv41.VisionRows, dev: mlx.mlx_array) !mlx.mlx_array {
    const mdl = m.mdl;
    var one: [1]u32 = undefined;
    const gpu_decode = !all and pos > 0 and (dev.ctx != null or host_ids.len == 1) and mdl.pair == null;
    const ids = if (dev.ctx != null and !gpu_decode) blk: {
        one[0] = try v4plugin.hostToken(dev);
        if (one[0] >= m.cfg.vocab_size) return error.TokenPastVocabulary;
        break :blk one[0..];
    } else host_ids;
    const logits = if (all) blk: {
        if (pos == 0) try dsv41.reset(mdl, gpa);
        break :blk try dsv41.extendAllResumable(mdl, gpa, ids);
    } else if (gpu_decode) blk: {
        const tok = if (dev.ctx != null) try v4plugin.deviceToken(dev, &.{ 1, 1 }, m.s) else blk2: {
            const id: i32 = @intCast(ids[0]);
            break :blk2 mlx.mlx_array_new_data(&id, &[_]c_int{ 1, 1 }, 2, .int32);
        };
        defer _ = mlx.mlx_array_free(tok);
        break :blk try dsv41.extendResumableGpu(mdl, gpa, tok);
    } else try runPrompt(m, ids, pos, rows);
    defer _ = mlx.mlx_array_free(logits);
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_reshape(&o, logits, &[_]c_int{ -1, @intCast(mdl.vocab) }, 2, m.s));
    return o;
}

/// The rows of one prompt call: a streamed EXL3 bank's up to `bill.stream_call_rows`; an MLX pack's the most whose pairs
/// one group's fetch takes (`max_call_ids`).
fn callRows(m: *const Model, n: usize) usize {
    if (comptime !build_options.mlx_stream) return n;
    const b = m.bank orelse return n;
    return switch (b) {
        .exl3 => bill_mod.stream_call_rows,
        .pack => streamed.max_call_ids / m.cfg.num_experts_per_tok,
    };
}

/// A prompt from `pos` (0: a new request), in calls of at most `callRows`.
fn runPrompt(m: *Model, ids: []const u32, pos: usize, rows: ?dsv41.VisionRows) !mlx.mlx_array {
    const mdl = m.mdl;
    if (pos == 0) try dsv41.reset(mdl, gpa);
    const call = callRows(m, ids.len);
    var vr = rows;
    var at: usize = 0;
    while (true) {
        const piece = ids[at..][0..@min(call, ids.len - at)];
        const out = try dsv41.extendResumable(mdl, gpa, piece, vr);
        at += piece.len;
        if (at == ids.len) return out;
        _ = mlx.mlx_array_free(out);
        if (vr) |*v| v.offset += std.mem.count(u32, piece, &.{m.cfg.image_token_id});
    }
}

/// `CacheHooks.resume_prompt`: restores the longest prefix of `prompt` the model kept (images keyed by their rows)
/// and returns its length; a cold prompt, or a failed restore, resets the state and returns 0.
pub fn resumePrompt(mp: *anyopaque, slot: u32, prompt: abi.Span(u32), media_rows: abi.Array) callconv(.c) u64 {
    const m: *Model = @ptrCast(@alignCast(mp));
    const rows: ?mlx.mlx_array = if (media_rows.ctx != null) .{ .ctx = media_rows.ctx } else null;
    if (auxEnabled(mp)) return 0;
    // A parking failure leaves the other live slot untouched.
    activate(m, slot) catch return 0;
    if (m.mdl.dec_state == null) if (m.idle) |k| {
        m.mdl.dec_state = k.st;
        m.mdl.state_gen = k.gen;
        m.idle = null;
    };
    return dsv41.resumePrompt(m.mdl, prompt.slice(), rows) catch blk: {
        // The forward at position 0 establishes the cold state even if this reset fails.
        dsv41.reset(m.mdl, gpa) catch {};
        break :blk 0;
    };
}

/// `CacheHooks.state_bytes`: the bytes the request's state holds on the GPU, for the engine's cache budget.
pub fn stateBytes(mp: *anyopaque) callconv(.c) u64 {
    const m: *Model = @ptrCast(@alignCast(mp));
    var bytes: u64 = if (m.mdl.dec_state) |*st| dsv41.stateBytes(st) else 0;
    var it = m.parked.valueIterator();
    while (it.next()) |k| bytes += dsv41.stateBytes(&k.st);
    if (m.idle) |*k| bytes += dsv41.stateBytes(&k.st);
    return bytes;
}

const aux_hooks = abi.AuxHooks{
    .enabled = &auxEnabled,
    .describe = &auxDescribe,
    .prefix_key = &auxPrefixKey,
    .begin_prompt = &auxBeginPrompt,
    .import_plan = &auxImportPlan,
    .export_arrays = &auxExportArrays,
    .import_arrays = &auxImportArrays,
    .reserve = &auxReserve,
    .commit = &auxCommit,
    .snapshot = &auxSnapshot,
    .restore_at = &auxRestoreAt,
    .slice = &auxSlice,
    .handoff = &auxHandoff,
    .free = &auxFree,
    .bytes = &auxBytes,
};

fn auxEnabled(mp: *anyopaque) callconv(.c) bool {
    const m: *Model = @ptrCast(@alignCast(mp));
    return !m.cfg.dsv41_streamed and m.bank == null and m.mdl.streamed == null and m.mdl.pair == null;
}

fn liveState(m: *Model, slot: u32) ?*dsv41.State {
    if (slot == m.owner) return if (m.mdl.dec_state) |*st| st else null;
    const kept = m.parked.getPtr(slot) orelse return null;
    return &kept.st;
}

fn snapOf(handle: abi.Handle) ?*dsv41.Snap {
    return @ptrCast(@alignCast(handle));
}

fn auxDescribe(_: *anyopaque, handle: abi.Handle, out: *abi.AuxSnapshotInfo) callconv(.c) Verdict {
    const sn = snapOf(handle) orelse return Verdict.refuse(.unsupported, "deepseek_v41: no snapshot to describe");
    if (out.size < @sizeOf(abi.AuxSnapshotInfo)) return Verdict.refuse(.unsupported, "deepseek_v41: snapshot metadata is too short");
    if (sn.n != sn.ids.len) return Verdict.refuse(.failed, "deepseek_v41: snapshot boundary differs from its ids");
    out.* = .{ .n = sn.n, .ids = .of(sn.ids), .prefix_key = sn.vision_key };
    return .ok;
}

fn auxPrefixKey(mp: *anyopaque, ids: abi.Span(u32), rows: abi.Array, out: *u64) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    out.* = dsv41.prefixKey(m.mdl, ids.slice(), if (rows.ctx != null) .{ .ctx = rows.ctx } else null) catch |e| return Verdict.named(.failed, @errorName(e));
    return .ok;
}

fn auxBeginPrompt(mp: *anyopaque, slot: u32, prompt: abi.Span(u32), rows: abi.Array, restored_at: u64, capture: bool) callconv(.c) Verdict {
    if (!auxEnabled(mp)) return Verdict.refuse(.unsupported, "deepseek_v41: auxiliary cache requires resident unpaired weights");
    const m: *Model = @ptrCast(@alignCast(mp));
    if (restored_at != 0) {
        const st = liveState(m, slot) orelse return Verdict.refuse(.failed, "deepseek_v41: restored slot has no state");
        if (st.n != restored_at or restored_at >= prompt.len) return Verdict.refuse(.failed, "deepseek_v41: restored position differs from the prompt boundary");
    }
    activate(m, slot) catch |e| return Verdict.named(.failed, @errorName(e));
    if (restored_at == 0) {
        dsv41.reset(m.mdl, gpa) catch |e| return Verdict.named(.failed, @errorName(e));
    }
    dsv41.armPrompt(m.mdl, prompt.slice(), if (rows.ctx != null) .{ .ctx = rows.ctx } else null, capture) catch |e| return Verdict.named(.failed, @errorName(e));
    return .ok;
}

fn auxReserve(_: *anyopaque, _: u32, _: u64) callconv(.c) Verdict {
    return .ok;
}

fn auxCommit(mp: *anyopaque, slot: u32, len: u64) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    const st = liveState(m, slot) orelse return Verdict.refuse(.failed, "deepseek_v41: no live slot to commit");
    if (len != st.n and (st.snap == null or st.snap.?.n != len)) return Verdict.refuse(.unsupported, "deepseek_v41: commit requires a completed boundary");
    return .ok;
}

fn auxSnapshot(mp: *anyopaque, slot: u32, out: *abi.Handle) callconv(.c) Verdict {
    out.* = null;
    if (!auxEnabled(mp)) return Verdict.refuse(.unsupported, "deepseek_v41: auxiliary cache requires resident unpaired weights");
    const m: *Model = @ptrCast(@alignCast(mp));
    const st = liveState(m, slot) orelse return .ok;
    const sn = st.snap orelse return .ok;
    if (sn.n != sn.ids.len) return Verdict.refuse(.failed, "deepseek_v41: snapshot boundary differs from its ids");
    const owned = gpa.create(dsv41.Snap) catch |e| return Verdict.named(.failed, @errorName(e));
    owned.* = sn;
    st.snap = null;
    out.* = owned;
    return .ok;
}

fn auxRestoreAt(mp: *anyopaque, slot: u32, handle: abi.Handle, pos: u64) callconv(.c) Verdict {
    if (!auxEnabled(mp)) return Verdict.refuse(.unsupported, "deepseek_v41: auxiliary cache requires resident unpaired weights");
    const sn = snapOf(handle) orelse return Verdict.refuse(.unsupported, "deepseek_v41: no snapshot to restore");
    if (pos != sn.n or sn.n != sn.ids.len) return Verdict.refuse(.unsupported, "deepseek_v41: restore requires the snapshot's exact position");
    const m: *Model = @ptrCast(@alignCast(mp));
    var fresh = dsv41.stateFromSnapshot(m.mdl, gpa, sn) catch |e| return Verdict.named(.failed, @errorName(e));
    activate(m, slot) catch |e| {
        fresh.deinit();
        return Verdict.named(.failed, @errorName(e));
    };
    if (m.mdl.dec_state) |*old| old.deinit();
    m.mdl.dec_state = fresh;
    m.mdl.gens += 1;
    m.mdl.state_gen = m.mdl.gens;
    return .ok;
}

fn auxSlice(_: *anyopaque, handle: abi.Handle, len: u64, out: *abi.Handle) callconv(.c) Verdict {
    out.* = null;
    const sn = snapOf(handle) orelse return Verdict.refuse(.unsupported, "deepseek_v41: no snapshot to slice");
    if (len != sn.n) return Verdict.refuse(.unsupported, "deepseek_v41: a snapshot cannot be shortened through its rings");
    const owned = gpa.create(dsv41.Snap) catch |e| return Verdict.named(.failed, @errorName(e));
    owned.* = sn.clone(gpa) catch |e| {
        gpa.destroy(owned);
        return Verdict.named(.failed, @errorName(e));
    };
    out.* = owned;
    return .ok;
}

fn auxHandoff(mp: *anyopaque, from: u32, to: u32) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (to == 0 or to == m.owner or m.parked.contains(to)) return Verdict.refuse(.busy, "deepseek_v41: handoff destination is in use");
    if (liveState(m, from) == null) return Verdict.refuse(.failed, "deepseek_v41: handoff source has no state");
    if (from == m.owner) {
        m.owner = to;
        return .ok;
    }
    m.parked.ensureUnusedCapacity(gpa, 1) catch |e| return Verdict.named(.failed, @errorName(e));
    const kept = m.parked.fetchRemove(from).?.value;
    m.parked.putAssumeCapacity(to, kept);
    return .ok;
}

fn auxFree(_: *anyopaque, handle: abi.Handle) callconv(.c) void {
    const sn = snapOf(handle) orelse return;
    sn.deinit(gpa);
    gpa.destroy(sn);
}

fn auxBytes(_: *anyopaque, handle: abi.Handle) callconv(.c) u64 {
    const sn = snapOf(handle) orelse return 0;
    return sn.bytes();
}

fn auxImportPlan(mp: *anyopaque, header: abi.Str, out: *u64) callconv(.c) Verdict {
    out.* = 0;
    if (!auxEnabled(mp)) return Verdict.refuse(.unsupported, "deepseek_v41: auxiliary cache requires resident unpaired weights");
    const m: *Model = @ptrCast(@alignCast(mp));
    out.* = dsv41.snapshotImportPlan(m.mdl, header.slice()) catch |e| return Verdict.named(.failed, @errorName(e));
    return .ok;
}

fn auxExportArrays(mp: *anyopaque, handle: abi.Handle, out: *abi.AuxExport) callconv(.c) Verdict {
    if (out.size < @sizeOf(abi.AuxExport)) return Verdict.refuse(.unsupported, "deepseek_v41: export storage is too short");
    out.tensor_len = 0;
    out.names_len = 0;
    out.header_len = 0;
    if (!auxEnabled(mp)) return Verdict.refuse(.unsupported, "deepseek_v41: auxiliary cache requires resident unpaired weights");
    const sn = snapOf(handle) orelse return Verdict.refuse(.failed, "deepseek_v41: no snapshot to export");
    const m: *Model = @ptrCast(@alignCast(mp));
    dsv41.exportSnapshot(m.mdl, sn, out) catch |e| return Verdict.named(.failed, @errorName(e));
    return .ok;
}

fn auxImportArrays(mp: *anyopaque, ids: abi.Span(u32), header: abi.Str, arrays: abi.Span(abi.AuxTensor), out: *abi.Handle) callconv(.c) Verdict {
    out.* = null;
    if (!auxEnabled(mp)) return Verdict.refuse(.unsupported, "deepseek_v41: auxiliary cache requires resident unpaired weights");
    const m: *Model = @ptrCast(@alignCast(mp));
    out.* = dsv41.importSnapshot(m.mdl, gpa, ids.slice(), header.slice(), arrays.slice()) catch |e| return Verdict.named(.failed, @errorName(e));
    return .ok;
}

/// The load reads its Engram tables from disk per lookup and drops the tensors it never loads
/// (`dsv41.residentDiskBytes`); the `mtp` setting decides the DSpark stages.
fn residentBytes(p: *const abi.Parsed, model_dir: abi.Str, settings_json: abi.Str, shards: u64, out: *u64) callconv(.c) Verdict {
    out.* = dsv41.residentDiskBytes(gpa, model_dir.slice(), config.of(p), mtpSetting(settings_json.slice()), shards) catch |e| return Verdict.named(.failed, @errorName(e));
    return .ok;
}

/// The tiny pack the conformance suite runs: the oracle's pipe layout without Engram or DSpark.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    return .{ .files = try fixture.v41Files(a, 7) };
}

// ── tests ──

/// `residentBytes` over the Engram fixture (`engram.writeAffineFixture`) in `dir`, from `shards` bytes.
fn engramResident(dir: []const u8, settings_json: []const u8, shards: u64) !u64 {
    var c: config.Config = .{ .model_type = "deepseek_v41", .dsv41_n_engram_layers = 1, .dsv41_engram_head_dim = 64 };
    c.dsv41_engram_layers[0] = 1;
    c.dsv41_engram_rows[0] = 4;
    var out: u64 = 0;
    const v = residentBytes(&.{ .private = &c }, .of(dir), .of(settings_json), shards, &out);
    if (!v.isOk()) return error.Refused;
    return out;
}

test "deepseek_v41 resident bill: the Engram tables it preads are not billed" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    try @import("engram.zig").writeAffineFixture(io, td.dir, "");
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    try std.testing.expectEqual(@as(u64, 10_000 - 288), try engramResident(dir, "", 10_000));
}

test "deepseek_v41 resident bill: neither the vision tower nor DSpark stages that will not load" {
    @import("../../env.zig").unset(.MLX_SERVE_DSV4_DSPARK); // test-order hygiene: `--dspark` (and tests) force the stages on
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    try @import("engram.zig").writeAffineFixture(io, td.dir, "");
    const hdr = "{\"layers.0.attn_norm.weight\":{\"dtype\":\"BF16\",\"shape\":[8],\"data_offsets\":[0,16]}," ++
        "\"vision.patch.weight\":{\"dtype\":\"BF16\",\"shape\":[16],\"data_offsets\":[16,48]}," ++
        "\"language_model.mtp.0.norm.weight\":{\"dtype\":\"BF16\",\"shape\":[32],\"data_offsets\":[48,112]}}";
    var shard: [8 + hdr.len + 112]u8 = @splat(0);
    std.mem.writeInt(u64, shard[0..8], hdr.len, .little);
    @memcpy(shard[8..][0..hdr.len], hdr);
    try td.dir.writeFile(io, .{ .sub_path = "model-00002.safetensors", .data = &shard });
    const e = "layers.1.engram.embed.";
    try td.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{" ++
        "\"" ++ e ++ "weight\":\"model-00001.safetensors\",\"" ++ e ++ "scales\":\"model-00001.safetensors\",\"" ++ e ++ "biases\":\"model-00001.safetensors\"," ++
        "\"layers.0.attn_norm.weight\":\"model-00002.safetensors\",\"vision.patch.weight\":\"model-00002.safetensors\"," ++
        "\"language_model.mtp.0.norm.weight\":\"model-00002.safetensors\"}}" });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    // The stages load by default; the tower never does.
    try std.testing.expectEqual(@as(u64, 10_000 - 288 - 32), try engramResident(dir, "", 10_000));
    try std.testing.expectEqual(@as(u64, 10_000 - 288 - 32 - 64), try engramResident(dir, "{\"mtp\":false}", 10_000));
}

test {
    _ = config;
    _ = bill_mod;
    _ = media;
    _ = @import("contract.zig");
    _ = @import("decode/decode.zig");
}
