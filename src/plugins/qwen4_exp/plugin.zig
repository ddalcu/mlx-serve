//! Qwen3.8-Flash-Next (`qwen4_exp`): the family's own forward behind the model table. The plugin's trunk module
//! (`trunk.zig`: hyper-connection streams, the n-gram PLE, QSA sparse attention, GatedDeltaNet + MoE layers, the MTP
//! head) runs `forwardQwen4With` and the batched GDN tick on each slot's caches.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const model = @import("../../sdk/config/model.zig");
const engine_cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const CaptureLayers = @import("../../engines/mlx/forward_ctx.zig").CaptureLayers;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const nextSsmMemberGen = @import("../../engines/mlx/ssm/group.zig").nextSsmMemberGen;
const ssmFreeQsaState = @import("qsa/history.zig").ssmFreeQsaState;
const QsaArmCounts = @import("qsa/arms.zig").QsaArmCounts;
const ple = @import("ple.zig");
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const MAX_BATCH_ROWS = @import("../../engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const bill_mod = @import("bill.zig");
const fixture = @import("fixture.zig");
const toModelConfig = @import("model_config.zig").toModelConfig;
const trunk_load = @import("trunk_load.zig");
const distributed = @import("../../engines/mlx/cluster/distributed.zig");
const lmHeadMxfp8On = @import("../../sdk/quant/mxfp8.zig").lmHeadMxfp8On;
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
    .forward_settle = &forwardSettle,
    .release_slot = &releaseSlot,
    .meters = &meters,
    .cache = &.{ .layout = &layout },
    .weights = &.{ .resident_bytes = &residentBytes },
};

/// P8 debt (`engines/mlx/plugin/intree.zig`): the engine's context of the serial forward in flight when the MLX
/// engine serves this plugin in-tree; the slot's M-RoPE table rides it.
pub fn engineCtx() *?*const ForwardCtx {
    return EngineCtx.get();
}
pub const EngineCtx = @import("../../sdk/util/shared.zig").SharedThreadLocal(?*const ForwardCtx, "qwen4_engine_ctx", null);

const Verdict = abi.Verdict;
const Array = mlx.mlx_array;
const Trunk = @import("trunk.zig").Trunk;
const gpa = std.heap.c_allocator;

/// GatedDeltaNet layers keep a recurrent state; every `full_attention_interval`-th layer caches KV at the shell's
/// geometry (QSA reads a sparse subset of it past the indexer budget).
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    const c = config.of(p);
    const first: u32 = if (span) |s| s.first else 0;
    const n = @min(if (span) |s| s.count else c.num_hidden_layers, cap);
    for (out[0..n], first..) |*l, li| l.* = if (c.isFullLayer(@intCast(li)))
        .{ .kind = .full, .kv_heads = c.num_key_value_heads, .k_dim = c.head_dim, .v_dim = c.head_dim }
    else
        .{ .kind = .linear };
    return n;
}

/// A slot's state when the host's caches are not the engine's: the KV cache, the per-layer recurrent entries (GDN
/// pair, PLE window and history, QSA keys) and the next position. Position 0 starts it over.
const OwnSlot = struct {
    id: u32,
    cache: KVCache,
    ssm: []SSMCacheEntry,
    pos: u64 = 0,

    fn init(id: u32, layers: u32) !OwnSlot {
        var cache = try KVCache.init(gpa, layers);
        errdefer cache.deinit();
        const ssm = try gpa.alloc(SSMCacheEntry, layers);
        const gen = nextSsmMemberGen();
        for (ssm) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false, .member_gen = gen };
        return .{ .id = id, .cache = cache, .ssm = ssm };
    }

    fn deinit(self: *OwnSlot, xfm: *Trunk) void {
        xfm.ssmGroupDrop(self.ssm);
        for (self.ssm) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        }
        gpa.free(self.ssm);
        self.cache.deinit();
    }
};

/// What a slot id carries between forwards: the n-gram PLE work a `defer_host` forward left owed, the QSA arm
/// tally for its meter line, and the engine entries its last forward ran on (a batched group may hold them).
const Rec = struct {
    pending: ?ple.PlePending = null,
    arms: QsaArmCounts = .{},
    ssm: ?[]SSMCacheEntry = null,
};

pub const Model = struct {
    cfg: model.ModelConfig,
    weights: model.Weights,
    trunk: Trunk,
    /// The cache a settle-only context names (it reads no cache).
    scratch: KVCache,
    caps: abi.Caps,
    own: std.ArrayList(OwnSlot) = .empty,
    recs: std.AutoHashMapUnmanaged(u32, Rec) = .empty,
    meter_buf: [192]u8 = undefined,

    fn rec(m: *Model, id: u32) !*Rec {
        const g = try m.recs.getOrPut(gpa, id);
        if (!g.found_existing) g.value_ptr.* = .{};
        return g.value_ptr;
    }

    /// A context for host work on `r`'s owed record alone (no forward reads its cache).
    fn settleCtx(m: *Model, r: *Rec, offset: *usize) ForwardCtx {
        return .{ .cache = &m.scratch, .moe_seq_offset = offset, .ssm_entries = r.ssm, .capture_hidden = null, .vision_embeddings = null, .ple_pending = r.pending };
    }

    fn ownSlot(m: *Model, id: u32, pos: u64) !*OwnSlot {
        for (m.own.items) |*o| if (o.id == id) {
            if (pos == 0) {
                o.deinit(&m.trunk);
                o.* = try OwnSlot.init(id, m.cfg.num_hidden_layers);
            } else if (o.pos != pos) return error.PositionDoesNotFollow;
            return o;
        };
        if (pos != 0) return error.PositionDoesNotFollow;
        try m.own.append(gpa, try OwnSlot.init(id, m.cfg.num_hidden_layers));
        return &m.own.items[m.own.items.len - 1];
    }
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    return loadModel(ctx) catch |e| {
        out.* = switch (e) {
            error.OutOfMemory => Verdict.refuse(.failed, "qwen4_exp: out of memory"),
            else => Verdict.named(.failed, @errorName(e)),
        };
        return null;
    };
}

/// `lm_head_mxfp8` as the load settings state it; null = the default (on for a quantized pack on an integrated CUDA
/// GPU).
fn headMxfp8Setting(settings_json: []const u8) ?bool {
    if (settings_json.len == 0) return null;
    const S = struct { lm_head_mxfp8: ?bool = null };
    const parsed = std.json.parseFromSlice(S, gpa, settings_json, .{ .ignore_unknown_fields = true }) catch return null;
    defer parsed.deinit();
    return parsed.value.lm_head_mxfp8;
}

/// The trunk module keeps the loader's whole map (folds and n-gram shards included): the engine's own when the
/// driver loaded it with this config (`Resident.take`), else read here. The vision tower stays the engine's.
fn loadModel(ctx: *const abi.LoadCtx) !*Model {
    const io = std.Io.Threaded.global_single_threaded.io();
    const m = try gpa.create(Model);
    errdefer gpa.destroy(m);
    m.* = .{ .cfg = undefined, .weights = undefined, .trunk = undefined, .scratch = try KVCache.init(gpa, 0), .caps = ctx.parsed.caps };
    errdefer m.scratch.deinit();
    m.cfg = try toModelConfig(gpa, config.of(ctx.parsed), ctx.model_dir.slice());
    errdefer m.cfg.deinit(gpa);
    const taken = if (ctx.weights) |src| Resident.take(src) else null;
    m.weights = taken orelse try model.loadModelWeights(io, gpa, ctx.model_dir.slice(), &m.cfg, false);
    errdefer m.weights.deinit();
    // The driver's load planned a Thunderbolt pair on the engine's config, and its map holds this Mac's part: the
    // same plan here (the plugin's own load applies it itself).
    if (taken != null) if (try distributed.planFor(ctx.model_dir.slice(), &m.cfg)) |part| switch (part) {
        .span => |sp| try m.cfg.applyLayerSpan(sp),
        .shard => |sh| try m.cfg.applyShard(sh),
        .remote => {},
    };
    model.resolveWeightPrefix(&m.cfg, &m.weights);
    m.trunk = try trunk_load.load(io, gpa, &m.cfg, &m.weights, .{
        .head_mxfp8 = lmHeadMxfp8On(headMxfp8Setting(ctx.settings_json.slice()), null, mlx.cudaIntegrated(), true, m.cfg.quant_bits),
    });
    m.trunk.prepare();
    return m;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Model = @ptrCast(@alignCast(mp));
    var it = m.recs.iterator();
    while (it.next()) |e| if (e.value_ptr.pending) |p| ple.releasePlePending(p);
    m.recs.deinit(gpa);
    for (m.own.items) |*o| o.deinit(&m.trunk);
    m.own.deinit(gpa);
    m.trunk.deinit();
    m.scratch.deinit();
    m.weights.deinit();
    m.cfg.deinit(gpa);
    gpa.destroy(m);
}

/// One slot's caches for this forward: the engine's own (its `cache_ops.Slot`) or the model's.
const SlotCaches = struct { kv: *KVCache, ssm: []SSMCacheEntry, packed_reads: bool = false, own: ?*OwnSlot = null };

fn slotCaches(m: *Model, f: *const abi.Forward, i: usize) !SlotCaches {
    const pos: u64 = if (f.positions.len > i) f.positions.slice()[i] else 0;
    if (f.cache_ops != null and f.cache_ops.?.sdkSlots() and f.caches.len > i) {
        const s: *engine_cache_ops.Slot = @ptrCast(@alignCast(f.caches.slice()[i].?));
        if (s.ssm.len != m.cfg.num_hidden_layers) return error.SlotWithoutSsmEntries;
        return .{ .kv = s.kv, .ssm = s.ssm, .packed_reads = s.packed_reads };
    }
    const id: u32 = if (f.slots.len > i) f.slots.slice()[i] else @intCast(i);
    const o = try m.ownSlot(id, pos);
    return .{ .kv = &o.cache, .ssm = o.ssm, .own = o };
}

fn tokenArray(f: *const abi.Forward, slots: usize) !Array {
    if (f.ids.ctx != null) {
        var a = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&a, .{ .ctx = f.ids.ctx }));
        return a;
    }
    var buf: [4096]i32 = undefined;
    const t = f.tokens.slice();
    if (t.len > buf.len or t.len != slots * f.rows) return error.TokenCount;
    for (t, 0..) |id, k| buf[k] = @intCast(id);
    return mlx.mlx_array_new_data(&buf, &[_]c_int{ @intCast(slots), @intCast(f.rows) }, 2, .int32);
}

pub fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (!abi.declaresMode(m.caps, f.mode)) return Verdict.refuse(.unsupported, "qwen4_exp: forward mode not declared");
    // Pooled QSA keys rope at block starts off the slot's whole M-RoPE table, which only the engine's context carries.
    if (f.mrope.ctx != null and engineCtx().* == null) return Verdict.refuse(.unsupported, "qwen4_exp: M-RoPE needs the slot's position table");
    if (f.rows == 0) return Verdict.refuse(.failed, "qwen4_exp: a forward of no rows");
    const slots = @max(f.positions.len, 1);
    if (f.mode != .batched_rows and slots != 1) return Verdict.refuse(.unsupported, "qwen4_exp: one slot's rows at a time outside batched rows");
    if (f.ids.ctx == null) for (f.tokens.slice()) |t| if (t >= m.cfg.vocab_size) return Verdict.refuse(.failed, "qwen4_exp: token id past the vocabulary");
    const r = if (f.mode == .batched_rows) runBatched(m, f, out) else runOne(m, f, out);
    r catch |e| return switch (e) {
        error.PositionDoesNotFollow => Verdict.refuse(.failed, "qwen4_exp: the position does not follow the slot's state"),
        error.SlotWithoutSsmEntries => Verdict.refuse(.failed, "qwen4_exp: the engine slot carries no recurrent entries"),
        else => Verdict.refuse(.failed, "qwen4_exp: the forward failed"),
    };
    return .ok;
}

/// `logits` `[B, S, V]` as the table returns it: every row, or each slot's last one.
fn shapeLogits(m: *Model, logits: Array, all: bool) !Array {
    const sh = mlx.getShape(logits);
    const s = m.trunk.s;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (all) {
        try mlx.check(mlx.mlx_reshape(&out, logits, &[_]c_int{ sh[0] * sh[1], sh[2] }, 2, s));
        return out;
    }
    var last = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(last);
    try mlx.check(mlx.mlx_slice(&last, logits, &[_]c_int{ 0, sh[1] - 1, 0 }, 3, &[_]c_int{ sh[0], sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    try mlx.check(mlx.mlx_reshape(&out, last, &[_]c_int{ sh[0], sh[2] }, 2, s));
    return out;
}

fn runOne(m: *Model, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const sc = try slotCaches(m, f, 0);
    const ids = try tokenArray(f, 1);
    defer _ = mlx.mlx_array_free(ids);
    var offset: usize = @intCast(if (f.positions.len > 0) f.positions.slice()[0] else 0);
    var hidden = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden);
    var captured: [64]Array = undefined;
    const cap_ids = f.capture_layers.slice();
    if (cap_ids.len > captured.len) return error.TooManyCaptures;
    for (captured[0..cap_ids.len]) |*c| c.* = mlx.mlx_array_new();
    defer for (captured[0..cap_ids.len]) |c| {
        _ = mlx.mlx_array_free(c);
    };
    var cl: CaptureLayers = .{ .ids = cap_ids, .out = captured[0..cap_ids.len] };
    const r = try m.rec(if (f.slots.len > 0) f.slots.slice()[0] else 0);
    r.ssm = sc.ssm;
    // On the M-RoPE table the slot's position is the engine's own (`Forward.positions` carries its delta past it).
    const ec = engineCtx().*;
    const on_table = if (ec) |e| e.mrope_pos != null else false;
    if (on_table) offset = ec.?.moe_seq_offset.*;
    // A verify leaves each recurrent row's state for the partial-accept rollback; the stream is the head's input.
    var ctx: ForwardCtx = .{
        .cache = sc.kv,
        .moe_seq_offset = &offset,
        .ssm_entries = sc.ssm,
        .capture_hidden = null,
        .capture_hidden_all = if (f.want_hidden and f.logits != .none) &hidden else null,
        .capture_layers = if (cap_ids.len > 0) &cl else null,
        .capture_ssm_seq = f.mode == .verify,
        .vision_embeddings = if (f.media.ctx != null) Array{ .ctx = f.media.ctx } else null,
        .vision_splice_offset = f.media_offset,
        .mrope_pos = if (on_table) ec.?.mrope_pos else null,
        .mrope_total = if (on_table) ec.?.mrope_total else 0,
        .mrope_delta = if (on_table) ec.?.mrope_delta else 0,
        .spec_verify = f.mode == .verify,
        .kv_attn_fused = sc.packed_reads,
        .skip_lm_head = f.logits == .none,
        .ple_defer = f.defer_host,
        .ple_pending = r.pending,
    };
    r.pending = null;
    defer {
        r.pending = ctx.ple_pending;
        r.arms.add(ctx.qsa_arms);
    }
    try m.trunk.ssmGroupRelease(&ctx);
    const res = m.trunk.forwardQwen4With(&ctx, ids) catch |e| {
        ple.discardDeferredPle(&m.trunk, &ctx);
        return e;
    };
    defer _ = mlx.mlx_array_free(res);
    if (sc.own) |o| o.pos = offset;
    if (ctx.ple_pending) |p| out.owed = if (p.gpu) .history else .input;
    if (f.logits == .none) {
        // No head: the result is the mixer output, the vector the lm_head reads.
        var h = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&h, res);
        out.hidden = .{ .ctx = h.ctx };
    } else out.logits = .{ .ctx = (try shapeLogits(m, res, f.logits == .all_rows)).ctx };
    if (f.want_hidden and f.logits != .none) {
        var h = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&h, hidden);
        out.hidden = .{ .ctx = h.ctx };
    }
    if (out.captured) |dst| for (captured[0..cap_ids.len], 0..) |c, k| {
        var h = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&h, c);
        dst[k] = .{ .ctx = h.ctx };
    };
}

/// One row per slot at its own position: the batched GDN tick when every slot's recurrent state is live (the
/// engine's gate), else each slot's serial step.
fn runBatched(m: *Model, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const n = f.positions.len;
    if (n == 0 or n > MAX_BATCH_ROWS) return error.BatchShape;
    if (f.rows > 1) return runBatchedVerify(m, f, out);
    if (f.ids.ctx != null or f.tokens.len != n) return error.BatchShape;
    var caches: [MAX_BATCH_ROWS]SlotCaches = undefined;
    var offsets: [MAX_BATCH_ROWS]usize = undefined;
    var ctxs: [MAX_BATCH_ROWS]ForwardCtx = undefined;
    var ptrs: [MAX_BATCH_ROWS]*ForwardCtx = undefined;
    for (0..n) |i| {
        caches[i] = try slotCaches(m, f, i);
        offsets[i] = f.positions.slice()[i];
        ctxs[i] = .{ .cache = caches[i].kv, .moe_seq_offset = &offsets[i], .ssm_entries = caches[i].ssm, .capture_hidden = null, .vision_embeddings = null, .kv_attn_fused = caches[i].packed_reads };
        ptrs[i] = &ctxs[i];
    }
    var rows: [MAX_BATCH_ROWS]Array = undefined;
    var made: usize = 0;
    defer for (rows[0..made]) |r| {
        _ = mlx.mlx_array_free(r);
    };
    var hidden: [MAX_BATCH_ROWS]Array = undefined;
    var made_hidden: usize = 0;
    defer for (hidden[0..made_hidden]) |h| {
        _ = mlx.mlx_array_free(h);
    };
    // The legacy tick's PLE gather defers inside it and settles once, before it returns: nothing is left owed.
    if (n > 1 and m.trunk.batchedGdnReady(ptrs[0..n])) {
        var hrows: ?[]Array = null;
        const got = try m.trunk.forwardMoeBatchedDecode(f.tokens.slice(), ptrs[0..n], f.positions.slice(), if (f.want_hidden) &hrows else null);
        defer gpa.free(got);
        @memcpy(rows[0..n], got);
        made = n;
        if (hrows) |hr| {
            defer gpa.free(hr);
            @memcpy(hidden[0..n], hr);
            made_hidden = n;
        }
    } else for (0..n) |i| {
        var id: i32 = @intCast(f.tokens.slice()[i]);
        const ids = mlx.mlx_array_new_data(&id, &[_]c_int{ 1, 1 }, 2, .int32);
        defer _ = mlx.mlx_array_free(ids);
        try m.trunk.ssmGroupRelease(ptrs[i]);
        if (f.want_hidden) {
            hidden[i] = mlx.mlx_array_new();
            made_hidden += 1;
            ptrs[i].capture_hidden = &hidden[i];
        }
        rows[i] = try m.trunk.forwardQwen4With(ptrs[i], ids);
        made += 1;
    }
    for (0..n) |i| {
        if (caches[i].own) |o| o.pos = f.positions.slice()[i] + 1;
        if (f.slots.len > i) {
            const r = try m.rec(f.slots.slice()[i]);
            r.ssm = caches[i].ssm;
            r.arms.add(ctxs[i].qsa_arms);
        }
    }
    out.logits = .{ .ctx = (try shapeLogits(m, try concatRows(m, rows[0..n]), false)).ctx };
    if (made_hidden == n and f.want_hidden) out.hidden = .{ .ctx = (try concatRows(m, hidden[0..n])).ctx };
}

/// Every slot's `[t1, drafts…]` row of `f.ids` `[N, S]` at its own position: the legacy batched verify, each
/// recurrent row's state captured for the slot's own rollback; logits for every row, the stream at every row.
fn runBatchedVerify(m: *Model, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const n = f.positions.len;
    if (f.ids.ctx == null) return error.BatchShape;
    var caches: [MAX_BATCH_ROWS]SlotCaches = undefined;
    var offsets: [MAX_BATCH_ROWS]usize = undefined;
    var ctxs: [MAX_BATCH_ROWS]ForwardCtx = undefined;
    var ptrs: [MAX_BATCH_ROWS]*ForwardCtx = undefined;
    for (0..n) |i| {
        caches[i] = try slotCaches(m, f, i);
        offsets[i] = f.positions.slice()[i];
        ctxs[i] = .{ .cache = caches[i].kv, .moe_seq_offset = &offsets[i], .ssm_entries = caches[i].ssm, .capture_hidden = null, .vision_embeddings = null, .kv_attn_fused = caches[i].packed_reads };
        ptrs[i] = &ctxs[i];
    }
    var last = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(last);
    var all = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(all);
    const logits = try m.trunk.forwardMoeBatchedVerify(.{ .ctx = f.ids.ctx }, ptrs[0..n], f.positions.slice(), &last, &all);
    defer _ = mlx.mlx_array_free(logits);
    for (0..n) |i| {
        if (caches[i].own) |o| o.pos = f.positions.slice()[i] + f.rows;
        if (f.slots.len > i) {
            const r = try m.rec(f.slots.slice()[i]);
            r.ssm = caches[i].ssm;
            r.arms.add(ctxs[i].qsa_arms);
        }
    }
    out.logits = .{ .ctx = (try shapeLogits(m, logits, true)).ctx };
    out.hidden = .{ .ctx = all.ctx };
}

/// `xs` stacked on the batch axis (a new array).
fn concatRows(m: *Model, xs: []const Array) !Array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (xs) |x| _ = mlx.mlx_vector_array_append_value(vec, x);
    var cat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cat);
    try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, m.trunk.s));
    return cat;
}

/// Settles what `slots` were left owed: the n-gram rows of a deferred leaf (`input`) or the history of the GPU
/// arm (`history`), from the device or the engine's `tokens`; `.discard` drops it and the spec slot it claimed.
fn forwardSettle(mp: *anyopaque, slots: abi.Span(u32), how: abi.Settle, tokens: *const abi.Span(u32)) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    settleSlots(m, slots.slice(), how, tokens.slice()) catch |e| return Verdict.named(.failed, @errorName(e));
    return .ok;
}

fn settleSlots(m: *Model, ids: []const u32, how: abi.Settle, tokens: []const u32) !void {
    if (ids.len > MAX_BATCH_ROWS) return error.PleGroupTooWide;
    var offs: [MAX_BATCH_ROWS]usize = @splat(0);
    var ctxs: [MAX_BATCH_ROWS]ForwardCtx = undefined;
    var ptrs: [MAX_BATCH_ROWS]*ForwardCtx = undefined;
    var recs: [MAX_BATCH_ROWS]*Rec = undefined;
    var n: usize = 0;
    for (ids) |id| {
        const r = m.recs.getPtr(id) orelse continue;
        if (r.pending == null) continue;
        recs[n] = r;
        ctxs[n] = m.settleCtx(r, &offs[n]);
        ptrs[n] = &ctxs[n];
        r.pending = null;
        n += 1;
    }
    if (n == 0) return;
    // A failed settle leaves nothing owed behind it: the engine discards the forward.
    defer for (ctxs[0..n]) |*c| if (c.ple_pending) |p| ple.releasePlePending(p);
    switch (how) {
        .discard => for (ctxs[0..n]) |*c| ple.discardDeferredPle(&m.trunk, c),
        .flush => if (tokens.len > 0) {
            if (n != 1) return error.PleSettleWidth;
            try ple.settleDeferredPle(&m.trunk, &ctxs[0], tokens);
        } else if (n == 1) try ple.flushDeferredPle(&m.trunk, &ctxs[0]) else try ple.flushDeferredPleGroup(&m.trunk, ptrs[0..n]),
    }
}

/// The slot id is gone: what it was owed dies with it, a batched group lets its entries go, and its own state
/// (another host's) is freed.
fn releaseSlot(mp: *anyopaque, slot: u32) callconv(.c) void {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (m.recs.fetchRemove(slot)) |kv| {
        var r = kv.value;
        var off: usize = 0;
        var c = m.settleCtx(&r, &off);
        ple.discardDeferredPle(&m.trunk, &c);
        if (r.ssm) |entries| m.trunk.ssmGroupDrop(entries);
    }
    m.trunk.markQsaPooledRopeStale();
    for (m.own.items, 0..) |*o, i| if (o.id == slot) {
        o.deinit(&m.trunk);
        _ = m.own.swapRemove(i);
        break;
    };
}

/// The slot's `[qsa-arms]` line (the legacy request summary's), its tally reset.
fn meters(mp: *anyopaque, slot: u32, out: *abi.Str) callconv(.c) void {
    const m: *Model = @ptrCast(@alignCast(mp));
    out.* = .{};
    const r = m.recs.getPtr(slot) orelse return;
    const c = r.arms.take() orelse return;
    const arm = c.majority() orelse return;
    const line = std.fmt.bufPrint(&m.meter_buf, "[qsa-arms] qsa={s} calls={d} mask={d} decode={d} verify={d} prefill={d}\n", .{ @tagName(arm), c.total(), c.mask, c.decode, c.verify, c.prefill }) catch return;
    out.* = abi.Str.of(line);
}

/// An n-gram table embedded in the shards is read from disk per lookup: never resident.
fn residentBytes(p: *const abi.Parsed, model_dir: abi.Str, _: abi.Str, shards: u64, out: *u64) callconv(.c) Verdict {
    out.* = shards;
    const c = config.of(p);
    if (!c.embedded_ngram) return .ok;
    var mc = toModelConfig(gpa, c, null) catch |e| return Verdict.named(.failed, @errorName(e));
    defer mc.deinit(gpa);
    const spec = model.qwen4EmbeddedSpec(&mc) catch |e| return Verdict.named(.failed, @errorName(e));
    const info = (@import("qwen4_exp.zig").inspectEmbedded(model_dir.slice(), spec) catch |e| return Verdict.named(.failed, @errorName(e))) orelse return .ok;
    if (info.payload_bytes > shards) return Verdict.named(.failed, "InvalidEmbeddedNgramTable");
    out.* = shards - info.payload_bytes;
    return .ok;
}

/// The tiny pack the conformance suite runs: four layers (GDN and QSA attention), the PLE, bf16 weights.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    return .{ .files = try fixture.files(a, 7, .bf16) };
}

// ── tests ──

test "qwen4_exp resident bill: an embedded n-gram table's payload is not resident" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    try @import("ple_table.zig").writeFixture(&td, .valid);
    var path: [std.fs.max_path_bytes]u8 = undefined;
    const dir = path[0..try td.dir.realPath(io, &path)];
    var c: config.Config = .{ .vocab_size = 1, .ngram_size = 3, .heads_per_ngram = 1, .ngram_vocab_base = 2, .ngram_vocab_divisor = 6, .ple_embed_dim = 64, .split_ngram_parts = 3, .ple_layer_idx = 1, .embedded_ngram = true };
    var out: u64 = 0;
    try std.testing.expect(residentBytes(&.{ .private = &c }, .of(dir), .{}, 1000, &out).isOk());
    try std.testing.expectEqual(@as(u64, 1000 - 120), out);
    try std.testing.expect(!residentBytes(&.{ .private = &c }, .of(dir), .{}, 100, &out).isOk());
    c.embedded_ngram = false;
    try std.testing.expect(residentBytes(&.{ .private = &c }, .of(dir), .{}, 1000, &out).isOk());
    try std.testing.expectEqual(@as(u64, 1000), out);
}

test {
    _ = config;
    _ = bill_mod;
    _ = fixture;
    _ = @import("model_config.zig");
    _ = @import("proof_test.zig");
    _ = @import("contract.zig");
    _ = @import("served_test.zig");
    _ = @import("oracle_test.zig");
    _ = @import("cost_profile.zig");
    _ = @import("trunk.zig");
}
