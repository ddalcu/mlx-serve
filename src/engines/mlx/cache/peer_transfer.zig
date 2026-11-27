//! Inference-thread RAM prefix-cache export (`exportWireModule`), peer import (`importWireModule`), and side-queue RPC.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const plugin_abi = @import("../../../abi/plugin.zig");
const plugin_driver = @import("../plugin/driver.zig");
const kv_quant = @import("../../../sdk/blocks/attention/kv_quant.zig");
const decode_attn_quant = @import("../../../sdk/blocks/attention/decode_attn_quant.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const log = @import("../../../sdk/util/log.zig");
const transformer_mod = @import("../transformer.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const peer_wire = @import("peer_wire.zig");
const prefill_mod = @import("../prefill.zig");
const pack_identity_mod = @import("pack_identity.zig");
const env_vars = @import("../../../env.zig");
const forceGpuFamilyFallback = @import("../../../mlx/device.zig").forceGpuFamilyFallback;
const publishHotCacheResidency = @import("budget.zig").publishHotCacheResidency;
const stagedHostBytes = @import("budget.zig").stagedHostBytes;
const prefillHeadroomNow = @import("../admission/prefill_chunk.zig").prefillHeadroomNow;
const module_record = @import("ssd/module_record.zig");
const nowMsMonotonic = io_util.nowMsMonotonic;
const requests = @import("../../../host/gpu_thread/requests.zig");
const LoadedModel = @import("../../../host/registry/registry.zig").LoadedModel;
const Scheduler = @import("../scheduler.zig").Scheduler;
const http_server = @import("../../../host/http/server.zig");

const Transformer = transformer_mod.Transformer;
const ModelConfig = model_mod.ModelConfig;

pub fn modelLayoutKey(cfg: *const ModelConfig, xfm: *const Transformer) u64 {
    var h = std.hash.Wyhash.init(0x6c61_796f_7574_3630);
    if (xfm.plugin) |b| {
        h.update(b.table.name.slice());
        if (b.table.cache.layout) |layout_fn| if (cfg.plugin) |pb| {
            var layers: [128]plugin_abi.LayerCache = undefined;
            const n = layout_fn(&pb.parsed, null, &layers, layers.len);
            for (layers[0..n]) |lc| {
                std.hash.autoHash(&h, .{ @backingInt(lc.kind), lc.nope, lc.kv_source, lc.window, lc.kv_heads, lc.k_dim, lc.v_dim });
            }
        };
    } else {
        h.update(cfg.model_type);
    }
    if (cfg.cacheLayoutNamespace()) |ns| {
        h.update("\x00");
        h.update(ns);
    }
    std.hash.autoHash(&h, .{
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.moe_intermediate_size,
        cfg.num_experts,
        cfg.vocab_size,
        cfg.sliding_window,
    });
    if (cfg.dist_shard) |sh| {
        std.hash.autoHash(&h, .{ @as(u8, 1), sh.rank, sh.size });
    } else {
        std.hash.autoHash(&h, @as(u8, 0));
    }
    return h.final();
}

fn optBoolTag(v: ?bool) u8 {
    return if (v) |b| (if (b) 2 else 1) else 0;
}

pub fn modelSettingsKey(lm: *const LoadedModel, cfg: *const ModelConfig, xfm: *const Transformer) u64 {
    var h = std.hash.Wyhash.init(0x7365_7474_696e_6773);
    const head_mxfp8 = cfg.lm_head_mxfp8_override orelse transformer_mod.cli().lm_head_mxfp8_cli;
    const mtp_loaded = lm.mtp != null or xfm.dsparkStages() > 0;
    const dq_active = decode_attn_quant.decodeAttnQuantEnabled() and xfm.dense_attn_proj;
    std.hash.autoHash(&h, .{
        optBoolTag(head_mxfp8),
        mtp_loaded,
        optBoolTag(cfg.mtp_override),
        optBoolTag(cfg.int8_prefill_override),
        dq_active,
        decode_attn_quant.decodeAttnQuantExplicit(),
        Transformer.mtpHeadKvQuantEnabled(),
    });
    if (cfg.stream_experts_override) |se| {
        switch (se) {
            .on => std.hash.autoHash(&h, .{ @as(u8, 1), @as(u64, 0) }),
            .rows => |r| std.hash.autoHash(&h, .{ @as(u8, 2), @as(u64, r) }),
        }
    } else {
        std.hash.autoHash(&h, @as(u8, 0));
    }
    std.hash.autoHash(&h, plugin_driver.auxSettingsKey(xfm));
    std.hash.autoHash(&h, .{ env_vars.int(.LAYER_CAP, usize, 0), env_vars.on(.DSV4_SKIP_MOE), forceGpuFamilyFallback() });
    if (model_mod.getConfigOverrides()) |raw| {
        h.update("\x00");
        h.update(raw);
    }
    return h.final();
}

fn rawArrayBytes(a: mlx.mlx_array) ?[*]const u8 {
    return switch (mlx.mlx_array_dtype(a)) {
        .bool_ => @ptrCast(mlx.mlx_array_data_bool(a) orelse return null),
        .uint8, .int8 => @ptrCast(mlx.mlx_array_data_uint8(a) orelse return null),
        .uint16, .int16 => @ptrCast(mlx.mlx_array_data_uint16(a) orelse return null),
        .uint32, .int32 => @ptrCast(mlx.mlx_array_data_uint32(a) orelse return null),
        .uint64, .int64 => @ptrCast(mlx.mlx_array_data_uint64(a) orelse return null),
        .float32 => @ptrCast(mlx.mlx_array_data_float32(a) orelse return null),
        .float16 => @ptrCast(mlx.mlx_array_data_float16(a) orelse return null),
        .bfloat16 => @ptrCast(mlx.mlx_array_data_bfloat16(a) orelse return null),
        else => null,
    };
}

fn materializeExportTensors(tensors: []plugin_abi.AuxTensor, s: mlx.mlx_stream) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (tensors) |*t| {
        const src: mlx.mlx_array = .{ .ctx = t.array.ctx };
        var cont = mlx.mlx_array_new();
        {
            errdefer _ = mlx.mlx_array_free(cont);
            try mlx.check(mlx.mlx_contiguous(&cont, src, false, s));
        }
        _ = mlx.mlx_array_free(src);
        t.array.ctx = cont.ctx;
        try mlx.check(mlx.mlx_vector_array_append_value(vec, cont));
    }
    try mlx.check(mlx.mlx_eval(vec));
    if (mlx.errorPending()) return error.MlxError;
}

pub fn exportWireModule(
    allocator: std.mem.Allocator,
    io: std.Io,
    sch: *const Scheduler,
    lm: *LoadedModel,
    prompt_ids: []const u32,
    has_tools: bool,
    kv_quant_override: ?kv_quant.KVQuantConfig,
    variant: u64,
    cache_key: u64,
    max_bytes: usize,
) !requests.CacheExportOutcome {
    return exportWireModuleWithBudget(allocator, io, sch, lm, prompt_ids, has_tools, kv_quant_override, variant, cache_key, 0, max_bytes, false);
}

const RamModuleHit = prefix_cache_mod.HotPrefixCache.RamModuleHit;

/// The snapshot an SSD record was read back into, freed when the export is done with it.
const DiskSession = struct {
    xfm: ?*const Transformer = null,
    snapshot: plugin_abi.Handle = null,

    fn release(self: *DiskSession) void {
        if (self.snapshot != null) plugin_driver.auxFree(self.xfm.?, self.snapshot);
        self.snapshot = null;
    }
};

const SessionLookup = union(enum) { none, hit: RamModuleHit, refused: requests.CacheExportOutcome };

/// Which entry of a session an export builds: its prompt boundary, or the decode-end entry beside it.
const Pick = enum { boundary, tail };

const ExportOk = @FieldType(requests.CacheExportOutcome, "ok");

pub var after_aux_import_for_test: ?*const fn () void = null;

/// The prompt boundary a session's newest tip names, else the next boundary of the session that has a record, in RAM or
/// on the SSD (`tipHit`). A session whose every tip is gone is `none`: the caller sees `session_not_found`. With
/// `.tail`, the decode-end entry the session committed after its newest tip's boundary (`ramTailOfTip`), RAM only.
fn sessionHit(
    allocator: std.mem.Allocator,
    sch: *const Scheduler,
    cfg: *const ModelConfig,
    hc: *prefix_cache_mod.HotPrefixCache,
    xfm: *Transformer,
    session: u64,
    max_bytes: usize,
    holder: *DiskSession,
    pick: Pick,
) SessionLookup {
    var walk: prefix_cache_mod.HotPrefixCache.TipWalk = .{ .session = session };
    if (pick == .tail) {
        const tip = walk.next(hc) orelse return .none;
        return if (hc.ramTailOfTip(tip, session)) |hit| .{ .hit = hit } else .none;
    }
    while (walk.next(hc)) |tip| {
        const had_error = mlx.errorPending();
        const found = tipHit(allocator, sch, cfg, hc, xfm, session, tip, max_bytes, holder);
        if (found != .none) return found;
        mlx.dropLatchedErrorUnless(had_error);
    }
    return .none;
}

/// The prompt boundary a tip names, found by identity: the RAM entry with that id and length, else the SSD record with
/// it. The record is read back into a snapshot (`holder` frees it) so it exports the way a RAM entry does, under the
/// variant the tip was committed with. The envelope carries the asking session, so the importer records it as an owner
/// even when another session wrote the entry. A tip whose entry is gone, or whose record cannot be read, is `none`.
fn tipHit(
    allocator: std.mem.Allocator,
    sch: *const Scheduler,
    cfg: *const ModelConfig,
    hc: *prefix_cache_mod.HotPrefixCache,
    xfm: *Transformer,
    session: u64,
    tip: prefix_cache_mod.HotPrefixCache.SessionTip,
    max_bytes: usize,
    holder: *DiskSession,
) SessionLookup {
    if (hc.ramHitOfTip(tip, session)) |hit| return .{ .hit = hit };
    const d = if (hc.disk) |*dd| dd else return .none;
    const ref = d.moduleByBoundary(tip.id, tip.n) orelse return .none;
    const binding = prefix_cache_mod.ModuleBinding.forVariant(xfm, tip.variant) orelse return .none;
    const header = d.readModuleHeader(ref.id) catch return .none;
    defer d.allocator.free(header);
    const inventory = d.moduleInventory(ref.id) orelse return .none;
    var stored: u64 = 0;
    for (inventory) |t| stored += t.bytes;
    if (stored > max_bytes) return .{ .refused = .{ .too_large = @intCast(stored) } };
    var plan_peak: u64 = 0;
    if (!plugin_driver.auxImportPlan(binding.xfm, header, &plan_peak).isOk()) return .none;
    const peak = prefill_mod.moduleImportPeak(plan_peak, inventory) catch return .{ .refused = .no_memory };
    if (peak > prefillHeadroomNow(cfg, stagedHostBytes(@constCast(sch)))) return .{ .refused = .no_memory };

    var arrays = d.loadModuleParts(ref.id) catch return .none;
    defer d.freeNamed(&arrays);
    const tensors = allocator.alloc(plugin_abi.AuxTensor, arrays.items.len) catch return .{ .refused = .no_memory };
    defer allocator.free(tensors);
    for (arrays.items, tensors) |array, *t| t.* = .{ .name = .of(array.key), .array = .{ .ctx = array.arr.ctx } };
    var snapshot: plugin_abi.Handle = null;
    const imported = plugin_driver.auxImportArrays(binding.xfm, ref.tokens, header, tensors, &snapshot).isOk();
    if (after_aux_import_for_test) |probe| probe();
    if (!imported or snapshot == null or mlx.errorPending()) {
        if (snapshot != null) plugin_driver.auxFree(binding.xfm, snapshot);
        return .none;
    }
    holder.* = .{ .xfm = binding.xfm, .snapshot = snapshot };
    return .{ .hit = .{
        .payload = .{
            .binding = binding,
            .snapshot = snapshot,
            .n = ref.tokens.len,
            .prefix_key = ref.prefix_key,
            .aux_bytes = plugin_driver.auxBytes(binding.xfm, snapshot),
            .disk_id = ref.id,
        },
        .tokens = ref.tokens,
        .quant_config = ref.quant,
        .cache_key = 0,
        .has_tools = ref.has_tools,
        .media = ref.media,
        .session = session,
    } };
}

/// A session export sends the session's prompt boundary and, when it has one, the decode-end entry it committed after
/// it, both in one bundle (`peer_wire.writeBundle`): the receiver keeps both, so a next prompt that extends the reply
/// restores the tail and one that drops it restores the boundary whole. Anything that keeps the tail out (it is
/// gone, over `max_bytes` with the boundary, refused by the transfer budget or memory, an export error) sends the
/// boundary alone.
fn exportWireModuleWithBudget(
    allocator: std.mem.Allocator,
    io: std.Io,
    sch: *const Scheduler,
    lm: *LoadedModel,
    prompt_ids: []const u32,
    has_tools: bool,
    kv_quant_override: ?kv_quant.KVQuantConfig,
    variant: u64,
    cache_key: u64,
    session: u64,
    max_bytes: usize,
    admit_transfer_budget: bool,
) !requests.CacheExportOutcome {
    var tail_from: usize = 0;
    const base = try exportWireModuleOnce(allocator, io, sch, lm, prompt_ids, has_tools, kv_quant_override, variant, cache_key, session, max_bytes, admit_transfer_budget, .boundary, &tail_from);
    if (session == 0 or base != .ok) return base;
    const remaining = max_bytes -| peer_wire.bundleSize(&.{ base.ok.bytes.len, 0 });
    if (remaining == 0) return base;
    const tail = exportWireModuleOnce(allocator, io, sch, lm, prompt_ids, has_tools, kv_quant_override, variant, cache_key, session, remaining, admit_transfer_budget, .tail, &tail_from) catch return base;
    if (tail != .ok) return base;
    return .{ .ok = bundleOrBoundary(allocator, base.ok, tail.ok, tail_from, admit_transfer_budget) };
}

/// Both members stay reserved until the bundle copy is complete.
fn bundleOrBoundary(allocator: std.mem.Allocator, base: ExportOk, tail: ExportOk, tail_from: usize, admit: bool) ExportOk {
    const total = peer_wire.bundleSize(&.{ base.bytes.len, tail.bytes.len });
    defer {
        allocator.free(tail.bytes);
        if (admit) http_server.releaseCacheExportBytes(@intCast(tail.bytes.len));
    }
    if (admit and !http_server.reserveCacheExportBytes(total)) return base;
    const bytes = allocator.alloc(u8, total) catch {
        if (admit) http_server.releaseCacheExportBytes(total);
        return base;
    };
    peer_wire.writeBundle(bytes, &.{ .{ .tail_from = 0, .body = base.bytes }, .{ .tail_from = tail_from, .body = tail.bytes } });
    allocator.free(base.bytes);
    if (admit) http_server.releaseCacheExportBytes(@intCast(base.bytes.len));
    const peak = if (admit) @max(base.planned_peak -| base.bytes.len, tail.planned_peak -| tail.bytes.len) +| total else 0;
    log.info("[hot-cache] peer module export: bundle of {d} + {d} tokens, bytes={d}\n", .{ base.n, tail.n, total });
    return .{ .bytes = bytes, .n = tail.n, .export_ms = base.export_ms + tail.export_ms, .planned_peak = peak };
}

fn exportWireModuleOnce(
    allocator: std.mem.Allocator,
    io: std.Io,
    sch: *const Scheduler,
    lm: *LoadedModel,
    prompt_ids: []const u32,
    has_tools: bool,
    kv_quant_override: ?kv_quant.KVQuantConfig,
    variant: u64,
    cache_key: u64,
    session: u64,
    max_bytes: usize,
    admit_transfer_budget: bool,
    pick: Pick,
    tail_from: *usize,
) !requests.CacheExportOutcome {
    const t0 = io_util.nowUsMonotonic(io);
    const had_error = mlx.errorPending();
    defer mlx.dropLatchedErrorUnless(had_error);

    const xfm = lm.transformer orelse return .not_found;
    const cfg = lm.config orelse return .not_found;
    const hc = lm.prefix_cache orelse return .not_found;
    const pack_identity = switch (pack_identity_mod.status(lm.pack_identity)) {
        .ready => |id| id,
        .pending => return .identity_pending,
        .unsupported => return .transfer_unsupported,
    };
    const quant_config = kv_quant_override orelse cfg.kv_quant_override orelse sch.kv_quant_config;

    var from_disk: DiskSession = .{};
    defer from_disk.release();
    const hit = if (session != 0) switch (sessionHit(allocator, sch, cfg, hc, xfm, session, max_bytes, &from_disk, pick)) {
        .hit => |h| h,
        .none => return .not_found,
        .refused => |outcome| return outcome,
    } else blk: {
        const by_prompt = prefix_cache_mod.ModuleBinding.forVariant(xfm, variant) orelse return .not_found;
        break :blk hc.findRamModule(by_prompt, prompt_ids, has_tools, quant_config) orelse return .not_found;
    };
    const binding = hit.payload.binding;
    tail_from.* = hit.payload.tail_from;

    var info: plugin_abi.AuxSnapshotInfo = .{};
    if (!plugin_driver.auxDescribe(binding.xfm, hit.payload.snapshot, &info).isOk() or
        info.n != hit.payload.n or
        info.n != hit.tokens.len or
        prefix_cache_mod.variantKey(info.prefix_key, binding.variant) != hit.payload.prefix_key or
        !std.mem.eql(u32, info.ids.slice(), hit.tokens) or
        mlx.errorPending())
    {
        return .not_found;
    }

    const exp = try prefix_cache_mod.ModuleExport.create(allocator, hit.payload);
    defer exp.destroy(allocator);

    var inv_arena = std.heap.ArenaAllocator.init(allocator);
    defer inv_arena.deinit();
    const inv = try exp.inventory(inv_arena.allocator());

    const total_len = try peer_wire.encodedSize(hit.tokens.len, hit.media.len, hit.session, exp.out.header_len, inv);
    if (total_len > max_bytes) {
        return .{ .too_large = @intCast(total_len) };
    }
    var planned_peak: u64 = 0;
    if (admit_transfer_budget) {
        var plugin_import_peak: u64 = 0;
        if (!plugin_driver.auxImportPlan(binding.xfm, exp.header[0..exp.out.header_len], &plugin_import_peak).isOk()) return .no_memory;
        planned_peak = try std.math.add(u64, try prefill_mod.moduleImportPeak(plugin_import_peak, inv), @intCast(total_len));
    }

    if (admit_transfer_budget and planned_peak > prefillHeadroomNow(cfg, stagedHostBytes(@constCast(sch))) +| hc.residentBytes()) return .no_memory;
    var transfer_reserved = false;
    if (admit_transfer_budget) {
        const amount: u64 = @intCast(total_len);
        if (!http_server.reserveCacheExportBytes(amount)) return .no_memory;
        transfer_reserved = true;
    }
    defer if (transfer_reserved) http_server.releaseCacheExportBytes(@intCast(total_len));

    const layout_key = modelLayoutKey(cfg, xfm);
    const settings_key = modelSettingsKey(lm, cfg, xfm);

    try materializeExportTensors(exp.tensors[0..exp.out.tensor_len], xfm.s);

    const out = try allocator.alloc(u8, total_len);
    errdefer allocator.free(out);

    const eff_cache_key = if (cache_key != 0) cache_key else hit.cache_key;
    const payload_off = try peer_wire.writeEnvelopePrefix(out, .{
        .layout_key = layout_key,
        .pack_identity = pack_identity,
        .settings = settings_key,
        .prefix_key = info.prefix_key,
        .cache_key = eff_cache_key,
        .variant = binding.variant,
        .has_tools = hit.has_tools,
        .quant_config = hit.quant_config,
        .ids = hit.tokens,
        .plugin_header = exp.header[0..exp.out.header_len],
        .tensors = inv,
        .media = hit.media,
        .session = hit.session,
    });

    var cursor = payload_off;
    for (exp.tensors[0..exp.out.tensor_len], inv) |t, item| {
        const nb: usize = @intCast(item.bytes);
        const src = rawArrayBytes(.{ .ctx = t.array.ctx }) orelse return error.UnreadableExportTensor;
        @memcpy(out[cursor..][0..nb], src[0..nb]);
        cursor += nb;
    }
    peer_wire.stampChecksum(out);

    const elapsed_ms: u64 = @intCast(@max(@divTrunc(io_util.nowUsMonotonic(io) - t0, 1000), 0));
    log.info("[hot-cache] peer module export: n={d} bytes={d} ({d} ms)\n", .{ hit.tokens.len, out.len, elapsed_ms });
    transfer_reserved = false;
    return .{ .ok = .{ .bytes = out, .n = hit.tokens.len, .export_ms = elapsed_ms, .planned_peak = planned_peak } };
}

fn mapDecodeError(err: peer_wire.DecodeError) error{OutOfMemory}!requests.CacheImportRefusal {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.TruncatedEnvelope => .truncated_body,
        error.BadEnvelope, error.BadTensorName, error.BadMediaSpans => .bad_envelope,
        error.UnsupportedEnvelopeVersion => .unknown_version,
        error.BadQuantConfig => .bad_quant_config,
        error.BadTensorDtype => .bad_dtype,
        error.BadTensorShape => .bad_shape,
        error.BadTensorOffsets => .bad_offsets,
        error.BadChecksum => .bad_checksum,
    };
}

fn logRefusal(refusal: requests.CacheImportRefusal, tokens: usize) void {
    log.info("[hot-cache] peer module import refused ({s}, n={d})\n", .{ @tagName(refusal), tokens });
}

/// The refusals `commitModule` would give, asked on the envelope's own byte counts before anything is evicted.
fn budgetRefusal(hc: *const prefix_cache_mod.HotPrefixCache, destination_bytes: u64) ?requests.CacheImportRefusal {
    if (!hc.ram_enabled or hc.max_entries == 0) return .commit_declined;
    if (hc.max_kv_bytes == 0) return null;
    if (destination_bytes > hc.max_kv_bytes) return .commit_declined;
    return null;
}

/// A failed import on an out-of-memory latch is the receiver's memory, not a bad pair of boxes.
fn importArraysRefusal() requests.CacheImportRefusal {
    return if (mlx.latchedErrorIs(mlx.mlxErrorIsMemory)) .no_memory else .import_arrays;
}

const ImportedBoundary = struct {
    env: peer_wire.DecodedEnvelope,
    commit: prefix_cache_mod.ModuleCommit,
};

const TailImport = struct {
    boundary: *const ImportedBoundary,
    from: usize,

    fn matches(self: TailImport, env: peer_wire.DecodedEnvelope) bool {
        const base = self.boundary.env;
        return env.session == base.session and env.variant == base.variant and env.has_tools == base.has_tools and
            std.meta.eql(env.quant_config, base.quant_config) and env.media.len == 0 and base.media.len == 0 and
            self.from >= base.ids.len and self.from <= base.ids.len + 1 and self.from < env.ids.len and
            std.mem.startsWith(u32, env.ids, base.ids);
    }
};

const ImportMember = union(enum) { single, boundary: *?ImportedBoundary, tail: TailImport };

/// A bundle may add a reply only to the prompt boundary this import committed or claimed.
pub fn importWireModule(
    allocator: std.mem.Allocator,
    io: std.Io,
    sch: *Scheduler,
    lm: *LoadedModel,
    body: []const u8,
) !requests.CacheImportOutcome {
    if (!peer_wire.isBundle(body)) return importEnvelope(allocator, io, sch, lm, body, .single);
    var members: [peer_wire.MAX_BUNDLE_MEMBERS]peer_wire.BundleMember = undefined;
    const list = peer_wire.readBundle(body, &members) catch |err| {
        const refusal = try mapDecodeError(err);
        logRefusal(refusal, 0);
        return .{ .refused = refusal };
    };
    if (list.len == 1) return importEnvelope(allocator, io, sch, lm, list[0].body, .single);
    var boundary: ?ImportedBoundary = null;
    const base = try importEnvelope(allocator, io, sch, lm, list[0].body, .{ .boundary = &boundary });
    const imported = if (boundary) |*b| b else return base;
    defer imported.env.deinit();

    const hc = lm.prefix_cache.?;
    if (!hc.pinPeerBoundary(imported.commit.entry_id)) return base;
    defer {
        hc.unpinPeerBoundary();
        publishHotCacheResidency(sch);
    }

    const tail_from = std.math.cast(usize, list[1].tail_from) orelse return .{ .refused = .bad_envelope };
    const tail: ImportMember = .{ .tail = .{ .boundary = imported, .from = tail_from } };
    const out = importEnvelope(allocator, io, sch, lm, list[1].body, tail) catch |err| {
        log.warn("[hot-cache] optional peer tail import failed: {s}\n", .{@errorName(err)});
        return base;
    };
    return if (out == .ok or out.refused == .bad_envelope) out else base;
}

fn importEnvelope(
    allocator: std.mem.Allocator,
    io: std.Io,
    sch: *Scheduler,
    lm: *LoadedModel,
    body: []const u8,
    member: ImportMember,
) !requests.CacheImportOutcome {
    const t0 = io_util.nowUsMonotonic(io);
    const had_error = mlx.errorPending();
    defer mlx.dropLatchedErrorUnless(had_error);

    const local_pack_id = switch (pack_identity_mod.status(lm.pack_identity)) {
        .ready => |id| id,
        .pending => {
            logRefusal(.identity_pending, 0);
            return .{ .refused = .identity_pending };
        },
        .unsupported => {
            logRefusal(.transfer_unsupported, 0);
            return .{ .refused = .transfer_unsupported };
        },
    };

    var env = peer_wire.decode(allocator, body) catch |err| {
        const r = try mapDecodeError(err);
        logRefusal(r, 0);
        return .{ .refused = r };
    };
    var boundary_commit: ?prefix_cache_mod.ModuleCommit = null;
    defer if (boundary_commit) |c| {
        member.boundary.* = .{ .env = env, .commit = c };
    } else env.deinit();
    const t_decode = io_util.nowUsMonotonic(io);
    var stats: requests.CacheImportStats = .{
        .n = env.ids.len,
        .bytes = body.len,
        .decode_ms = @intCast(@max(@divTrunc(t_decode - t0, 1000), 0)),
        .admit_ms = 0,
        .import_ms = 0,
        .commit_ms = 0,
    };
    if ((member == .boundary and env.session == 0) or (member == .tail and !member.tail.matches(env))) {
        logRefusal(.bad_envelope, env.ids.len);
        return .{ .refused = .bad_envelope };
    }

    const xfm = lm.transformer orelse {
        logRefusal(.unsupported_model, env.ids.len);
        return .{ .refused = .unsupported_model };
    };
    const cfg = lm.config orelse {
        logRefusal(.unsupported_model, env.ids.len);
        return .{ .refused = .unsupported_model };
    };
    const hc = lm.prefix_cache orelse {
        logRefusal(.unsupported_model, env.ids.len);
        return .{ .refused = .unsupported_model };
    };
    const binding = prefix_cache_mod.ModuleBinding.forVariant(xfm, env.variant) orelse {
        logRefusal(.unsupported_model, env.ids.len);
        return .{ .refused = .unsupported_model };
    };

    if (env.layout_key != modelLayoutKey(cfg, xfm)) {
        logRefusal(.layout_key, env.ids.len);
        return .{ .refused = .layout_key };
    }
    if (env.pack_identity != local_pack_id) {
        logRefusal(.pack_identity, env.ids.len);
        return .{ .refused = .pack_identity };
    }
    if (env.settings != modelSettingsKey(lm, cfg, xfm)) {
        logRefusal(.settings, env.ids.len);
        return .{ .refused = .settings };
    }
    const local_quant = cfg.kv_quant_override orelse sch.kv_quant_config;
    if (!std.meta.eql(env.quant_config, local_quant)) {
        logRefusal(.quant_config, env.ids.len);
        return .{ .refused = .quant_config };
    }

    if (member == .tail) {
        if (hc.exactModuleBoundary(binding, env.ids, env.media, env.has_tools, env.quant_config) != null) {
            logRefusal(.stale, env.ids.len);
            return .{ .refused = .stale };
        }
    } else if (hc.claimCovering(binding, env.ids, env.media, env.has_tools, env.quant_config, env.session)) |claim| {
        if (member == .boundary and claim.seq != 0) {
            boundary_commit = claim;
            return .{ .ok = stats };
        }
        const refusal: requests.CacheImportRefusal = if (member == .boundary and env.session == 0) .bad_envelope else .stale;
        logRefusal(refusal, env.ids.len);
        return .{ .refused = refusal };
    }

    var plan: plugin_abi.AuxImportPlan = .{};
    if (!plugin_driver.auxImportPlanDetails(binding.xfm, env.plugin_header, &plan).isOk()) {
        logRefusal(.import_plan, env.ids.len);
        return .{ .refused = .import_plan };
    }
    const destination_bytes = if (plan.destination == 0) blk: {
        var bytes: u64 = 0;
        for (env.tensors) |tensor| bytes = try std.math.add(u64, bytes, tensor.bytes);
        break :blk bytes;
    } else plan.destination;
    if (budgetRefusal(hc, destination_bytes)) |r| {
        logRefusal(r, env.ids.len);
        return .{ .refused = r };
    }
    var peak = try prefill_mod.moduleImportPeak(plan.peak, env.tensors);
    const body_bytes: u64 = @intCast(body.len);
    peak = try std.math.add(u64, peak, body_bytes);
    var probe: prefill_mod.ModuleImportProbe = .{ .sch = sch, .cfg = cfg, .peak = peak, .body_bytes = body_bytes };
    const admitted = if (member == .tail) prefill_mod.ModuleImportProbe.fits(&probe) else blk: {
        const staged = stagedHostBytes(sch) -| body_bytes;
        if (peak > prefillHeadroomNow(cfg, staged) +| hc.residentBytes()) break :blk false;
        const report = hc.evictLruToAdmit(@intCast(env.ids.len), &probe, prefill_mod.ModuleImportProbe.fits, false);
        publishHotCacheResidency(sch);
        break :blk report.admitted;
    };
    if (!admitted) {
        logRefusal(.no_memory, env.ids.len);
        return .{ .refused = .no_memory };
    }
    const t_admit = io_util.nowUsMonotonic(io);

    const wrappers = try allocator.alloc(mlx.mlx_array, env.tensors.len);
    for (wrappers) |*w| w.* = .{ .ctx = null };
    defer {
        for (wrappers) |w| {
            if (w.ctx != null) _ = mlx.mlx_array_free(w);
        }
        allocator.free(wrappers);
    }
    const aux_tensors = try allocator.alloc(plugin_abi.AuxTensor, env.tensors.len);
    defer allocator.free(aux_tensors);

    for (env.tensors, env.descriptors, wrappers, aux_tensors) |t, desc, *w, *aux| {
        var c_shape: [8]c_int = undefined;
        for (t.shape, 0..) |dim, k| c_shape[k] = @intCast(dim);
        const slice = env.payload[@intCast(desc.offset)..@intCast(desc.end)];
        w.* = mlx.mlx_array_new_data(slice.ptr, &c_shape, @intCast(t.shape.len), desc.mlx_dtype);
        if (w.ctx == null) {
            const r = importArraysRefusal();
            logRefusal(r, env.ids.len);
            return .{ .refused = r };
        }
        aux.* = .{ .name = .of(t.name), .array = .{ .ctx = w.ctx } };
    }

    var snapshot: plugin_abi.Handle = null;
    if (!plugin_driver.auxImportArrays(binding.xfm, env.ids, env.plugin_header, aux_tensors, &snapshot).isOk() or
        snapshot == null or
        mlx.errorPending())
    {
        if (snapshot != null) plugin_driver.auxFree(binding.xfm, snapshot);
        const r = importArraysRefusal();
        logRefusal(r, env.ids.len);
        return .{ .refused = r };
    }
    const t_import = io_util.nowUsMonotonic(io);

    var info: plugin_abi.AuxSnapshotInfo = .{};
    if (!plugin_driver.auxDescribe(binding.xfm, snapshot, &info).isOk() or
        info.n != env.ids.len or
        info.prefix_key != env.prefix_key or
        prefix_cache_mod.variantKey(info.prefix_key, binding.variant) != prefix_cache_mod.variantKey(env.prefix_key, binding.variant) or
        !std.mem.eql(u32, info.ids.slice(), env.ids) or
        mlx.errorPending())
    {
        plugin_driver.auxFree(binding.xfm, snapshot);
        logRefusal(.describe_mismatch, env.ids.len);
        return .{ .refused = .describe_mismatch };
    }

    const committed = hc.commitModuleTail(
        binding,
        snapshot,
        env.ids,
        env.prefix_key,
        env.media,
        env.has_tools,
        env.quant_config,
        env.cache_key,
        env.session,
        if (member == .tail) member.tail.from else 0,
        if (member == .tail) member.tail.boundary.commit.seq else 0,
    ) catch |err| {
        if (member != .tail) publishHotCacheResidency(sch);
        return err;
    };
    if (member != .tail) publishHotCacheResidency(sch);
    const t_commit = io_util.nowUsMonotonic(io);

    switch (committed) {
        .ok => |c| {
            if (member == .boundary) {
                if (env.session == 0) {
                    logRefusal(.bad_envelope, env.ids.len);
                    return .{ .refused = .bad_envelope };
                }
                if (c.seq != 0) boundary_commit = c;
            }
            stats.n = c.n;
            stats.admit_ms = @intCast(@max(@divTrunc(t_admit - t_decode, 1000), 0));
            stats.import_ms = @intCast(@max(@divTrunc(t_import - t_admit, 1000), 0));
            stats.commit_ms = @intCast(@max(@divTrunc(t_commit - t_import, 1000), 0));
            log.info("[hot-cache] peer module import: n={d} bytes={d} (decode={d}ms admit={d}ms import={d}ms commit={d}ms)\n", .{
                stats.n, stats.bytes, stats.decode_ms, stats.admit_ms, stats.import_ms, stats.commit_ms,
            });
            return .{ .ok = stats };
        },
        else => {
            logRefusal(.commit_declined, env.ids.len);
            return .{ .refused = .commit_declined };
        },
    }
}

pub fn runCacheExportRequest(sch: *Scheduler, req: *requests.CacheExportRequest) void {
    if (deadlinePassed(sch.io, req.deadline_ms)) {
        req.done_mu.lockUncancelable(sch.io);
        defer req.done_mu.unlock(sch.io);
        req.outcome = .deadline_passed;
        req.done = true;
        req.done_cond.broadcast(sch.io);
        return;
    }
    const out = exportWireModuleWithBudget(
        req.allocator,
        sch.io,
        sch,
        req.model,
        req.prompt_ids,
        req.has_tools,
        req.kv_quant_override,
        req.variant,
        req.cache_key,
        req.session,
        req.max_bytes,
        req.admit_transfer_budget,
    ) catch |err| {
        log.warn("[hot-cache] peer cache export failed: {s}\n", .{@errorName(err)});
        req.done_mu.lockUncancelable(sch.io);
        defer req.done_mu.unlock(sch.io);
        req.error_name = @errorName(err);
        req.done = true;
        req.done_cond.broadcast(sch.io);
        return;
    };
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    req.outcome = out;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

pub fn runCacheImportRequest(sch: *Scheduler, req: *requests.CacheImportRequest) void {
    if (deadlinePassed(sch.io, req.deadline_ms)) {
        req.done_mu.lockUncancelable(sch.io);
        defer req.done_mu.unlock(sch.io);
        req.outcome = .{ .refused = .deadline_passed };
        req.done = true;
        req.done_cond.broadcast(sch.io);
        return;
    }
    const out = importWireModule(req.allocator, sch.io, sch, req.model, req.body) catch |err| {
        log.warn("[hot-cache] peer cache import failed: {s}\n", .{@errorName(err)});
        req.done_mu.lockUncancelable(sch.io);
        defer req.done_mu.unlock(sch.io);
        req.error_name = @errorName(err);
        req.done = true;
        req.done_cond.broadcast(sch.io);
        return;
    };
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    req.outcome = out;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

fn deadlinePassed(io: std.Io, deadline_ms: ?u64) bool {
    const deadline = deadline_ms orelse return false;
    const now: u64 = @intCast(@max(nowMsMonotonic(io), 0));
    return now > deadline;
}

pub fn exportCache(sch: *Scheduler, req: *requests.CacheExportRequest) !void {
    sch.queue_mu.lockUncancelable(sch.io);
    sch.cache_export_queue.append(sch.allocator, req) catch |err| {
        sch.queue_mu.unlock(sch.io);
        return err;
    };
    sch.queue_cond.broadcast(sch.io);
    sch.queue_mu.unlock(sch.io);

    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    while (!req.done) {
        req.done_cond.waitUncancelable(sch.io, &req.done_mu);
    }
    if (req.error_name != null) {
        req.error_name = null;
        return error.CacheExportFailed;
    }
}

pub fn importCache(sch: *Scheduler, req: *requests.CacheImportRequest) !void {
    sch.queue_mu.lockUncancelable(sch.io);
    sch.cache_import_queue.append(sch.allocator, req) catch |err| {
        sch.queue_mu.unlock(sch.io);
        return err;
    };
    sch.queue_cond.broadcast(sch.io);
    sch.queue_mu.unlock(sch.io);

    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    while (!req.done) {
        req.done_cond.waitUncancelable(sch.io, &req.done_mu);
    }
    if (req.error_name != null) {
        req.error_name = null;
        return error.CacheImportFailed;
    }
}

test "peer import budget uses resident size before evicting cache entries" {
    const testing = std.testing;
    var hc = prefix_cache_mod.HotPrefixCache.initWithMem(testing.allocator, 4, 128);
    defer hc.deinit();
    hc.current_kv_bytes = 64;
    const wire_tensors = [_]module_record.Tensor{.{ .name = "state", .dtype = "bf16", .shape = &.{4}, .bytes = 32 }};

    const wire_bytes = wire_tensors[0].bytes;
    try testing.expect(wire_bytes < 128);
    const refusal = budgetRefusal(&hc, 160);

    try testing.expectEqual(@as(?requests.CacheImportRefusal, .commit_declined), refusal);
    try testing.expectEqual(@as(u64, 64), hc.residentBytes());
}
