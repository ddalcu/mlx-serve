//! A model loaded onto an engine beside MLX: the settings each engine opens with, and the registry entry it is
//! installed on. GGUFs open on the inference thread; a Neural Engine build may run on a thread of its own.

const std = @import("std");
const abi = @import("../../../abi/engine.zig");
const tokenizer_mod = @import("../../../sdk/tokenizer/tokenizer.zig");
const tokenize_cache_mod = @import("../../../host/tokenizer/cache.zig");
const generate_mod = @import("../generate.zig");
const dflash_mod = @import("../spec/dflash/dflash.zig");
const log = @import("../../../sdk/util/log.zig");
const served_mod = @import("../../../host/engines/served.zig");
const Served = served_mod.Served;
const Kind = served_mod.Kind;
const GgufRoute = @import("thread.zig").GgufRoute;
const Scheduler = @import("../scheduler.zig").Scheduler;
const scheduler_mod = @import("../scheduler.zig");
const publishHotCacheResidency = scheduler_mod.publishHotCacheResidency;
const makeRoomInRam = @import("load_requests.zig").makeRoomInRam;
const drafterFor = @import("cold_load.zig").drafterFor;
const availForLoad = @import("preflight.zig").availForLoad;
const memInsufficientForLoad = @import("preflight.zig").memInsufficientForLoad;
const loadRequirementBytes = @import("preflight.zig").loadRequirementBytes;
const ds4_load = @import("../../ds4/load.zig");
const AneSettings = @import("../../ane/engine.zig").Settings;

/// A GGUF on ds4 or llama.cpp: open, install, ready. The stub config's context is the one asked for; the engine's
/// answer replaces it.
pub fn loadGguf(sch: *Scheduler, params: anytype, route: GgufRoute) !void {
    const kind = served_mod.fromGguf(route.engine);
    const settings = switch (kind) {
        .ds4 => try std.json.Stringify.valueAlloc(sch.allocator, ds4_load.Flags{
            .mtp = params.ds4_mtp,
            .dspark = params.ds4_dspark,
            .ssd_streaming = params.ds4_ssd_streaming,
            .prefill_chunk = generate_mod.explicitPrefillChunkU32(),
        }, .{}),
        else => try std.json.Stringify.valueAlloc(sch.allocator, params.llama, .{}),
    };
    defer sch.allocator.free(settings);
    const s = try Served.open(sch.allocator, kind, .{
        .model_dir = route.path,
        .context = params.config.max_position_embeddings,
        .max_concurrent = sch.queue_cap,
        .settings_json = settings,
    }, &params.entry.load_progress, &sch.engine_gauges, sch.metrics);
    errdefer s.close();
    params.config.max_position_embeddings = s.info.context_window;
    // The host renders a GGUF's embedded template through its own Jinja (tool synthesis included).
    if (s.chatTemplate()) |tmpl| {
        const cc = params.chat_config;
        const dup = try cc.allocator.dupe(u8, tmpl);
        cc.allocator.free(cc.chat_template);
        cc.chat_template = dup;
    }
    install(sch, params, s, "");
    // A GGUF load takes the scheduler's current model, as the inference thread once decoded it.
    sch.current_model = params.entry;
    sch.xfm = null;
    sch.weights = null;
    sch.vision_encoder = null;
    sch.drafter = null;
    sch.dflash = null;
    publishHotCacheResidency(sch);
    if (scheduler_mod.hot_cache_budget_invalidate) |f| f();
}

/// The Neural Engine's own preflight: its layout runs there, its RAM is made and its bill fits; false serves the
/// model on the GPU (by name in the log). Inference thread: making room frees MLX models.
pub fn aneAdmit(sch: *Scheduler, params: anytype) bool {
    const t = served_mod.table(.ane).?;
    var v: abi.Verdict = .{};
    if (!t.probe(.of(params.model_dir), .{}, &v)) {
        log.warn("[ane] {s} ({s}) {s}; serving it on the GPU\n", .{ params.model_dir, params.config.model_type, v.why.slice() });
        return false;
    }
    // Its own preflight: the int8 copies outweigh a 4-bit pack's shards, and the GPU arm checks those.
    const need = aneBytes(sch.allocator, params.model_dir, params.ane_slots) orelse return false;
    makeRoomInRam(sch, params.entry.id, need);
    if (!scheduler_mod.skip_mem_preflight) {
        const avail = availForLoad(sch.io, need);
        if (memInsufficientForLoad(need, avail)) {
            log.warn("[ane] {s} needs ~{d:.1} GB on the Neural Engine, {d:.1} GB available; serving it on the GPU\n", .{
                params.model_dir,
                @as(f64, @floatFromInt(loadRequirementBytes(need))) / 1_073_741_824.0,
                @as(f64, @floatFromInt(avail)) / 1_073_741_824.0,
            });
            return false;
        }
    }
    return true;
}

/// What the Neural Engine holds for `slots` engines on the pack at `dir`: its weights and every slot's KV (the
/// table's `load_bytes`); null when the pack does not run there.
pub fn aneBytes(gpa: std.mem.Allocator, dir: []const u8, slots: u32) ?u64 {
    const settings = std.json.Stringify.valueAlloc(gpa, AneSettings{ .slots = slots }, .{}) catch return null;
    defer gpa.free(settings);
    const host: abi.Host = .{ .ctx = undefined, .log = undefined, .wake = undefined };
    const o: abi.Open = .{ .model_dir = .of(dir), .model_type = .{}, .config_json = .{}, .context = 0, .max_concurrent = slots, .memory_budget = 0, .settings_json = .of(settings), .options_json = .{}, .host = &host };
    var bytes: u64 = 0;
    return if (served_mod.table(.ane).?.load_bytes(&o, &bytes).code == .ok) bytes else null;
}

/// Build (or restore) the programs and publish the model; false when the open failed (the GPU then serves it).
/// Never calls MLX, so a cold load runs it on its own thread.
pub fn aneOpen(sch: *Scheduler, params: anytype) !bool {
    log.info("[ane] opening {s}\n", .{params.model_dir});
    const gpa = sch.allocator;
    // The drafter resolves as on the GPU: launch flags, the per-model setting, else the pack's own.
    const chosen = drafterFor(params.no_drafter, params.drafter_dir, params.config.drafter_override);
    const in_dir: ?[]u8 = if (chosen == null) dflash_mod.resolveInDirDrafter(sch.io, gpa, params.model_dir) else null;
    defer if (in_dir) |d| gpa.free(d);
    const drafter: ?[]const u8 = if (chosen) |d| (if (d.len > 0) d else null) else in_dir;
    // The GPU's sampling rule (`installSuppressMask`): reserved specials and padding rows are never drawn.
    const suppress_ids: []u32 = tokenizer_mod.reservedOutputIds(gpa, params.tok.flagged_specials, params.chat_config.chat_template, params.config.eosTokenSlice()) catch &.{};
    defer gpa.free(suppress_ids);
    const settings = try std.json.Stringify.valueAlloc(gpa, AneSettings{
        .slots = params.ane_slots,
        .drafter = drafter,
        .suppress_ids = suppress_ids,
        .defined_vocab = if (params.config.unpadded_vocab_size > 0) params.config.unpadded_vocab_size else @intCast(params.tok.definedVocabSize()),
        // The hot prefix cache's switches: its entry cap counts checkpoints here, each a full DeltaNet state.
        .cache = if (params.prefix_cache_capacity > 0 and params.prefix_cache_ram_enabled and params.ssm_checkpoint_stride > 0)
            .{ .max_bytes = params.prefix_cache_mem_bytes, .max_states = params.prefix_cache_capacity, .stride = params.ssm_checkpoint_stride }
        else
            null,
    }, .{});
    defer gpa.free(settings);
    const s = Served.open(gpa, .ane, .{
        .model_dir = params.model_dir,
        .model_type = params.config.model_type,
        .max_concurrent = params.ane_slots,
        .settings_json = settings,
    }, &params.entry.load_progress, &sch.engine_gauges, sch.metrics) catch {
        log.err("[ane] {s} not loaded on the Neural Engine; serving it on the GPU\n", .{params.model_dir});
        return false;
    };
    errdefer s.close();
    const drafter_path: []const u8 = if (drafter != null and s.caps().drafts) try gpa.dupe(u8, drafter.?) else "";
    // What the engine holds is the context it advertises, whatever `--ctx-size` asks.
    const window = s.info.context_window;
    params.config.max_position_embeddings = window;
    params.config.ctx_override = if (params.config.ctx_override > 0) @min(params.config.ctx_override, window) else window;
    install(sch, params, s, drafter_path);
    return true;
}

/// The engine on the entry, the stub (or pack) CPU state moved over, the entry ready at what the engine holds.
fn install(sch: *Scheduler, params: anytype, s: *Served, drafter_path: []const u8) void {
    const entry = params.entry;
    entry.engine = s;
    entry.releaseRetainedCpuState();
    entry.config = params.config;
    entry.tokenizer = params.tok;
    entry.chat_config = params.chat_config;
    entry.weights = null;
    entry.transformer = null;
    entry.vision_encoder = null;
    entry.drafter = null;
    entry.dflash = null;
    entry.drafter_block_size = 0;
    entry.drafter_path = drafter_path;
    entry.prefix_cache = null;
    if (params.tokenize_cache_entries > 0) {
        entry.tokenize_cache = tokenize_cache_mod.TokenizeCache.init(sch.allocator, params.tokenize_cache_entries);
    }
    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.markReadyLocked(entry, s.residentBytes());
    sch.registry.mutex.unlock(sch.io);
}
