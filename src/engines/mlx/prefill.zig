//! Prefill of one slot.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const transformer_mod = @import("transformer.zig");
const generate_mod = @import("generate.zig");
const gen_mod = @import("../../host/media_jobs/gen.zig");
const model_mod = @import("../../sdk/config/model.zig");
const prefix_cache_mod = @import("cache/prefix_cache.zig");
const restore_dump = @import("cache/restore_dump.zig");
const kv_disk_cache = @import("cache/ssd/disk_tier.zig");
const log = @import("../../sdk/util/log.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const plugin_driver = @import("plugin/driver.zig");
const plugin_abi = @import("../../abi/plugin.zig");
const plugin_settle = @import("plugin/settle.zig");
const mlx_stream = @import("modules/stream.zig");
const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;
const Generator = generate_mod.Generator;
const dflash_mod = @import("spec/dflash/dflash.zig");
const testing = std.testing;

const publishHotCacheResidency = @import("cache/budget.zig").publishHotCacheResidency;
const prefillDecodeShare = @import("prefill_interleave.zig").prefillDecodeShare;
const InterleaveCtx = @import("prefill_interleave.zig").InterleaveCtx;
const companyPrefillChunk = @import("prefill_interleave.zig").companyPrefillChunk;
const promptLogprobsWidthCap = @import("prefill_interleave.zig").promptLogprobsWidthCap;
const decodeShareAdmissionCap = @import("prefill_interleave.zig").decodeShareAdmissionCap;
const resetRestoredSlot = @import("prefill_interleave.zig").resetRestoredSlot;
const prefillWriteThroughCb = @import("prefill_interleave.zig").prefillWriteThroughCb;
const WriteThroughCtx = @import("prefill_interleave.zig").WriteThroughCtx;
const writeThroughArmed = @import("prefill_interleave.zig").writeThroughArmed;
const ChunkWidthCtx = @import("prefill_interleave.zig").ChunkWidthCtx;
const chunkWidenConfirmCb = @import("prefill_interleave.zig").chunkWidenConfirmCb;
const chunkWidthCb = @import("prefill_interleave.zig").chunkWidthCb;
const interleaveDecodeTickCb = @import("prefill_interleave.zig").interleaveDecodeTickCb;
const Scheduler = @import("scheduler.zig").Scheduler;
const postEvictionPrefillChunk = @import("scheduler/params.zig").postEvictionPrefillChunk;
const adaptiveChunkWidthFor = @import("scheduler/params.zig").adaptiveChunkWidthFor;
const Slot = @import("slot.zig").Slot;
const dflashGateMinimum = @import("spec/dflash_gate.zig").dflashGateMinimum;
const specInitWiring = @import("spec/wiring.zig").specInitWiring;
const publishLiveKvResidency = @import("step.zig").publishLiveKvResidency;
const scheduler_mod = @import("scheduler.zig");
const intree = @import("plugin/intree.zig");

fn beginModulePrompt(slot: *Slot, binding: prefix_cache_mod.ModuleBinding, n: usize) !void {
    const id = plugin_settle.slotId(slot.model.transformer.?.plugin.?, &slot.ctx);
    const capture = if (slot.model.prefix_cache) |hc| hc.ram_enabled and hc.max_entries > 0 else false;
    const v = plugin_driver.auxBeginPrompt(binding.xfm, id, slot.full_prompt, slot.vision_embeddings, n, capture);
    if (!v.isOk()) {
        log.err("[hot-cache] module prompt setup failed: {s}\n", .{v.why.slice()});
        return error.PluginForwardFailed;
    }
    slot.cache.step = n;
    slot.moe_seq_offset = n;
}

fn restoreModulePrompt(sch: *Scheduler, slot: *Slot, binding: prefix_cache_mod.ModuleBinding) !usize {
    const id = plugin_settle.slotId(slot.model.transformer.?.plugin.?, &slot.ctx);
    const selected = modulePromptMatch(slot, binding);
    var n = if (selected) |match| restoreModuleMatch(sch, slot, binding, match, id) else 0;
    slot.cached_tokens = 0;
    slot.restored_entry = 0;
    const had_error = mlx.errorPending();
    beginModulePrompt(slot, binding, n) catch |err| {
        if (n == 0) return err;
        mlx.dropLatchedErrorUnless(had_error);
        if (selected) |match| if (match.source == .disk) slot.model.prefix_cache.?.disk.?.retireModule(match.disk_id);
        n = 0;
        try beginModulePrompt(slot, binding, 0);
    };
    if (n > 0) if (selected) |match| {
        log.info("[hot-cache] module restored source={s} id={d} position={d}\n", .{ @tagName(match.source), if (match.source == .ram) match.entry_id else match.disk_id, n });
    };
    return n;
}

fn modulePromptMatch(slot: *Slot, binding: prefix_cache_mod.ModuleBinding) ?prefix_cache_mod.ModuleMatch {
    if (slot.skip_prefix_cache) return null;
    const hc = slot.model.prefix_cache orelse return null;
    if (!hc.ram_enabled or hc.max_entries == 0) return null;
    return hc.findModuleMatch(binding, slot.full_prompt, slot.has_tools, slot.media, slot.vision_embeddings, slot.cache.config);
}

fn restoreModuleMatch(sch: *Scheduler, slot: *Slot, binding: prefix_cache_mod.ModuleBinding, match: prefix_cache_mod.ModuleMatch, id: u32) usize {
    if (match.source == .ram) return slot.model.prefix_cache.?.consumeModule(binding, match.entry_id, id);
    const restored = restoreDiskModule(sch, slot, binding, match, id) catch |err| {
        log.warn("[disk-cache] module import id={d} failed: {s}; prefilling cold\n", .{ match.disk_id, @errorName(err) });
        return 0;
    };
    return if (restored) match.n else 0;
}

const ModuleImportProbe = struct {
    sch: *Scheduler,
    cfg: *const model_mod.ModelConfig,
    peak: u64,

    fn fits(ctx: ?*anyopaque) bool {
        const self: *@This() = @ptrCast(@alignCast(ctx.?));
        const staged = @import("cache/budget.zig").stagedHostBytes(self.sch);
        return self.peak <= @import("admission/prefill_chunk.zig").prefillHeadroomNow(self.cfg, staged);
    }
};

/// Plan and admit before loading any part. Keep all inputs private until exact restore and prompt setup succeed.
fn restoreDiskModule(sch: *Scheduler, slot: *Slot, binding: prefix_cache_mod.ModuleBinding, match: prefix_cache_mod.ModuleMatch, id: u32) !bool {
    const hc = slot.model.prefix_cache orelse return false;
    const d = if (hc.disk) |*disk| disk else return false;
    const had_error = mlx.errorPending();
    defer mlx.dropLatchedErrorUnless(had_error);
    errdefer d.retireModule(match.disk_id);
    const header = try d.readModuleHeader(match.disk_id);
    defer d.allocator.free(header);
    var peak: u64 = 0;
    if (!plugin_driver.auxImportPlan(binding.xfm, header, &peak).isOk()) return error.ModuleImportPlanFailed;
    const inventory = d.moduleInventory(match.disk_id) orelse return false;
    peak = try moduleImportPeak(peak, inventory);
    var probe: ModuleImportProbe = .{ .sch = sch, .cfg = slot.model.config orelse return false, .peak = peak };
    const admitted = hc.evictLruToAdmit(slot.full_prompt.len, &probe, ModuleImportProbe.fits, false).admitted;
    publishHotCacheResidency(sch);
    if (!admitted) {
        log.info("[disk-cache] module import id={d} skipped: peak {d} bytes does not fit\n", .{ match.disk_id, peak });
        return false;
    }
    try importModuleSnapshot(slot, binding, match, id, d, header);
    return true;
}

/// A matching inventory uses the planner's full peak; a mismatching one must be rejected
/// before destination allocation. Bound its loader overlap, maps and header/name copies too.
fn moduleImportPeak(planned: u64, inventory: []const kv_disk_cache.module_record.Tensor) !u64 {
    var loaded: u64 = 0;
    for (inventory) |tensor| loaded = try std.math.add(u64, loaded, tensor.bytes);
    const host = try std.math.add(u64, 4 * kv_disk_cache.module_record.MAX_ENVELOPE_BYTES, try std.math.mul(u64, inventory.len, 4096));
    const load_peak = try std.math.add(u64, host, try std.math.mul(u64, loaded, 3));
    return @max(planned, load_peak);
}

fn importModuleSnapshot(slot: *Slot, binding: prefix_cache_mod.ModuleBinding, match: prefix_cache_mod.ModuleMatch, id: u32, d: *kv_disk_cache.DiskTier, header: []const u8) !void {
    // Admission can evict and swap-remove disk entries. Re-find the immutable ID.
    var arrays = try d.loadModuleParts(match.disk_id);
    defer d.freeNamed(&arrays);
    const tensors = try d.allocator.alloc(plugin_abi.AuxTensor, arrays.items.len);
    defer d.allocator.free(tensors);
    for (arrays.items, tensors) |array, *tensor| tensor.* = .{ .name = .of(array.key), .array = .{ .ctx = array.arr.ctx } };
    var snapshot: plugin_abi.Handle = null;
    defer plugin_driver.auxFree(binding.xfm, snapshot);
    const ids = slot.full_prompt[0..match.n];
    if (!plugin_driver.auxImportArrays(binding.xfm, ids, header, tensors, &snapshot).isOk()) return error.ModuleImportFailed;
    var info: plugin_abi.AuxSnapshotInfo = .{};
    if (!plugin_driver.auxDescribe(binding.xfm, snapshot, &info).isOk()) return error.ModuleImportDescriptionFailed;
    if (info.n != match.n or info.prefix_key != match.prefix_key or !std.mem.eql(u32, info.ids.slice(), ids)) return error.ModuleImportIdentityMismatch;
    if (!plugin_driver.auxRestoreAt(binding.xfm, id, snapshot, match.n).isOk()) return error.ModuleRestoreFailed;
}

pub fn runPrefill(sch: *Scheduler, slot: *Slot) !void {
    // Mark the phase for the whole of prefill, and clear both signals on EVERY
    // exit path (success, cancel, error). `requests_prefilling` flips at entry
    // so the panel isn't blind until the first chunk lands; the chunk loop in
    // generate.zig stores absolute progress into `inflight_prefill_tokens`.
    //
    // Gated on `--metrics`, per the observability contract: when it's off the
    // prefill path executes NO extra instruction at all (the chunk loop's hook
    // is null too — see the `prefill_progress` option below).
    const observe = sch.metrics != null;
    if (observe) {
        _ = sch.requests_prefilling.fetchAdd(1, .monotonic);
        const module_bytes = @import("cache/budget.zig").moduleStateBytes(sch);
        sch.queue_mu.lockUncancelable(sch.io);
        defer sch.queue_mu.unlock(sch.io);
        publishLiveKvResidency(sch, slot, module_bytes);
    }
    defer if (observe) {
        _ = sch.requests_prefilling.fetchSub(1, .monotonic);
        sch.inflight_prefill_tokens.store(0, .monotonic);
        sch.inflight_prefill_expected.store(0, .monotonic);
    };

    const sampling = slot.sampling;
    // Refresh ctx in case slot was relocated (paranoia — slot is heap so no,
    // but cheap).
    slot.ctx.cache = &slot.cache;
    slot.ctx.moe_seq_offset = &slot.moe_seq_offset;
    slot.ctx.ssm_entries = slot.ssm_entries;
    slot.ctx.vision_embeddings = slot.vision_embeddings;
    slot.ctx.mrope_pos = slot.mrope_pos;
    slot.ctx.mrope_total = slot.mrope_total;
    slot.ctx.mrope_delta = slot.mrope_delta;
    slot.ctx.capture_hidden = null;
    slot.ctx.kv_attn_fused = slot.kv_attn_fused;
    if (slot.model.transformer) |xfm| try xfm.ssmGroupRelease(&slot.ctx);

    // deepseek_v4: PLD/drafter/qwen-MTP verify passes through forwardWith
    // would APPEND draft tokens to module-owned state and corrupt every later
    // step, so their handles stay off here. The request's spec INTENT is
    // passed through anyway (as pld_enabled) so the Generator chokepoint —
    // the single authority since the DSpark port — can arm dsv4's OWN draft
    // mode (stage-bearing checkpoint + clean-greedy request) or zero
    // everything. skip_lazy_preforward deliberately ignores the intent bit:
    // a non-armed dsv4 request keeps today's synchronous-t1 serial init.
    // A module-owned arch that CAN rewind (`moduleStateSpecRollback`) may run
    // its own MTP head; PLD/drafter stay off there for the reasons in
    // `specInitWiring`. DSpark rides the MTP flag alone (the
    // "model's native head" semantics): the server defaults enable_mtp ON for a
    // stage-bearing dsv4, the n-gram prompt gate never touches it, and
    // enable_mtp:false opts out.
    // Module CLASS (owned or shared-readonly), not ownership alone: qwen4
    // batches its plain slots but its spec wiring stays the module one.
    const owns_module_state = slot.model.transformer != null and
        slot.model.transformer.?.moduleSpecWiring();
    const has_native_draft = slot.model.transformer != null and
        (intree.dsv4(slot.model.transformer.?) != null or intree.dsv41(slot.model.transformer.?) != null or plugin_driver.nativeDrafter(slot.model.transformer.?) != null or slot.model.transformer.?.mlx_stream_ext != null);
    const module_spec_rollback = slot.model.transformer != null and
        slot.model.transformer.?.moduleStateSpecRollback();
    const wiring = specInitWiring(
        owns_module_state,
        module_spec_rollback,
        has_native_draft,
        slot.enable_mtp,
        slot.mtp != null,
        slot.enable_drafter,
        slot.drafter != null,
        slot.dflash != null,
        slot.enable_pld,
    );
    const use_mtp = wiring.use_mtp;
    const use_drafter = wiring.use_drafter;
    const use_dflash = wiring.use_dflash;
    const use_pld = wiring.use_pld;
    const dsv4_spec_intent = wiring.native_intent;
    log.debug("[spec-wiring] mtp={} dflash={} drafter={} pld={} (slot: drafter_flag={} dflash_handle={} drafter_handle={})\n", .{
        use_mtp,             use_dflash,          use_drafter,          use_pld,
        slot.enable_drafter, slot.dflash != null, slot.drafter != null,
    });

    // Phase A6: prefill source-of-truth is `slot.full_prompt` — the conn
    // thread's `reuseKVCache` may have trimmed `slot.prompt_ids` based on
    // `xfm.cache` (the legacy global cache), but the slot has its own cache
    // which started empty. Using `slot.prompt_ids` would cause the model to
    // attend to only the trailing portion with empty cache, producing
    // garbage. Always start from the full prompt and let the hot prefix
    // cache (if configured) trim it back via the slot's own cache state.
    //
    // Vision-bearing slots: skip the hot cache altogether. Image tokens
    // have identical IDs but the underlying vision embeddings differ
    // per-request, so prefix matching would reuse stale features.
    var prefill_tokens: []const u32 = slot.full_prompt;
    var hot_matched: u32 = 0;
    // Are the restored rows the slot's own? A RAM checkout or a disk restore both credit them.
    var hot_checked_out: bool = false;
    // The DFlash assistant's context rides the prefix cache: a restore
    // forwards no trunk layers, so without it the assistant starts every
    // reused turn blind and drafts against nothing. Measured on Muse 4-bit,
    // 160-token generations: a full-prefix hit cost 92.6% -> 66.5% per-draft
    // acceptance and 80.2 -> 60.9 tok/s. Adopted by the Generator below.
    var dflash_restored: ?dflash_mod.DflashCtx = null;
    errdefer if (dflash_restored) |*dc| dc.deinit();
    // The MTP head's committed history rides the prefix cache the same way:
    // it is built from trunk hiddens, a restore forwards nothing, and a
    // blind start collapses acceptance (measured ~70 -> ~38 tok/s on warm
    // Qwen3.6-27B echo). Adopted by the Generator below.
    var mtp_restored: ?generate_mod.MtpRestored = null;
    errdefer if (mtp_restored) |*mr| mr.cache.deinit();
    // Phase D: per-slot model — pull transformer + prefix cache off the
    // slot's LoadedModel. Both stay resident for the slot's lifetime
    // because the conn thread holds a refcount on slot.model.
    const xfm_ptr: *Transformer = slot.model.transformer.?;
    const module_binding = prefix_cache_mod.ModuleBinding.init(xfm_ptr);
    errdefer if (module_binding != null) {
        @import("finish.zig").releasePluginSlot(slot);
        publishHotCacheResidency(sch);
    };
    if (module_binding) |binding| {
        hot_matched = @intCast(try restoreModulePrompt(sch, slot, binding));
        prefill_tokens = slot.full_prompt[hot_matched..];
        hot_checked_out = hot_matched > 0;
        publishHotCacheResidency(sch);
    } else {
        // DeepSeek-V4.1 keeps its own state: a prompt that extends its last one resumes there.
        const resumed: u64 = if (plugin_driver.resumePrompt(xfm_ptr, &slot.ctx, slot.full_prompt, slot.vision_embeddings)) |n|
            n
        else if (xfm_ptr.mlx_stream_ext) |m|
            try mlx_stream.begin(m, slot.full_prompt, slot.max_tokens, mlx_stream.contextLength(&xfm_ptr.config))
        else
            0;
        if (resumed > 0) {
            hot_matched = @intCast(resumed);
            prefill_tokens = slot.full_prompt[resumed..];
            slot.cache.step = resumed;
            log.info("[dsv41] resumed {d} of {d} prompt tokens\n", .{ resumed, slot.full_prompt.len });
        }
    }
    if (module_binding == null) if (slot.model.prefix_cache) |hc| {
        {
            // Only build a restore target when this request will actually
            // draft — a non-dflash turn leaves the payload in the entry for
            // the next one that does.
            var dfl_target: ?dflash_mod.DflashCtx = if (use_dflash and slot.dflash != null)
                dflash_mod.DflashCtx.init(slot.allocator, slot.dflash.?, 0) catch null
            else
                null;
            errdefer if (dfl_target) |*dc| dc.deinit();
            var dfl_base: usize = 0;
            var mtp_target: ?generate_mod.MtpCacheRef = if (use_mtp and slot.mtp != null)
                slot.mtp.?.makeCache(slot.allocator) catch null
            else
                null;
            errdefer if (mtp_target) |*mc| mc.deinit();
            var mtp_base: usize = 0;
            var mtp_tail: mlx.mlx_array = .{ .ctx = null };
            defer if (mtp_tail.ctx != null) {
                _ = mlx.mlx_array_free(mtp_tail);
            };
            const mtp_kv: ?*KVCache = if (mtp_target) |*mc| mc.kv() else null;
            // qwen4_exp: the head's QSA half travels with its KV; adoption is all-or-nothing.
            const mtp_head = if (mtp_target) |*mc| mc.head() else null;
            // Restore by move: the slot names itself, opting into the checkout; `finishSlot`
            // releases it on every path that ends the slot.
            const lookup = hc.lookupAndRestoreWithMedia(
                &slot.cache,
                &slot.moe_seq_offset,
                slot.ssm_entries,
                xfm_ptr.s,
                slot.full_prompt,
                slot.has_tools,
                slot.media,
                if (dfl_target) |*dc| .{ .cache = &dc.cache, .base_pos = &dfl_base } else null,
                if (mtp_kv) |k| .{ .cache = k, .base_pos = &mtp_base, .head = mtp_head, .tail_hidden = &mtp_tail } else null,
                @intFromPtr(slot),
                slot.skip_prefix_cache,
            ) catch |err| blk: {
                log.warn("[hot-cache] lookup failed: {s} — proceeding with cold prefill\n", .{@errorName(err)});
                break :blk prefix_cache_mod.LookupResult{ .matched = 0, .full_match = false };
            };
            if (lookup.matched > 0 and lookup.matched <= slot.full_prompt.len) {
                hot_matched = @intCast(lookup.matched);
                prefill_tokens = slot.full_prompt[hot_matched..];
                hot_checked_out = lookup.checked_out or lookup.slot_owned;
                slot.restored_entry = lookup.entry_id;
            }
            if (dfl_target) |*dc| {
                // Adopt only a context that lines up EXACTLY with the trunk
                // cursor — `nextDflash` asserts `absLen() == cache.step`, and
                // a blind start is always a valid fallback.
                if (lookup.dflash_base != null and dfl_base + dc.cache.step == hot_matched) {
                    dc.base_pos = dfl_base;
                    dflash_restored = dc.*;
                } else {
                    dc.deinit();
                }
                dfl_target = null;
            }
            if (mtp_target) |*mc| {
                // A decode-end restore lands one row past the history: append that row first.
                if (lookup.mtp_base != null and mtp_tail.ctx != null and mtp_base + mc.step() + 1 == hot_matched and hot_matched < slot.full_prompt.len) {
                    slot.mtp.?.appendHistory(xfm_ptr, mc, slot.full_prompt[hot_matched .. hot_matched + 1], mtp_tail, @intCast(mc.step()), null, slot.allocator) catch |err| {
                        log.warn("[hot-cache] mtp history row append failed: {s} — head starts blind\n", .{@errorName(err)});
                    };
                }
                // Same exact-alignment rule (the Generator asserts
                // `base + step == ssm_cp_offset` on adoption).
                if (lookup.mtp_base != null and mtp_base + mc.step() == hot_matched) {
                    mtp_restored = .{ .cache = mc.*, .base = mtp_base };
                } else {
                    mc.deinit();
                }
                mtp_target = null;
            }
        }
    };

    // A request admitted at another prefill's chunk boundary must not turn into a long prefill of its
    // own: the restore covered less than the pre-filter peeked (SSD tier, lien, QSA gap, a checkout
    // taken meanwhile). Nothing has been forwarded yet, so hand the entry back and let it wait.
    if (slot.boundary_max_tail) |max_tail| if (prefill_tokens.len > max_tail) {
        resetRestoredSlot(slot);
        return error.BoundaryTailTooLarge;
    };

    // Phase 1 (perf-plan): forward the SSM-checkpoint stride from the
    // LoadedModel so the prefill loop snapshots SSM state at stride-aligned
    // positions for hybrid archs. Plain-attn models have empty ssm_entries
    // and ignore the stride entirely (no-op even at stride > 0). When the
    // hot prefix cache is disabled or off, set stride to 0 to skip
    // snapshot work that would just be discarded.
    const cp_stride: u32 = if (slot.model.prefix_cache != null) slot.model.ssm_checkpoint_stride else 0;
    const cp_max: u32 = slot.model.ssm_checkpoint_max;

    // Evict the hot cache to admit (#353). Here, not on the connection thread: the inference
    // thread is the sole mlx caller and the restore has already happened, so the entry this
    // request uses is the MRU one. The estimator is re-asked after every eviction.
    // `admitted_prefill_chunk` is the pre-eviction width (0 = no pass ran); the width is
    // re-asked after the pass against the live delta.
    var admitted_prefill_chunk: u32 = 0;
    var evicted_live_bytes: u64 = 0;
    if (scheduler_mod.prefill_admission_fits) |fits_fn| {
        if (slot.model.config) |cfg| {
            // Bills the request's own kv-quant scheme and vision chunking, not the process defaults.
            const Probe = struct {
                cfg: *const model_mod.ModelConfig,
                seq: usize,
                max_tokens: u32,
                kv_cfg: transformer_mod.KVQuantConfig,
                /// The prefix the hot cache restored and the capacity its buffers hold: already
                /// inside active memory, so billing them again invents a copy.
                warm_matched: u64,
                warm_capacity: u64,
                /// Only a checked-out restore is credited: a refcount share is copied by the first append.
                warm_will_donate: bool,
                enable_mtp: bool,
                fits: *const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, u64, u64, bool, bool) bool,
                fn call(ctx: ?*anyopaque) bool {
                    const self: *@This() = @ptrCast(@alignCast(ctx.?));
                    return self.fits(self.cfg, self.seq, self.max_tokens, self.kv_cfg, self.warm_matched, self.warm_capacity, self.warm_will_donate, self.enable_mtp);
                }
            };
            var probe = Probe{
                .cfg = cfg,
                .seq = slot.full_prompt.len,
                .max_tokens = slot.max_tokens,
                .kv_cfg = slot.cache.config,
                .warm_matched = hot_matched,
                .warm_capacity = slot.cache.residentCapacityTokens(),
                .warm_will_donate = hot_checked_out,
                .enable_mtp = slot.enable_mtp,
                .fits = fits_fn,
            };
            var fits = Probe.call(&probe);
            // The encoder cache is the cheapest memory to give back: a miss costs one encode.
            if (!fits) if (slot.model.vision_encoder) |ve| if (ve.emb_cache.bytes > 0) {
                log.info("[scheduler] prefill does not fit: dropping {d}MB of cached media embeddings\n", .{ve.emb_cache.bytes / (1024 * 1024)});
                ve.emb_cache.clear();
                fits = Probe.call(&probe);
            };
            // A resident video pack is memory its next request can reload: give it back before refusing.
            if (!fits) if (gen_mod.releaseMediaResidency() > 0) {
                log.info("[scheduler] prefill does not fit: released the resident media cache\n", .{});
                fits = Probe.call(&probe);
            };
            // A shared restore is billed a whole second copy; taking the entry over moves it instead.
            if (!fits and !hot_checked_out and hot_matched > 0) if (slot.model.prefix_cache) |hc| {
                if (hc.checkoutRestored(@intFromPtr(slot), slot.full_prompt.len)) {
                    hot_checked_out = true;
                    probe.warm_will_donate = true;
                    fits = Probe.call(&probe);
                }
            };
            if (!fits) {
                // The width admission was billed at, read before anything is evicted.
                if (scheduler_mod.prefill_request_chunk) |pick_pre| {
                    admitted_prefill_chunk = pick_pre(cfg, slot.full_prompt.len, slot.max_tokens, slot.cache.config, probe.warm_matched, probe.warm_capacity, probe.warm_will_donate, probe.enable_mtp);
                }
                // Per-model, off the slot: `sch.hot_prefix_cache` is whichever model loaded last.
                // Captured by pointer: a `|hc|` capture would evict from a stack copy.
                const report = if (slot.model.prefix_cache) |hc|
                    // Never evict the entry this request restored from: its buffers are shared.
                    hc.evictLruToAdmit(slot.full_prompt.len, &probe, Probe.call, true)
                else
                    prefix_cache_mod.EvictionReport{ .admitted = false };
                publishHotCacheResidency(sch);
                // What the allocator returned, not what the cache was billed for.
                evicted_live_bytes = report.bytes;
                fits = report.admitted;
                if (!fits and hot_matched > 0) if (module_binding) |binding| {
                    // Consumed snapshots have no donor left to evict. Drop their live state once.
                    try beginModulePrompt(slot, binding, 0);
                    hot_matched = 0;
                    hot_checked_out = false;
                    prefill_tokens = slot.full_prompt;
                    slot.restored_entry = 0;
                    probe.warm_matched = 0;
                    probe.warm_capacity = 0;
                    probe.warm_will_donate = false;
                    publishHotCacheResidency(sch);
                    // Dropping the restore can also unpin buffers shared with remaining entries.
                    const cold_report = slot.model.prefix_cache.?.evictLruToAdmit(slot.full_prompt.len, &probe, Probe.call, false);
                    evicted_live_bytes += cold_report.bytes;
                    fits = cold_report.admitted;
                    publishHotCacheResidency(sch);
                };
                if (!fits) {
                    log.warn("[scheduler] prefill refused: {d} tokens do not fit even with an empty hot cache\n", .{slot.full_prompt.len});
                    if (scheduler_mod.prefill_admission_refused_log) |report_fn| {
                        slot.refused_bill = report_fn(cfg, slot.full_prompt.len, slot.max_tokens, slot.cache.config, probe.warm_matched, probe.warm_capacity, probe.warm_will_donate, probe.enable_mtp);
                    }
                    // Not `error.OutOfMemory` (the MLX latch's name, a 503): this is a request the
                    // machine cannot hold, a named 400.
                    return error.PrefillDoesNotFit;
                }
            }
        }
    }

    // The prefill width for this request, chosen after the admission pass evicted. Falls
    // back to the load-time pin without the hook or the arch opt-in.
    const req_prefill_chunk: u32 = if (slot.model.config) |cfg| blk: {
        const pin = cfg.pinned_prefill_chunk;
        const pick = scheduler_mod.prefill_request_chunk orelse break :blk pin;
        const reasked = pick(
            cfg,
            slot.full_prompt.len,
            slot.max_tokens,
            slot.cache.config,
            hot_matched,
            slot.cache.residentCapacityTokens(),
            hot_checked_out,
            slot.enable_mtp,
        );
        const decision = postEvictionPrefillChunk(admitted_prefill_chunk, reasked);
        if (decision.widened) {
            log.info("[prefill] re-ask: width {d} -> {d} after the eviction pass returned {d} MB\n", .{ admitted_prefill_chunk, decision.width, evicted_live_bytes >> 20 });
        } else if (decision.moved) {
            log.warn("[prefill] re-ask: width {d} is NARROWER than the admitted {d} after the eviction pass returned {d} MB — memory moved between the two reads; the re-ask is the live reading, so it runs\n", .{ decision.width, admitted_prefill_chunk, evicted_live_bytes >> 20 });
        }
        break :blk decision.width;
    } else 0;

    // Chunk-boundary decode yields: the hook advances already-decoding
    // streams between this prefill's chunks. Ticks hosted here are billed
    // out of prefill_ns below (the decoding slots got the time).
    var interleave_ctx = InterleaveCtx{ .sch = sch, .slot = slot, .chunk_sw = io_util.Stopwatch.init(sch.io) };
    var write_through_ctx = WriteThroughCtx{ .slot = slot };
    // Per-chunk prefill width context. Stack-scoped like `interleave_ctx`.
    var width_ctx = ChunkWidthCtx{
        .sch = sch,
        .cfg = slot.model.config,
        .kv_bits = if (slot.cache.config.scheme == .off) 16 else slot.cache.config.bits,
    };

    // Ownership of the restored spec caches transfers AT THE CALL:
    // initWithOptions adopts them and frees them via its own errdefers on
    // any failure past adoption (a mid-prefill disconnect throws
    // error.Cancelled from its chunk loop). MtpCacheRef/DflashCtx hold the
    // KVCache BY VALUE, so a second deinit from our errdefers walked freed
    // mlx handles — SIGSEGV in freeKVEntry (issue #266). Clear FIRST.
    const dflash_pass = dflash_restored;
    dflash_restored = null;
    const mtp_pass = mtp_restored;
    mtp_restored = null;
    // Restore by move, the transfer: the last point at which nothing has written to `slot.cache`.
    // Everything above can still refuse and `releaseCheckout` then hands the entry back whole;
    // nothing between here and `initWithOptions` may fail.
    if (slot.model.prefix_cache) |hc| hc.donateCheckout(@intFromPtr(slot));
    const graphs_were = mlx.setCudaGraphs(mlx.prefillKeepsCudaGraphs());
    defer _ = mlx.setCudaGraphs(graphs_were);
    var gen = try Generator.initWithOptions(
        sch.io,
        slot.allocator,
        xfm_ptr,
        slot.model.tokenizer.?,
        prefill_tokens,
        slot.max_tokens,
        sampling,
        slot.eos_token_ids,
        .{
            .pld_enabled = use_pld,
            .drafter_enabled = use_drafter,
            .drafter = if (use_drafter) slot.drafter else null,
            .drafter_block_size = slot.drafter_block_size,
            .dflash_enabled = use_dflash,
            .dflash = if (use_dflash) slot.dflash else null,
            .dflash_policy = use_dflash and generate_mod.Generator.dflashPolicyFor(slot.model.config.?),
            // The dflash-resolved block rides the shared drafter_block_size.
            .dflash_block_size = slot.drafter_block_size,
            // Use the resolved thinking mode and normalize the M5/block-16
            // calibration to this machine's effective assistant width.
            .dflash_min_accepted_per_round = dflashGateMinimum(
                slot.drafter_block_size,
                slot.enable_thinking,
                slot.model.config.?.isMoe(),
            ),
            .mtp_enabled = use_mtp or dsv4_spec_intent,
            .mtp_acceptance = slot.model.config.?.mtpAcceptance(generate_mod.mtp_acceptance_default),
            .mtp_greedy_tail = generate_mod.mtpGreedyTailFor(slot.model.config.?.mtp_greedy_tail_override),
            .mtp = if (use_mtp) slot.mtp else null,
            // The model's head before this request's opt-out (`entry.mtp` already ANDs `--no-mtp`).
            .model_has_mtp = slot.mtp != null,
            .mtp_depth = slot.mtp_depth,
            .lookup_prompt = slot.full_prompt,
            .ctx = slot.ctx,
            // Regular path: skip the lazy preforward so cache.step lands at
            // exactly prompt_len with t1 NOT in cache. Generator.next's
            // transition shim sync-forwards [t1] on the first decode call.
            // PLD/drafter/MTP init paths already skip preforward unconditionally.
            .skip_lazy_preforward = !use_pld and !use_drafter and !use_mtp and !use_dflash,
            .speech = slot.speech,
            .ssm_checkpoint_stride = cp_stride,
            .ssm_checkpoint_max = cp_max,
            .ssm_low_anchors = if (xfm_ptr.config.longCtxGated()) kv_disk_cache.SSM_DISK_LOW_ANCHORS else 0,
            .ssm_checkpoint_pos_offset = hot_matched,
            // A restored prefix already holds its image rows: the splice
            // resumes at the placeholder count inside the matched prefix.
            .vision_rows_before = if (slot.vision_embeddings != null and hot_matched > 0)
                generate_mod.countSpliceRows(@ptrCast(slot.full_prompt[0..hot_matched]), xfm_ptr.config.image_token_id, xfm_ptr.config.audio_token_id, xfm_ptr.config.video_token_id)
            else
                0,
            // The width the admission guard billed for this request; the forward can never run wider.
            .pinned_prefill_chunk = companyPrefillChunk(req_prefill_chunk, scheduler_mod.decodingCount(sch)),
            .decode_share_width_cap = promptLogprobsWidthCap(slot.prompt_logprobs, decodeShareAdmissionCap(scheduler_mod.liveDecodingCount(sch), prefillDecodeShare())),
            .dflash_ctx_restored = dflash_pass,
            .mtp_cache_restored = mtp_pass,
            // Abandoned-prefill abort: the conn thread sets slot.cancelled
            // when the client disconnects; the chunk loop checks it between
            // chunks so a ghost 40K prefill stops within one chunk.
            .cancel_flag = &slot.cancelled,
            // Salvage sink: on abort the Generator moves its captured SSM
            // stride checkpoints here (they die with the failed
            // construction otherwise) so the cancelled-prefill commit can
            // restore hybrids too.
            .cancelled_checkpoint_sink = &slot.cancelled_prefill,
            .prefill_progress = if (observe) &sch.inflight_prefill_tokens else null,
            .prefill_expected = if (observe) &sch.inflight_prefill_expected else null,
            .interleave_hook = .{ .ctx = &interleave_ctx, .call = interleaveDecodeTickCb },
            .write_through_hook = if (writeThroughArmed(slot, prefill_tokens.len))
                .{ .ctx = &write_through_ctx, .call = prefillWriteThroughCb }
            else
                null,
            .chunk_width_hook = if (scheduler_mod.prefill_chunk_adapt != null and !slot.prompt_logprobs)
                .{ .ctx = &width_ctx, .call = chunkWidthCb, .confirm = chunkWidenConfirmCb }
            else
                null,
            // The install above is process-wide, so it is not the arch gate.
            .adaptive_chunk_width = adaptiveChunkWidthFor(width_ctx.cfg),
            // Init reads logprobs in the split-prefill final-token forward,
            // before any post-init field write.
            .logprobs_n = slot.logprobs_n,
            .prompt_logprobs = if (slot.prompt_logprobs) slot.freshPromptLogprobs() else null,
        },
    );
    slot.prefill_interleaved_ns = interleave_ctx.decode_ns;
    gen.timeout_ns = slot.timeout_ns;
    gen.logprobs_n = slot.logprobs_n;

    slot.legacy_gen = gen;
    // The last chunks' transient is freed AFTER the loop's own per-chunk clear, so it
    // parks in MLX's pool up to the cap and the first decode tick allocates on top of
    // it. Returned once here, at the handover — long-context gate only.
    if (slot.legacy_gen) |*g| g.clearPoolBeforeDecode();
    // The conn thread's `cached_tokens` counted against `xfm.cache` (legacy
    // global cache) which the slot doesn't use. The slot's `cached_tokens`
    // is the hot-cache match (or 0 if the hot cache missed / isn't
    // configured) — those are the only tokens actually present in slot.cache
    // before this turn's prefill. With this override, slot.prompt_tokens
    // reports the full prompt length: gen.prompt_tokens (= prefill_tokens.len)
    // covers the un-cached tail, and `hot_matched` covers the restored prefix.
    slot.cached_tokens = hot_matched;
    slot.prompt_tokens = gen.prompt_tokens + slot.cached_tokens;
    slot.state = .decoding;
    if (hot_matched == 0) {
        const cold_pos = if (slot.moe_seq_offset > 0) slot.moe_seq_offset else slot.cache.step;
        _ = restore_dump.dumpRestoreIfEnabled(&slot.cache, slot.ssm_entries, xfm_ptr.s, .{
            .kind = "cold",
            .pos = cold_pos,
            .source = "cold",
        });
    }
}

/// Sum the in-flight generated tokens over the active slots for the live-tok/s
/// gauge, EXCLUDING any slot that already finished/cancelled/errored this tick
/// (its tokens were counted into `generation_tokens_total` by `finishSlot`, so
/// including them here would double-count — and would leave a stale non-zero
/// aggregate once the last slot finishes, breaking the "at rest ⇒ live == total"
/// invariant). Generic over the slot type so the exact filter is unit-testable
/// with a lightweight stub; the real caller passes `[]*Slot`.
pub fn sumInflightGeneratedTokens(active: anytype) u64 {
    var inflight: u64 = 0;
    for (active) |s| {
        if (s.finished or s.error_code != null or s.cancelled.load(.acquire)) continue;
        inflight += @as(u64, s.completion_tokens);
    }
    return inflight;
}

// ── tests ──

test "module import admission bounds envelope loads and rejects overflowing bills" {
    const Tensor = kv_disk_cache.module_record.Tensor;
    const tensors = [_]Tensor{ .{ .name = "a", .dtype = "F32", .shape = &.{4}, .bytes = 16 }, .{ .name = "b", .dtype = "F32", .shape = &.{8}, .bytes = 32 } };
    const peak = try moduleImportPeak(100, &tensors);
    try testing.expect(peak >= 100 + 3 * (16 + 32) + 2 * @sizeOf(plugin_abi.AuxTensor));
    try testing.expectEqual(@as(u64, 1 << 40), try moduleImportPeak(1 << 40, &tensors));
    const too_large = [_]Tensor{.{ .name = "a", .dtype = "F32", .shape = &.{1}, .bytes = std.math.maxInt(u64) }};
    try testing.expectError(error.Overflow, moduleImportPeak(1, &too_large));
    try testing.expectEqual(std.math.maxInt(u64), try moduleImportPeak(std.math.maxInt(u64), &tensors));
}

test "sumInflightGeneratedTokens sums active slots, excludes finished/cancelled/errored" {
    // Lightweight stub carrying exactly the fields the aggregate reads — proves
    // the live-gauge filter without constructing a real (mlx-backed) Slot.
    const StubSlot = struct {
        completion_tokens: u32,
        finished: bool,
        error_code: ?[]const u8,
        cancelled: std.atomic.Value(bool),
        fn make(tok: u32) @This() {
            return .{ .completion_tokens = tok, .finished = false, .error_code = null, .cancelled = std.atomic.Value(bool).init(false) };
        }
    };
    var a = StubSlot.make(10);
    var b = StubSlot.make(20);
    var done = StubSlot.make(99);
    done.finished = true; // already counted in generation_tokens_total
    var errd = StubSlot.make(50);
    errd.error_code = "OutOfMemory";
    var cxl = StubSlot.make(7);
    cxl.cancelled.store(true, .release);

    // Two still-decoding slots (10 + 20); the finished/errored/cancelled ones
    // are excluded, so the aggregate is 30 — never double-counting the tail.
    const active = [_]*StubSlot{ &a, &b, &done, &errd, &cxl };
    try testing.expectEqual(@as(u64, 30), sumInflightGeneratedTokens(active[0..]));

    // At rest: the last decoding slot finishes ⇒ aggregate collapses to 0,
    // pinning the "live == total at rest" invariant test_metrics.sh checks.
    a.finished = true;
    b.finished = true;
    try testing.expectEqual(@as(u64, 0), sumInflightGeneratedTokens(active[0..]));
}
