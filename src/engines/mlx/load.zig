//! MLX model load on the inference thread.

const std = @import("std");
const modules = @import("modules/call.zig");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const transformer_mod = @import("transformer.zig");
const generate_mod = @import("generate.zig");
const gen_mod = @import("../../host/media_jobs/gen.zig");
const drafter_mod = @import("../../plugins/gemma4/assistant_drafter.zig");
const mtp_graft = @import("../../plugins/prism_hadamard_qwen35/mtp_graft.zig");
const mtp_mod = @import("../../sdk/spec/mtp/mtp.zig");
const ane_mod = @import("../../sdk/ane/offload.zig");
const model_mod = @import("../../sdk/config/model.zig");
const vision_mod = @import("vision_encoder.zig");
const prefix_cache_mod = @import("cache/prefix_cache.zig");
const kv_disk_cache = @import("cache/ssd/disk_tier.zig");
const tokenize_cache_mod = @import("../../host/tokenizer/cache.zig");
const model_registry_mod = @import("../../host/registry/registry.zig");
const log = @import("../../sdk/util/log.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const status = @import("../../host/status.zig");
const Transformer = transformer_mod.Transformer;
const plugin_driver = @import("plugin/driver.zig");
const distributed = @import("cluster/distributed.zig");
const plugin_head = @import("plugin/head.zig");
const plugin_media = @import("plugin/media.zig");
const DrafterModel = drafter_mod.DrafterModel;
const dflash_mod = @import("spec/dflash/dflash.zig");
const round_cost_mod = @import("spec/round_cost.zig");
const DflashModel = dflash_mod.DflashModel;
const VisionEncoder = vision_mod.VisionEncoder;
const Weights = model_mod.Weights;
const ModelRegistry = model_registry_mod.ModelRegistry;
const testing = std.testing;

const engines_served = @import("../../host/engines/served.zig");
const engines_load = @import("runtime/engine_load.zig");
const publishHotCacheResidency = @import("cache/budget.zig").publishHotCacheResidency;
const Scheduler = @import("scheduler.zig").Scheduler;
const logWiredPolicy = @import("runtime/load_requests.zig").logWiredPolicy;
const makeRoomInRam = @import("runtime/load_requests.zig").makeRoomInRam;
const doLoadGenOnInferenceThread = @import("runtime/media_load.zig").doLoadGenOnInferenceThread;
const drafterFor = @import("runtime/cold_load.zig").drafterFor;
const pleTableBill = @import("runtime/preflight.zig").pleTableBill;
const modelDiskBytes = @import("runtime/preflight.zig").modelDiskBytes;
const residentModelDiskBytes = @import("runtime/preflight.zig").residentModelDiskBytes;
const residentGateBytes = @import("runtime/preflight.zig").residentGateBytes;
const committedTextBytes = @import("runtime/preflight.zig").committedTextBytes;
const availForLoad = @import("runtime/preflight.zig").availForLoad;
const mtpSidecarDiskBytes = @import("runtime/preflight.zig").mtpSidecarDiskBytes;
const drafterResidentBytes = @import("runtime/preflight.zig").drafterResidentBytes;
const preflightVerdict = @import("runtime/preflight.zig").preflightVerdict;
const loadRequirementBytes = @import("runtime/preflight.zig").loadRequirementBytes;
const scheduler_mod = @import("scheduler.zig");

/// Phase A1 → Plan 05: do the full model load on the inference thread.
/// mlx ops here bind to this thread's GPU stream from t0; subsequent
/// forwards stay on the same thread.
///
/// `params` is duck-typed (`anytype`): both `LoadParams` (startup) and
/// `*LoadRequest` (on-demand) supply the same field set — `entry`,
/// `config`/`tok`/`chat_config` (heap pointers), `model_dir`,
/// `drafter_dir`, `load_vision`, `warmup_eager`, `draft_block_size`,
/// `draft_block_size_explicit`, `kv_quant_config`, `prefix_cache_capacity`,
/// `prefix_cache_mem_bytes`. The function reads them by name.
///
/// On any error the partial state has already been freed via errdefer; the
/// caller decides how to surface (startup → recordLoadError + signal
/// started; on-demand → req.error_name + done broadcast).
pub fn doLoadOnInferenceThread(sch: *Scheduler, params: anytype) !void {
    // ── An engine beside MLX: a GGUF on ds4 or llama.cpp (the stub config/tok/chat_config move onto the entry),
    //    or an MLX pack on the Neural Engine. `params` is `LoadParams` (startup) or `*LoadRequest` (cold load).
    const Ty = @TypeOf(params);
    const TyInfo = @typeInfo(Ty);
    const Inner = if (TyInfo == .pointer) TyInfo.pointer.child else Ty;
    params.entry.load_progress.start(io_util.nowMsMonotonic(sch.io));
    const ane_declined = if (@hasField(Inner, "ane_declined")) params.ane_declined else false;
    if (params.gguf == null and engines_served.aneWanted(params.config, params.ane) and !ane_declined and engines_load.aneAdmit(sch, params) and try engines_load.aneOpen(sch, params)) return;
    if (params.gguf) |route| return engines_load.loadGguf(sch, params, route);
    // ── media-gen fast path: the (stub) config's model_type marks an image/
    //    audio/video model (FLUX / Qwen3-TTS / LTX). Build the modality engine
    //    instead of the MLX safetensors path. Works for BOTH the startup
    //    gen-primary path (main builds the stub config) and the cold-load path
    //    (preloadCpuState builds it), since both dispatch off model_type.
    if (gen_mod.modalityFromType(params.config.model_type)) |modality| {
        try doLoadGenOnInferenceThread(sch, params, modality);
        return;
    }
    const model_bytes = params.resident_model_bytes orelse blk: {
        const scanned = try residentModelDiskBytes(sch.allocator, sch.io, params.model_dir, params.config);
        break :blk residentGateBytes(scanned, params.entry.bytes_on_disk, true).?;
    };

    // GPU-memory pre-flight (MLX path). A Metal OOM during weight load / warmup
    // is thrown by MLX as a C++ exception that can't be caught across the C ABI,
    // so it terminates the whole process. Refuse the load up front instead, with
    // an actionable error, when free RAM clearly can't hold the weights + warmup
    // headroom — catches the common "restarted before the prior server released
    // its memory" case. Bypass with --skip-mem-preflight.
    // The sidecar is resolved here so the preflight bills it with the weights:
    // launch flags, then the per-model setting; otherwise the checkpoint's own
    // `drafter/` subdir (dflash.resolveInDirDrafter). That is what makes the
    // drafter a LOAD-time dependency rather than a launch flag: a hot model
    // switch brings its own, and no pairing table has to decide which sidecar
    // goes with which checkpoint.
    const chosen_drafter = drafterFor(params.no_drafter, params.drafter_dir, params.config.drafter_override);
    var in_dir_drafter: ?[]u8 = if (chosen_drafter == null)
        dflash_mod.resolveInDirDrafter(sch.io, sch.allocator, params.model_dir)
    else
        null;
    defer if (in_dir_drafter) |p| sch.allocator.free(p);
    makeRoomInRam(sch, params.entry.id, model_bytes);
    if (!scheduler_mod.skip_mem_preflight) {
        const gb = 1024.0 * 1024.0 * 1024.0;
        const sidecar: []const u8 = chosen_drafter orelse in_dir_drafter orelse "";
        const drafter_bytes: u64 = if (sidecar.len == 0) 0 else drafterResidentBytes(modelDiskBytes(sch.io, sidecar), dflash_mod.sidecarQuantBits(sch.io, sch.allocator, sidecar));
        const mtp_bytes = mtpSidecarDiskBytes(sch.io, sch.allocator, params.model_dir, params.config.mtp_override orelse params.mtp_enabled);
        const weights_bytes = model_bytes + drafter_bytes + mtp_bytes;
        const avail_bytes = availForLoad(sch.io, weights_bytes);
        log.info("[preflight] weights ~{d:.2} GB, available {d:.2} GB\n", .{
            @as(f64, @floatFromInt(model_bytes)) / gb,
            @as(f64, @floatFromInt(avail_bytes)) / gb,
        });
        if (drafter_bytes > 0) log.info("[preflight] drafter ~{d:.2} GB at {s}\n", .{ @as(f64, @floatFromInt(drafter_bytes)) / gb, sidecar });
        if (mtp_bytes > 0) log.info("[preflight] mtp sidecar ~{d:.2} GB\n", .{@as(f64, @floatFromInt(mtp_bytes)) / gb});
        switch (preflightVerdict(model_bytes, drafter_bytes, avail_bytes, in_dir_drafter != null)) {
            .fits => {},
            .drop_drafter => {
                log.warn("[dflash] sidecar at {s} skipped: model + drafter need ~{d:.1} GB free, {d:.1} GB available; the model loads without it\n", .{
                    sidecar,
                    @as(f64, @floatFromInt(loadRequirementBytes(weights_bytes))) / gb,
                    @as(f64, @floatFromInt(avail_bytes)) / gb,
                });
                sch.allocator.free(in_dir_drafter.?);
                in_dir_drafter = null;
            },
            .refuse => {
                log.err("Insufficient memory to load model: needs ~{d:.1} GB free ({d:.1} GB of weights{s} plus headroom for warmup buffers and a baseline KV cache) but only {d:.1} GB is available. Close other models/apps (or wait for a prior mlx-serve to fully exit) and retry; pass --skip-mem-preflight to override.{s}\n", .{
                    @as(f64, @floatFromInt(loadRequirementBytes(weights_bytes))) / gb,
                    @as(f64, @floatFromInt(weights_bytes)) / gb,
                    if (drafter_bytes > 0) " and drafter" else "",
                    @as(f64, @floatFromInt(avail_bytes)) / gb,
                    if (params.config.exl3 != null) " This Sushi pack keeps its experts resident here; `sushi` can stream them from SSD." else "",
                });
                return error.InsufficientMemory;
            },
        }
    }

    {
        const sidecar = chosen_drafter orelse in_dir_drafter orelse "";
        const total = model_bytes + (if (sidecar.len > 0) modelDiskBytes(sch.io, sidecar) else 0);
        params.entry.load_progress.beginReading(total, status.getAppMemFootprintMb(), io_util.nowMsMonotonic(sch.io));
    }

    // Allocate the drafter_path dupe up front so the post-publish step
    // (lower down) has no fallible operations — once we start assigning
    // pointers onto `params.entry`, an OOM during a dupe would leave the
    // entry holding pointers that the per-ptr errdefers would double-free.
    var drafter_path_owned: []u8 = &[_]u8{};
    errdefer if (drafter_path_owned.len > 0) sch.allocator.free(drafter_path_owned);
    if (params.drafter_dir.len > 0) {
        drafter_path_owned = try sch.allocator.dupe(u8, params.drafter_dir);
    }

    // Weights — first mlx call. Binds the stream on this thread.
    const weights_ptr = try sch.allocator.create(Weights);
    errdefer sch.allocator.destroy(weights_ptr);
    // A Thunderbolt pair plans the model before it loads (a type with no multi-Mac path is refused by name there).
    if (params.config.plugin != null) _ = try distributed.planFor(params.model_dir, params.config);
    // A served plugin reads the pack's tensors itself (`plugin/driver.zig`): the entry's map holds only the vision
    // tower, for the encoder below.
    const served = params.config.plugin != null;
    // Only a plugin or an mlx-stream module serves a pack: refuse the rest before reading any weights.
    if (!served and !params.config.mlxStream()) return error.PluginDeclinedNoLegacy;
    weights_ptr.* = if (served) Weights.init(sch.allocator) else try model_mod.loadModelWeights(sch.io, sch.allocator, params.model_dir, params.config, params.load_vision);
    errdefer weights_ptr.deinit();
    if (!served) {
        model_mod.resolveWeightPrefix(params.config, weights_ptr);
        try model_mod.narrowHadamardPackTables(params.config, weights_ptr, mlx.gpuStream());
    }

    // Transformer — owns the bulk of the GPU memory.
    const xfm_ptr = try sch.allocator.create(Transformer);
    errdefer sch.allocator.destroy(xfm_ptr);
    xfm_ptr.* = if (served)
        try plugin_driver.loadWith(sch.io, sch.allocator, params.config, params.model_dir, mlx.gpuStream(), if (params.load_vision) weights_ptr else null)
    else
        try Transformer.init(sch.io, sch.allocator, params.config.*);
    errdefer xfm_ptr.deinit();

    // Reserved-token suppression mask (never sample `<|fim_hole|>`-class
    // specials): derived per model from tokenizer + template + eos.
    generate_mod.installSuppressMask(xfm_ptr, params.tok, params.chat_config.chat_template, params.config.eosTokenSlice());

    // Propagate the kv-quant config to the Transformer's own cache. Slot
    // caches in serve mode honor this independently in `Slot.init`; this
    // call covers any path that still touches `xfm.cache` directly (legacy
    // single-slot fallbacks, prompt-cache reuse).
    // Per-model settings (stamped on the config at BOTH construction sites) outrank the flags.
    const kv_quant_config = params.config.kv_quant_override orelse params.kv_quant_config;
    const mtp_enabled = params.config.mtp_override orelse params.mtp_enabled;
    const tail_override = params.config.mtp_greedy_tail_override;
    if (mtp_enabled and (tail_override != null or generate_mod.mtp_greedy_tail_default)) log.info("[mtp] greedy tail {s} ({s})\n", .{
        if (generate_mod.mtpGreedyTailFor(tail_override)) "on" else "off",
        if (tail_override != null) "model-settings.json" else "--mtp-greedy-tail",
    });
    if (kv_quant_config.scheme != .off) {
        try xfm_ptr.cache.reinit(params.config.num_hidden_layers, kv_quant_config);
    }
    transformer_mod.cli().mtp_head_kv_quant = params.mtp_head_kv_quant;
    if (xfm_ptr.qwen4Trunk()) |t| try t.qwen4MtpApplyKvQuant(Transformer.qwen4MtpHeadKvConfig(kv_quant_config));

    // Wire model weights into GPU memory (prevents paging, matches mlx-lm).
    // Policy in mlx.applyWiredPolicy; re-applied in runLoadRequest /
    // runUnloadRequest so `fit` capacity tracks the live set across
    // load/unload churn (this early call covers warmup's forwards).
    logWiredPolicy(mlx.applyWiredPolicy());

    // JIT-compile activation kernels. These are bound to THIS thread's mlx
    // stream — that's exactly the point of doing them here. Skipped entirely
    // when there's no GPU backend (iOS Simulator's CPU-only MLX): mlx_compile
    // is Metal kernel fusion and requests a GPU stream that doesn't exist; the
    // runtime falls back to the uncompiled paths when compiled_* stays null. A served shell forwards through its
    // plugin, which compiles its own; nothing on the shell reads these.
    if (!mlx.noGpuBackend() and !served) {
        if (params.config.hidden_act == .gelu_approx) {
            xfm_ptr.compileGelu();
            xfm_ptr.compileGeglu();
        }
        if (xfm_ptr.moe_layers != null) {
            xfm_ptr.compileMoeRouting();
        }
        if (params.config.linear_num_key_heads > 0) {
            xfm_ptr.compileGdnGate();
        }
    }

    // Vision encoder if requested. `MissingVisionWeights` is a benign opt-out
    // (model declares vision in config but the safetensors didn't ship the
    // tower); other errors fail the whole load.
    var vision_ptr: ?*VisionEncoder = null;
    // An outside plugin's tower is the plugin's own (`MediaHooks.tower`), not a pack's tensors the engine reads.
    if (params.load_vision) if (xfm_ptr.plugin) |b| if (params.config.plugin.?.parsed.caps.images and params.config.plugin.?.outside) if (b.table.media) |hooks| {
        const v = try sch.allocator.create(VisionEncoder);
        v.* = VisionEncoder.initPlugin(sch.allocator, params.config.*, hooks, b.model);
        vision_ptr = v;
    };
    // An in-tree plugin's towers run on the engine: decoded media in, its own prepare and tower.
    if (params.load_vision and vision_ptr == null) if (pluginMedia(sch.allocator, params.config, xfm_ptr)) |pm| {
        const v = try sch.allocator.create(VisionEncoder);
        v.* = VisionEncoder.initPluginMedia(sch.allocator, params.config.*, pm);
        vision_ptr = v;
    };
    if (params.load_vision and vision_ptr == null) {
        const v = try sch.allocator.create(VisionEncoder);
        if (VisionEncoder.init(sch.allocator, params.config.*, weights_ptr)) |encoder| {
            v.* = encoder;
            vision_ptr = v;
        } else |err| {
            sch.allocator.destroy(v);
            if (err == error.MissingVisionWeights) {
                log.warn("Vision weights missing — vision disabled (model may have been quantized without vision tower)\n", .{});
            } else {
                return err;
            }
        }
    }
    errdefer if (vision_ptr) |v| {
        v.deinit();
        sch.allocator.destroy(v);
    };

    // The measured round-cost table (`round_cost.zig`) is keyed per (chip,
    // model, quant, OS build); restored here, written at request end.
    {
        var quant_buf: [32]u8 = undefined;
        const quant = std.fmt.bufPrint(&quant_buf, "q{d}g{d}", .{
            params.config.quant_bits,
            params.config.quant_group_size,
        }) catch "q?";
        var os_buf: [64]u8 = undefined;
        const os_build = transformer_mod.macosProductVersion(&os_buf) orelse "";
        // The measured round-cost table rides the same identity: restored
        // here, written at the end of any request that folded new samples.
        // The bucket grid and store version are the arch's (only qwen4_exp gets the long
        // grid); every other arch keeps the `rc1` table 26.9.1 wrote and boots warm.
        const rc_layout: round_cost_mod.Layout = round_cost_mod.layoutFor(params.config);
        xfm_ptr.round_cost.layout = rc_layout;
        const rc_key = round_cost_mod.cacheKey(&xfm_ptr.round_cost_key_buf, ane_mod.chipBrand(), params.model_dir, quant, os_build, rc_layout, round_cost_mod.engineBuildId());
        xfm_ptr.round_cost_key_len = @intCast(rc_key.len);
        if (round_cost_mod.loadCached(sch.allocator, sch.io, rc_key, rc_layout)) |t| {
            xfm_ptr.round_cost = t;
            log.info("[spec-cost] round-cost table restored ({d} width cells, {d} serial cells)\n", .{ t.restored, t.restored_serial });
            if (t.restored_dropped > 0) log.info("[spec-cost] dropped {d} implausible persisted cell(s)\n", .{t.restored_dropped});
        }
    }

    // Assistant sidecar (optional). Loaded only when `drafter_dir` is
    // non-empty. The sidecar KIND is decided by its config CONTRACT: a
    // config declaring block_size + mask_token_id + target_layer_ids is a
    // DFlash block-drafter (any `*_assistant` family); anything else goes
    // to the Gemma cross-attention drafter loader.
    var drafter_ptr: ?*DrafterModel = null;
    var dflash_ptr: ?*DflashModel = null;
    errdefer if (drafter_ptr) |d| {
        d.deinit();
        sch.allocator.destroy(d);
    };
    errdefer if (dflash_ptr) |d| {
        d.deinit();
        sch.allocator.destroy(d);
    };
    const drafter_found: []const u8 = chosen_drafter orelse in_dir_drafter orelse "";
    // A served model without the trunk's table and head (or layer capture) serves without its drafter.
    const drafter_dir: []const u8 = if (served and drafter_found.len > 0 and !xfm_ptr.plugin.?.drafters) blk: {
        log.warn("[plugin] {s}: the drafter at {s} needs the plugin's trunk_weight and layer capture: served without it\n", .{ xfm_ptr.plugin.?.table.name.slice(), drafter_found });
        break :blk "";
    } else drafter_found;
    if (drafter_dir.len > 0 and dflash_mod.probeIsDflash(sch.io, sch.allocator, drafter_dir)) {
        {
            const d = try sch.allocator.create(DflashModel);
            d.* = dflash_mod.loadDflash(sch.io, sch.allocator, mlx.gpuStream(), drafter_dir) catch |err| {
                sch.allocator.destroy(d);
                log.err("Failed to load DFlash assistant at {s}: {s}\n", .{ drafter_dir, @errorName(err) });
                return err;
            };
            d.bind(xfm_ptr) catch |err| {
                d.deinit();
                sch.allocator.destroy(d);
                log.err(
                    "DFlash assistant at {s} is incompatible with target: {s}\n" ++
                        "  (assistant+target must share hidden_size, the mask token and\n" ++
                        "  target_layer_ids must exist in the target, and the target must\n" ++
                        "  run the standard dense-attention forward path)\n",
                    .{ drafter_dir, @errorName(err) },
                );
                return err;
            };
            dflash_ptr = d;
            const wide_lane = dflash_mod.wideVerifyLaneAvailable() or xfm_ptr.ternaryWideVerify();
            const block_cap = dflash_mod.blockCapForMachine(ane_mod.chipBrand(), d.selector != null and xfm_ptr.specTreeSupported());
            sch.drafter_block_size = dflash_mod.resolveBlockSize(
                d.config.block_size,
                params.draft_block_size,
                params.draft_block_size_explicit,
                wide_lane,
                block_cap.cap,
            );
            if (params.draft_block_size_explicit and params.draft_block_size > sch.drafter_block_size)
                log.warn("--draft-block-size {d} is past the drafter's trained block; using {d}\n", .{ params.draft_block_size, sch.drafter_block_size });
            // Trees on the tensor units draft past the drafter's trained block (TensorFold's
            // tree_block): the lattice reaches depth 15 and a 16-row window costs about an 8-row one.
            if (!params.draft_block_size_explicit and d.selector != null and xfm_ptr.specTreeSupported() and transformer_mod.naxAvailable())
                sch.drafter_block_size = dflash_mod.TREE_NAX_BLOCK;
            var cap_note_buf: [96]u8 = undefined;
            const cap_note: []const u8 = if (params.draft_block_size_explicit)
                ", user-clamped"
            else if (!wide_lane and d.config.block_size > sch.drafter_block_size)
                std.fmt.bufPrint(&cap_note_buf, ", capped ({s} cap {d})", .{
                    block_cap.label,
                    block_cap.cap,
                }) catch ", capped"
            else
                "";
            log.info("DFlash drafter ready (block_size={d}{s}, wide_verify_lane={}, targets={any}).\n", .{
                sch.drafter_block_size,
                cap_note,
                wide_lane,
                d.config.target_layer_ids,
            });
            const tiled_bytes = try xfm_ptr.tileLaneWeights();
            if (tiled_bytes > 0) log.info("[lane] trunk projections tiled in place ({d:.1} GB, no copy)\n", .{@as(f64, @floatFromInt(tiled_bytes)) / (1 << 30)});
            const drafter_tiled = try d.tileLaneWeights(xfm_ptr.actDtype(), mlx.gpuStream());
            if (drafter_tiled > 0) log.info("[lane] drafter projections tiled in place ({d:.2} GB, no copy)\n", .{@as(f64, @floatFromInt(drafter_tiled)) / (1 << 30)});
        }
    } else if (drafter_dir.len > 0) {
        const d = try sch.allocator.create(DrafterModel);
        d.* = drafter_mod.loadDrafter(sch.io, sch.allocator, mlx.gpuStream(), drafter_dir) catch |err| {
            sch.allocator.destroy(d);
            log.err("Failed to load drafter at {s}: {s}\n", .{ drafter_dir, @errorName(err) });
            return err;
        };
        d.bind(xfm_ptr) catch |err| {
            d.deinit();
            sch.allocator.destroy(d);
            log.err(
                "Drafter checkpoint at {s} is incompatible with target: {s}\n" ++
                    "  (drafter+target must share backbone_hidden_size, vocab_size, and have\n" ++
                    "  matching layer types in the target's non-shared K/V layers)\n",
                .{ drafter_dir, @errorName(err) },
            );
            return err;
        };
        drafter_ptr = d;

        // Auto-detect block_size unless the user pinned it explicitly.
        if (!params.draft_block_size_explicit) {
            const auto_bs = drafter_mod.recommendedBlockSize(params.config);
            sch.drafter_block_size = auto_bs;
            log.info(
                "Drafter ready (block_size={d}, auto-detected for {s}/{d}-layer{s}).\n",
                .{
                    auto_bs,
                    params.config.model_type,
                    params.config.num_hidden_layers,
                    if (params.config.isMoe()) ",moe" else "",
                },
            );
        } else {
            log.info("Drafter ready (block_size={d}, user override).\n", .{params.draft_block_size});
        }

        if (params.config.isMoe()) {
            log.warn(
                "Drafter loaded but target is MoE ({s}); per-request " ++
                    "enable_drafter defaults to OFF — drafter+MoE regresses " ++
                    "at single-stream batch=1 (verify forward expert-routing " ++
                    "penalty). Pass enable_drafter:true per request to opt-in.\n",
                .{params.config.model_type},
            );
        }
    }
    params.config.row_exact_covered = xfm_ptr.config.row_exact_covered;
    params.config.dflash_bound = xfm_ptr.config.dflash_bound;
    if (params.config.dflash_bound and std.meta.activeTag(params.config.mtp_acceptance_override orelse generate_mod.mtp_acceptance_default) != .exact)
        log.info("[dflash] MTP acceptance forced to exact while the drafter is bound\n", .{});

    // After the drafter binds: an exact-mode arch runs its row kernels only then.
    // DIAGNOSTIC (diag decode-fwd-ubench=N): time N decode-width forward
    // passes back to back, with NO sampling, detokenization, stop-checking or
    // cache bookkeeping around them. The server reports `predicted_ms` around
    // the whole decode LOOP, so this is the only way to say how much of a
    // token is the model and how much is everything else. Resets the KV cache
    // afterwards so the probe cannot pollute real requests.
    if (env.get(.DECODE_FWD_UBENCH)) |raw| {
        const n = std.fmt.parseInt(usize, std.mem.sliceTo(raw, 0), 10) catch 0;
        if (n > 0) {
            const io_u = @import("../../sdk/util/io_util.zig");
            const tio = std.Io.Threaded.global_single_threaded.io();
            var ctx = xfm_ptr.defaultCtx();
            // diag decode-fwd-ubench-s=<rows>: verify-width forwards
            // (per-position SSM capture on, as spec verify runs them).
            // diag decode-fwd-ubench-kv=<tokens>: prefill that many
            // tokens first so the meter runs at a real context length.
            const rows: usize = blk: {
                const r = env.get(.DECODE_FWD_UBENCH_S) orelse break :blk 1;
                break :blk @max(1, std.fmt.parseInt(usize, std.mem.sliceTo(r, 0), 10) catch 1);
            };
            const kv_pre: usize = blk: {
                const r = env.get(.DECODE_FWD_UBENCH_KV) orelse break :blk 0;
                break :blk std.fmt.parseInt(usize, std.mem.sliceTo(r, 0), 10) catch 0;
            };
            const tok_slice = try sch.allocator.alloc(i32, @min(rows, 4096));
            defer sch.allocator.free(tok_slice);
            for (tok_slice, 0..) |*v, i| v.* = @intCast(1 + (i % 997));
            const tok = tok_slice.ptr;
            const tsh = [_]c_int{ 1, @intCast(tok_slice.len) };
            if (kv_pre > 0) {
                var done_pre: usize = 0;
                const pre_buf = try sch.allocator.alloc(i32, 2048);
                defer sch.allocator.free(pre_buf);
                for (pre_buf, 0..) |*v, i| v.* = @intCast(1 + (i % 1000));
                while (done_pre < kv_pre) {
                    const n_chunk = @min(2048, kv_pre - done_pre);
                    const psh = [_]c_int{ 1, @intCast(n_chunk) };
                    const ti = mlx.mlx_array_new_data(pre_buf.ptr, &psh, 2, .int32);
                    defer _ = mlx.mlx_array_free(ti);
                    const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                    _ = mlx.mlx_array_eval(lg);
                    _ = mlx.mlx_array_free(lg);
                    done_pre += n_chunk;
                }
                log.info("[fwd-ubench] prefilled {d} tokens\n", .{done_pre});
            }
            // Rows past 16 are a prefill chunk unless `_KV` says a context stands behind them (verify widths up to 32).
            const prefill_rows = rows > 16 and kv_pre == 0;
            ctx.capture_ssm_seq = rows > 1 and rows <= 32 and !prefill_rows and ctx.ssm_entries != null; // verify widths capture, prefill chunks do not
            // diag decode-fwd-ubench-tree at 16 rows: verify a fixed draft
            // tree (a 9-row trunk, siblings at depths 1-4, their children), as a round does.
            const tree_parents = [16]i32{ -1, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 9, 10, 11 };
            var tree_depth: [16]i32 = undefined;
            for (tree_parents, 0..) |p, i| tree_depth[i] = if (p < 0) 0 else tree_depth[@intCast(p)] + 1;
            const spec_tree = generate_mod.Generator.specTreeFor(&tree_parents, &tree_depth, 8);
            defer spec_tree.deinit();
            if (rows == 16 and env.get(.DECODE_FWD_UBENCH_TREE) != null) ctx.tree = &spec_tree;
            // Prefill widths: every forward starts from an empty cache (else each
            // one attends over the previous ones' rows) and skips the lm_head,
            // which a real intermediate chunk never evaluates.
            ctx.skip_lm_head = prefill_rows;
            log.info("[fwd-ubench] rows={d} capture={} prefill={}\n", .{ tok_slice.len, ctx.capture_ssm_seq, prefill_rows });
            // Warm: first forward pays kernel JIT + lazy weight materialization.
            for (0..3) |_| {
                if (prefill_rows) try xfm_ptr.resetCache();
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                _ = mlx.mlx_array_eval(lg);
                _ = mlx.mlx_array_free(lg);
            }
            // DIAGNOSTIC (diag decode-graph-dump=<path>): print the lazy
            // graph of ONE forward (every primitive with its shape) before it
            // is evaluated, to map which ops a decode step dispatches.
            if (env.get(.DECODE_GRAPH_DUMP)) |path| {
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                if (xfm_ptr.forwardWith(&ctx, ti)) |lg| {
                    defer _ = mlx.mlx_array_free(lg);
                    if (std.c.fopen(path, "w")) |f| {
                        const outs = mlx.mlx_vector_array_new_value(lg);
                        defer _ = mlx.mlx_vector_array_free(outs);
                        const namer = mlx.mlx_node_namer_new();
                        defer _ = mlx.mlx_node_namer_free(namer);
                        _ = mlx.mlx_print_graph(f, namer, outs);
                        _ = std.c.fclose(f);
                        log.info("[fwd-ubench] graph dumped to {s}\n", .{std.mem.sliceTo(path, 0)});
                    }
                    _ = mlx.mlx_array_eval(lg);
                } else |_| {}
            }
            // Split CPU graph CONSTRUCTION from GPU execution. MLX is lazy, so
            // `forwardWith` only issues ops — if that half dominates, the token
            // is bounded by op count / FFI overhead, not by memory bandwidth,
            // and no kernel-level optimization can reach it.
            var sw = io_u.Stopwatch.init(tio);
            var build_ns: u64 = 0;
            var eval_ns: u64 = 0;
            var encode_ns: u64 = 0;
            var ops_total: u64 = 0;
            var done: usize = 0;
            for (0..n) |_| {
                if (prefill_rows) try xfm_ptr.resetCache();
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const ops_before = mlx.op_count.load(.monotonic);
                var swb = io_u.Stopwatch.init(tio);
                const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                build_ns += swb.read();
                ops_total += mlx.op_count.load(.monotonic) - ops_before;
                // The eval is two host phases: encoding every kernel (MLX runs `eval_gpu` on the
                // calling thread, throttled to MAX_ACTIVE_TASKS open command buffers), then the
                // wait for the GPU. The split says which side bounds the forward.
                var swe = io_u.Stopwatch.init(tio);
                const lgv = mlx.mlx_vector_array_new_value(lg);
                _ = mlx.mlx_async_eval(lgv);
                _ = mlx.mlx_vector_array_free(lgv);
                encode_ns += swe.read();
                _ = mlx.mlx_array_eval(lg);
                eval_ns += swe.read();
                _ = mlx.mlx_array_free(lg);
                done += 1;
            }
            const dn: f64 = @floatFromInt(@max(done, 1));
            const ms = if (prefill_rows) @as(f64, @floatFromInt(build_ns + eval_ns)) / 1.0e6 / dn else @as(f64, @floatFromInt(sw.read())) / 1.0e6 / dn;
            transformer_mod.decodeProfReport();
            log.info("[fwd-ubench] {d} decode forwards, eval-per-step: {d:.3} ms/forward (build {d:.3} ms CPU + eval {d:.3} ms GPU [encode {d:.3} + wait {d:.3}], {d:.0} ops/forward)\n", .{
                done,
                ms,
                @as(f64, @floatFromInt(build_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(eval_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(encode_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(eval_ns - encode_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(ops_total)) / dn,
            });

            // Same forward with the vocab projection suppressed. lm_head is
            // terminal — nothing downstream depends on it — so dropping it
            // cannot change the work the rest of the graph does, which makes
            // this the one sound ablation in the probe.
            ctx.skip_lm_head = true;
            for (0..3) |_| {
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                _ = mlx.mlx_array_eval(lg);
                _ = mlx.mlx_array_free(lg);
            }
            var sw_nolm = io_u.Stopwatch.init(tio);
            var done_nolm: usize = 0;
            for (0..n) |_| {
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                _ = mlx.mlx_array_eval(lg);
                _ = mlx.mlx_array_free(lg);
                done_nolm += 1;
            }
            const ms_nolm = @as(f64, @floatFromInt(sw_nolm.read())) / 1.0e6 / @as(f64, @floatFromInt(@max(done_nolm, 1)));
            ctx.skip_lm_head = false;
            log.info("[fwd-ubench] without lm_head: {d:.3} ms/forward  => lm_head = {d:.3} ms\n", .{ ms_nolm, ms - ms_nolm });
            xfm_ptr.diagProjBench(20, &ctx);
            log.info("[fwd-ubench] done\n", .{});
            xfm_ptr.resetCache() catch {};
        }
    }

    // Qwen native MTP head (optional). Auto-loaded when the model dir ships
    // one — an `mtp/weights.safetensors`-class sidecar file OR in-checkpoint
    // `[language_model.]mtp.*` tensors in the trunk shards; a failed load or
    // bind only disables the head — the model still serves.
    var mtp_ptr: ?*mtp_mod.MtpModel = null;
    var mtp_cost_profile: mtp_mod.MtpCostProfile = .generic;
    if (mtp_enabled) mtp_graft.ensure(sch.allocator, sch.io, params.model_dir, params.config);
    // A served model's head is its plugin's (`SpecHooks.head`), never a legacy arm reading the shell's config.
    var plugin_mtp: ?*plugin_head.Head = null;
    if (served) {
        if (mtp_enabled) plugin_mtp = plugin_head.load(sch.io, sch.allocator, xfm_ptr, params.model_dir);
        if (plugin_mtp) |h| {
            mtp_cost_profile = h.m5NaxCostProfile(xfm_ptr);
            log.info("MTP head ready (depth={d}, profile={s}).\n", .{
                generate_mod.Generator.resolveMtpDepthCapForProfile(params.config.mtpDepth(params.mtp_depth), mtp_cost_profile),
                @tagName(mtp_cost_profile),
            });
        }
    } else if (mtp_enabled and mtp_mod.hasMtpHead(sch.io, sch.allocator, params.model_dir)) {
        if (sch.allocator.create(mtp_mod.MtpModel)) |h| {
            if (mtp_mod.loadMtp(sch.io, sch.allocator, mlx.gpuStream(), params.model_dir)) |loaded| {
                h.* = loaded;
                if (h.bind(xfm_ptr)) {
                    mtp_ptr = h;
                    mtp_cost_profile = h.m5NaxCostProfile(xfm_ptr);
                    // Price the DRAFT side. The verify ladder above measures
                    // the trunk forward and nothing else, but an m-deep round
                    // is that forward PLUS m sequential head steps — which on
                    // a 27B dominate the per-position marginal. Fitting the
                    // EV surface without this under-prices depth ~9x (live
                    // forward marginal 0.8 ms/position against a
                    // hand-measured composite of 7.6). A cached curve already
                    // carries it, so this is paid once per (chip, model,
                    // quant, OS build) like the ladder itself.
                    log.info("MTP head ready (depth={d}, profile={s}).\n", .{
                        generate_mod.Generator.resolveMtpDepthCapForProfile(params.config.mtpDepth(params.mtp_depth), mtp_cost_profile),
                        @tagName(mtp_cost_profile),
                    });
                } else |bind_err| {
                    log.warn("MTP sidecar incompatible with target ({s}) — disabled.\n", .{@errorName(bind_err)});
                    h.deinit();
                    sch.allocator.destroy(h);
                }
            } else |load_err| {
                log.warn("Failed to load MTP sidecar: {s} — disabled.\n", .{@errorName(load_err)});
                sch.allocator.destroy(h);
            }
        } else |_| {}
    } else if (mtp_enabled) {
        // A quiet fallback to mode=pld cost a tester a day: nothing logged
        // when the probe finds no head. Debug-level — most checkpoints have
        // no MTP head and an info line per load would be noise.
        log.debug(
            "[mtp] no head found: no mtp/ sidecar and no [language_model.]mtp.* " ++
                "keys resolvable from the index at {s} — MTP off\n",
            .{params.model_dir},
        );
    }
    errdefer if (mtp_ptr) |h| {
        h.deinit();
        sch.allocator.destroy(h);
    };
    errdefer if (plugin_mtp) |h| plugin_head.destroy(h, sch.allocator);

    // The qwen4_exp in-checkpoint head's twin of the sidecar's own coarse
    // rerank build (inside `bind`, above): a `requantizeRows` of the whole
    // trunk lm_head plus a synchronous eval of ~240 MB. Lazily it ran inside
    // the FIRST request's draft chain — on this thread, mid-round, with the
    // stream drained — i.e. first-token latency for whoever loaded the model.
    // Gated exactly like `entry.mtp`'s `.qwen4` arm below: `--no-mtp` never
    // drafts, so it never pays. One-shot, so the draft path's ask stays a
    // pure read; if this ever does not run, that ask still builds.
    if (mtp_ptr == null and mtp_enabled) if (xfm_ptr.qwen4Trunk()) |t| {
        _ = modules.call("qwen4_build_draft_rerank", .{t});
    };

    // ANE prefill-MLP offload (`--ane-prefill`, perf-plan-aug-17 P5): built
    // HERE because the mlx dequant must run on the inference thread (sole
    // MLX caller), with the chunk width resolved through the server's own
    // pin (idempotent — the later pinAutoContext keeps this value), so the
    // compiled fixed-shape tile matches the width the forward will run.
    if (params.ane_prefill) {
        const ane_force: ?[]const u8 = env.span(.MLX_SERVE_ANE_FORCE);
        if (!ane_mod.anePrefillAllowed(transformer_mod.verifyQmmNaxAvailable(), ane_force)) {
            // ANE prefill is M4-and-below: on NAX machines it measured a
            // loss (M5 Max). `/props` ane stays absent, as off.
            log.info(
                "[ane] --ane-prefill disabled: NAX-class GPU prefill already outruns the ANE seam " ++
                    "(measured a loss on M5 Max, PR #223); MLX_SERVE_ANE_FORCE=1 overrides\n",
                .{},
            );
        } else if (sch.ane_chunk_resolver) |resolve| {
            const pinned = resolve(@constCast(params.config));
            // The forward's chunk is the pinned width run through the SAME
            // per-request policy every prefill applies (effectivePrefillChunk:
            // the MoE 4096 / dense-hd-256 8192 caps + the --prefill-chunk and
            // env overrides) — compiling the tile at the pinned width alone
            // left every MoE program built at 8192 while the forward chunked
            // at 4096: built, never dispatched (A7). total_ctx is
            // representative-large: under the default fused-causal mode the
            // policy arm is ctx-independent, and under the composed fallback
            // the chunk is ctx-dependent anyway (fixed shapes cannot follow
            // it, and the seam's width equality just never engages).
            const cfg = params.config;
            const chunk: u32 = @intCast(generate_mod.effectivePrefillChunk(
                cfg.prefillScoreHeadDim(),
                cfg.num_attention_heads,
                1 << 20,
                cfg.has_sliding_window,
                cfg.isMoe(),
                cfg.longCtxGated(),
                pinned,
            ));
            xfm_ptr.buildAnePrefill(sch.io, chunk, ane_mod.splitShare(), sch.ane_headroom_resolver);
        } else {
            log.warn("[ane] --ane-prefill: no prefill-chunk resolver on this load path — disabled\n", .{});
        }
    }

    // Eager warmup: faults weight pages + compiles the decode-path kernels
    // on this thread's stream. ~600-900 ms at boot but the first user request
    // skips a cold path — observed savings on Gemma 4 E4B 4-bit.
    if (params.warmup_eager) {
        params.entry.load_progress.begin(.warming_up, 0, io_util.nowMsMonotonic(sch.io));
        const warmup_start = std.Io.Timestamp.now(sch.io, .awake);
        xfm_ptr.warmup() catch |err| {
            log.warn("Warmup failed ({s}); continuing without it — first request may be slow.\n", .{@errorName(err)});
        };
        const warmup_ns: u64 = @intCast(warmup_start.untilNow(sch.io, .awake).nanoseconds);
        log.info("Warmup complete ({d} ms).\n", .{warmup_ns / std.time.ns_per_ms});
        if (mtp_enabled) if (xfm_ptr.qwen4Trunk()) |t| if (t.qwen4_mtp != null) {
            const cap = generate_mod.Generator.resolveMtpDepthCapForProfile(params.mtp_depth, mtp_cost_profile);
            modules.call("qwen4_warmup_spec_verify", .{ t, cap, params.kv_quant_config }) catch |err| {
                log.warn("[spec-warmup] failed ({s}); the first round at each width pays its kernel compile inside the round.\n", .{@errorName(err)});
            };
        };
    }

    // ── Phase 05: install everything onto the LoadedModel entry, mark
    //    ready, and update the scheduler's borrowed views. The registry
    //    mutex guards the state transition + `current_resident_bytes`
    //    accounting; `state_cond.broadcast` (inside markReadyLocked) wakes
    //    any waiter blocked in ensureLoaded.
    //
    //    Everything below this comment must be infallible — once we begin
    //    assigning to `params.entry`, the per-ptr errdefers above would
    //    double-free if we error-return. `drafter_path_owned` was alloc'd
    //    up front for exactly this reason; the per-model prefix cache
    //    init is a struct literal (no fallible alloc).
    const entry = params.entry;
    entry.weights = weights_ptr;
    entry.transformer = xfm_ptr;
    entry.vision_encoder = vision_ptr;
    entry.drafter = drafter_ptr;
    entry.dflash = dflash_ptr;
    if (entry.config) |c| c.drafter_ctx_bytes_per_token = if (dflash_ptr) |d| dflash_mod.ctxBytesPerToken(&d.config) else 0;
    entry.drafter_block_size = sch.drafter_block_size;
    const head_ref: ?generate_mod.MtpHeadRef = if (mtp_ptr) |h|
        generate_mod.MtpHeadRef{ .qwen = h }
    else if (plugin_mtp) |h|
        generate_mod.MtpHeadRef{ .plugin = h }
    else if (mtp_enabled and xfm_ptr.qwen4Trunk() != null and xfm_ptr.qwen4Trunk().?.qwen4_mtp != null)
        generate_mod.MtpHeadRef{ .qwen4 = xfm_ptr.qwen4Trunk().? }
    else
        null;
    entry.mtp = if (head_ref) |r| blk: {
        const p = try sch.allocator.create(generate_mod.MtpHeadRef);
        p.* = r;
        break :blk p;
    } else null;
    // Resolve the auto (0) cap here so every downstream reader of
    // `lm.mtp_depth` (server log lines, slot params) sees the real value.
    const mtp_depth_cfg = params.config.mtpDepth(params.mtp_depth);
    entry.mtp_depth = generate_mod.Generator.resolveMtpDepthCapForProfile(mtp_depth_cfg, mtp_cost_profile);
    xfm_ptr.mtp_depth_free = generate_mod.Generator.mtpDepthCapFree(mtp_depth_cfg);
    // A MERGED drafter has no `--drafter` to echo, so the reported path comes
    // from what was actually resolved — `drafter_loaded` and `drafter_path`
    // must not disagree about the same sidecar.
    if (drafter_path_owned.len == 0 and drafter_dir.len > 0 and
        (dflash_ptr != null or drafter_ptr != null))
    {
        drafter_path_owned = try sch.allocator.dupe(u8, drafter_dir);
    }
    entry.drafter_path = drafter_path_owned;
    drafter_path_owned = &[_]u8{}; // disarm the errdefer
    // Transfer ownership of the heap-allocated CPU state from `params` to
    // the entry. The caller (main.zig) MUST NOT free these — `LoadedModel.deinit`
    // walks them in the same `*X` pointer form they came in.
    entry.releaseRetainedCpuState();
    entry.config = params.config;
    entry.tokenizer = params.tok;
    entry.chat_config = params.chat_config;
    // Per-model hot prefix cache (Plan 03 → Plan 05 move). Hybrid recurrent
    // archs are accepted iff `ssm_checkpoint_stride > 0` (Phase 1 of the
    // performance plan): with per-stride SSM checkpoints we can rewind both
    // KV and SSM to a snapshotted prefix; without them, divergence forces a
    // full reset, so we keep the legacy single-slot path for hybrid.
    const enable_ssm_cps = params.ssm_checkpoint_stride > 0;
    const ram_prefix_cache = params.prefix_cache_ram_enabled;
    const disk_prefix_cache = params.prefix_cache_disk_bytes > 0;
    const module_binding = prefix_cache_mod.ModuleBinding.init(xfm_ptr);
    if (plugin_driver.auxBinding(xfm_ptr) != null and !plugin_driver.auxEnabled(xfm_ptr)) {
        log.info("[hot-cache] module auxiliary retention is disabled for this loaded instance; using legacy resume\n", .{});
    }
    // Such a model keeps no conversation of its own any more: without a RAM prefix cache every turn prefills cold.
    if (module_binding != null and (params.prefix_cache_capacity == 0 or !ram_prefix_cache)) {
        log.warn("[hot-cache] no RAM prefix cache (--prefix-cache-entries 0 or --no-prefix-cache-ram): this model retains no conversation, every prompt prefills cold\n", .{});
    }
    if (params.prefix_cache_capacity > 0 and (ram_prefix_cache or disk_prefix_cache) and
        prefix_cache_mod.HotPrefixCache.shouldUseInstance(params.config, enable_ssm_cps, xfm_ptr) and
        (module_binding == null or ram_prefix_cache))
    {
        // The weights are resident here, so the resolver's active-memory read
        // is honest; the raw launch budget never reaches initWithMem (a 40 GB
        // cap beside a ~70 GB pack was the uncatchable Metal OOM).
        // Disk-only mode retains no reusable KV in RAM, so it asks for no RAM budget.
        var ssd_idle_mem: u64 = 0;
        const clamped_prefix_mem: u64 = if (!ram_prefix_cache)
            0
        else if (sch.prefix_cache_mem_resolver) |resolve|
            resolve(params.config, params.prefix_cache_mem_bytes, .{}, &ssd_idle_mem)
        else
            params.prefix_cache_mem_bytes;
        const new_hc = try sch.allocator.create(prefix_cache_mod.HotPrefixCache);
        new_hc.* = prefix_cache_mod.HotPrefixCache.initWithMem(
            sch.allocator,
            if (ram_prefix_cache) params.prefix_cache_capacity else 0,
            clamped_prefix_mem,
        );
        entry.prefix_cache = new_hc;
        entry.prefix_cache.?.qsa_history_required = params.config.indexer_budget != 0;
        // Checkpoint-retention arch gate, mirrored once: `HotPrefixCache`/`DiskTier` never
        // see a ModelConfig. The ungated value names the previous behaviour at each site.
        entry.prefix_cache.?.cp_thin = if (params.config.longCtxGated()) .min_span_recency else .min_span;
        // Same gate, same constant as the disk tier below (#794):
        // merge/shed thinning never migrates the grid floor upward on the long-context line,
        // so the dense low anchors survive from generation all the way to disk.
        entry.prefix_cache.?.ssm_low_anchors = if (params.config.longCtxGated()) kv_disk_cache.SSM_DISK_LOW_ANCHORS else 0;
        entry.prefix_cache.?.ssd_idle_mem = ssd_idle_mem;
        // SSD tier (`--prefix-cache-disk`). Phase 3 persists hybrid recurrent
        // state too: the disk tier is allowed whenever the RAM tier accepted
        // the arch — i.e. pure-attention always, hybrid iff SSM checkpoints
        // are enabled (`enable_ssm_cps`, the same gate `shouldUse` applied).
        // Every failure mode is caught: persistence silently stays off, the
        // RAM cache is unaffected.
        const has_ssm_layers = params.config.hasSsmLayers();
        entry.prefix_cache.?.hybrid = has_ssm_layers;
        const disk_ok = module_binding != null or !has_ssm_layers or enable_ssm_cps;
        if (params.prefix_cache_disk_bytes > 0 and disk_ok) attach: {
            const fp = kv_disk_cache.modelFingerprintWithLayout(
                sch.allocator,
                sch.io,
                entry.path,
                params.config.cacheLayoutNamespace(),
            ) catch |err| {
                log.warn("[disk-cache] fingerprint failed: {s} — persistence off for this model\n", .{@errorName(err)});
                break :attach;
            };
            defer sch.allocator.free(fp);
            const base = kv_disk_cache.defaultBaseDir(sch.allocator) catch break :attach;
            defer sch.allocator.free(base);
            entry.prefix_cache.?.disk = initPrefixDisk(
                sch.allocator,
                sch.io,
                base,
                fp,
                params.prefix_cache_disk_bytes,
                kv_disk_cache.DEFAULT_CHUNK_TOKENS,
                module_binding,
                entry.path,
            ) catch |err| {
                log.warn("[disk-cache] init failed: {s} — persistence off for this model\n", .{@errorName(err)});
                break :attach;
            };
            entry.prefix_cache.?.disk.?.cp_thin =
                if (params.config.longCtxGated() or !ram_prefix_cache) .min_span_recency else .oldest;
            entry.prefix_cache.?.disk.?.ssm_max_per_entry = if (params.config.longCtxGated() or !ram_prefix_cache)
                kv_disk_cache.SSM_DISK_MAX_PER_ENTRY
            else
                kv_disk_cache.SSM_DISK_MAX_PER_ENTRY_LEGACY;
            // Same gate: the long-context thin keeps the two lowest checkpoints so a
            // prompt that forked early (a client-side injection block) still has a
            // shallow restore anchor; every other arch keeps the plain spread.
            entry.prefix_cache.?.disk.?.ssm_low_anchors = if (params.config.longCtxGated() or !ram_prefix_cache)
                kv_disk_cache.SSM_DISK_LOW_ANCHORS
            else
                0;
        }
        // SSD-first: arch + env switch + a live disk tier. Below the attach because the tier
        // is part of the answer; without `--prefix-cache-disk` qwen4_exp takes the RAM arm.
        entry.prefix_cache.?.armDiskTier(params.config);
        entry.ssm_checkpoint_stride = params.ssm_checkpoint_stride;
        entry.ssm_checkpoint_max = params.ssm_checkpoint_max;
        // The cache re-applies the cap after a replace-path merge; without this
        // it defaults to 0 (unlimited) and multi-turn entries grow unbounded.
        entry.prefix_cache.?.ssm_checkpoint_max = params.ssm_checkpoint_max;
    }
    // Iteration 2 (perf-plan Phase 4 #3): tokenize cache for warm-path
    // chat-template renders. Applies to every engine — they all funnel
    // through `cachedFormatChat` at the handler boundary.
    // Default capacity is small (4 entries) because chat conversations
    // mutate the messages list every turn; the goal is to catch warm
    // reuse benches and repeated agent-loop probes, not to memoize a
    // full session.
    if (params.tokenize_cache_entries > 0) {
        entry.tokenize_cache = tokenize_cache_mod.TokenizeCache.init(
            sch.allocator,
            params.tokenize_cache_entries,
        );
    }
    // Commit the same validated weight estimate used by the eviction gate
    // and preflight (zero means no disk estimate), plus a GPU-resident n-gram table.
    const bytes_resident = committedTextBytes(model_bytes, params.config) + pleTableBill(sch.io, params.config);

    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.markReadyLocked(entry, bytes_resident);
    sch.registry.mutex.unlock(sch.io);

    // Set borrowed views for scheduler-internal code (and `current_model`
    // for ensureLoaded → request handlers in Phase C).
    sch.current_model = entry;
    sch.weights = weights_ptr;
    sch.xfm = xfm_ptr;
    sch.vision_encoder = vision_ptr;
    sch.drafter = drafter_ptr;
    sch.dflash = dflash_ptr;
    // The connection thread bills chat against the view, so a model that cannot chat never takes it.
    if (entry.prefix_cache) |hc| {
        if (ModelRegistry.chatCapable(entry)) sch.hot_prefix_cache = hc;
    }
    publishHotCacheResidency(sch);
}

/// A module tier borrows its loaded transformer and keeps its compatibility identity in its own root.
pub fn initPrefixDisk(a: std.mem.Allocator, io: std.Io, base: []const u8, fingerprint: []const u8, cap: u64, chunk: u32, binding: ?prefix_cache_mod.ModuleBinding, path: []const u8) !kv_disk_cache.DiskTier {
    const b = binding orelse return kv_disk_cache.DiskTier.init(a, io, base, fingerprint, cap, chunk);
    const identity = try modulePackFingerprint(a, io, path, fingerprint);
    defer a.free(identity);
    var tier = try kv_disk_cache.DiskTier.initWithModule(a, io, base, identity, cap, chunk, .{
        .identity = identity,
        .user = @constCast(b.xfm),
        .validate = &validateModuleHeader,
    });
    tier.module_context.?.identity = std.fs.path.basename(tier.root);
    return tier;
}

fn validateModuleHeader(user: ?*anyopaque, header: []const u8, _: *const kv_disk_cache.module_record.Envelope) bool {
    const xfm: *const Transformer = @ptrCast(@alignCast(user orelse return false));
    const had_error = mlx.errorPending();
    defer mlx.dropLatchedErrorUnless(had_error);
    var peak: u64 = 0;
    const verdict = plugin_driver.auxImportPlan(xfm, header, &peak);
    if (!verdict.isOk()) log.debug("[disk-cache] module header declined: {s}\n", .{verdict.why.slice()});
    return verdict.isOk();
}

fn modulePackFingerprint(a: std.mem.Allocator, io: std.Io, path: []const u8, fingerprint: []const u8) ![]u8 {
    if (!std.fs.path.isAbsolute(path)) return error.BadDiskCacheDir;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    var dir = try std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true });
    defer dir.close(io);
    const resident = @import("../../sdk/weights/resident.zig");
    var selected = resident.indexShardSet(io, dir);
    defer if (selected) |*s| resident.freeShardSet(s);
    var names = std.ArrayList([]const u8).empty;
    if (selected) |*s| {
        var keys = s.keyIterator();
        while (keys.next()) |name| try appendPackFile(aa, &names, name.*);
    } else {
        var it = dir.iterate();
        while (try it.next(io)) |file| {
            if (file.kind != .file and file.kind != .sym_link) continue;
            if (!std.mem.endsWith(u8, file.name, ".safetensors")) continue;
            try appendPackFile(aa, &names, file.name);
        }
    }
    try appendPackFile(aa, &names, "engram/engram-residents.safetensors");
    const config: []const u8 = dir.readFileAlloc(io, "config.json", aa, .limited(64 << 20)) catch |err| blk: {
        if (err != error.FileNotFound) return err;
        break :blk "{}";
    };
    try configuredPackFiles(aa, &names, try std.json.parseFromSliceLeaky(std.json.Value, aa, config, .{}));
    const token_map: kv_disk_cache.module_record.TokenMap = blk: {
        for ([_][]const u8{ "engram-token-map.u32", "engram_token_map.json" }) |name| {
            _ = dir.statFile(io, name, .{}) catch |err| {
                if (err == error.FileNotFound) continue;
                return err;
            };
            break :blk .{ .file = name };
        }
        break :blk .builtin;
    };
    const stamp = try kv_disk_cache.module_record.packStamp(aa, io, path, names.items, token_map);
    return std.fmt.allocPrint(a, "{s}-m{x}", .{ fingerprint, std.hash.Wyhash.hash(0, stamp) });
}

fn appendPackFile(a: std.mem.Allocator, names: *std.ArrayList([]const u8), name: []const u8) !void {
    for (names.items) |old| if (std.mem.eql(u8, old, name)) return;
    try names.append(a, try a.dupe(u8, name));
}

fn configuredPackFiles(a: std.mem.Allocator, names: *std.ArrayList([]const u8), value: std.json.Value) std.mem.Allocator.Error!void {
    switch (value) {
        .object => |obj| {
            var it = obj.iterator();
            while (it.next()) |entry| {
                const key = entry.key_ptr.*;
                const v = entry.value_ptr.*;
                if (v == .string and (std.mem.eql(u8, key, "weight_file") or std.mem.eql(u8, key, "scale_file") or std.mem.eql(u8, key, "bias_file"))) {
                    try appendPackFile(a, names, v.string);
                    continue;
                }
                try configuredPackFiles(a, names, v);
            }
        },
        .array => |arr| for (arr.items) |v| try configuredPackFiles(a, names, v),
        else => {},
    }
}

/// The towers of a plugin that declares media hooks, over its loaded model.
fn pluginMedia(a: std.mem.Allocator, config: *const model_mod.ModelConfig, xfm: *const Transformer) ?plugin_media.PluginMedia {
    if (!config.plugin_media) return null;
    const b = config.plugin orelse return null;
    const hooks = b.table.media orelse return null;
    const bound = xfm.plugin orelse return null;
    return .{ .hooks = hooks, .model = bound.model, .parsed = &b.parsed, .a = a };
}

// ── tests ──

test "module pack identity follows selected weights and token map precedence" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const io = testing.io;
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = "weights" });
    try tmp.dir.writeFile(io, .{ .sub_path = "unused.safetensors", .data = "unused" });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"w\":\"model.safetensors\"}}" });
    var buf: [512]u8 = undefined;
    const path = buf[0..try tmp.dir.realPath(io, &buf)];
    const first = try modulePackFingerprint(testing.allocator, io, path, "root");
    defer testing.allocator.free(first);
    try tmp.dir.writeFile(io, .{ .sub_path = "unused.safetensors", .data = "changed unused shard" });
    const same = try modulePackFingerprint(testing.allocator, io, path, "root");
    defer testing.allocator.free(same);
    try testing.expectEqualStrings(first, same);
    try tmp.dir.writeFile(io, .{ .sub_path = "engram_token_map.json", .data = "[0,1]" });
    const json = try modulePackFingerprint(testing.allocator, io, path, "root");
    defer testing.allocator.free(json);
    try testing.expect(!std.mem.eql(u8, first, json));
    try tmp.dir.writeFile(io, .{ .sub_path = "engram-token-map.u32", .data = "binary" });
    const binary = try modulePackFingerprint(testing.allocator, io, path, "root");
    defer testing.allocator.free(binary);
    try testing.expect(!std.mem.eql(u8, json, binary));
    try tmp.dir.writeFile(io, .{ .sub_path = "engram_token_map.json", .data = "[1,2,3]" });
    const same_binary = try modulePackFingerprint(testing.allocator, io, path, "root");
    defer testing.allocator.free(same_binary);
    try testing.expectEqualStrings(binary, same_binary);
    try tmp.dir.createDirPath(io, "engram");
    try tmp.dir.writeFile(io, .{ .sub_path = "engram/engram-residents.safetensors", .data = "resident side weights" });
    const side = try modulePackFingerprint(testing.allocator, io, path, "root");
    defer testing.allocator.free(side);
    try testing.expect(!std.mem.eql(u8, binary, side));
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"tables\":{\"layer\":{\"weight_file\":\"engram/table.safetensors\"}}}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "engram/table.safetensors", .data = "table" });
    const table = try modulePackFingerprint(testing.allocator, io, path, "root");
    defer testing.allocator.free(table);
    try tmp.dir.writeFile(io, .{ .sub_path = "engram/table.safetensors", .data = "changed table weights" });
    const changed_table = try modulePackFingerprint(testing.allocator, io, path, "root");
    defer testing.allocator.free(changed_table);
    try testing.expect(!std.mem.eql(u8, table, changed_table));
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = "changed selected weights" });
    const changed = try modulePackFingerprint(testing.allocator, io, path, "root");
    defer testing.allocator.free(changed);
    try testing.expect(!std.mem.eql(u8, binary, changed));
}
