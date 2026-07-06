//! iOS in-process entry point for the mlx-serve engine.
//!
//! On macOS the Swift app launches `mlx-serve` as a subprocess and talks to it
//! over localhost HTTP. iOS forbids spawning separate executables, so the engine
//! is compiled into a static library (`libmlxserve.a`, see `zig build ios-lib`)
//! and linked directly into the app. The Swift side calls `mlxserve_start` on a
//! background thread; it boots the HTTP server bound to 127.0.0.1:<port> and the
//! Swift client talks to it exactly as the macOS app talks to the subprocess —
//! zero changes to the networking layer.
//!
//! This mirrors the HEADLESS serve path in `main.zig` (`runHeadlessServe`):
//! the server boots with no model resident, discovers checkpoints under
//! `models_dir`, and the app loads/unloads them on demand via `/v1/load-model`
//! + `/v1/unload-model` (by id or absolute path). That gives the iPhone app the
//! same flow the macOS app uses — and lets a chat LM and a TTS model be
//! resident together (voice clone), bounded by `max_resident`.
//!
//! ds4 / llama.cpp GGUF engines are macOS-only and stubbed out for iOS via
//! `build_options.ios`; iOS serves MLX safetensors models only.
//!
//! `mlxserve_start` blocks for the lifetime of the server (it runs the accept
//! loop), so call it from a dedicated thread and treat a return as shutdown.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

// Zig's default Darwin panic handler symbolicates stack traces via
// `_dyld_get_image_header_containing_address`, which is a macOS-only libSystem
// symbol absent from the iOS SDK (link failure). Use the simple panic handler —
// it prints the message and traps without the dyld-based symbolizer. The stub
// engine panics (ds4/llama unavailable on iOS) are never reached at runtime
// anyway; this only governs how an unexpected panic terminates.
pub const panic = std.debug.simple_panic;
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const transformer_mod = @import("transformer.zig");
const model_discovery = @import("model_discovery.zig");
const model_registry_mod = @import("model_registry.zig");
const scheduler_mod = @import("scheduler.zig");
const server_mod = @import("server.zig");
const gen_mod = @import("gen.zig");
const log = @import("log.zig");

/// Returns the engine version string (NUL-terminated, static lifetime).
export fn mlxserve_version() [*:0]const u8 {
    return std.fmt.comptimePrint("{s}", .{build_options.version});
}

/// Boot the in-process server (headless: models load on demand). Blocks until
/// the server stops (or never, in the happy path). Returns 0 on clean
/// shutdown, non-zero on a setup error.
///
/// - `models_dir`: absolute path to the app's models root; scanned so
///   discovered checkpoints appear in /v1/models and load by id. May be ""
///   (load by absolute path via /v1/load-model instead).
/// - `host`: bind address, typically "127.0.0.1".
/// - `port`: bind port.
/// - `ctx_size`: max context (0 = auto/model default).
/// - `max_resident`: how many models may be resident at once (0 → 2 = chat + TTS).
export fn mlxserve_start(
    models_dir_z: [*:0]const u8,
    host_z: [*:0]const u8,
    port: u16,
    ctx_size: u32,
    max_resident: u32,
) c_int {
    const models_dir = std.mem.sliceTo(models_dir_z, 0);
    const host = std.mem.sliceTo(host_z, 0);
    run(models_dir, host, port, ctx_size, if (max_resident == 0) 2 else max_resident) catch |err| {
        log.err("[ios] mlxserve_start failed: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}

fn run(models_dir: []const u8, host: []const u8, port: u16, ctx_size: u32, max_resident: u32) !void {
    // c_allocator (malloc) — fast, and libSystem is always linked into the app.
    const allocator = std.heap.c_allocator;

    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var ver = mlx.mlx_string_new();
    defer _ = mlx.mlx_string_free(ver);
    try mlx.check(mlx.mlx_version(&ver));
    log.info("mlx-serve {s} (MLX {s}) [iOS headless]\n", .{ build_options.version, mlx.mlx_string_data(ver) });
    log.info("[ios] models-dir: {s}, serve: {s}:{d}, ctx-size={d}, max-resident={d}\n", .{ models_dir, host, port, ctx_size, max_resident });

    // Device selection. Default: GPU when Metal is available (the real
    // on-device path). `MLXSERVE_DEVICE=cpu` forces the CPU backend — used to
    // validate the full inference pipeline in the iOS Simulator, whose Metal
    // can't run MLX's kernels. CPU needs no metallib.
    var metal_avail: bool = false;
    try mlx.check(mlx.mlx_metal_is_available(&metal_avail));
    const force_cpu = if (std.c.getenv("MLXSERVE_DEVICE")) |d| std.mem.eql(u8, std.mem.sliceTo(d, 0), "cpu") else false;
    log.info("[ios] Metal GPU available: {}, force_cpu: {}\n", .{ metal_avail, force_cpu });
    if (force_cpu) {
        const cpu_dev = mlx.mlx_device_new_type(.cpu, 0);
        defer _ = mlx.mlx_device_free(cpu_dev);
        try mlx.check(mlx.mlx_set_default_device(cpu_dev));
    } else if (metal_avail) {
        const gpu_dev = mlx.mlx_device_new_type(.gpu, 0);
        defer _ = mlx.mlx_device_free(gpu_dev);
        try mlx.check(mlx.mlx_set_default_device(gpu_dev));
    }
    _ = mlx.mlx_random_seed(@intCast(std.Io.Timestamp.now(io, .real).toMilliseconds()));

    // Vision stays ON: the app's chat supports image attachments (Gemma's
    // SigLIP tower costs a few hundred MB but "what is this picture" is a
    // headline feature). Text-only checkpoints skip it automatically.

    // Cap MLX's buffer cache. On macOS the allocator may park freed buffers
    // without bound (the wired limit is huge); under iOS jetsam ceilings that
    // cache growth DURING a diffusion denoise loop is what kills the app —
    // the post-generation clear in the scheduler only helps between requests.
    // 384 MB keeps step-to-step buffer reuse (the perf win) while returning
    // everything else to the OS as it frees.
    var old_cache_limit: usize = 0;
    _ = mlx.mlx_set_cache_limit(&old_cache_limit, 384 * 1024 * 1024);
    log.info("[ios] mlx cache limit: 384 MB (was {d} MB)\n", .{old_cache_limit / (1024 * 1024)});

    // Discover models under the app's models root (empty/missing dir → none;
    // the app can still load by absolute path).
    var discovery_storage: ?model_discovery.DiscoveryResult = null;
    if (models_dir.len > 0 and std.fs.path.isAbsolute(models_dir)) {
        discovery_storage = model_discovery.discoverModels(io, allocator, models_dir) catch |err| blk: {
            log.warn("[ios] model discovery failed under {s}: {s}\n", .{ models_dir, @errorName(err) });
            break :blk null;
        };
        if (discovery_storage) |d| log.info("[ios] discovered {d} model(s)\n", .{d.models.len});
    }

    // Stub CPU state so the scheduler can boot with no model resident
    // (mirrors main.runHeadlessServe).
    var stub = try gen_mod.buildStubCpuState(allocator, .image);
    defer gen_mod.freeStubCpuState(allocator, &stub);

    // Resident-memory cap: 80% of the GPU's recommended working set when
    // available (same derivation as main.zig); 0 = unlimited (CPU sim).
    const max_resident_mem: u64 = blk: {
        var dev = mlx.mlx_device{ .ctx = null };
        _ = mlx.mlx_get_default_device(&dev);
        var info = mlx.mlx_device_info_new();
        defer _ = mlx.mlx_device_info_free(info);
        if (mlx.mlx_device_info_get(&info, dev) != 0) break :blk 0;
        var max_rec: usize = 0;
        if (mlx.mlx_device_info_get_size(&max_rec, info, "max_recommended_working_set_size") != 0 or max_rec == 0) break :blk 0;
        break :blk @as(u64, max_rec) * 4 / 5;
    };

    const registry = try model_registry_mod.ModelRegistry.init(allocator, io, discovery_storage, max_resident, max_resident_mem, null);
    defer registry.deinit();

    // Carrier entry for LoadParams (required field), never loaded here
    // (`no_initial_load`). Prefer a discovered stub (so it's listed in
    // /v1/models); else a throwaway placeholder. No default model is set, so
    // a request that omits `model` gets a clean 503 until a model is loaded.
    var placeholder = model_registry_mod.LoadedModel{
        .allocator = allocator,
        .id = "",
        .path = "",
        .bytes_on_disk = null,
        .arch_hint = "",
        .config = null,
        .weights = null,
        .transformer = null,
        .tokenizer = null,
        .chat_config = null,
        .vision_encoder = null,
        .drafter = null,
        .drafter_path = "",
        .drafter_block_size = 0,
        .prefix_cache = null,
        .refcount = std.atomic.Value(u32).init(0),
        .last_used_ns = 0,
        .bytes_resident = 0,
        .state = .unloaded,
        .error_name = null,
    };
    const carrier: *model_registry_mod.LoadedModel = blk: {
        var it = registry.entries.valueIterator();
        if (it.next()) |e| break :blk e.*;
        log.info("[ios] no models discovered; load by path via /v1/load-model.\n", .{});
        break :blk &placeholder;
    };

    const params = scheduler_mod.LoadParams{
        .registry = registry,
        .entry = carrier,
        .config = stub.config,
        .tok = stub.tok,
        .chat_config = stub.chat_config,
        .model_dir = "",
        .no_initial_load = true,
        .load_vision = false,
        // Skip eager warmup when there's no GPU backend (CPU-only Simulator
        // build): a warmup forward on CPU is slow and only pre-JITs Metal
        // kernels that don't exist there.
        .warmup_eager = !mlx.noGpuBackend(),
        .draft_block_size = 0,
        .kv_quant_config = transformer_mod.KVQuantConfig.dense,
        // Phone RAM is tight relative to the Mac: keep the hot prefix cache
        // small (1 entry, 256 MB) so chat-history reuse works without
        // hoarding memory.
        .prefix_cache_capacity = 1,
        .prefix_cache_mem_bytes = 256 * 1024 * 1024,
        .tokenize_cache_entries = 4,
    };

    try server_mod.serve(io, allocator, params, stub.config, host, port, .{
        .max_context_size = ctx_size,
        .request_timeout_sec = 0,
        .default_reasoning_budget = -1,
        .default_temperature = null,
        .default_top_p = null,
        .default_top_k = null,
        .default_enable_pld = true,
        .default_pld_draft_len = 5,
        .default_pld_key_len = 3,
        .default_kv_attn_fused = false,
    });
}
