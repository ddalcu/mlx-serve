const std = @import("std");
const build_options = @import("build_options");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const tokenizer_mod = @import("tokenizer.zig");
const transformer_mod = @import("transformer.zig");
const generate_mod = @import("generate.zig");
const model_discovery = @import("model_discovery.zig");
const gguf_meta = @import("gguf_meta.zig");
const model_registry_mod = @import("model_registry.zig");
const drafter_mod = @import("drafter.zig");
const mtp_mod = @import("mtp.zig");
const chat_mod = @import("chat.zig");
const server_mod = @import("server.zig");
const scheduler_mod = @import("scheduler.zig");
const vision_mod = @import("vision.zig");
const ds4_arch = @import("arch/ds4.zig");
const llama_arch = @import("arch/llama.zig");
const ds4_ffi = @import("ds4_ffi.zig");
const gen_mod = @import("gen.zig");
const cli_mod = @import("cli.zig");
const log = @import("log.zig");
const metrics_mod = @import("metrics.zig");

pub const VERSION: []const u8 = build_options.version;

const DEFAULT_MODEL_DIR = ""; // pass --model <path> to specify

// --ssd-streaming (issue #39): ds4 weight-streaming toggle. Set during arg
// parsing, read by the ds4 serve + offline open paths. Module-level to avoid
// threading it through runDs4Serve's already-long parameter list.
var ds4_ssd_streaming: bool = false;

/// `mlx-serve run` REPL thread: chats against the in-process server over
/// its own Ollama /api/chat endpoint, then brings the server down cleanly
/// (SIGTERM → the serve loop's shutdown path) when the user exits.
fn replThreadMain(allocator: std.mem.Allocator, io: std.Io, port: u16) void {
    cli_mod.runRepl(allocator, io, port) catch |err| {
        log.warn("chat REPL exited: {s}\n", .{@errorName(err)});
    };
    std.posix.raise(std.posix.SIG.TERM) catch {};
}

fn printUsage(io: std.Io) void {
    var stdout_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writer(io, &stdout_buf);
    stdout_w.interface.writeAll(
        \\mlx-serve — MLX inference server for Apple Silicon
        \\
        \\Usage: mlx-serve <command> [options]
        \\       mlx-serve [options]
        \\
        \\Commands:
        \\  run <model>         Download if needed, serve it, and chat right here
        \\                      (short name like "gemma4", "qwen3.6:27b", or any
        \\                      HuggingFace "org/repo")
        \\  pull <model>        Download a model into ~/.mlx-serve/models
        \\  list                Show downloaded models
        \\  serve               Start the server over ~/.mlx-serve/models
        \\                      (every pulled model loads on demand by name)
        \\
        \\Options:
        \\  --model <dir>       Path to MLX model directory (required)
        \\  --serve             Start HTTP server mode
        \\  --host <ip>         Bind address (default: 0.0.0.0)
        \\  --port <n>          Bind port (default: 11234)
        \\  --ctx-size <n>      Maximum context length (default: model max)
        \\  --prompt <text>     Run single prompt (interactive mode)
        \\  --stream            Stream tokens as they are generated (with --prompt)
        \\  --max-tokens <n>    Max tokens to generate (default: 100)
        \\  --temp <f>          Temperature. Offline: sampling temp (default 0.0).
        \\                      Serve: default for requests that omit `temperature`
        \\                      (otherwise the model's generation_config.json, then 1.0)
        \\  --top-p <f>         Serve-mode default top_p for requests that omit it
        \\                      (otherwise generation_config.json, then 1.0 = off)
        \\  --top-k <n>         Serve-mode default top_k for requests that omit it
        \\                      (otherwise generation_config.json, then 0 = off)
        \\  --timeout <n>       Stall timeout in seconds: abort a request after n seconds
        \\                      WITHOUT producing a token (default: 300, 0=none). A request
        \\                      that keeps generating never times out, however long it runs.
        \\  --reasoning-budget <n>  Max thinking tokens per request (default: unlimited)
        \\  --no-vision         Disable vision encoder (saves memory)
        \\  --no-safety         Disable the image-gen NSFW content filter (on by
        \\                      default; or set "safety":false per request)
        \\  --skip-mem-preflight  Bypass the model-load free-RAM pre-flight that
        \\                        refuses a load whose weights + warmup headroom
        \\                        look too big for current free memory. The check
        \\                        is conservative (macOS reclaims file cache as
        \\                        MLX allocates); use this if a load you know fits
        \\                        is being refused. A genuine over-commit can
        \\                        hard-crash the server.
        \\  --pld               Enable Prompt Lookup Decoding (default: ON).
        \\                        Model-agnostic speculative decoding via n-gram
        \\                        matches in the prompt + generated tokens. Big
        \\                        wins on echo-heavy workloads (code editing, RAG,
        \\                        agentic loops). Adaptive prompt-time gate
        \\                        auto-disables it on novel content. Pass
        \\                        --no-pld to force-disable.
        \\  --no-pld            Force-disable Prompt Lookup Decoding.
        \\  --pld-draft-len <n> Max draft tokens per PLD step (default: 5).
        \\  --pld-key-len <n>   N-gram match key length for PLD (default: 3).
        \\  --drafter <dir>     Path to a Gemma 4 assistant drafter checkpoint.
        \\                        When set, the drafter is loaded at startup,
        \\                        bound to the target model, and used as the
        \\                        default draft source for new requests
        \\                        (priority: drafter > PLD > regular).
        \\  --draft-block-size <n>  Tokens per drafter round. Default is
        \\                        auto-detected per Gemma 4 target (E2B=2,
        \\                        E4B=4, 26B-A4B=4, 31B=8); pass to override.
        \\  --no-mtp            Disable the Qwen native MTP head (auto-loaded
        \\                        when the model dir ships mtp/weights.safetensors;
        \\                        priority: MTP > drafter > PLD).
        \\  --mtp-depth <n>     Max tokens drafted per MTP round (default: 1).
        \\                        Depths >1 adapt down per-request when the
        \\                        acceptance rate sags.
        \\  --kv-quant <mode>   KV-cache quantization scheme:
        \\                        off (default), 4, 8     — affine group quant.
        \\                        turbo2, turbo4          — Hadamard-rotated
        \\                          affine at 2/4 bits; lower distortion at
        \\                          comparable storage. Per-request override
        \\                          via the `kv_quant` body field.
        \\  --kv-attn-mode {{dense|fused}}
        \\                      Attention path for quantized KV. `dense`
        \\                        (default) dequantizes K/V before SDPA;
        \\                        `fused` consumes the quant triples directly
        \\                        via mlx_quantized_matmul (opt-in; only
        \\                        effective at --kv-quant 4 or 8).
        \\  --prefix-cache-mem <n>{{KB,MB,GB}}
        \\                      Hot prefix cache KV-bytes budget (default: 2GB).
        \\                      Evicts LRU entries until the budget fits.
        \\                      Pass 0/off to disable the byte budget.
        \\  --prefix-cache-disk <n>{{KB,MB,GB}}
        \\                      SSD tier for the prefix cache (default: off).
        \\                      Seen prefixes persist under ~/.mlx-serve/kv-cache
        \\                        and are restored across restarts and RAM
        \\                        evictions instead of recomputed. Can use many
        \\                        GB of disk, so it's opt-in; e.g. 10GB. 0/off
        \\                        disables.
        \\  --tokenize-cache-entries <n>
        \\                      Per-model LRU cache of chat-template render +
        \\                        tokenize results (default: 4). Skips re-
        \\                        rendering identical messages on warm reuse.
        \\                        0 disables.
        \\  --llama-cache-entries <n>
        \\                      For GGUF models served via llama.cpp, the max
        \\                        number of resident KV sessions (default: 4).
        \\                        N > 1 keeps the N most-recently-used prompts
        \\                        hot so alternating multi-doc workloads don't
        \\                        cold-prefill on every flip.
        \\  --engine {{auto|ds4|llama}}
        \\                      Engine selector for `.gguf` inputs ONLY.
        \\                        Safetensors models always run on the native
        \\                        MLX engine and ignore this flag. For
        \\                        GGUF: `auto` (default) reads the file's
        \\                        `general.architecture` metadata and routes
        \\                        deepseek4 + ds4-MLA quants to the embedded
        \\                        ds4 engine, everything else to llama.cpp.
        \\                        Override when auto-detection is wrong
        \\                        (e.g. an unusual ds4 quant whose metadata
        \\                        layout differs).
        \\  --ssd-streaming     ds4 / DeepSeek-V4-Flash only: stream expert
        \\                        weights from SSD instead of holding the whole
        \\                        model in RAM (skips full residency + warmup).
        \\                        Use when the model is larger than available
        \\                        memory. Ignored by the MLX + llama.cpp engines.
        \\  --model-dir <dir>   Directory of MLX models to discover at startup.
        \\                        Discovered siblings appear in /v1/models and
        \\                        can be loaded on-demand via /v1/load-model
        \\                        (or by sending a request with model=<id>).
        \\  --max-resident-models <n>
        \\                      Maximum loaded models in memory (default: 3).
        \\                        ensureLoaded evicts LRU before exceeding.
        \\  --max-resident-mem <n>{{KB,MB,GB}}|auto
        \\                      Summed resident-bytes cap across all loaded
        \\                        models. Default 'auto' = 80% of MLX wired
        \\                        limit at startup. Pass 0 to disable.
        \\  --idle-evict-secs <n>
        \\                      Evict .ready entries with refcount==0 if
        \\                        idle for this many seconds. Default: off.
        \\  --metrics           Enable Prometheus metrics at GET /metrics and a
        \\                        live metrics panel on the index page (opt-in;
        \\                        zero cost when off). Also GET /metrics.json.
        \\  --no-tool-autocorrect
        \\                      Disable tool-call ARGUMENT auto-correct — the
        \\                        coercion of parsed args to the tool schema's
        \\                        declared types (e.g. Python `False` -> JSON
        \\                        `false`). Args then pass through as the model
        \\                        emitted them (still valid JSON). Default: on.
        \\  --api-key <token>   Require this key on every request (OpenAI/
        \\                        Anthropic/Ollama APIs + index page + metrics).
        \\                        Accepts Authorization: Bearer, x-api-key, HTTP
        \\                        Basic (key = password), or ?api_key=. /health
        \\                        stays open. Unset = no auth (default).
        \\  --log-level <lvl>   Log level: error, warn, info, debug (default: info)
        \\  --log-file <path>   Persist the server log ("off" disables).
        \\                      Default: ~/.mlx-serve/logs/mlx-serve-<port>.log
        \\  --version           Print version and exit
        \\  --help              Show this help
        \\
    ) catch {};
    stdout_w.interface.flush() catch {};
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    // Materialize CLI args from the iterator API into a flat slice
    var args_iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_iter.deinit();
    var args_list: std.ArrayList([]const u8) = .empty;
    defer {
        for (args_list.items) |a| allocator.free(a);
        args_list.deinit(allocator);
    }
    while (args_iter.next()) |arg| {
        try args_list.append(allocator, try allocator.dupe(u8, arg));
    }
    const args = args_list.items;

    if (args.len == 1) {
        printUsage(io);
        return;
    }

    // ── Subcommands (Ollama-grade CLI): `mlx-serve run|pull|list|serve` ──
    // `pull` and `list` finish here; `run` and `serve` fall through into the
    // normal flag parse (skipping the consumed positionals) and serve path.
    var arg_start: usize = 1;
    var run_model_dir: ?[]u8 = null;
    defer if (run_model_dir) |d| allocator.free(d);
    var use_default_models_root = false;
    var repl_after_serve = false;
    if (args.len >= 2 and args[1].len > 0 and args[1][0] != '-') {
        const cmd = args[1];
        if (std.mem.eql(u8, cmd, "pull")) {
            if (args.len < 3) {
                log.err("usage: mlx-serve pull <model>\n", .{});
                std.process.exit(1);
            }
            try cli_mod.cmdPull(allocator, io, args[2]);
            return;
        } else if (std.mem.eql(u8, cmd, "list")) {
            try cli_mod.cmdList(allocator, io);
            return;
        } else if (std.mem.eql(u8, cmd, "run")) {
            if (args.len < 3) {
                log.err("usage: mlx-serve run <model> [options]\n", .{});
                std.process.exit(1);
            }
            run_model_dir = try cli_mod.ensureModelAvailable(allocator, io, args[2]);
            // `run` is the chat UX — refuse non-chat models up front with
            // the serve alternative instead of booting a server whose chat
            // surface can only 400 (pre-guard it SIGSEGV'd: the media stub
            // tokenizer yields 0 tokens and prefill derefs a null
            // transformer — see server.zig textGenRejectReason).
            if (model_discovery.classifyModelPath(io, allocator, run_model_dir.?)) |kind| {
                if (kind != .chat) {
                    log.err("'{s}' is {s} — `mlx-serve run` starts a chat REPL, which it can't serve.\n", .{ args[2], kind.describe() });
                    if (kind.genEndpoint()) |ep| {
                        log.err("serve it for API/app use instead:\n", .{});
                        log.err("  mlx-serve --model \"{s}\" --serve\n", .{run_model_dir.?});
                        log.err("  then POST {s}\n", .{ep});
                    }
                    std.process.exit(1);
                }
            }
            arg_start = 3;
            use_default_models_root = true;
            repl_after_serve = std.Io.File.stdin().isTty(io) catch false;
        } else if (std.mem.eql(u8, cmd, "serve")) {
            arg_start = 2;
            use_default_models_root = true;
        } else {
            log.err("unknown command '{s}' (expected run, pull, list, or serve)\n", .{cmd});
            std.process.exit(1);
        }
    }

    var model_dir: []const u8 = DEFAULT_MODEL_DIR;
    var models_root: ?[]const u8 = null; // --model-dir for plan 05 discovery
    var port: u16 = 11234;
    var host: []const u8 = "0.0.0.0";
    // `--log-file <path|off>`. null = default (`~/.mlx-serve/logs/mlx-serve-<port>.log`).
    var log_file_arg: ?[]const u8 = null;
    var serve_mode = false;
    var stream_mode = false;
    var prompt: ?[]const u8 = null;
    var max_tokens: u32 = 100;
    var temperature: f32 = 0.0;
    // Serve-mode sampling defaults for requests that omit the field
    // (request > flag > model generation_config.json > hardcoded). `--temp`
    // doubles as the offline --prompt sampling temp, so track whether it was
    // explicitly given — only then does it become the serve default.
    var temp_explicit = false;
    var top_p_flag: ?f32 = null;
    var top_k_flag: ?u32 = null;
    var ctx_size: u32 = 0; // 0 = use model default
    var timeout: u32 = 300; // seconds, 0 = no timeout
    var reasoning_budget: i32 = -1; // -1 = unlimited
    var no_vision = false;
    var enable_pld = true; // Prompt Lookup Decoding (on by default; --no-pld to disable)
    var pld_draft_len: u32 = 5;
    var pld_key_len: u32 = 3;
    var drafter_dir: ?[]const u8 = null; // Path to Gemma 4 assistant drafter checkpoint
    var draft_block_size: u32 = drafter_mod.DEFAULT_BLOCK_SIZE;
    var draft_block_size_explicit: bool = false; // user passed --draft-block-size?
    var enable_mtp = true; // Qwen native MTP head (auto when sidecar present; --no-mtp to disable)
    var mtp_depth: u32 = mtp_mod.DEFAULT_DEPTH;
    // Plan 04 Phase 1: pre-fault weights and pre-compile kernels at boot.
    // Default ON in serve mode — small boot-time cost, big cold-prefill win.
    // --no-warmup-eager opts out for benchmarking / minimal-footprint deployments.
    var warmup_eager: bool = true;
    var kv_quant_config: transformer_mod.KVQuantConfig = transformer_mod.KVQuantConfig.dense;
    // Phase 2 (Plan ricky): fused attention reads K/V triples directly via
    // mlx_quantized_matmul instead of dequantizing through DenseKVView.
    // Off by default — only `.affine` cache scheme is supported by the
    // v1 fused path; TurboQuant + dense schemes ignore it.
    var kv_attn_fused_default: bool = false;
    // Plan 05 Phase D: multi-model caps. Defaults aim for "comfortable on
    // 32–64 GB systems running Gemma 4 E4B-class models". Override via the
    // CLI flags below; the Swift app exposes them under Advanced settings.
    var max_resident_models: u32 = 3;
    var max_resident_mem: u64 = 0; // 0 = auto (80% of wired limit at startup)
    var max_resident_mem_explicit: bool = false;
    var idle_evict_secs: ?u32 = null;
    var metrics_enabled = false;
    // GGUF engine routing override. null → auto (decided by gguf_meta on
    // file inspection); set explicitly via --engine to force ds4 or llama.
    var engine_override: ?gguf_meta.Engine = null;
    var log_level_explicit = false;
    var i: usize = arg_start;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--version")) {
            var ver_buf: [64]u8 = undefined;
            var ver_w = std.Io.File.stdout().writer(io, &ver_buf);
            ver_w.interface.writeAll("mlx-serve " ++ VERSION ++ "\n") catch {};
            ver_w.interface.flush() catch {};
            return;
        } else if (std.mem.eql(u8, args[i], "--help") or std.mem.eql(u8, args[i], "-h")) {
            printUsage(io);
            return;
        } else if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
            i += 1;
            model_dir = args[i];
        } else if (std.mem.eql(u8, args[i], "--port") and i + 1 < args.len) {
            i += 1;
            port = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--host") and i + 1 < args.len) {
            i += 1;
            host = args[i];
        } else if (std.mem.eql(u8, args[i], "--serve")) {
            serve_mode = true;
        } else if (std.mem.eql(u8, args[i], "--stream")) {
            stream_mode = true;
        } else if (std.mem.eql(u8, args[i], "--prompt") and i + 1 < args.len) {
            i += 1;
            prompt = args[i];
        } else if (std.mem.eql(u8, args[i], "--max-tokens") and i + 1 < args.len) {
            i += 1;
            max_tokens = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--temp") and i + 1 < args.len) {
            i += 1;
            temperature = try std.fmt.parseFloat(f32, args[i]);
            temp_explicit = true;
        } else if (std.mem.eql(u8, args[i], "--top-p") and i + 1 < args.len) {
            i += 1;
            top_p_flag = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, args[i], "--top-k") and i + 1 < args.len) {
            i += 1;
            top_k_flag = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--ctx-size") and i + 1 < args.len) {
            i += 1;
            ctx_size = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--timeout") and i + 1 < args.len) {
            i += 1;
            timeout = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--no-vision")) {
            no_vision = true;
            // Module global so on-demand /v1/load-model cold loads honor the
            // flag too (they used to hardcode vision from config.has_vision).
            scheduler_mod.no_vision_global = true;
        } else if (std.mem.eql(u8, args[i], "--skip-mem-preflight")) {
            scheduler_mod.skip_mem_preflight = true;
        } else if (std.mem.eql(u8, args[i], "--no-safety")) {
            server_mod.image_safety_filter = false;
        } else if (std.mem.eql(u8, args[i], "--pld")) {
            enable_pld = true;
        } else if (std.mem.eql(u8, args[i], "--no-tool-autocorrect")) {
            server_mod.g_tool_autocorrect = false;
        } else if (std.mem.eql(u8, args[i], "--no-pld")) {
            enable_pld = false;
        } else if (std.mem.eql(u8, args[i], "--pld-draft-len") and i + 1 < args.len) {
            i += 1;
            pld_draft_len = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--pld-key-len") and i + 1 < args.len) {
            i += 1;
            pld_key_len = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--drafter") and i + 1 < args.len) {
            i += 1;
            drafter_dir = args[i];
        } else if (std.mem.eql(u8, args[i], "--draft-block-size") and i + 1 < args.len) {
            i += 1;
            draft_block_size = try std.fmt.parseInt(u32, args[i], 10);
            draft_block_size_explicit = true;
        } else if (std.mem.eql(u8, args[i], "--metrics")) {
            metrics_enabled = true;
        } else if (std.mem.eql(u8, args[i], "--api-key") and i + 1 < args.len) {
            i += 1;
            // Borrowed from argv (lives for the process). Empty ⇒ leave open.
            if (args[i].len > 0) server_mod.g_api_key = args[i];
        } else if (std.mem.eql(u8, args[i], "--no-mtp")) {
            enable_mtp = false;
        } else if (std.mem.eql(u8, args[i], "--mtp-depth") and i + 1 < args.len) {
            i += 1;
            mtp_depth = @min(mtp_mod.MAX_DEPTH, @max(1, try std.fmt.parseInt(u32, args[i], 10)));
        } else if (std.mem.eql(u8, args[i], "--reasoning-budget") and i + 1 < args.len) {
            i += 1;
            reasoning_budget = try std.fmt.parseInt(i32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--log-level") and i + 1 < args.len) {
            i += 1;
            if (log.Level.fromString(args[i])) |level| {
                log.setLevel(level);
                log_level_explicit = true;
            }
        } else if (std.mem.eql(u8, args[i], "--log-file") and i + 1 < args.len) {
            i += 1;
            log_file_arg = args[i];
        } else if (std.mem.eql(u8, args[i], "--warmup-eager")) {
            warmup_eager = true;
        } else if (std.mem.eql(u8, args[i], "--no-warmup-eager")) {
            warmup_eager = false;
        } else if (std.mem.eql(u8, args[i], "--prefill-chunk") and i + 1 < args.len) {
            i += 1;
            const v = std.fmt.parseInt(usize, args[i], 10) catch 8192;
            generate_mod.prefill_chunk_override = v;
        } else if (std.mem.eql(u8, args[i], "--prefill-trace")) {
            generate_mod.prefill_trace_force = true;
        } else if (std.mem.eql(u8, args[i], "--prefix-cache-entries") and i + 1 < args.len) {
            i += 1;
            server_mod.prefix_cache_capacity = std.fmt.parseInt(u32, args[i], 10) catch 1;
        } else if (std.mem.eql(u8, args[i], "--prefix-cache-mem") and i + 1 < args.len) {
            // Wave 1.B — KV-bytes budget for the hot prefix cache. Accepts
            // bare numbers (bytes), a suffix of `MB`/`GB`/`KB` (case-
            // insensitive), or `0`/`off` to disable the byte budget entirely
            // (count cap from --prefix-cache-entries still applies).
            i += 1;
            server_mod.prefix_cache_mem_bytes = parseSizeArg(args[i]) catch {
                log.err("--prefix-cache-mem: expected '<n>{{MB,GB,KB}}' or '0'/'off'; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, args[i], "--prefix-cache-disk") and i + 1 < args.len) {
            // SSD tier for the hot prefix cache: previously-seen prefixes are
            // persisted as chunked safetensors and restored across restarts
            // and RAM evictions instead of recomputed. Byte budget with the
            // same size grammar as --prefix-cache-mem; `0`/`off` disables.
            i += 1;
            server_mod.prefix_cache_disk_bytes = parseSizeArg(args[i]) catch {
                log.err("--prefix-cache-disk: expected '<n>{{MB,GB,KB}}' or '0'/'off'; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, args[i], "--tokenize-cache-entries") and i + 1 < args.len) {
            // Iteration 2 (perf-plan Phase 4 #3): caps the per-LoadedModel
            // chat-template tokenize cache. 0 = off (every request re-
            // renders+re-tokenizes, mirrors pre-Iteration-2 behavior).
            i += 1;
            server_mod.tokenize_cache_entries = std.fmt.parseInt(u32, args[i], 10) catch 4;
        } else if (std.mem.eql(u8, args[i], "--llama-cache-entries") and i + 1 < args.len) {
            // Iteration 3-5 (perf-plan Phase 5 #1): max concurrent
            // llama.cpp KV sessions per model. 1 = legacy single-session
            // (every prefill fights for the one slot). > 1 enables the
            // best-prefix-match LRU.
            i += 1;
            server_mod.llama_cache_entries = std.fmt.parseInt(u32, args[i], 10) catch 4;
        } else if (std.mem.eql(u8, args[i], "--ssm-checkpoint-stride") and i + 1 < args.len) {
            // Phase 1 (perf-plan): per-position SSM/conv state snapshots during
            // chunked prefill enable multi-turn warm reuse on hybrid SSM
            // architectures. 0 disables (legacy behavior: hybrid bypasses the
            // hot prefix cache); default 128.
            i += 1;
            server_mod.ssm_checkpoint_stride = std.fmt.parseInt(u32, args[i], 10) catch 128;
        } else if (std.mem.eql(u8, args[i], "--ssm-checkpoint-max") and i + 1 < args.len) {
            i += 1;
            server_mod.ssm_checkpoint_max = std.fmt.parseInt(u32, args[i], 10) catch 32;
        } else if (std.mem.eql(u8, args[i], "--llama-kv-quant") and i + 1 < args.len) {
            // Phase 5 #2: KV-cache quantization for the embedded llama.cpp
            // engine. Accepts `off`/`f16` (default; F16), `q8`/`8`/`Q8_0`
            // (~2× compression, near-lossless), `q4`/`4`/`Q4_0` (~4×
            // compression, some quality impact). Auto-enables flash-attn
            // in the shim because llama's plain SDPA needs F16/F32 KV.
            i += 1;
            const arch_llama = @import("arch/llama.zig");
            if (arch_llama.LlamaKvQuant.fromString(args[i])) |q| {
                server_mod.llama_kv_quant = q;
            } else {
                log.err("--llama-kv-quant: expected off|q8|q4 (or 8/4), got '{s}'\n", .{args[i]});
                std.process.exit(1);
            }
        } else if (std.mem.eql(u8, args[i], "--max-concurrent") and i + 1 < args.len) {
            i += 1;
            server_mod.max_concurrent = std.fmt.parseInt(u32, args[i], 10) catch 1;
        } else if (std.mem.eql(u8, args[i], "--model-dir") and i + 1 < args.len) {
            i += 1;
            models_root = args[i];
        } else if (std.mem.eql(u8, args[i], "--max-resident-models") and i + 1 < args.len) {
            // Plan 05 Phase D: cap on .ready entries in the registry.
            // ensureLoaded evicts LRU before loading when this would be exceeded.
            i += 1;
            max_resident_models = std.fmt.parseInt(u32, args[i], 10) catch 3;
            if (max_resident_models == 0) max_resident_models = 1;
        } else if (std.mem.eql(u8, args[i], "--max-resident-mem") and i + 1 < args.len) {
            // Plan 05 Phase D: cap on summed resident bytes. Accepts the
            // same suffixes as --prefix-cache-mem. Special string "auto"
            // (or default 0) → 80% of mlx_set_wired_limit at server start.
            i += 1;
            if (std.mem.eql(u8, args[i], "auto")) {
                max_resident_mem = 0;
            } else {
                max_resident_mem = parseSizeArg(args[i]) catch {
                    log.err("--max-resident-mem: expected '<n>{{MB,GB,KB}}' or 'auto'; got '{s}'\n", .{args[i]});
                    std.process.exit(1);
                };
                max_resident_mem_explicit = true;
            }
        } else if (std.mem.eql(u8, args[i], "--idle-evict-secs") and i + 1 < args.len) {
            // Plan 05 Phase D: idle-tick eviction window. When set, the
            // inference loop's idle path evicts .ready entries (refcount==0)
            // whose last_used_ns is older than this. Default off — eviction
            // is on-demand only.
            i += 1;
            const n = std.fmt.parseInt(u32, args[i], 10) catch 0;
            idle_evict_secs = if (n > 0) n else null;
        } else if (std.mem.eql(u8, args[i], "--kv-quant") and i + 1 < args.len) {
            i += 1;
            if (std.mem.eql(u8, args[i], "off") or std.mem.eql(u8, args[i], "0")) {
                kv_quant_config = transformer_mod.KVQuantConfig.dense;
            } else if (std.mem.eql(u8, args[i], "4")) {
                kv_quant_config = transformer_mod.KVQuantConfig.affine(4);
            } else if (std.mem.eql(u8, args[i], "8")) {
                kv_quant_config = transformer_mod.KVQuantConfig.affine(8);
            } else if (std.mem.eql(u8, args[i], "turbo2")) {
                kv_quant_config = transformer_mod.KVQuantConfig.turboquant(2);
            } else if (std.mem.eql(u8, args[i], "turbo4")) {
                kv_quant_config = transformer_mod.KVQuantConfig.turboquant(4);
            } else {
                log.err("--kv-quant: expected one of {{off, 4, 8, turbo2, turbo4}}; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            }
        } else if (std.mem.eql(u8, args[i], "--engine") and i + 1 < args.len) {
            i += 1;
            if (std.mem.eql(u8, args[i], "auto")) {
                engine_override = null;
            } else if (std.mem.eql(u8, args[i], "ds4")) {
                engine_override = .ds4;
            } else if (std.mem.eql(u8, args[i], "llama")) {
                engine_override = .llama;
            } else {
                log.err("--engine: expected one of {{auto, ds4, llama}}; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            }
        } else if (std.mem.eql(u8, args[i], "--ssd-streaming")) {
            ds4_ssd_streaming = true;
        } else if (std.mem.eql(u8, args[i], "--kv-attn-mode") and i + 1 < args.len) {
            i += 1;
            if (std.mem.eql(u8, args[i], "dense")) {
                kv_attn_fused_default = false;
            } else if (std.mem.eql(u8, args[i], "fused")) {
                kv_attn_fused_default = true;
            } else {
                log.err("--kv-attn-mode: expected 'dense' or 'fused'; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            }
        }
    }

    // Subcommand plumbing: `run <model>` supplies the model dir + serve
    // mode; `run`/`serve` default the discovery root to ~/.mlx-serve/models
    // so every pulled model is loadable by name (Ollama-style).
    var default_models_root_storage: ?[]u8 = null;
    defer if (default_models_root_storage) |r| allocator.free(r);
    if (run_model_dir) |d| {
        model_dir = d;
        serve_mode = true;
    }
    if (use_default_models_root) {
        serve_mode = true;
        if (models_root == null) {
            const home = std.mem.span(std.c.getenv("HOME") orelse "/tmp");
            default_models_root_storage = try cli_mod.modelsRootPath(allocator, home);
            models_root = default_models_root_storage;
        }
    }

    // `mlx-serve run` on a TTY quiets logs to warn (unless --log-level was
    // given) BEFORE the models-root scan below — discovery's per-directory
    // `[discovery] skip …` info lines would otherwise spam the chat REPL.
    if (repl_after_serve and !log_level_explicit) log.setLevel(.warn);

    // Persist the server log. The macOS app only keeps stderr in a 64 KB
    // in-memory ring, so a server that crashed or was restarted takes its
    // history with it — exactly when you need it (see the 2026-07-08 pi
    // session post-mortem). Serving paths only; `pull`/`list` stay quiet.
    //
    // NOTE: opened AFTER `--log-level` is parsed (so the level gates what
    // reaches disk) and BEFORE model discovery/loading (so weight-load and
    // auto-context lines land in the file).
    if (serve_mode or repl_after_serve) {
        var log_path_buf: [1024]u8 = undefined;
        const chosen: ?[]const u8 = if (log_file_arg) |a|
            (if (std.mem.eql(u8, a, "off") or std.mem.eql(u8, a, "none")) null else a)
        else if (std.c.getenv("HOME")) |h|
            log.defaultLogPath(&log_path_buf, std.mem.span(h), port) catch null
        else
            null;
        if (chosen) |p| {
            if (log.openFile(p, log.default_max_bytes)) |_| {
                log.info("Logging to {s} (rotates at {d} MB)\n", .{ p, log.default_max_bytes / (1024 * 1024) });
            } else |e| {
                log.warn("could not open log file {s}: {s} (stderr only)\n", .{ p, @errorName(e) });
            }
        }
    }
    defer log.closeFile();

    // Plan 05 Phase 1: model discovery. When --model-dir is passed, scan
    // the directory for subdirectories containing config.json. The
    // discovered list is published via /v1/models. v1: routing still goes
    // to a single loaded model — if --model isn't set, pick the first
    // discovered. v2 (plan 05 phases 2-5) adds on-demand load and LRU.
    var discovery_storage: ?model_discovery.DiscoveryResult = null;
    defer if (discovery_storage) |*d| d.deinit();
    if (models_root) |root| {
        discovery_storage = model_discovery.discoverModels(io, allocator, root) catch |err| blk: {
            log.warn("--model-dir scan failed ({s}): {s}\n", .{ root, @errorName(err) });
            break :blk null;
        };
        if (discovery_storage) |*d| {
            log.info("Discovered {d} model(s) under {s}:\n", .{ d.models.len, root });
            for (d.models) |m| {
                if (m.bytes_on_disk) |b| {
                    log.info("  - {s} ({d:.1} GB)\n", .{ m.id, @as(f64, @floatFromInt(b)) / 1_073_741_824.0 });
                } else {
                    log.info("  - {s}\n", .{m.id});
                }
            }
            // No auto-select: when `--model` is omitted but `--model-dir` is
            // present, the server starts HEADLESS (no primary model). All
            // discovered models are registered as stubs and load on demand via
            // `/v1/load-model` — chat OR media. The headless branch in the
            // serve block below handles this.
        }
    }
    // In serve mode, check if the port is already in use before loading the model
    // (model loading takes seconds — fail fast instead of wasting time)
    if (serve_mode) {
        if (portInUse(io, port)) {
            log.err("Port {d} is already in use — another mlx-serve instance may be running.\n", .{port});
            log.err("Stop it first (pkill -f mlx-serve) or use a different port (--port {d}).\n", .{port + 1});
            std.process.exit(1);
        }
    }

    // `mlx-serve run` on a TTY: chat REPL on a side thread. It polls
    // /health until the model is up, then drives the server's own /api/chat
    // (Ollama NDJSON) endpoint. (Logs were already quieted to warn above,
    // before discovery, so streamed tokens aren't interleaved with [info]
    // lines.)
    if (repl_after_serve and serve_mode) {
        const t = std.Thread.spawn(.{}, replThreadMain, .{ allocator, io, port }) catch |err| blk: {
            log.warn("could not start chat REPL: {s}\n", .{@errorName(err)});
            break :blk null;
        };
        if (t) |thread| thread.detach();
    }

    // Observability: allocate the metrics core once (when --metrics is on) and
    // publish it via the server-global `g_metrics`. Declared here — above every
    // serve-dispatch path (GGUF/ds4/llama, headless, media, and the primary MLX
    // path) — so `server_mod.serve()` spawns the gauge sampler + routes /metrics
    // regardless of engine, and each LoadParams builder reads it back into the
    // scheduler's per-request sink via `.metrics = server_mod.g_metrics`. The
    // instance lives on this stack frame for the whole process lifetime; the
    // defer clears the global so an early serve() failure can't leave it
    // dangling. Off (the default) → null: a single per-request branch, no cost.
    var metrics_instance: ?metrics_mod.Metrics = if (metrics_enabled) metrics_mod.Metrics.init() else null;
    if (metrics_instance) |*m| server_mod.g_metrics = m;
    defer server_mod.g_metrics = null;

    // ── GGUF early-branch: route to an embedded engine ──
    //
    // mlx-serve serves GGUF models through embedded engines (no MLX path). The
    // backend is picked at load time by file extension + family: DeepSeek-V4-Flash
    // goes to `lib/ds4/` (antirez/ds4, a bespoke engine for that architecture);
    // every other `.gguf` goes to the embedded llama.cpp engine (`lib/llama_shim/`
    // + `src/arch/llama.zig`). Any path ending in `.gguf` (or a directory
    // containing one) bypasses the MLX safetensors path entirely. Both offline
    // (`--prompt`) and serve (`--serve`) modes are wired; serve constructs a stub
    // LoadedModel whose request handlers route through the engine.
    if (isGgufPath(io, model_dir)) {
        const chosen = chooseGgufEngine(io, allocator, model_dir, engine_override);
        if (serve_mode) {
            switch (chosen) {
                .ds4 => try runDs4Serve(io, allocator, model_dir, host, port, ctx_size, timeout, reasoning_budget,
            if (temp_explicit) temperature else null,
            top_p_flag,
            top_k_flag, max_resident_models, max_resident_mem, max_resident_mem_explicit, idle_evict_secs),
                .llama => try runLlamaServe(io, allocator, model_dir, host, port, ctx_size, timeout, reasoning_budget,
            if (temp_explicit) temperature else null,
            top_p_flag,
            top_k_flag, max_resident_models, max_resident_mem, max_resident_mem_explicit, idle_evict_secs),
            }
            return;
        }
        const prompt_text = prompt orelse {
            log.err("GGUF offline mode requires --prompt <text>\n", .{});
            std.process.exit(2);
        };
        switch (chosen) {
            .ds4 => try runDs4Offline(io, allocator, model_dir, prompt_text, max_tokens, temperature, ctx_size),
            .llama => try runLlamaOffline(io, allocator, model_dir, prompt_text, max_tokens, temperature),
        }
        return;
    }

    // Print MLX version
    var ver = mlx.mlx_string_new();
    defer _ = mlx.mlx_string_free(ver);
    try mlx.check(mlx.mlx_version(&ver));
    log.info("mlx-serve {s} (MLX {s})\n", .{ VERSION, mlx.mlx_string_data(ver) });

    // Echo the resolved arguments — makes drafter/target mismatches obvious
    // from the log without having to scroll through the whole launch line in
    // the parent's process listing.
    log.info("[args] model: {s}\n", .{model_dir});
    if (drafter_dir) |dir| {
        log.info("[args] drafter: {s} (block_size={d}{s})\n", .{
            dir,
            draft_block_size,
            if (draft_block_size_explicit) "" else ", auto",
        });
    } else {
        log.info("[args] drafter: <none>\n", .{});
    }
    if (serve_mode) {
        log.info("[args] serve: {s}:{d}, ctx-size={d}, pld={s}, no-vision={}\n", .{
            host,
            port,
            ctx_size,
            if (enable_pld) "on" else "off",
            no_vision,
        });
    }
    switch (kv_quant_config.scheme) {
        .off => log.info("[args] kv-quant: off\n", .{}),
        .affine => log.info("[args] kv-quant: affine {d}-bit (group={d})\n", .{ kv_quant_config.bits, kv_quant_config.group_size }),
        .turboquant_2, .turboquant_4 => log.info("[args] kv-quant: turboquant {d}-bit (group={d}, Hadamard rotation)\n", .{ kv_quant_config.bits, kv_quant_config.group_size }),
    }
    log.info("[args] kv-attn-mode: {s}\n", .{if (kv_attn_fused_default) "fused" else "dense"});

    // Set GPU as default
    var metal_avail: bool = false;
    try mlx.check(mlx.mlx_metal_is_available(&metal_avail));
    log.info("Metal GPU: {}\n", .{metal_avail});

    if (metal_avail) {
        const gpu_dev = mlx.mlx_device_new_type(.gpu, 0);
        defer _ = mlx.mlx_device_free(gpu_dev);
        try mlx.check(mlx.mlx_set_default_device(gpu_dev));
    }

    // Seed MLX RNG with current wall-clock time for non-deterministic sampling
    _ = mlx.mlx_random_seed(@intCast(std.Io.Timestamp.now(io, .real).toMilliseconds()));

    if (serve_mode) {
        // Headless boot: `--model-dir` given, no `--model`. Start with no
        // primary model; everything (chat + media) loads on demand through the
        // registry. The app uses this so a single server hosts chat + image +
        // audio + video, coexisting under one memory budget.
        if (model_dir.len == 0) {
            const discovery_for_registry = discovery_storage;
            discovery_storage = null; // ownership moves to the registry
            try runHeadlessServe(io, allocator, discovery_for_registry, host, port, ctx_size, timeout, reasoning_budget, max_resident_models, max_resident_mem, max_resident_mem_explicit, idle_evict_secs);
            return;
        }

        // Native media generation (image FLUX / audio Qwen3-TTS / video LTX):
        // these bypass the MLX transformer but are now hosted by the ONE main
        // server via the registry (modality engine on the LoadedModel). Peek
        // model_type from config.json BEFORE the transformer-shaped parseConfig
        // and route the primary model through the media serve path.
        if (gen_mod.detectModality(io, allocator, model_dir)) |modality| {
            const discovery_for_registry = discovery_storage;
            discovery_storage = null; // ownership moves to the registry
            try runGenServe(io, allocator, model_dir, modality, discovery_for_registry, host, port, ctx_size, timeout, reasoning_budget, max_resident_models, max_resident_mem, max_resident_mem_explicit, idle_evict_secs);
            return;
        }
    }

    // Parse config — heap allocate so the LoadedModel can take ownership
    // (Plan 05). Free path in serve_mode = registry.deinit; offline mode =
    // explicit defer on `config_storage`.
    const config_storage = try allocator.create(model_mod.ModelConfig);
    var config_owned_by_registry = false;
    // defer-only, NOT errdefer + defer: a plain `defer` already runs on the
    // error-return path, so pairing it with an errdefer that has the same body
    // frees the resource twice on error (double-free / SIGSEGV). The runtime
    // `owned_by_registry` guard makes the single defer correct on every exit.
    defer if (!config_owned_by_registry) allocator.destroy(config_storage);
    config_storage.* = try model_mod.parseConfig(io, allocator, model_dir);
    const config = config_storage;
    log.info("Model: {s} ({d} layers, {d}-dim, head_dim={d}, {d}h/{d}kv, {d}-bit {s} quant)\n", .{
        config.model_type,
        config.num_hidden_layers,
        config.hidden_size,
        config.head_dim,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.quant_bits,
        @tagName(config.quant_mode),
    });

    // Load tokenizer — heap-allocated, ownership transfers to registry on serve_mode.
    log.info("Loading tokenizer...\n", .{});
    const tok = try allocator.create(tokenizer_mod.Tokenizer);
    var tok_owned_by_registry = false;
    tok.* = tokenizer_mod.loadTokenizer(io, allocator, model_dir) catch |err| {
        // Raw memory only — nothing initialized to deinit. The cleanup defer
        // below must NOT be registered yet: deinit on the undefined pointee
        // was a live SIGSEGV on a partially-downloaded model dir (the
        // preloadCpuState errdefer-after-init pattern applies here too).
        allocator.destroy(tok);
        log.err("failed to load tokenizer from {s}: {s} (incomplete download? `mlx-serve pull` the model again to resume, or delete the dir)\n", .{ model_dir, @errorName(err) });
        return err;
    };
    // defer-only (see config note above): errdefer + defer with the same body
    // double-frees on the error-return path.
    defer if (!tok_owned_by_registry) {
        tok.deinit();
        allocator.destroy(tok);
    };

    // Load chat config — heap-allocated, ownership transfers to registry on serve_mode.
    const chat_config = try allocator.create(chat_mod.ChatConfig);
    var chat_config_owned_by_registry = false;
    chat_config.* = chat_mod.loadChatConfig(io, allocator, model_dir) catch |err| {
        // Raw memory only — see the tokenizer catch above.
        allocator.destroy(chat_config);
        return err;
    };
    // defer-only (see config note above): errdefer + defer with the same body
    // double-frees on the error-return path — this is the one that crashed in
    // the #45 GPU-OOM pre-flight refusal (ChatConfig.deinit ran twice).
    defer if (!chat_config_owned_by_registry) {
        chat_config.deinit();
        allocator.destroy(chat_config);
    };

    // Merge the tokenizer's chat-terminator EOS into the stop set — ALWAYS,
    // even when config.json already specified an eos_token_id. Some checkpoints
    // (e.g. Qwen2.5-Coder-7B) set config.json eos_token_id to <|endoftext|>
    // (151643) but their chat template ends turns with <|im_end|> (151645);
    // stopping only on config's id leaks <|im_end|> into the output (breaks
    // structured-JSON / tool-calling). Additive + dedup-guarded: this can only
    // ADD a model-declared stop token, never remove one.
    if (chat_config.eos_token) |eos_str| {
        if (tok.special_tokens.get(eos_str)) |eos_id| {
            if (!config.isEosToken(eos_id)) {
                config.addEosToken(eos_id);
                log.info("EOS token from tokenizer: {s} (id={d})\n", .{ eos_str, eos_id });
            }
        }
    }
    // Also add <|endoftext|> if it exists and wasn't already added.
    if (tok.special_tokens.get("<|endoftext|>")) |eot_id| {
        if (!config.isEosToken(eot_id)) {
            config.addEosToken(eot_id);
        }
    }

    // Treat <pad> as a stop token, but only if it's not token ID 0
    // (ID 0 can be produced spuriously by models under long/confusing prompts)
    if (tok.special_tokens.get("<pad>")) |pad_id| {
        if (pad_id > 0 and !config.isEosToken(pad_id)) {
            config.addEosToken(pad_id);
            log.info("Added <pad> as stop token (id={d})\n", .{pad_id});
        }
    }

    // Pre-encode the user-turn marker so vision-image insertion can locate the
    // latest user turn at request time, regardless of architecture.
    try config.populateUserTurnMarker(allocator, tok, chat_config.chat_template);

    const load_vision = config.has_vision and !no_vision;

    if (serve_mode) {
        // ── Plan 05: build the ModelRegistry, register a stub for the
        //    loaded model, and pass everything to serve(). The registry
        //    takes ownership of `discovery_storage` (if any) and, once
        //    the inference thread completes loading, ownership of
        //    config/tok/chat_config too.
        const model_id = blk: {
            var p = model_dir;
            while (p.len > 0 and p[p.len - 1] == '/') p = p[0 .. p.len - 1];
            if (p.len == 0) break :blk config.model_type;
            if (std.mem.lastIndexOfScalar(u8, p, '/')) |slash_idx| break :blk p[slash_idx + 1 ..];
            break :blk p;
        };

        const discovery_for_registry = discovery_storage;
        discovery_storage = null; // ownership moves to the registry

        // Plan 05 Phase D: compute the effective max_resident_mem. When the
        // user didn't pass an explicit cap, derive 80% of mlx's wired limit
        // (mlx_set_wired_limit returns a value the platform considers safe
        // for sustained GPU work). The wired limit was already applied in
        // the inference thread's load path; here we mirror that calculation
        // so the registry's eviction gate stays in sync. 0 disables the cap.
        const effective_max_resident_mem: u64 = if (max_resident_mem_explicit)
            max_resident_mem
        else blk: {
            var dev = mlx.mlx_device{ .ctx = null };
            _ = mlx.mlx_get_default_device(&dev);
            var info = mlx.mlx_device_info_new();
            defer _ = mlx.mlx_device_info_free(info);
            if (mlx.mlx_device_info_get(&info, dev) != 0) break :blk 0;
            var max_rec: usize = 0;
            if (mlx.mlx_device_info_get_size(&max_rec, info, "max_recommended_working_set_size") != 0 or max_rec == 0) break :blk 0;
            break :blk @as(u64, max_rec) * 4 / 5;
        };
        if (effective_max_resident_mem > 0) {
            log.info("[registry] max_resident_models={d}, max_resident_mem={d:.1} GB\n", .{
                max_resident_models,
                @as(f64, @floatFromInt(effective_max_resident_mem)) / 1_073_741_824.0,
            });
        } else {
            log.info("[registry] max_resident_models={d}, max_resident_mem=unlimited\n", .{max_resident_models});
        }

        const registry = try model_registry_mod.ModelRegistry.init(
            allocator,
            io,
            discovery_for_registry,
            max_resident_models,
            effective_max_resident_mem,
            idle_evict_secs,
        );
        defer registry.deinit();

        // Register the loaded model. Use the pre-registered discovery entry
        // when available (so id/path/bytes_on_disk are consistent across
        // /v1/models listings); otherwise create a fresh stub.
        const entry = if (registry.peek(model_id)) |e|
            e
        else if (registry.peekByPath(model_dir)) |e|
            // Discovered under an org/name id whose basename differs from
            // model_id — reuse it, never register the same path twice.
            e
        else
            try registry.registerStub(model_id, model_dir, null);
        try registry.setDefault(entry.id);

        // Ownership-transfer defer: registry takes ownership of
        // config/tok/chat_config IF the inference-thread load installed
        // them on `entry` (entry.config != null). Declared AFTER
        // registry.deinit so it fires BEFORE it on scope exit — by the
        // time registry.deinit walks the entry we've already decided who
        // owns the heap pointers, so the early defers can no-op.
        defer if (entry.config != null) {
            config_owned_by_registry = true;
            tok_owned_by_registry = true;
            chat_config_owned_by_registry = true;
        };

        const params = scheduler_mod.LoadParams{
            .registry = registry,
            .entry = entry,
            .config = config,
            .tok = tok,
            .chat_config = chat_config,
            .model_dir = model_dir,
            .ctx_size = ctx_size,
            .drafter_dir = drafter_dir orelse "",
            .mtp_enabled = enable_mtp,
            .mtp_depth = mtp_depth,
            .load_vision = load_vision,
            .warmup_eager = warmup_eager,
            .draft_block_size = draft_block_size,
            .draft_block_size_explicit = draft_block_size_explicit,
            .kv_quant_config = kv_quant_config,
            .prefix_cache_capacity = server_mod.prefix_cache_capacity,
            .prefix_cache_mem_bytes = server_mod.prefix_cache_mem_bytes,
            .prefix_cache_disk_bytes = server_mod.prefix_cache_disk_bytes,
            .ssm_checkpoint_stride = server_mod.ssm_checkpoint_stride,
            .ssm_checkpoint_max = server_mod.ssm_checkpoint_max,
            .tokenize_cache_entries = server_mod.tokenize_cache_entries,
            .llama_cache_entries = server_mod.llama_cache_entries,
            .llama_kv_type_k = server_mod.llama_kv_quant.ggmlType(),
            .llama_kv_type_v = server_mod.llama_kv_quant.ggmlType(),
            .metrics = server_mod.g_metrics,
        };
        try server_mod.serve(io, allocator, params, config, host, port, .{
            .max_context_size = ctx_size,
            .request_timeout_sec = timeout,
            .default_reasoning_budget = reasoning_budget,
            .default_temperature = if (temp_explicit) temperature else null,
            .default_top_p = top_p_flag,
            .default_top_k = top_k_flag,
            .default_enable_pld = enable_pld,
            .default_pld_draft_len = pld_draft_len,
            .default_pld_key_len = pld_key_len,
            .default_kv_attn_fused = kv_attn_fused_default,
        });
    } else {
        // ── Offline single-prompt mode. mlx ops run on this thread, no
        //    scheduler. The same load path as pre-A1.
        log.info("Loading weights...\n", .{});
        var weights = if (load_vision)
            try model_mod.loadWeightsWithVision(io, allocator, model_dir)
        else
            try model_mod.loadWeights(io, allocator, model_dir);
        defer weights.deinit();

        var xfm = try transformer_mod.Transformer.init(io, allocator, config.*, &weights);
        defer xfm.deinit();

        // Honor --kv-quant in offline mode too. The serve path threads this
        // through Slot caches via the scheduler; here we swap the
        // Transformer's own legacy cache to match.
        if (kv_quant_config.scheme != .off) {
            xfm.cache.deinit();
            xfm.cache = try transformer_mod.KVCache.initWithConfigAndHeadDim(allocator, config.num_hidden_layers, kv_quant_config, config.head_dim);
        }

        // JIT-compile + wire memory limits.
        {
            var dev = mlx.mlx_device{ .ctx = null };
            _ = mlx.mlx_get_default_device(&dev);
            var info = mlx.mlx_device_info_new();
            if (mlx.mlx_device_info_get(&info, dev) == 0) {
                var max_rec: usize = 0;
                if (mlx.mlx_device_info_get_size(&max_rec, info, "max_recommended_working_set_size") == 0 and max_rec > 0) {
                    var old_limit: usize = 0;
                    _ = mlx.mlx_set_wired_limit(&old_limit, max_rec);
                    log.debug("Wired limit set to {d} MB\n", .{max_rec / (1024 * 1024)});
                }
                _ = mlx.mlx_device_info_free(info);
            }
        }
        if (config.hidden_act == .gelu_approx) {
            xfm.compileGelu();
            xfm.compileGeglu();
        }
        if (config.final_logit_softcapping > 0.0) {
            xfm.compileSoftcap();
        }
        if (xfm.moe_layers != null) {
            xfm.compileMoeRouting();
        }
        if (config.linear_num_key_heads > 0) {
            xfm.compileGdnGate();
        }
        log.info("Model ready.\n", .{});

        // Qwen native MTP head — auto-load when the model ships a sidecar.
        var mtp_head: ?mtp_mod.MtpModel = null;
        defer if (mtp_head) |*h| h.deinit();
        if (enable_mtp and mtp_mod.hasMtpSidecar(io, model_dir)) {
            mtp_head = try mtp_mod.loadMtp(io, allocator, xfm.s, model_dir);
            mtp_head.?.bind(&xfm) catch |err| {
                log.warn("[mtp] sidecar incompatible with target ({any}) — disabled\n", .{err});
                mtp_head.?.deinit();
                mtp_head = null;
            };
        }

        const user_prompt = prompt orelse "What is 2+2? Answer in one sentence.";
        const messages = [_]chat_mod.Message{
            .{ .role = "user", .content = user_prompt },
        };

        const prompt_ids = try chat_mod.formatChat(allocator, tok, &messages, chat_config, null, null, false);
        defer allocator.free(prompt_ids);

        // Reset peak memory before generation
        _ = mlx.mlx_reset_peak_memory();

        const eos_slice = config.eosTokenSlice();
        const sampling = generate_mod.SamplingParams{ .temperature = temperature };

        var stdout_buf: [16 * 1024]u8 = undefined;
        var stdout_w_state = std.Io.File.stdout().writer(io, &stdout_buf);
        const stdout_w = &stdout_w_state.interface;
        defer stdout_w.flush() catch {};

        if (stream_mode) {
            // Streaming: print tokens as they're generated
            const prefill_start = std.Io.Timestamp.now(io, .awake);
            var gen = try generate_mod.Generator.init(io, allocator, &xfm, tok, prompt_ids, max_tokens, sampling, eos_slice);
            defer gen.deinit(allocator);

            const prefill_ns: u64 = @intCast(prefill_start.untilNow(io, .awake).nanoseconds);
            const prefill_tps: f64 = if (prefill_ns > 0)
                @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
            else
                0.0;

            try stdout_w.writeAll("==========\n");
            const decode_start = std.Io.Timestamp.now(io, .awake);
            var completion_tokens: u32 = 0;
            while (try gen.next(allocator)) |token_id| {
                const ids = [_]u32{token_id};
                const piece = try tok.decode(allocator, &ids, completion_tokens == 0);
                defer allocator.free(piece);
                if (piece.len > 0) {
                    try stdout_w.writeAll(piece);
                    try stdout_w.flush();
                }
                completion_tokens += 1;
            }
            const decode_ns: u64 = @intCast(decode_start.untilNow(io, .awake).nanoseconds);
            const decode_tps: f64 = if (decode_ns > 0)
                @as(f64, @floatFromInt(completion_tokens)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
            else
                0.0;

            try stdout_w.writeAll("\n==========\n");
            try stdout_w.print("Prompt: {d} tokens, {d:.3} tokens-per-sec\n", .{ prompt_ids.len, prefill_tps });
            try stdout_w.print("Generation: {d} tokens, {d:.3} tokens-per-sec\n", .{ completion_tokens, decode_tps });
        } else {
            // Non-streaming: generate all tokens then print
            const result = if (mtp_head) |*h|
                try generate_mod.generateMtp(io, allocator, &xfm, h, tok, prompt_ids, max_tokens, sampling, eos_slice, 0, mtp_depth, null)
            else
                try generate_mod.generate(io, allocator, &xfm, tok, prompt_ids, max_tokens, sampling, eos_slice, 0, 0);
            defer allocator.free(result.text);
            defer allocator.free(result.token_ids);

            try stdout_w.writeAll("==========\n");
            try stdout_w.writeAll(result.text);
            try stdout_w.writeAll("\n==========\n");
            try stdout_w.print("Prompt: {d} tokens, {d:.3} tokens-per-sec\n", .{ result.prompt_tokens, result.prefill_tps });
            try stdout_w.print("Generation: {d} tokens, {d:.3} tokens-per-sec\n", .{ result.completion_tokens, result.decode_tps });
        }

        var peak_mem: usize = 0;
        _ = mlx.mlx_get_peak_memory(&peak_mem);
        const peak_gb = @as(f64, @floatFromInt(peak_mem)) / (1024.0 * 1024.0 * 1024.0);
        try stdout_w.print("Peak memory: {d:.3} GB\n", .{peak_gb});
    }
}

/// Check if a port is already in use by trying to connect to it.
fn portInUse(io: std.Io, port: u16) bool {
    const addr: std.Io.net.IpAddress = .{ .ip4 = std.Io.net.Ip4Address.loopback(port) };
    const stream = addr.connect(io, .{ .mode = .stream }) catch return false;
    stream.close(io);
    return true;
}

// GGUF path helpers (`isGgufModelPath` / `resolveGgufFile` /
// `logResolveGgufError`) live in `model_discovery.zig` — shared with
// discovery and the scheduler's cold-load path, and hermetically tested
// there (main.zig is the executable root and not in the test pool).
const isGgufPath = model_discovery.isGgufModelPath;
const resolveGgufFile = model_discovery.resolveGgufFile;
const logResolveGgufError = model_discovery.logResolveGgufError;

/// Decide which embedded engine serves a `.gguf` file (or dir containing one).
///
/// Priority: explicit `--engine` override wins. Otherwise we read the file's
/// GGUF metadata (cheap, header-only) and route on `general.architecture`:
/// `deepseek4` + the antirez-style MLA key → ds4; everything else → llama.cpp.
/// Issue #15 — the previous basename heuristic mis-routed two real-world
/// files; see `src/gguf_meta.zig` for the rule.
///
/// On any inspection failure (file unreadable, malformed header, etc.) we
/// default to llama.cpp and log the reason. Caller still gets a sane attempt
/// (libllama will produce its own actionable error if the file isn't loadable).
fn chooseGgufEngine(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    override: ?gguf_meta.Engine,
) gguf_meta.Engine {
    if (override) |e| {
        log.info("[gguf] engine: {s} (forced via --engine)\n", .{@tagName(e)});
        return e;
    }
    const gguf_path = resolveGgufFile(io, allocator, path) catch |err| {
        log.warn("[gguf] route: cannot resolve gguf file ({s}); defaulting to llama\n", .{@errorName(err)});
        return .llama;
    };
    defer allocator.free(gguf_path);

    var info = gguf_meta.readFromFile(io, allocator, gguf_path) catch |err| {
        log.warn("[gguf] route: metadata read failed ({s}); defaulting to llama\n", .{@errorName(err)});
        return .llama;
    };
    defer info.deinit(allocator);

    const e = gguf_meta.preferredEngine(info);
    log.info("[gguf] engine: {s} (arch={s}, ds4-lora={})\n", .{
        @tagName(e),
        info.architecture orelse "?",
        info.has_ds4_lora_rank,
    });
    return e;
}

/// Offline single-prompt generation through the embedded ds4 engine.
/// Skips the MLX/safetensors scaffolding entirely — there's no `Transformer`,
/// no `Generator`, no scheduler. ds4 owns its own tokenizer, KV cache, and
/// sampler; we just feed it the user prompt and stream the decoded tokens.
fn runDs4Offline(
    io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    prompt: []const u8,
    max_tokens: u32,
    temp: f32,
    ctx_size: u32,
) !void {
    const gguf_path = resolveGgufFile(io, allocator, model_dir) catch |err| {
        logResolveGgufError(model_dir, err);
        return err;
    };
    defer allocator.free(gguf_path);

    log.info("[ds4] backend: Metal, model: {s}\n", .{gguf_path});

    var engine = ds4_arch.Ds4Engine.open(allocator, gguf_path, .{
        .backend = .metal,
        .warm_weights = true,
        .ssd_streaming = ds4_ssd_streaming,
    }) catch |err| {
        log.err("[ds4] engine open failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer engine.close();

    log.info("[ds4] engine ready (EOS={d}, has_mtp={})\n", .{ engine.eosToken(), engine.hasMtp() });

    // Render the prompt through ds4's built-in chat template. `prompt` is the
    // raw user text; the engine adds BOS, system markers, and the assistant
    // prefix according to the GGUF's vocab.
    const prompt_ids = try engine.encodeChatPrompt(allocator, null, prompt, .none);
    defer allocator.free(prompt_ids);

    log.info("[ds4] prompt: {d} tokens\n", .{prompt_ids.len});

    // ds4's session API decouples cache lifetime from a single request — one
    // session can be reused across multiple `sync` calls. ds4 sizes its
    // prefill buffers against the requested ctx (`prefill_chunk = 2048` per
    // the CLI default), and sessions smaller than the prefill chunk produce
    // junk output — so the user's --ctx-size is floored at the chunk; 0/unset
    // → ds4's default of 32768.
    const sess_ctx: i32 = @intCast(ds4_arch.clampSessionCtx(ctx_size));
    var sess = try engine.createSession(sess_ctx);
    defer sess.free();

    try sess.sync(prompt_ids);

    var rng: u64 = @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds());

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
    const out_w = &stdout.interface;
    try out_w.writeAll("\n");

    const eos = engine.eosToken();
    var generated: u32 = 0;
    while (generated < max_tokens) : (generated += 1) {
        const next_id: i32 = if (temp <= 0.0)
            sess.argmax()
        else
            sess.sample(temp, 0, 1.0, 0.05, &rng);

        if (next_id == eos) break;

        const piece = try engine.detokenizeOne(allocator, next_id);
        defer allocator.free(piece);
        try out_w.writeAll(piece);
        try out_w.flush();

        try sess.eval(next_id);
    }

    try out_w.writeAll("\n");
    try out_w.flush();
    log.info("[ds4] generated {d} tokens (max={d})\n", .{ generated, max_tokens });
}

/// Registry resident-memory cap: the user's explicit value, or 80% of mlx's
/// wired limit at startup (mirrors the MLX serve block). 0 = query failed →
/// unlimited (the count cap still applies).
fn autoResidentMemBytes(explicit: bool, val: u64) u64 {
    if (explicit) return val;
    var dev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&dev);
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) != 0) return 0;
    var max_rec: usize = 0;
    if (mlx.mlx_device_info_get_size(&max_rec, info, "max_recommended_working_set_size") != 0 or max_rec == 0) return 0;
    return @as(u64, max_rec) * 4 / 5;
}

fn dirBasename(path: []const u8) []const u8 {
    var p = path;
    while (p.len > 0 and p[p.len - 1] == '/') p = p[0 .. p.len - 1];
    if (p.len == 0) return p;
    if (std.mem.lastIndexOfScalar(u8, p, '/')) |i| return p[i + 1 ..];
    return p;
}

/// Native media-generation serve mode (image / audio / video) with the media
/// model as the PRIMARY (default) model. Builds a modality stub
/// (config/tok/chat_config) and hands it to `Scheduler.init`; the gen load arm
/// dispatches off the stub config's media `model_type` and opens the modality
/// engine on the inference thread. Coexists with on-demand chat/media loads via
/// the registry (`discovery`). Mirrors `runDs4Serve`.
fn runGenServe(
    io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    modality: gen_mod.Modality,
    discovery: ?model_discovery.DiscoveryResult,
    host: []const u8,
    port: u16,
    ctx_size: u32,
    timeout: u32,
    reasoning_budget: i32,
    max_resident_models: u32,
    max_resident_mem: u64,
    max_resident_mem_explicit: bool,
    idle_evict_secs: ?u32,
) !void {
    log.info("mlx-serve {s} (native {s} engine)\n", .{ VERSION, @tagName(modality) });
    log.info("[args] model: {s}\n", .{model_dir});
    log.info("[args] serve: {s}:{d}\n", .{ host, port });

    var stub = try gen_mod.buildStubCpuState(allocator, modality);
    var config_owned_by_registry = false;
    errdefer if (!config_owned_by_registry) gen_mod.freeStubCpuState(allocator, &stub);

    const model_id = dirBasename(model_dir);

    const effective_max_resident_mem = autoResidentMemBytes(max_resident_mem_explicit, max_resident_mem);
    if (effective_max_resident_mem > 0) {
        log.info("[registry] max_resident_models={d}, max_resident_mem={d:.1} GB\n", .{ max_resident_models, @as(f64, @floatFromInt(effective_max_resident_mem)) / 1_073_741_824.0 });
    } else {
        log.info("[registry] max_resident_models={d}, max_resident_mem=unlimited\n", .{max_resident_models});
    }

    const registry = try model_registry_mod.ModelRegistry.init(allocator, io, discovery, max_resident_models, effective_max_resident_mem, idle_evict_secs);
    defer registry.deinit();

    const entry = if (registry.peek(model_id)) |e|
        e
    else if (registry.peekByPath(model_dir)) |e|
        e
    else
        try registry.registerStubWithArch(model_id, model_dir, null, modality.modelType());
    try registry.setDefault(entry.id);

    // Registry takes ownership of the stub if the inference thread installed it.
    defer if (entry.config != null) {
        config_owned_by_registry = true;
    };

    const params = scheduler_mod.LoadParams{
        .registry = registry,
        .entry = entry,
        .config = stub.config,
        .tok = stub.tok,
        .chat_config = stub.chat_config,
        .model_dir = model_dir,
        .ctx_size = ctx_size,
        .load_vision = false,
        .warmup_eager = false,
        .draft_block_size = 0,
        .draft_block_size_explicit = false,
        .kv_quant_config = transformer_mod.KVQuantConfig.dense,
        .prefix_cache_capacity = 0,
        .prefix_cache_mem_bytes = 0,
        .tokenize_cache_entries = 0,
        .metrics = server_mod.g_metrics,
    };

    try server_mod.serve(io, allocator, params, stub.config, host, port, .{
        .max_context_size = ctx_size,
        .request_timeout_sec = timeout,
        .default_reasoning_budget = reasoning_budget,
        .default_temperature = null,
        .default_top_p = null,
        .default_top_k = null,
        .default_enable_pld = false,
        .default_pld_draft_len = 5,
        .default_pld_key_len = 3,
        .default_kv_attn_fused = false,
    });
}

/// Headless serve mode: start with NO primary model. The registry holds all
/// discovery stubs; chat AND media models load on demand via `/v1/load-model`
/// (or a request targeting a discovered id), coexisting under one memory
/// budget. The scheduler's borrowed-view fields are seeded from a throwaway
/// stub that's never installed on an entry (`no_initial_load`).
fn runHeadlessServe(
    io: std.Io,
    allocator: std.mem.Allocator,
    discovery: ?model_discovery.DiscoveryResult,
    host: []const u8,
    port: u16,
    ctx_size: u32,
    timeout: u32,
    reasoning_budget: i32,
    max_resident_models: u32,
    max_resident_mem: u64,
    max_resident_mem_explicit: bool,
    idle_evict_secs: ?u32,
) !void {
    log.info("mlx-serve {s} (headless — models load on demand)\n", .{VERSION});
    log.info("[args] serve: {s}:{d}\n", .{ host, port });

    var stub = try gen_mod.buildStubCpuState(allocator, .image);
    defer gen_mod.freeStubCpuState(allocator, &stub);

    const effective_max_resident_mem = autoResidentMemBytes(max_resident_mem_explicit, max_resident_mem);
    if (effective_max_resident_mem > 0) {
        log.info("[registry] max_resident_models={d}, max_resident_mem={d:.1} GB\n", .{ max_resident_models, @as(f64, @floatFromInt(effective_max_resident_mem)) / 1_073_741_824.0 });
    } else {
        log.info("[registry] max_resident_models={d}, max_resident_mem=unlimited\n", .{max_resident_models});
    }

    const registry = try model_registry_mod.ModelRegistry.init(allocator, io, discovery, max_resident_models, effective_max_resident_mem, idle_evict_secs);
    defer registry.deinit();

    // Carrier entry for LoadParams (required field), never loaded here
    // (`no_initial_load`). Prefer a discovered stub (so it's listed in
    // /v1/models); else a throwaway placeholder — headless is valid with empty
    // discovery because the app loads media/chat models by ABSOLUTE PATH via
    // /v1/load-model (registerByPath), regardless of what --model-dir scans.
    // No default is set, so a request that omits `model` gets a clean 503
    // until a model is loaded.
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
        log.info("Headless: no models under --model-dir; load by path via /v1/load-model.\n", .{});
        break :blk &placeholder;
    };

    const params = scheduler_mod.LoadParams{
        .registry = registry,
        .entry = carrier,
        .config = stub.config,
        .tok = stub.tok,
        .chat_config = stub.chat_config,
        .model_dir = "",
        .ctx_size = ctx_size,
        .no_initial_load = true,
        .load_vision = false,
        .warmup_eager = false,
        .draft_block_size = 0,
        .kv_quant_config = transformer_mod.KVQuantConfig.dense,
        // Seed the scheduler's prefix-cache config from the server globals so
        // on-demand (headless/discover-mode) loads get the SAME hot prefix
        // cache as a `--model` startup load. Previously hardcoded to 0, which
        // left `Scheduler.prefix_cache_capacity == 0` → every model loaded via
        // `ensureLoaded` skipped `HotPrefixCache` init → cross-turn KV reuse
        // was silently dead for the entire headless serving mode (the default
        // `serve` path). Mirrors the LoadParams built in `main()`.
        .prefix_cache_capacity = server_mod.prefix_cache_capacity,
        .prefix_cache_mem_bytes = server_mod.prefix_cache_mem_bytes,
        .prefix_cache_disk_bytes = server_mod.prefix_cache_disk_bytes,
        .ssm_checkpoint_stride = server_mod.ssm_checkpoint_stride,
        .ssm_checkpoint_max = server_mod.ssm_checkpoint_max,
        .tokenize_cache_entries = server_mod.tokenize_cache_entries,
        .metrics = server_mod.g_metrics,
    };

    try server_mod.serve(io, allocator, params, stub.config, host, port, .{
        .max_context_size = ctx_size,
        .request_timeout_sec = timeout,
        .default_reasoning_budget = reasoning_budget,
        .default_temperature = null,
        .default_top_p = null,
        .default_top_k = null,
        .default_enable_pld = false,
        .default_pld_draft_len = 5,
        .default_pld_key_len = 3,
        .default_kv_attn_fused = false,
    });
}

/// ds4 serve mode. Builds a stub LoadedModel + ModelConfig + ChatConfig
/// (the engine owns the real tokenizer and chat template internally) and
/// hands them to `Scheduler.init` via `LoadParams.ds4_path` — the scheduler's
/// inference thread opens the engine on the right GPU-stream thread. All
/// MLX-specific load steps (weights, Transformer, vision, drafter, JIT,
/// warmup) are skipped.
fn runDs4Serve(
    io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    host: []const u8,
    port: u16,
    ctx_size: u32,
    timeout: u32,
    reasoning_budget: i32,
    default_temperature: ?f32,
    default_top_p: ?f32,
    default_top_k: ?u32,
    max_resident_models: u32,
    max_resident_mem: u64,
    max_resident_mem_explicit: bool,
    idle_evict_secs: ?u32,
) !void {
    // Resolve the GGUF file once on this thread so the engine's open() call
    // (running on the inference thread) gets an absolute path.
    const gguf_path_owned = resolveGgufFile(io, allocator, model_dir) catch |err| {
        logResolveGgufError(model_dir, err);
        return err;
    };
    defer allocator.free(gguf_path_owned);

    log.info("mlx-serve {s} (ds4 engine, GGUF backend)\n", .{VERSION});
    log.info("[args] model: {s}\n", .{gguf_path_owned});
    log.info("[args] serve: {s}:{d}, ctx-size={d}\n", .{ host, port, ctx_size });

    // Build a stub ModelConfig. The fields below are read by various parts
    // of server.zig + scheduler.zig but the ds4 path bypasses anything that
    // actually consumes the model architecture (Transformer, KV shapes,
    // SSM cache, MoE routing). The values picked keep `modelBatchable`
    // returning false (we're routed through `runSingleDecodeTick`), and
    // `getEffectiveContextLength` returning the runtime ctx size.
    const config_storage = try allocator.create(model_mod.ModelConfig);
    var config_owned_by_registry = false;
    errdefer if (!config_owned_by_registry) allocator.destroy(config_storage);
    config_storage.* = model_mod.ModelConfig{
        .model_type = "deepseek_v4",
        .weight_prefix = "model",
        .num_hidden_layers = 61,
        .hidden_size = 7168,
        .head_dim = 128,
        .num_attention_heads = 56,
        .num_key_value_heads = 56,
        // Carry the user-supplied --ctx-size (floored at ds4's prefill chunk;
        // 0/unset → ds4's default) on the standard field. `runPrefillDs4` reads
        // it back to size the ds4 session, and `getEffectiveContextLength` /
        // /v1/models report it.
        .max_position_embeddings = ds4_arch.clampSessionCtx(ctx_size),
        .is_encoder_only = false,
    };

    // Stub tokenizer. Most server.zig fast paths read `lm.tokenizer.?` —
    // we build a minimal empty Tokenizer here. The chat handlers route
    // through `chat_mod.decodeViaDs4` / `encodeChatViaDs4` when
    // `lm.ds4_engine != null`, so the stub never actually services
    // encode/decode on the happy path.
    const tok_storage = try allocator.create(tokenizer_mod.Tokenizer);
    var tok_owned_by_registry = false;
    errdefer if (!tok_owned_by_registry) {
        tok_storage.deinit();
        allocator.destroy(tok_storage);
    };
    var byte_map: [256]u21 = undefined;
    var b: usize = 0;
    while (b < 256) : (b += 1) byte_map[b] = @intCast(b);
    tok_storage.* = .{
        .vocab = std.StringHashMap(u32).init(allocator),
        .id_to_token = std.AutoHashMap(u32, []const u8).init(allocator),
        .merge_ranks = @TypeOf(tok_storage.merge_ranks).init(allocator),
        .allocator = allocator,
        .special_tokens = std.StringHashMap(u32).init(allocator),
        .tok_type = .byte_level_bpe,
        .byte_to_unicode = byte_map,
        .unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator),
        .bos_id = null,
        .eos_id = null,
        .parsed_json = null,
    };

    // Stub chat config — chat template stays empty. The ds4 path renders
    // chat via the engine; the stub just keeps `lm.chat_config.?` reads
    // from crashing.
    const chat_config_storage = try allocator.create(chat_mod.ChatConfig);
    var chat_config_owned_by_registry = false;
    errdefer if (!chat_config_owned_by_registry) {
        allocator.destroy(chat_config_storage);
    };
    chat_config_storage.* = .{
        .chat_template = try allocator.dupe(u8, ""),
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    // ── Registry + scheduler scaffolding. Mirror the MLX serve branch. ──
    const model_id = blk: {
        var p = gguf_path_owned;
        while (p.len > 0 and p[p.len - 1] == '/') p = p[0 .. p.len - 1];
        if (std.mem.lastIndexOfScalar(u8, p, '/')) |slash_idx| {
            const name = p[slash_idx + 1 ..];
            break :blk if (std.mem.endsWith(u8, name, ".gguf")) name[0 .. name.len - 5] else name;
        }
        break :blk p;
    };

    const effective_max_resident_mem: u64 = if (max_resident_mem_explicit) max_resident_mem else 0;
    if (effective_max_resident_mem > 0) {
        log.info("[registry] max_resident_models={d}, max_resident_mem={d:.1} GB\n", .{
            max_resident_models,
            @as(f64, @floatFromInt(effective_max_resident_mem)) / 1_073_741_824.0,
        });
    } else {
        log.info("[registry] max_resident_models={d}, max_resident_mem=unlimited\n", .{max_resident_models});
    }

    const registry = try model_registry_mod.ModelRegistry.init(
        allocator,
        io,
        null,
        max_resident_models,
        effective_max_resident_mem,
        idle_evict_secs,
    );
    defer registry.deinit();

    // Stat the GGUF so the registry knows its on-disk size — used by
    // /v1/models, /props (memory indicator), and the LRU eviction gate.
    // Without it the Swift GPU-memory bar stays at 0 for the whole session.
    // Path is absolute; split into parent dir + basename so we can use the
    // 0.16-era `Dir.statFile` API.
    const gguf_bytes: ?u64 = blk: {
        const slash = std.mem.lastIndexOfScalar(u8, gguf_path_owned, '/') orelse break :blk null;
        const parent = gguf_path_owned[0..slash];
        const name = gguf_path_owned[slash + 1 ..];
        var dir = std.Io.Dir.openDirAbsolute(io, parent, .{}) catch break :blk null;
        defer dir.close(io);
        const st = dir.statFile(io, name, .{}) catch break :blk null;
        break :blk @as(u64, @intCast(st.size));
    };
    const entry = try registry.registerStub(model_id, gguf_path_owned, gguf_bytes);
    try registry.setDefault(model_id);

    // Once the inference thread hands ownership of the stub
    // config/tok/chat_config to the entry, the entry's deinit owns them —
    // we mustn't double-free here.
    defer if (entry.config != null) {
        config_owned_by_registry = true;
        tok_owned_by_registry = true;
        chat_config_owned_by_registry = true;
    };

    // ds4's process-wide flock makes >1 in-flight session per process
    // untested; clamp serial.
    server_mod.max_concurrent = 1;

    const params = scheduler_mod.LoadParams{
        .registry = registry,
        .entry = entry,
        .config = config_storage,
        .tok = tok_storage,
        .chat_config = chat_config_storage,
        .model_dir = gguf_path_owned, // unused on the ds4 branch but kept symmetric
        .ctx_size = ctx_size,
        .drafter_dir = "",
        .load_vision = false,
        .warmup_eager = false,
        .draft_block_size = 0,
        .draft_block_size_explicit = false,
        .kv_quant_config = transformer_mod.KVQuantConfig.dense,
        .prefix_cache_capacity = 0,
        .prefix_cache_mem_bytes = 0,
        // Iteration 2: tokenize cache for ds4 too.
        .tokenize_cache_entries = server_mod.tokenize_cache_entries,
        .ds4_path = gguf_path_owned,
        .ds4_ssd_streaming = ds4_ssd_streaming,
        .metrics = server_mod.g_metrics,
    };

    try server_mod.serve(io, allocator, params, config_storage, host, port, .{
        .max_context_size = ctx_size,
        .request_timeout_sec = timeout,
        .default_reasoning_budget = reasoning_budget,
        .default_temperature = default_temperature,
        .default_top_p = default_top_p,
        .default_top_k = default_top_k,
        .default_enable_pld = false,
        .default_pld_draft_len = 5,
        .default_pld_key_len = 3,
        .default_kv_attn_fused = false,
    });
}

/// Offline single-prompt generation through the embedded llama.cpp engine.
/// Renders the prompt via the GGUF's built-in chat template (falling back to a
/// raw tokenize when the template isn't a recognized format) and streams
/// decoded tokens to stdout. Mirrors `runDs4Offline`.
fn runLlamaOffline(
    io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    prompt: []const u8,
    max_tokens: u32,
    temp: f32,
) !void {
    const gguf_path = resolveGgufFile(io, allocator, model_dir) catch |err| {
        logResolveGgufError(model_dir, err);
        return err;
    };
    defer allocator.free(gguf_path);

    log.info("[llama] backend: Metal, model: {s}\n", .{gguf_path});

    var engine = llama_arch.LlamaEngine.open(allocator, gguf_path, .{}) catch |err| {
        log.err("[llama] engine open failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer engine.close();

    log.info("[llama] engine ready (EOS={d}, n_vocab={d})\n", .{ engine.eosToken(), engine.nVocab() });

    // Render the single user turn through the model's chat template; tokenize the
    // result with add_special=false (the template owns BOS). Fall back to a raw
    // add-special tokenize if the GGUF's template isn't recognized.
    const turns = [_]llama_arch.LlamaEngine.ChatTurn{.{ .role = "user", .content = prompt }};
    const prompt_ids: []i32 = blk: {
        if (engine.applyChatTemplate(allocator, &turns, true)) |rendered| {
            defer allocator.free(rendered);
            break :blk try engine.tokenizeText(allocator, rendered, false);
        } else |_| {
            break :blk try engine.tokenizeText(allocator, prompt, true);
        }
    };
    defer allocator.free(prompt_ids);

    log.info("[llama] prompt: {d} tokens\n", .{prompt_ids.len});

    var sess = try engine.createSession(8192);
    defer sess.free();

    _ = try sess.sync(prompt_ids); // cold session: cached count is 0, unused here

    var rng: u64 = @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds());

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
    const out_w = &stdout.interface;
    try out_w.writeAll("\n");

    var generated: u32 = 0;
    while (generated < max_tokens) : (generated += 1) {
        const next_id: i32 = if (temp < 0.01)
            sess.argmax()
        else
            sess.sample(temp, 0, 1.0, 0.0, &rng);

        if (next_id < 0 or engine.isEog(next_id)) break;

        const piece = try engine.detokenizeOne(allocator, next_id);
        defer allocator.free(piece);
        try out_w.writeAll(piece);
        try out_w.flush();

        try sess.eval(next_id);
    }

    try out_w.writeAll("\n");
    try out_w.flush();
    log.info("[llama] generated {d} tokens (max={d})\n", .{ generated, max_tokens });
}

/// llama.cpp serve mode. Builds a stub LoadedModel + ModelConfig + ChatConfig
/// and hands them to `Scheduler.init` via `LoadParams.llama_path` — the
/// scheduler's inference thread opens the engine on the GPU-stream thread and
/// adopts the GGUF's embedded chat template into the stub ChatConfig. Mirrors
/// `runDs4Serve`; all MLX-specific load steps are skipped.
fn runLlamaServe(
    io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    host: []const u8,
    port: u16,
    ctx_size: u32,
    timeout: u32,
    reasoning_budget: i32,
    default_temperature: ?f32,
    default_top_p: ?f32,
    default_top_k: ?u32,
    max_resident_models: u32,
    max_resident_mem: u64,
    max_resident_mem_explicit: bool,
    idle_evict_secs: ?u32,
) !void {
    const gguf_path_owned = resolveGgufFile(io, allocator, model_dir) catch |err| {
        logResolveGgufError(model_dir, err);
        return err;
    };
    defer allocator.free(gguf_path_owned);

    // Effective context: the user's --ctx-size, else a safe 8192 default (we
    // can't read the GGUF's trained context until the engine opens on the
    // inference thread). Used for BOTH the llama session size (via the stub
    // config's max_position_embeddings, read in runPrefillLlama) AND the
    // server's context guard (server_config.max_context_size), so they agree.
    const effective_ctx: u32 = if (ctx_size > 0) ctx_size else 8192;

    log.info("mlx-serve {s} (llama.cpp engine, GGUF backend)\n", .{VERSION});
    log.info("[args] model: {s}\n", .{gguf_path_owned});
    log.info("[args] serve: {s}:{d}, ctx-size={d}\n", .{ host, port, effective_ctx });

    // Stub ModelConfig. The llama path bypasses everything that consumes model
    // architecture (Transformer, KV shapes, SSM, MoE); only model_type (echoed
    // in /v1/models) and max_position_embeddings (session sizing) matter.
    const config_storage = try allocator.create(model_mod.ModelConfig);
    var config_owned_by_registry = false;
    errdefer if (!config_owned_by_registry) allocator.destroy(config_storage);
    config_storage.* = model_mod.ModelConfig{
        .model_type = "gguf",
        .weight_prefix = "model",
        .head_dim = 128,
        .max_position_embeddings = effective_ctx,
        .is_encoder_only = false,
    };

    // Stub tokenizer — the llama engine owns the real GGUF vocab; chat handlers
    // route through chat_mod.{encode,decode}ViaLlama when lm.llama_engine != null,
    // so this never services encode/decode on the happy path.
    const tok_storage = try allocator.create(tokenizer_mod.Tokenizer);
    var tok_owned_by_registry = false;
    errdefer if (!tok_owned_by_registry) {
        tok_storage.deinit();
        allocator.destroy(tok_storage);
    };
    var byte_map: [256]u21 = undefined;
    var b: usize = 0;
    while (b < 256) : (b += 1) byte_map[b] = @intCast(b);
    tok_storage.* = .{
        .vocab = std.StringHashMap(u32).init(allocator),
        .id_to_token = std.AutoHashMap(u32, []const u8).init(allocator),
        .merge_ranks = @TypeOf(tok_storage.merge_ranks).init(allocator),
        .allocator = allocator,
        .special_tokens = std.StringHashMap(u32).init(allocator),
        .tok_type = .byte_level_bpe,
        .byte_to_unicode = byte_map,
        .unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator),
        .bos_id = null,
        .eos_id = null,
        .parsed_json = null,
    };

    // Stub chat config — template starts empty and is replaced with the GGUF's
    // embedded template by doLoadLlamaOnInferenceThread once the engine opens.
    const chat_config_storage = try allocator.create(chat_mod.ChatConfig);
    var chat_config_owned_by_registry = false;
    errdefer if (!chat_config_owned_by_registry) {
        chat_config_storage.deinit();
        allocator.destroy(chat_config_storage);
    };
    chat_config_storage.* = .{
        .chat_template = try allocator.dupe(u8, ""),
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const model_id = blk: {
        var p = gguf_path_owned;
        while (p.len > 0 and p[p.len - 1] == '/') p = p[0 .. p.len - 1];
        if (std.mem.lastIndexOfScalar(u8, p, '/')) |slash_idx| {
            const name = p[slash_idx + 1 ..];
            break :blk if (std.mem.endsWith(u8, name, ".gguf")) name[0 .. name.len - 5] else name;
        }
        break :blk p;
    };

    const effective_max_resident_mem: u64 = if (max_resident_mem_explicit) max_resident_mem else 0;
    if (effective_max_resident_mem > 0) {
        log.info("[registry] max_resident_models={d}, max_resident_mem={d:.1} GB\n", .{
            max_resident_models,
            @as(f64, @floatFromInt(effective_max_resident_mem)) / 1_073_741_824.0,
        });
    } else {
        log.info("[registry] max_resident_models={d}, max_resident_mem=unlimited\n", .{max_resident_models});
    }

    const registry = try model_registry_mod.ModelRegistry.init(
        allocator,
        io,
        null,
        max_resident_models,
        effective_max_resident_mem,
        idle_evict_secs,
    );
    defer registry.deinit();

    const gguf_bytes: ?u64 = blk: {
        const slash = std.mem.lastIndexOfScalar(u8, gguf_path_owned, '/') orelse break :blk null;
        const parent = gguf_path_owned[0..slash];
        const name = gguf_path_owned[slash + 1 ..];
        var dir = std.Io.Dir.openDirAbsolute(io, parent, .{}) catch break :blk null;
        defer dir.close(io);
        const st = dir.statFile(io, name, .{}) catch break :blk null;
        break :blk @as(u64, @intCast(st.size));
    };
    const entry = try registry.registerStub(model_id, gguf_path_owned, gguf_bytes);
    try registry.setDefault(model_id);

    defer if (entry.config != null) {
        config_owned_by_registry = true;
        tok_owned_by_registry = true;
        chat_config_owned_by_registry = true;
    };

    // Serial for v1 — each llama session owns an independent context (memory
    // multiplies with concurrency); keep one in flight like the ds4 path.
    server_mod.max_concurrent = 1;

    const params = scheduler_mod.LoadParams{
        .registry = registry,
        .entry = entry,
        .config = config_storage,
        .tok = tok_storage,
        .chat_config = chat_config_storage,
        .model_dir = gguf_path_owned, // unused on the llama branch but kept symmetric
        .ctx_size = ctx_size,
        .drafter_dir = "",
        .load_vision = false,
        .warmup_eager = false,
        .draft_block_size = 0,
        .draft_block_size_explicit = false,
        .kv_quant_config = transformer_mod.KVQuantConfig.dense,
        .prefix_cache_capacity = 0,
        .prefix_cache_mem_bytes = 0,
        // Iteration 2 + 3-5: thread the tokenize cache + multi-session
        // LRU through the llama-specific LoadParams. doLoadLlamaOnInferenceThread
        // reads both fields.
        .tokenize_cache_entries = server_mod.tokenize_cache_entries,
        .llama_cache_entries = server_mod.llama_cache_entries,
        // Phase 5 #2: also thread the llama KV-quant types on this path
        // (the MLX branch sets them via the shared assignment, which we
        // don't reach for GGUF models).
        .llama_kv_type_k = server_mod.llama_kv_quant.ggmlType(),
        .llama_kv_type_v = server_mod.llama_kv_quant.ggmlType(),
        .llama_path = gguf_path_owned,
        .metrics = server_mod.g_metrics,
    };

    try server_mod.serve(io, allocator, params, config_storage, host, port, .{
        .max_context_size = effective_ctx,
        .request_timeout_sec = timeout,
        .default_reasoning_budget = reasoning_budget,
        .default_temperature = default_temperature,
        .default_top_p = default_top_p,
        .default_top_k = default_top_k,
        .default_enable_pld = false,
        .default_pld_draft_len = 5,
        .default_pld_key_len = 3,
        .default_kv_attn_fused = false,
    });
}

/// Parse a size-style CLI argument: bare integer = bytes, suffix `KB`/`MB`/
/// `GB` (case-insensitive) multiplies by 1024^N, "0"/"off" = 0. Used by
/// `--prefix-cache-mem`; returns `error.InvalidSize` on malformed input.
fn parseSizeArg(s: []const u8) !u64 {
    if (std.mem.eql(u8, s, "off") or std.mem.eql(u8, s, "0")) return 0;
    var end: usize = s.len;
    var mult: u64 = 1;
    if (std.mem.endsWith(u8, s, "GB") or std.mem.endsWith(u8, s, "gb")) {
        end -= 2;
        mult = 1024 * 1024 * 1024;
    } else if (std.mem.endsWith(u8, s, "MB") or std.mem.endsWith(u8, s, "mb")) {
        end -= 2;
        mult = 1024 * 1024;
    } else if (std.mem.endsWith(u8, s, "KB") or std.mem.endsWith(u8, s, "kb")) {
        end -= 2;
        mult = 1024;
    } else if (std.mem.endsWith(u8, s, "B") or std.mem.endsWith(u8, s, "b")) {
        end -= 1;
    }
    if (end == 0) return error.InvalidSize;
    const n = std.fmt.parseInt(u64, s[0..end], 10) catch return error.InvalidSize;
    return n * mult;
}

// Tests for the GGUF path helpers (`isMmprojGgufBasename`,
// `isGgufModelPath`, `resolveGgufFile`) live with the implementations in
// `src/model_discovery.zig` (where they get picked up by `zig build test`);
// main.zig itself is the executable root and is not in the test pool.
