const std = @import("std");
const build_options = @import("build_options");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const tokenizer_mod = @import("tokenizer.zig");
const transformer_mod = @import("transformer.zig");
const generate_mod = @import("generate.zig");
const chat_mod = @import("chat.zig");
const server_mod = @import("server.zig");
const vision_mod = @import("vision.zig");
const log = @import("log.zig");

pub const VERSION: []const u8 = build_options.version;

const DEFAULT_MODEL_DIR = ""; // pass --model <path> to specify

fn printUsage(io: std.Io) void {
    var stdout_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writer(io, &stdout_buf);
    stdout_w.interface.writeAll(
        \\mlx-serve — MLX inference server for Apple Silicon
        \\
        \\Usage: mlx-serve [options]
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
        \\  --temp <f>          Temperature (default: 0.0)
        \\  --timeout <n>       Request timeout in seconds (default: 300, 0=none)
        \\  --reasoning-budget <n>  Max thinking tokens per request (default: unlimited)
        \\  --no-vision         Disable vision encoder (saves memory)
        \\  --mtp               Enable MTP (Multi-Token Prediction) speculative decoding
        \\                        Requires a model with `num_nextn_predict_layers > 0`
        \\                        in config.json (Qwen3.5+, Qwen3-Next).
        \\  --pld               Enable Prompt Lookup Decoding (model-agnostic
        \\                        speculative decoding via n-gram matches in the
        \\                        prompt + generated tokens). Big wins on echo-heavy
        \\                        workloads (code editing, RAG, agentic loops).
        \\  --pld-draft-len <n> Max draft tokens per PLD step (default: 5).
        \\  --pld-key-len <n>   N-gram match key length for PLD (default: 3).
        \\  --log-level <lvl>   Log level: error, warn, info, debug (default: info)
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

    var model_dir: []const u8 = DEFAULT_MODEL_DIR;
    var port: u16 = 11234;
    var host: []const u8 = "0.0.0.0";
    var serve_mode = false;
    var stream_mode = false;
    var prompt: ?[]const u8 = null;
    var max_tokens: u32 = 100;
    var temperature: f32 = 0.0;
    var ctx_size: u32 = 0; // 0 = use model default
    var timeout: u32 = 300; // seconds, 0 = no timeout
    var reasoning_budget: i32 = -1; // -1 = unlimited
    var no_vision = false;
    var enable_mtp = false; // MTP self-speculative decoding (off by default)
    var enable_pld = false; // Prompt Lookup Decoding (off by default)
    var pld_draft_len: u32 = 5;
    var pld_key_len: u32 = 3;
    var i: usize = 1;
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
        } else if (std.mem.eql(u8, args[i], "--ctx-size") and i + 1 < args.len) {
            i += 1;
            ctx_size = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--timeout") and i + 1 < args.len) {
            i += 1;
            timeout = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--no-vision")) {
            no_vision = true;
        } else if (std.mem.eql(u8, args[i], "--mtp")) {
            enable_mtp = true;
        } else if (std.mem.eql(u8, args[i], "--no-mtp")) {
            enable_mtp = false;
        } else if (std.mem.eql(u8, args[i], "--pld")) {
            enable_pld = true;
        } else if (std.mem.eql(u8, args[i], "--no-pld")) {
            enable_pld = false;
        } else if (std.mem.eql(u8, args[i], "--pld-draft-len") and i + 1 < args.len) {
            i += 1;
            pld_draft_len = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--pld-key-len") and i + 1 < args.len) {
            i += 1;
            pld_key_len = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--reasoning-budget") and i + 1 < args.len) {
            i += 1;
            reasoning_budget = try std.fmt.parseInt(i32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--log-level") and i + 1 < args.len) {
            i += 1;
            if (log.Level.fromString(args[i])) |level| {
                log.setLevel(level);
            }
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

    // Print MLX version
    var ver = mlx.mlx_string_new();
    defer _ = mlx.mlx_string_free(ver);
    try mlx.check(mlx.mlx_version(&ver));
    log.info("mlx-serve {s} (MLX {s})\n", .{ VERSION, mlx.mlx_string_data(ver) });

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

    // Parse config
    var config = try model_mod.parseConfig(io, allocator, model_dir);
    log.info("Model: {s} ({d} layers, {d}-dim, head_dim={d}, {d}h/{d}kv, {d}-bit quant)\n", .{
        config.model_type,
        config.num_hidden_layers,
        config.hidden_size,
        config.head_dim,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.quant_bits,
    });

    // Load tokenizer
    log.info("Loading tokenizer...\n", .{});
    var tok = try tokenizer_mod.loadTokenizer(io, allocator, model_dir);
    defer tok.deinit();

    // Load chat config
    var chat_config = try chat_mod.loadChatConfig(io, allocator, model_dir);
    defer chat_config.deinit();

    // Resolve EOS tokens from tokenizer if config.json didn't specify any
    if (config.num_eos_tokens == 0) {
        if (chat_config.eos_token) |eos_str| {
            if (tok.special_tokens.get(eos_str)) |eos_id| {
                config.addEosToken(eos_id);
                log.info("EOS token from tokenizer: {s} (id={d})\n", .{ eos_str, eos_id });
            }
        }
        // Also add <|endoftext|> if it exists and wasn't already added
        if (tok.special_tokens.get("<|endoftext|>")) |eot_id| {
            if (!config.isEosToken(eot_id)) {
                config.addEosToken(eot_id);
            }
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
    try config.populateUserTurnMarker(allocator, &tok, chat_config.chat_template);

    // Load weights (include vision weights if model has vision config and not disabled)
    const load_vision = config.has_vision and !no_vision;
    log.info("Loading weights...\n", .{});
    var weights = if (load_vision)
        try model_mod.loadWeightsWithVision(io, allocator, model_dir)
    else
        try model_mod.loadWeights(io, allocator, model_dir);
    defer weights.deinit();

    // Initialize transformer
    var xfm = try transformer_mod.Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();

    // Initialize vision encoder if model supports it (and not disabled)
    var vision_enc: ?vision_mod.VisionEncoder = if (load_vision) blk: {
        log.info("Initializing vision encoder...\n", .{});
        break :blk vision_mod.VisionEncoder.init(allocator, config, &weights) catch |err| {
            if (err == error.MissingVisionWeights) {
                log.warn("Vision weights missing — vision disabled (model may have been quantized without vision tower)\n", .{});
                break :blk null;
            }
            return err;
        };
    } else null;
    defer {
        if (vision_enc) |*ve| ve.deinit();
    }

    // Wire model weights into GPU memory (prevents paging, matches mlx-lm behavior)
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

    // JIT-compile activation functions (fuses ops → single kernels, matching mlx-lm)
    if (config.hidden_act == .gelu_approx) {
        xfm.compileGelu();
        xfm.compileGeglu(); // gelu(gate) * up → 1 kernel
    }
    if (config.final_logit_softcapping > 0.0) {
        xfm.compileSoftcap(); // tanh(x/cap) * cap → 1 kernel
    }

    log.info("Model ready.\n", .{});

    if (serve_mode) {
        // Start HTTP server
        // MTP guardrail: --mtp on a model without an MTP head silently downgrades.
        const mtp_active = enable_mtp and config.has_mtp;
        if (enable_mtp and !config.has_mtp) {
            log.warn("--mtp requested but model has no MTP head (num_nextn_predict_layers=0); running without speculative decoding.\n", .{});
        }
        try server_mod.serve(io, allocator, &xfm, &tok, &chat_config, &config, if (vision_enc) |*ve| ve else null, model_dir, host, port, ctx_size, timeout, reasoning_budget, mtp_active, enable_pld, pld_draft_len, pld_key_len);
    } else {
        const user_prompt = prompt orelse "What is 2+2? Answer in one sentence.";
        const messages = [_]chat_mod.Message{
            .{ .role = "user", .content = user_prompt },
        };

        const prompt_ids = try chat_mod.formatChat(allocator, &tok, &messages, &chat_config, null, null, false);
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
            var gen = try generate_mod.Generator.init(io, allocator, &xfm, &tok, prompt_ids, max_tokens, sampling, eos_slice);
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
            const result = try generate_mod.generate(io, allocator, &xfm, &tok, prompt_ids, max_tokens, sampling, eos_slice, 0, 0);
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
