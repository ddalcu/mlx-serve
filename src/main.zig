const std = @import("std");
const build_options = @import("build_options");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const tokenizer_mod = @import("tokenizer.zig");
const transformer_mod = @import("transformer.zig");
const generate_mod = @import("generate.zig");
const chat_mod = @import("chat.zig");
const server_mod = @import("server.zig");
const log = @import("log.zig");

pub const VERSION: []const u8 = build_options.version;

const DEFAULT_MODEL_DIR = "/Volumes/Sandisk_1TB/Models/mlx-community/gemma-3-12b-it-qat-4bit";

fn printUsage() void {
    const stdout = std.fs.File.stdout();
    stdout.writeAll(
        \\mlx-serve — MLX inference server for Apple Silicon
        \\
        \\Usage: mlx-serve [options]
        \\
        \\Options:
        \\  --model <dir>       Path to MLX model directory (required)
        \\  --serve             Start HTTP server mode
        \\  --host <ip>         Bind address (default: 0.0.0.0)
        \\  --port <n>          Bind port (default: 8080)
        \\  --ctx-size <n>      Maximum context length (default: model max)
        \\  --prompt <text>     Run single prompt (interactive mode)
        \\  --max-tokens <n>    Max tokens to generate (default: 100)
        \\  --temp <f>          Temperature (default: 0.0)
        \\  --timeout <n>       Request timeout in seconds (default: 300, 0=none)
        \\  --log-level <lvl>   Log level: error, warn, info, debug (default: info)
        \\  --version           Print version and exit
        \\  --help              Show this help
        \\
    ) catch {};
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse CLI args (before model loading for --version/--help)
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var model_dir: []const u8 = DEFAULT_MODEL_DIR;
    var port: u16 = 8080;
    var host: []const u8 = "0.0.0.0";
    var serve_mode = false;
    var prompt: ?[]const u8 = null;
    var max_tokens: u32 = 100;
    var temperature: f32 = 0.0;
    var ctx_size: u32 = 0; // 0 = use model default
    var timeout: u32 = 300; // seconds, 0 = no timeout
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--version")) {
            const stdout = std.fs.File.stdout();
            stdout.writeAll("mlx-serve " ++ VERSION ++ "\n") catch {};
            return;
        } else if (std.mem.eql(u8, args[i], "--help") or std.mem.eql(u8, args[i], "-h")) {
            printUsage();
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
        } else if (std.mem.eql(u8, args[i], "--log-level") and i + 1 < args.len) {
            i += 1;
            if (log.Level.fromString(args[i])) |level| {
                log.setLevel(level);
            }
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

    // Seed MLX RNG with current time for non-deterministic sampling
    _ = mlx.mlx_random_seed(@intCast(std.time.milliTimestamp()));

    // Parse config
    var config = try model_mod.parseConfig(allocator, model_dir);
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
    var tok = try tokenizer_mod.loadTokenizer(allocator, model_dir);
    defer tok.deinit();

    // Load chat config
    var chat_config = try chat_mod.loadChatConfig(allocator, model_dir);
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

    // Load weights
    log.info("Loading weights...\n", .{});
    var weights = try model_mod.loadWeights(allocator, model_dir);
    defer weights.deinit();

    // Initialize transformer
    var xfm = try transformer_mod.Transformer.init(allocator, config, &weights);
    defer xfm.deinit();

    if (ctx_size > 0) {
        log.info("Context size: {d} tokens\n", .{ctx_size});
    }
    log.info("Model ready.\n", .{});

    if (serve_mode) {
        // Start HTTP server
        try server_mod.serve(allocator, &xfm, &tok, &chat_config, &config, host, port, ctx_size, timeout);
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
        const result = try generate_mod.generate(allocator, &xfm, &tok, prompt_ids, max_tokens, sampling, eos_slice, 0, 0);
        defer allocator.free(result.text);
        defer allocator.free(result.token_ids);

        // Print mlx_lm-style output to stdout
        const stdout = std.fs.File.stdout();
        try stdout.writeAll("==========\n");
        try stdout.writeAll(result.text);
        try stdout.writeAll("\n==========\n");

        var stat_buf: [256]u8 = undefined;
        var msg = std.fmt.bufPrint(&stat_buf, "Prompt: {d} tokens, {d:.3} tokens-per-sec\n", .{ result.prompt_tokens, result.prefill_tps }) catch unreachable;
        try stdout.writeAll(msg);
        msg = std.fmt.bufPrint(&stat_buf, "Generation: {d} tokens, {d:.3} tokens-per-sec\n", .{ result.completion_tokens, result.decode_tps }) catch unreachable;
        try stdout.writeAll(msg);

        var peak_mem: usize = 0;
        _ = mlx.mlx_get_peak_memory(&peak_mem);
        const peak_gb = @as(f64, @floatFromInt(peak_mem)) / (1024.0 * 1024.0 * 1024.0);
        msg = std.fmt.bufPrint(&stat_buf, "Peak memory: {d:.3} GB\n", .{peak_gb}) catch unreachable;
        try stdout.writeAll(msg);
    }
}
