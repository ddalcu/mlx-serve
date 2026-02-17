const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const tokenizer_mod = @import("tokenizer.zig");
const transformer_mod = @import("transformer.zig");
const generate_mod = @import("generate.zig");
const chat_mod = @import("chat.zig");
const server_mod = @import("server.zig");

const DEFAULT_MODEL_DIR = "/Volumes/Sandisk_1TB/Models/mlx-community/gemma-3-12b-it-qat-4bit";

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Print MLX version
    var ver = mlx.mlx_string_new();
    defer _ = mlx.mlx_string_free(ver);
    try mlx.check(mlx.mlx_version(&ver));
    std.debug.print("MLX version: {s}\n", .{mlx.mlx_string_data(ver)});

    // Set GPU as default
    var metal_avail: bool = false;
    try mlx.check(mlx.mlx_metal_is_available(&metal_avail));
    std.debug.print("Metal GPU: {}\n", .{metal_avail});

    if (metal_avail) {
        const gpu_dev = mlx.mlx_device_new_type(.gpu, 0);
        defer _ = mlx.mlx_device_free(gpu_dev);
        try mlx.check(mlx.mlx_set_default_device(gpu_dev));
    }

    // Seed MLX RNG with current time for non-deterministic sampling
    _ = mlx.mlx_random_seed(@intCast(std.time.milliTimestamp()));

    // Parse CLI args
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var model_dir: []const u8 = DEFAULT_MODEL_DIR;
    var port: u16 = 8080;
    var serve_mode = false;
    var prompt: ?[]const u8 = null;
    var max_tokens: u32 = 100;
    var temperature: f32 = 0.0;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
            i += 1;
            model_dir = args[i];
        } else if (std.mem.eql(u8, args[i], "--port") and i + 1 < args.len) {
            i += 1;
            port = try std.fmt.parseInt(u16, args[i], 10);
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
        }
    }

    // Parse config
    const config = try model_mod.parseConfig(allocator, model_dir);
    std.debug.print("Model: {s} ({d} layers, {d}-dim, {d}-bit quant)\n", .{
        config.model_type,
        config.num_hidden_layers,
        config.hidden_size,
        config.quant_bits,
    });

    // Load tokenizer
    std.debug.print("Loading tokenizer...\n", .{});
    var tok = try tokenizer_mod.loadTokenizer(allocator, model_dir);
    defer tok.deinit();

    // Load chat config
    var chat_config = try chat_mod.loadChatConfig(allocator, model_dir);
    defer chat_config.deinit();

    // Load weights
    std.debug.print("Loading weights...\n", .{});
    var weights = try model_mod.loadWeights(allocator, model_dir);
    defer weights.deinit();

    // Initialize transformer
    var xfm = try transformer_mod.Transformer.init(allocator, config, &weights);
    defer xfm.deinit();

    std.debug.print("Model ready.\n", .{});

    if (serve_mode) {
        // Start HTTP server
        try server_mod.serve(allocator, &xfm, &tok, &chat_config, &config, port);
    } else {
        const user_prompt = prompt orelse "What is 2+2? Answer in one sentence.";
        const messages = [_]chat_mod.Message{
            .{ .role = "user", .content = user_prompt },
        };

        const prompt_ids = try chat_mod.formatChat(allocator, &tok, &messages, &chat_config);
        defer allocator.free(prompt_ids);

        // Reset peak memory before generation
        _ = mlx.mlx_reset_peak_memory();

        const eos_slice = config.eosTokenSlice();
        const result = try generate_mod.generate(allocator, &xfm, &tok, prompt_ids, max_tokens, temperature, eos_slice);
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
