//! The offline single-prompt mode (`--prompt` without `--serve`): one generation on the calling thread, no scheduler.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const host = @import("../../host/engines/mlx.zig");
const model_mod = @import("../../sdk/config/model.zig");
const chat_mod = @import("../../host/chat/chat.zig");
const mtp_mod = @import("../../sdk/spec/mtp/mtp.zig");
const transformer_mod = @import("transformer.zig");
const generate_mod = @import("generate.zig");
const round_cost_mod = @import("spec/round_cost.zig");
const plugin_driver = @import("plugin/driver.zig");
const depth_bounds = @import("spec/mtp_depth_bounds.zig");
const mtp_graft = @import("../../plugins/prism_hadamard_qwen35/mtp_graft.zig");

pub fn run(o: *const host.Offline) ?[]const u8 {
    runOrErr(o) catch |e| return @errorName(e);
    return null;
}

fn runOrErr(o: *const host.Offline) !void {
    const io = o.io;
    const allocator = o.allocator;
    const config = o.config;
    const tok = o.tok;
    const chat_config = o.chat_config;
    const model_dir = o.model_dir;
    const load_vision = o.load_vision;
    const kv_quant_config = o.kv_quant;
    const enable_mtp = o.enable_mtp;
    const temperature = o.temperature;
    const max_tokens = o.max_tokens;
    const stream_mode = o.stream;
    // ── Offline single-prompt mode. mlx ops run on this thread, no
    //    scheduler. The same load path as pre-A1.
    log.info("Loading weights...\n", .{});
    // A served model loads through its plugin, as the serve path does (`engines/mlx/load.zig`).
    const served = config.plugin != null;
    if (!served and !config.mlxStream()) return error.PluginDeclinedNoLegacy;
    var weights = if (served) model_mod.Weights.init(allocator) else try model_mod.loadModelWeights(io, allocator, model_dir, config, load_vision);
    defer weights.deinit();
    if (!served) {
        model_mod.resolveWeightPrefix(config, &weights);
        try model_mod.narrowHadamardPackTables(config, &weights, mlx.gpuStream());
    }

    var xfm = if (served)
        try plugin_driver.loadWith(io, allocator, config, model_dir, mlx.gpuStream(), if (load_vision) &weights else null)
    else
        try transformer_mod.Transformer.init(io, allocator, config.*);
    defer xfm.deinit();

    xfm.round_cost.layout = round_cost_mod.layoutFor(config);

    // Reserved-token suppression, same derivation as the serve path.
    generate_mod.installSuppressMask(&xfm, tok, chat_config.chat_template, config.eosTokenSlice());

    // Honor --kv-quant in offline mode too. The serve path threads this
    // through Slot caches via the scheduler; here we swap the
    // Transformer's own legacy cache to match.
    if (kv_quant_config.scheme != .off) {
        try xfm.cache.reinit(config.num_hidden_layers, kv_quant_config);
    }
    if (xfm.qwen4Trunk()) |t| try t.qwen4MtpApplyKvQuant(transformer_mod.Transformer.qwen4MtpHeadKvConfig(kv_quant_config));

    // JIT-compile + wire memory limits (policy: mlx.applyWiredPolicy).
    {
        const wired = mlx.applyWiredPolicy();
        if (wired.target) |t| log.debug("[wired] mode={s} limit={d} MB\n", .{ @tagName(wired.mode), t / (1024 * 1024) });
    }
    if (config.hidden_act == .gelu_approx) {
        xfm.compileGelu();
        xfm.compileGeglu();
    }
    if (xfm.moe_layers != null) {
        xfm.compileMoeRouting();
    }
    if (config.linear_num_key_heads > 0) {
        xfm.compileGdnGate();
    }
    log.info("Model ready.\n", .{});

    // Qwen native MTP head — auto-load when the model ships one (sidecar
    // file or in-checkpoint tensors in the trunk shards).
    var mtp_head: ?mtp_mod.MtpModel = null;
    defer if (mtp_head) |*h| h.deinit();
    if (enable_mtp and !served) mtp_graft.ensure(allocator, io, model_dir, config);
    // A served model's head is its plugin's (`SpecHooks.head`), served by the scheduler path only.
    if (served and enable_mtp) log.info("[mtp] offline mode runs a served model without its MTP head\n", .{});
    if (!served and enable_mtp and mtp_mod.hasMtpHead(io, allocator, model_dir)) {
        // A failed load (e.g. a sidecar layout we can't bind yet) only
        // disables the head — mirrors the serve path's graceful degrade.
        if (mtp_mod.loadMtp(io, allocator, xfm.s, model_dir)) |loaded| {
            mtp_head = loaded;
            mtp_head.?.bind(&xfm) catch |err| {
                log.warn("[mtp] sidecar incompatible with target ({any}) — disabled\n", .{err});
                mtp_head.?.deinit();
                mtp_head = null;
            };
        } else |err| {
            log.warn("[mtp] failed to load sidecar ({any}) — disabled\n", .{err});
        }
    }

    const user_prompt = o.prompt;
    const messages = [_]chat_mod.Message{
        .{ .role = "user", .content = user_prompt },
    };

    const prompt_ids = try chat_mod.formatChat(allocator, tok, &messages, chat_config, null, null, false, null, false);
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
            try generate_mod.generateMtp(io, allocator, &xfm, h, tok, prompt_ids, max_tokens, sampling, eos_slice, 0, depth_bounds.active.max, null)
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
