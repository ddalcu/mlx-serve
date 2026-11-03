//! One-shot generation drivers (plain, PLD, drafter, MTP) and their result finishers.
const std = @import("std");
const transformer_mod = @import("../transformer.zig");
const tokenizer_mod = @import("../../../sdk/tokenizer/tokenizer.zig");
const log = @import("../../../sdk/util/log.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const drafter_mod = @import("../../../plugins/gemma4/assistant_drafter.zig");
const mtp_mod = @import("../../../sdk/spec/mtp/mtp.zig");
const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const DrafterModel = drafter_mod.DrafterModel;
const generate_mod = @import("../generate.zig");
const GenerationResult = @import("../../../sdk/request/result.zig").GenerationResult;
const Generator = @import("generator.zig").Generator;
const LogprobResult = @import("../sampling/logprobs.zig").LogprobResult;
const MtpHeadRef = @import("../spec/mtp_head_ref.zig").MtpHeadRef;
const SamplingParams = @import("../sampling/params.zig").SamplingParams;
const isEosId = @import("../../../sdk/request/stops.zig").isEosId;


/// Convenience: generate all tokens at once (non-streaming).
pub fn generate(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    logprobs_n: u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    // logprobs_n rides InitOptions so init's argmax-only gate sees it (a
    // post-init field write would let the split-prefill final-token forward
    // engage the pruned lm_head on a logprobs request).
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{ .logprobs_n = logprobs_n });
    gen.timeout_ns = timeout_ns;
    gen.logprobs_n = logprobs_n;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    var logprob_results = std.ArrayList(LogprobResult).empty;
    defer {
        if (logprobs_n == 0) {
            for (logprob_results.items) |*lp| allocator.free(lp.top_logprobs);
            logprob_results.deinit(allocator);
        }
    }

    timer.reset();
    while (try gen.next(allocator)) |token_id| {
        try output_ids.append(allocator, token_id);
        if (logprobs_n > 0) {
            if (gen.last_logprob) |lp| {
                try logprob_results.append(allocator, lp);
                gen.last_logprob = null; // Transfer ownership
            }
        }
    }

    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    log.debug("Decode: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        decode_ns / std.time.ns_per_ms,
        num_decoded,
        decode_tps,
    });

    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);

    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = if (logprobs_n > 0) try logprob_results.toOwnedSlice(allocator) else null,
    };
}

/// PLD-enabled non-streaming variant of `generate`. Model-agnostic — works on
/// every supported architecture, no extra weights required. Logprobs and
/// constrained sampling are unsupported (asserted out by `nextPld`).
///
/// `draft_len` and `key_len` come from server config (`--pld-draft-len` /
/// `--pld-key-len`); typical values are 5 and 3 respectively.
pub fn generatePld(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    draft_len: u32,
    key_len: u32,
    lookup_prompt: ?[]const u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{ .pld_enabled = true, .lookup_prompt = lookup_prompt });
    gen.timeout_ns = timeout_ns;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill (PLD): {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    timer.reset();

    // Decode loop. Each `nextPld` returns 1..=(1+draft_len) tokens. Stop on
    // EOS / max_tokens / timeout. We check stop conditions on every emitted
    // token (drafts can include EOS just like regular sampling) so the early
    // exit is correct.
    decode: while (!gen.done and gen.completion_tokens < max_tokens) {
        const result = (try gen.nextPld(allocator, draft_len, key_len)) orelse break;
        defer allocator.free(result.tokens);
        // Match `generate`'s convention: stop tokens are NOT included in
        // output_ids. Check before appending — the speculative path has to do
        // this explicitly because `nextPld` emits multiple tokens at once and
        // can't return-null mid-batch like the single-token `next` does.
        for (result.tokens) |tok_id| {
            if (isEosId(tok_id, eos_token_ids)) {
                gen.done = true;
                gen.finish_reason = "stop";
                break :decode;
            }
            try output_ids.append(allocator, tok_id);
            if (output_ids.items.len >= max_tokens) {
                gen.done = true;
                gen.finish_reason = "length";
                break :decode;
            }
        }
        if (timeout_ns > 0 and timer.read() >= timeout_ns) {
            gen.done = true;
            gen.finish_reason = "length";
            break;
        }
    }

    return finishPldResult(&gen, &output_ids, allocator, prefill_tps, timer, tok);
}

/// Drafter-enabled non-streaming variant of `generate`. Mirrors
/// `generatePld` (multi-token-per-step emit pattern) but the draft comes from
/// a Gemma 4 assistant drafter cross-attending into the target's KV cache
/// instead of an n-gram lookup.
///
/// `drafter` must already be `bind()`-ed to `xfm`. `block_size` is the
/// per-round token budget (drafter forwards = block_size - 1; verify forward
/// length = block_size).
pub fn generateDrafter(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    drafter: *DrafterModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    block_size: u32,
    lookup_prompt: ?[]const u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{
        .drafter_enabled = true,
        .drafter = drafter,
        .drafter_block_size = block_size,
        .lookup_prompt = lookup_prompt,
    });
    gen.timeout_ns = timeout_ns;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill (drafter): {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    timer.reset();

    decode: while (!gen.done and gen.completion_tokens < max_tokens) {
        const result = (try gen.nextDrafter(allocator)) orelse break;
        defer allocator.free(result.tokens);
        for (result.tokens) |tok_id| {
            if (isEosId(tok_id, eos_token_ids)) {
                gen.done = true;
                gen.finish_reason = "stop";
                break :decode;
            }
            try output_ids.append(allocator, tok_id);
            if (output_ids.items.len >= max_tokens) {
                gen.done = true;
                gen.finish_reason = "length";
                break :decode;
            }
        }
        if (timeout_ns > 0 and timer.read() >= timeout_ns) {
            gen.done = true;
            gen.finish_reason = "length";
            break;
        }
    }

    return finishDrafterResult(&gen, &output_ids, allocator, prefill_tps, timer, tok);
}

fn finishDrafterResult(
    gen: *Generator,
    output_ids: *std.ArrayList(u32),
    allocator: std.mem.Allocator,
    prefill_tps: f64,
    timer: io_util.Stopwatch,
    tok: *const Tokenizer,
) !GenerationResult {
    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    if (gen.drafter_attempted > 0) {
        const avg_acc: f64 = @as(f64, @floatFromInt(gen.drafter_accepted_tokens)) / @as(f64, @floatFromInt(gen.drafter_attempted));
        log.info("Decode (drafter): {d}ms ({d} tokens, {d:.3} tok/s; drafter accept={d}/{d} attempts, avg {d:.2} tokens/attempt)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
            gen.drafter_accepted_tokens,
            gen.drafter_attempted,
            avg_acc,
        });
    } else {
        log.debug("Decode (drafter): {d}ms ({d} tokens, {d:.3} tok/s; no draft attempts)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
        });
    }
    gen.logSpecStats();
    gen.logQsaArms();
    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);
    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = null,
    };
}

/// MTP-enabled non-streaming variant of `generate`. Mirrors `generateDrafter`
/// but drives `nextMtp` (the model's own multi-token-prediction head).
/// `head` must already be `bind()`-ed to `xfm`.
pub fn generateMtp(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    head: *mtp_mod.MtpModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    depth: u32,
    lookup_prompt: ?[]const u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{
        .mtp_enabled = true,
        .mtp_acceptance = generate_mod.mtp_acceptance_default,
        .mtp_greedy_tail = generate_mod.mtp_greedy_tail_default,
        .mtp = MtpHeadRef{ .qwen = head },
        .mtp_depth = depth,
        .lookup_prompt = lookup_prompt,
    });
    gen.timeout_ns = timeout_ns;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill (mtp): {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    timer.reset();

    decode: while (!gen.done and gen.completion_tokens < max_tokens) {
        const result = (try gen.nextMtp(allocator)) orelse break;
        defer allocator.free(result.tokens);
        for (result.tokens) |tok_id| {
            if (isEosId(tok_id, eos_token_ids)) {
                gen.done = true;
                gen.finish_reason = "stop";
                break :decode;
            }
            try output_ids.append(allocator, tok_id);
            if (output_ids.items.len >= max_tokens) {
                gen.done = true;
                gen.finish_reason = "length";
                break :decode;
            }
        }
        if (timeout_ns > 0 and timer.read() >= timeout_ns) {
            gen.done = true;
            gen.finish_reason = "length";
            break;
        }
    }

    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    if (gen.mtp_attempted > 0) {
        const avg_acc: f64 = @as(f64, @floatFromInt(gen.mtp_accepted_tokens)) / @as(f64, @floatFromInt(gen.mtp_attempted));
        log.info("Decode (mtp): {d}ms ({d} tokens, {d:.3} tok/s; mtp accept={d}/{d} attempts, avg {d:.2} tokens/attempt)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
            gen.mtp_accepted_tokens,
            gen.mtp_attempted,
            avg_acc,
        });
    } else {
        log.debug("Decode (mtp): {d}ms ({d} tokens, {d:.3} tok/s; no draft attempts)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
        });
    }
    gen.logSpecStats();
    gen.logQsaArms();
    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);
    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = null,
    };
}

fn finishPldResult(
    gen: *Generator,
    output_ids: *std.ArrayList(u32),
    allocator: std.mem.Allocator,
    prefill_tps: f64,
    timer: io_util.Stopwatch,
    tok: *const Tokenizer,
) !GenerationResult {
    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    if (gen.pld_attempted > 0) {
        // "Tokens saved" = accepted_tokens (drafts that landed) + 0 from
        // verify forwards that ran. Acceptance ratio is per-position, so we
        // compute average tokens accepted per attempt for visibility.
        const avg_acc: f64 = @as(f64, @floatFromInt(gen.pld_accepted_tokens)) / @as(f64, @floatFromInt(gen.pld_attempted));
        log.info("Decode (PLD): {d}ms ({d} tokens, {d:.3} tok/s; pld accept={d}/{d} attempts, avg {d:.2} tokens/attempt)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
            gen.pld_accepted_tokens,
            gen.pld_attempted,
            avg_acc,
        });
    } else {
        log.debug("Decode (PLD): {d}ms ({d} tokens, {d:.3} tok/s; no n-gram matches found)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
        });
    }
    gen.logSpecStats();
    gen.logQsaArms();
    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);
    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = null,
    };
}
