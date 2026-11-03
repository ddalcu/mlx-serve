//! The connection thread's side of MLX's memory, through its `abi.Admission` table: the context a model advertises
//! and the admission guard that refuses, by name and with its numbers, a prompt that does not fit.

const std = @import("std");
const abi = @import("../../abi/engine.zig");
const model_mod = @import("../../sdk/config/model.zig");
const KVQuantConfig = @import("../../sdk/blocks/attention/kv_quant.zig").KVQuantConfig;
const LoadedModel = @import("../registry/registry.zig").LoadedModel;
const Conn = @import("../http/conn.zig").Conn;
const errors = @import("../http/errors.zig");
const sendAnthropicError = @import("anthropic.zig").sendAnthropicError;

const ModelConfig = model_mod.ModelConfig;
const mlx_engine = @import("../engines/mlx.zig");

fn table() *const @import("../../abi/engine.zig").Admission {
    return mlx_engine.api().admission;
}

pub fn getEffectiveContextLength(config: *const ModelConfig) u32 {
    return table().context(config);
}

/// The largest context that fits right now, before the advertised one's headroom (`/props`).
pub fn computeMaxSafeContext(config: *const ModelConfig) u32 {
    return table().max_safe_context(config);
}

/// Freezes the model's auto-context; `announce` logs where it came from.
pub fn pinAutoContext(config: *ModelConfig, announce: bool) u32 {
    return table().pin_context(config, announce);
}

/// The hot prefix cache's resolved byte budget; 0 = none.
pub fn resolvedPrefixCacheMem() u64 {
    return table().cache_budget();
}

/// Whether the MLX-prefill attention-memory preflight applies to a request.
/// The guard (`checkAttentionMemory`) models the MLX
/// transformer's per-token working set. The embedded ds4 (DeepSeek-V4-Flash)
/// and llama.cpp engines NEVER take that path — they own their KV *outside*
/// MLX — so the estimate is pure fiction for them. Their stub `ModelConfig`
/// still advertises head/layer counts (ds4: 56 heads, 61 layers, hidden 7168),
/// so without this early-out the guard projected ~25 GB for an 8.6K-token ds4
/// prompt and 400-rejected it — a prompt the SAME server had just served on the
/// MLX qwen35 engine one model-switch earlier (live, pi + ds4). Skip
/// the memory guard whenever an embedded engine will serve; the context-length
/// guard still bounds the prompt against ds4's own session ctx.
pub fn mlxMemoryGuardApplies(engine_backed: bool) bool {
    return !engine_backed;
}

pub fn checkAttentionMemory(allocator: std.mem.Allocator, stream: *Conn, prompt_ids: []const u32, max_tokens: u32, config: *const ModelConfig, is_anthropic: bool, kv_override: ?KVQuantConfig, lm: *const LoadedModel, enable_mtp: bool, media_bytes: u64) !bool {
    if (!mlxMemoryGuardApplies(lm.engineBacked())) return true;
    const kv_bits: u8 = if (kv_override) |kv| (if (kv.isQuant()) kv.bits else 0) else abi.kv_configured;
    var bill: abi.Bill = .{};
    if (table().admit(config, .of(lm.id), .of(prompt_ids), max_tokens, kv_bits, enable_mtp, media_bytes, &bill)) return true;
    // A refusal quotes the numbers it compared, hot cache included; everything the cache holds
    // is evictable here by construction (the withheld case took the deferral arm).
    const msg = try errors.memoryRefusalMessage(allocator, prompt_ids.len, bill.needed / (1024 * 1024), bill.available / (1024 * 1024), bill);
    defer allocator.free(msg);
    errors.countRejected();
    if (is_anthropic) {
        try sendAnthropicError(allocator, stream, "invalid_request_error", msg, 400);
    } else {
        try errors.sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, 400);
    }
    return false;
}

// ── tests

test "mlxMemoryGuardApplies: engine-backed models (ds4/llama/ANE) skip the MLX-prefill memory guard" {
    const t = std.testing;
    // MLX model — the guard applies (real per-token working set).
    try t.expect(mlxMemoryGuardApplies(false));
    // ds4's stub config advertises 56 heads / 61 layers, so the guard would project ~25 GB for an
    // 8.6K-token prompt and 400-reject a request ds4 serves fine; no engine-backed model prefills in MLX.
    try t.expect(!mlxMemoryGuardApplies(true));
}
