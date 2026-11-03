//! MLX's `abi.Admission` table: the context a model advertises and the admission guard, over `admission/`. `model` is
//! the model's `ModelConfig`.

const abi = @import("../../abi/engine.zig");
const ModelConfig = @import("../../sdk/config/model.zig").ModelConfig;
const KVQuantConfig = @import("../../sdk/blocks/attention/kv_quant.zig").KVQuantConfig;
const context = @import("admission/context.zig");
const admit_mod = @import("admission/admit.zig");
const resolvedPrefixCacheMem = @import("admission/hot_cache_plan.zig").resolvedPrefixCacheMem;

pub const table: abi.Admission = .{
    .context = contextOf,
    .pin_context = pinContext,
    .max_safe_context = maxSafeContext,
    .cache_budget = cacheBudget,
    .admit = admit,
};

fn config(model: *const anyopaque) *const ModelConfig {
    return @ptrCast(@alignCast(model));
}

fn contextOf(model: *const anyopaque) callconv(.c) u32 {
    return context.getEffectiveContextLength(config(model));
}

fn pinContext(model: *anyopaque, announce: bool) callconv(.c) u32 {
    const c: *ModelConfig = @ptrCast(@alignCast(model));
    const pinned = context.pinAutoContext(c);
    if (announce) context.logPinnedContext(c, pinned);
    return pinned;
}

fn maxSafeContext(model: *const anyopaque) callconv(.c) u32 {
    return context.computeMaxSafeContext(config(model));
}

fn cacheBudget() callconv(.c) u64 {
    return resolvedPrefixCacheMem();
}

fn admit(model: *const anyopaque, keep: abi.Str, prompt: abi.Span(u32), max_tokens: u32, kv_bits: u8, mtp: bool, media_bytes: u64, out: *abi.Bill) callconv(.c) bool {
    const kv: ?KVQuantConfig = switch (kv_bits) {
        abi.kv_configured => null,
        0 => .dense,
        else => .affine(kv_bits),
    };
    return admit_mod.admitPrompt(config(model), abi.items(u8, keep), abi.items(u32, prompt), max_tokens, kv, mtp, media_bytes, out);
}
