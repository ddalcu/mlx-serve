//! Scheduler launch and submit parameters, the sibling promise ledger, and the prefill chunk hooks.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const tokenizer_mod = @import("../../../sdk/tokenizer/tokenizer.zig");
const generate_mod = @import("../generate.zig");
const drafter_mod = @import("../../../plugins/gemma4/assistant_drafter.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const chat_mod = @import("../../../host/chat/chat.zig");
const prefix_cache_mod = @import("../cache/prefix_cache.zig");
const metrics_mod = @import("../../../host/metrics/metrics.zig");
const model_registry_mod = @import("../../../host/registry/registry.zig");
const ModelConfig = model_mod.ModelConfig;
const Tokenizer = tokenizer_mod.Tokenizer;
const SamplingParams = generate_mod.SamplingParams;
const DrafterModel = drafter_mod.DrafterModel;
const dflash_mod = @import("../spec/dflash/dflash.zig");
const DflashModel = dflash_mod.DflashModel;
const ChatConfig = chat_mod.ChatConfig;
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;
const testing = std.testing;

const BudgetRevise = @import("../cache/budget.zig").BudgetRevise;
const scheduler_mod = @import("../scheduler.zig");

pub const LlamaSettings = @import("../../../sdk/quant/gguf/llama_settings.zig").Settings;
const GgufRoute = @import("../runtime/thread.zig").GgufRoute;

pub const LoadParams = @import("../../../host/gpu_thread/params.zig").LoadParams;
pub const SubmitParams = @import("../../../host/gpu_thread/params.zig").SubmitParams;
pub const PromiseLedger = @import("../../../host/gpu_thread/params.zig").PromiseLedger;
pub const PROMISE_MATERIALIZED_TOKENS = @import("../../../host/gpu_thread/params.zig").PROMISE_MATERIALIZED_TOKENS;
pub const SIBLING_MARGIN_BYTES = @import("../../../host/gpu_thread/params.zig").SIBLING_MARGIN_BYTES;
pub const promiseMaterialized = @import("../../../host/gpu_thread/params.zig").promiseMaterialized;

pub const PostEvictionWidth = struct {
    /// The width the prefill runs at. Always the re-ask.
    width: u32,
    widened: bool,
    /// The re-ask came back narrower than the admitted width: memory moved between the reads.
    moved: bool,
};

/// The prefill width to run, given the width admission was billed at before the eviction
/// pass and the width re-asked against live memory after it. The re-ask wins in both
/// directions: `@max` would widen on memory that is gone. `admitted == 0` = no pass ran.
pub fn postEvictionPrefillChunk(admitted: u32, reasked: u32) PostEvictionWidth {
    return .{
        .width = reasked,
        .widened = admitted != 0 and reasked > admitted,
        .moved = admitted != 0 and reasked < admitted,
    };
}

/// What `Generator.InitOptions.adaptive_chunk_width` gets for a slot.
pub fn adaptiveChunkWidthFor(cfg: ?*const model_mod.ModelConfig) bool {
    const c = cfg orelse return false;
    const enabled = scheduler_mod.prefill_chunk_adaptive_enabled orelse return false;
    return enabled(c);
}

// ── tests ──

test "postEvictionPrefillChunk: the re-asked width is the one that runs, in BOTH directions" {
    // 1. Admitted at 1024, affords 4096 after the pass: runs at 4096, logs the widen.
    {
        const d = postEvictionPrefillChunk(1024, 4096);
        try testing.expectEqual(@as(u32, 4096), d.width);
        try testing.expect(d.widened);
        try testing.expect(!d.moved);
    }
    try testing.expectEqual(@as(u32, 8192), postEvictionPrefillChunk(512, 8192).width);

    // 2. No eviction pass ran: the re-ask is the only ask, nothing reportable.
    {
        const d = postEvictionPrefillChunk(0, 1024);
        try testing.expectEqual(@as(u32, 1024), d.width);
        try testing.expect(!d.widened);
        try testing.expect(!d.moved);
    }
    try testing.expectEqual(@as(u32, 512), postEvictionPrefillChunk(0, 512).width);
    try testing.expectEqual(@as(u32, 0), postEvictionPrefillChunk(0, 0).width);

    // 3. A narrower re-ask still wins (`@max` would widen on memory that is gone).
    {
        const d = postEvictionPrefillChunk(4096, 1024);
        try testing.expectEqual(@as(u32, 1024), d.width);
        try testing.expect(d.moved);
        try testing.expect(!d.widened);
    }
    try testing.expectEqual(@as(u32, 512), postEvictionPrefillChunk(2048, 512).width);

    // 4. Unchanged memory is a no-op in both directions.
    inline for (.{ 512, 1024, 2048, 4096, 8192 }) |w| {
        const d = postEvictionPrefillChunk(w, w);
        try testing.expectEqual(@as(u32, w), d.width);
        try testing.expect(!d.widened and !d.moved);
    }

    // 5. Never below the ladder floor: this function computes no width of its own.
    inline for (.{ 512, 1024, 2048, 4096, 8192 }) |floor| {
        try testing.expectEqual(@as(u32, floor), postEvictionPrefillChunk(8192, floor).width);
    }
}
