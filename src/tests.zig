// Test root — imports all modules to run their embedded tests.
// Run with: zig build test

test {
    _ = @import("log.zig");
    _ = @import("chat.zig");
    _ = @import("format_corpus_test.zig");
    _ = @import("server.zig");
    _ = @import("model.zig");
    _ = @import("generate.zig");
    _ = @import("transformer.zig");
    _ = @import("vision.zig");
    _ = @import("qwen_vision.zig");
    _ = @import("mrope.zig");
    _ = @import("regex.zig");
    _ = @import("json_schema.zig");
    _ = @import("json_grammar.zig");
    _ = @import("token_mask.zig");
    _ = @import("responses.zig");
    _ = @import("ws.zig");
    _ = @import("pld_index.zig");
    _ = @import("kv_quant.zig");
    _ = @import("drafter.zig");
    _ = @import("mtp.zig");
    _ = @import("diffusion.zig");
    _ = @import("tokenizer.zig");
    _ = @import("prefix_cache.zig");
    _ = @import("model_discovery.zig");
    _ = @import("gguf_meta.zig");
    _ = @import("model_registry.zig");
    _ = @import("scheduler.zig");
    _ = @import("ds4_ffi.zig");
    _ = @import("arch/ds4.zig");
    _ = @import("llama_ffi.zig");
    _ = @import("arch/llama.zig");
}
