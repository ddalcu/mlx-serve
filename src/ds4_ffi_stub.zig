//! iOS stub for `ds4_ffi.zig`. The ds4 (DeepSeek-V4-Flash) engine is a
//! macOS-Metal-host C library that isn't built for iOS. iOS serves MLX
//! safetensors models only, so every ds4 code path is dead at runtime
//! (`LoadedModel.ds4_engine` is always null). These declarations exist purely
//! so the shared scheduler/server/chat code type-checks for the iOS target.
//!
//! Kept in sync with the public surface of `ds4_ffi.zig` that the engine
//! references. Selected via `build_options.ios` in build.zig.

pub const Backend = enum(c_int) {
    metal = 0,
    cuda = 1,
    cpu = 2,
};

pub const ThinkMode = enum(c_int) {
    none = 0,
    high = 1,
    max = 2,
};

pub const ContextMemory = extern struct {
    total_bytes: u64,
    raw_bytes: u64,
    compressed_bytes: u64,
    scratch_bytes: u64,
    prefill_cap: u32,
    raw_cap: u32,
    comp_cap: u32,
};

pub fn ds4_context_memory_estimate(backend: Backend, ctx_size: c_int) ContextMemory {
    _ = backend;
    _ = ctx_size;
    return .{
        .total_bytes = 0,
        .raw_bytes = 0,
        .compressed_bytes = 0,
        .scratch_bytes = 0,
        .prefill_cap = 0,
        .raw_cap = 0,
        .comp_cap = 0,
    };
}
