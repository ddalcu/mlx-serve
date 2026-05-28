// Zig FFI bindings for the mlx-serve llama.cpp shim (lib/llama_shim/llama_shim.h).
//
// 1:1 mirror of the shim's public header; do not add behavior here. The
// Zig-friendly wrapper that owns lifetimes and errors lives in
// `src/arch/llama.zig`. Mirroring the small purpose-built shim (rather than the
// raw, ABI-fragile llama.h structs) means an upstream header drift surfaces as a
// shim compile error in C, not as silent memory corruption in Zig — the same
// discipline as `src/ds4_ffi.zig` over `ds4.h`.

pub const Engine = opaque {};
pub const Session = opaque {};

pub extern fn mlx_llama_open(gguf_path: [*:0]const u8, n_gpu_layers: i32, err: ?[*]u8, errlen: usize) ?*Engine;
pub extern fn mlx_llama_close(e: ?*Engine) void;

pub extern fn mlx_llama_eos_token(e: *Engine) i32;
pub extern fn mlx_llama_is_eog(e: *Engine, token: i32) bool;
pub extern fn mlx_llama_n_vocab(e: *Engine) i32;

pub extern fn mlx_llama_tokenize(
    e: *Engine,
    text: [*]const u8,
    text_len: i32,
    add_special: bool,
    parse_special: bool,
    out: [*]i32,
    out_cap: i32,
) i32;
pub extern fn mlx_llama_token_to_piece(e: *Engine, token: i32, buf: [*]u8, buf_cap: i32) i32;

pub extern fn mlx_llama_chat_template(e: *Engine) ?[*:0]const u8;
pub extern fn mlx_llama_apply_chat_template(
    e: *Engine,
    roles: [*]const [*:0]const u8,
    contents: [*]const [*:0]const u8,
    n_msgs: i32,
    add_assistant: bool,
    buf: [*]u8,
    buf_cap: i32,
) i32;

pub extern fn mlx_llama_session_create(e: *Engine, n_ctx: i32, err: ?[*]u8, errlen: usize) ?*Session;
pub extern fn mlx_llama_session_create_kv_quant(
    e: *Engine,
    n_ctx: i32,
    type_k: i32,
    type_v: i32,
    err: ?[*]u8,
    errlen: usize,
) ?*Session;
pub extern fn mlx_llama_session_free(s: ?*Session) void;

/// ggml_type values from lib/llama/include/ggml.h that we expose for KV
/// quantization. F16 is the default (matches `mlx_llama_session_create`).
pub const GgmlType = struct {
    pub const F16: i32 = 1;
    pub const Q4_0: i32 = 2;
    pub const Q4_1: i32 = 3;
    pub const Q5_0: i32 = 6;
    pub const Q5_1: i32 = 7;
    pub const Q8_0: i32 = 8;
};

pub extern fn mlx_llama_session_sync(s: *Session, tokens: [*]const i32, n_tokens: i32, err: ?[*]u8, errlen: usize) i32;
pub extern fn mlx_llama_session_trim(s: *Session, n_keep: i32) i32;
pub extern fn mlx_llama_session_reset(s: *Session) void;
pub extern fn mlx_llama_session_eval(s: *Session, token: i32, err: ?[*]u8, errlen: usize) i32;
pub extern fn mlx_llama_session_sample(s: *Session, temperature: f32, top_k: i32, top_p: f32, min_p: f32, rng: *u64) i32;
pub extern fn mlx_llama_session_argmax(s: *Session) i32;
pub extern fn mlx_llama_session_pos(s: *Session) i32;
