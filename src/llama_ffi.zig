// Zig FFI bindings for the mlx-serve llama.cpp shim (lib/llama_shim/llama_shim.h).
//
// 1:1 mirror of the shim's public header; do not add behavior here. The
// Zig-friendly wrapper that owns lifetimes and errors lives in
// `src/arch/llama.zig`. Mirroring the small purpose-built shim (rather than the
// raw, ABI-fragile llama.h structs) means an upstream header drift surfaces as a
// shim compile error in C, not as silent memory corruption in Zig — the same
// discipline as `src/ds4_ffi.zig` over `ds4.h`.

pub const Engine = opaque {};
pub const Ctx = opaque {};

pub extern fn mlx_llama_open(
    gguf_path: [*:0]const u8,
    n_gpu_layers: i32,
    mtp_path: ?[*:0]const u8,
    load_mtp: bool,
    err: ?[*]u8,
    errlen: usize,
) ?*Engine;
pub extern fn mlx_llama_close(e: ?*Engine) void;

pub extern fn mlx_llama_eos_token(e: *Engine) i32;
pub extern fn mlx_llama_is_eog(e: *Engine, token: i32) bool;
pub extern fn mlx_llama_n_vocab(e: *Engine) i32;
pub extern fn mlx_llama_has_mtp(e: *Engine) bool;

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

pub const CtxParams = extern struct {
    n_ctx: i32 = 0,
    n_seq: i32 = 1,
    type_k: i32 = 0,
    type_v: i32 = 0,
    n_ubatch: i32 = 0,
    mtp_drafts: i32 = 0,
};

/// ggml_type values from lib/llama/include/ggml.h that we expose for KV
/// quantization. F16 is the default.
pub const GgmlType = struct {
    pub const F16: i32 = 1;
    pub const Q4_0: i32 = 2;
    pub const Q4_1: i32 = 3;
    pub const Q5_0: i32 = 6;
    pub const Q5_1: i32 = 7;
    pub const Q8_0: i32 = 8;
};

pub extern fn mlx_llama_ctx_create(e: *Engine, p: *const CtxParams, err: ?[*]u8, errlen: usize) ?*Ctx;
pub extern fn mlx_llama_ctx_free(c: ?*Ctx) void;
pub extern fn mlx_llama_ctx_mtp_drafts(c: *Ctx) i32;

pub extern fn mlx_llama_seq_prefill(c: *Ctx, seq: i32, tokens: [*]const i32, n_tokens: i32, err: ?[*]u8, errlen: usize) i32;
pub extern fn mlx_llama_step(c: *Ctx, seqs: [*]const i32, tokens: [*]const i32, n: i32, err: ?[*]u8, errlen: usize) i32;
pub extern fn mlx_llama_seq_trim(c: *Ctx, seq: i32, n_keep: i32) i32;
pub extern fn mlx_llama_seq_reset(c: *Ctx, seq: i32) void;
pub extern fn mlx_llama_seq_pos(c: *Ctx, seq: i32) i32;
pub extern fn mlx_llama_seq_sample(c: *Ctx, seq: i32, temperature: f32, top_k: i32, top_p: f32, min_p: f32, rng: *u64) i32;
pub extern fn mlx_llama_seq_spec_step(
    c: *Ctx,
    seq: i32,
    id_last: i32,
    max_drafts: i32,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    rng: *u64,
    out: [*]i32,
    err: ?[*]u8,
    errlen: usize,
) i32;
