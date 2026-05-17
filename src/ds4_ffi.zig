// Zig FFI bindings for the ds4 inference engine (lib/ds4/ds4.h).
//
// 1:1 mirror of the public header; do not add behavior here. The Zig-friendly
// wrapper that owns lifetimes, errors, and Metal-kernel extraction lives in
// `src/arch/ds4.zig`. Keeping this layer mechanical means an upstream `ds4.h`
// drift shows up as a Zig compile error here rather than in the bridge.
//
// Submodule pin: lib/ds4 @ 613e9b2.

const std = @import("std");

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

pub const LogType = enum(c_int) {
    default = 0,
    prefill = 1,
    generation = 2,
    kvcache = 3,
    tool = 4,
    warning = 5,
    timing = 6,
    ok = 7,
    err = 8,
};

pub const Tokens = extern struct {
    v: ?[*]c_int = null,
    len: c_int = 0,
    cap: c_int = 0,
};

pub const TokenScore = extern struct {
    id: c_int,
    logit: f32,
    logprob: f32,
};

pub const EngineOptions = extern struct {
    model_path: ?[*:0]const u8 = null,
    mtp_path: ?[*:0]const u8 = null,
    backend: Backend = .metal,
    n_threads: c_int = 0,
    mtp_draft_tokens: c_int = 0,
    mtp_margin: f32 = 0,
    directional_steering_file: ?[*:0]const u8 = null,
    directional_steering_attn: f32 = 0,
    directional_steering_ffn: f32 = 0,
    warm_weights: bool = false,
    quality: bool = false,
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

pub const SessionSnapshot = extern struct {
    ptr: ?[*]u8 = null,
    len: u64 = 0,
    cap: u64 = 0,
};

pub const SessionRewriteResult = enum(c_int) {
    err = -1,
    ok = 0,
    rebuild_needed = 1,
};

pub const Engine = opaque {};
pub const Session = opaque {};

pub const SessionProgressFn = ?*const fn (ud: ?*anyopaque, event: [*:0]const u8, current: c_int, total: c_int) callconv(.C) void;
pub const TokenEmitFn = ?*const fn (ud: ?*anyopaque, token: c_int) callconv(.C) void;
pub const GenerationDoneFn = ?*const fn (ud: ?*anyopaque) callconv(.C) void;

pub extern fn ds4_engine_open(out: *?*Engine, opt: *const EngineOptions) c_int;
pub extern fn ds4_engine_close(e: ?*Engine) void;
pub extern fn ds4_engine_summary(e: ?*Engine) void;
pub extern fn ds4_backend_name(backend: Backend) [*:0]const u8;
pub extern fn ds4_think_mode_enabled(mode: ThinkMode) bool;
pub extern fn ds4_think_mode_name(mode: ThinkMode) [*:0]const u8;
pub extern fn ds4_think_max_prefix() [*:0]const u8;
pub extern fn ds4_think_max_min_context() u32;
pub extern fn ds4_think_mode_for_context(mode: ThinkMode, ctx_size: c_int) ThinkMode;
pub extern fn ds4_context_memory_estimate(backend: Backend, ctx_size: c_int) ContextMemory;
pub extern fn ds4_log_is_tty(fp: ?*anyopaque) bool;

pub extern fn ds4_tokens_push(tv: *Tokens, token: c_int) void;
pub extern fn ds4_tokens_free(tv: *Tokens) void;
pub extern fn ds4_tokens_copy(dst: *Tokens, src: *const Tokens) void;
pub extern fn ds4_tokens_starts_with(tokens: *const Tokens, prefix: *const Tokens) bool;

pub extern fn ds4_tokenize_text(e: *Engine, text: [*:0]const u8, out: *Tokens) void;
pub extern fn ds4_tokenize_rendered_chat(e: *Engine, text: [*:0]const u8, out: *Tokens) void;
pub extern fn ds4_chat_begin(e: *Engine, tokens: *Tokens) void;
pub extern fn ds4_encode_chat_prompt(
    e: *Engine,
    system: ?[*:0]const u8,
    prompt: [*:0]const u8,
    think_mode: ThinkMode,
    out: *Tokens,
) void;
pub extern fn ds4_chat_append_max_effort_prefix(e: *Engine, tokens: *Tokens) void;
pub extern fn ds4_chat_append_message(e: *Engine, tokens: *Tokens, role: [*:0]const u8, content: [*:0]const u8) void;
pub extern fn ds4_chat_append_assistant_prefix(e: *Engine, tokens: *Tokens, think_mode: ThinkMode) void;

pub extern fn ds4_token_text(e: *Engine, token: c_int, len: *usize) ?[*]u8;
pub extern fn ds4_token_eos(e: *Engine) c_int;
pub extern fn ds4_token_user(e: *Engine) c_int;
pub extern fn ds4_token_assistant(e: *Engine) c_int;

pub extern fn ds4_session_create(out: *?*Session, e: *Engine, ctx_size: c_int) c_int;
pub extern fn ds4_session_free(s: ?*Session) void;
pub extern fn ds4_session_set_progress(s: *Session, f: SessionProgressFn, ud: ?*anyopaque) void;

pub extern fn ds4_session_sync(s: *Session, prompt: *const Tokens, err: ?[*]u8, errlen: usize) c_int;
pub extern fn ds4_session_rewrite_requires_rebuild(live_len: c_int, canonical_len: c_int, common: c_int) bool;
pub extern fn ds4_session_rewrite_from_common(
    s: *Session,
    prompt: *const Tokens,
    common: c_int,
    err: ?[*]u8,
    errlen: usize,
) SessionRewriteResult;
pub extern fn ds4_session_common_prefix(s: *Session, prompt: *const Tokens) c_int;
pub extern fn ds4_session_argmax(s: *Session) c_int;
pub extern fn ds4_session_argmax_excluding(s: *Session, excluded_id: c_int) c_int;
pub extern fn ds4_session_sample(
    s: *Session,
    temperature: f32,
    top_k: c_int,
    top_p: f32,
    min_p: f32,
    rng: *u64,
) c_int;
pub extern fn ds4_session_top_logprobs(s: *Session, out: [*]TokenScore, k: c_int) c_int;
pub extern fn ds4_session_token_logprob(s: *Session, token: c_int, out: *TokenScore) c_int;
pub extern fn ds4_session_eval(s: *Session, token: c_int, err: ?[*]u8, errlen: usize) c_int;
pub extern fn ds4_session_eval_speculative_argmax(
    s: *Session,
    first_token: c_int,
    max_tokens: c_int,
    eos_token: c_int,
    accepted: ?[*]c_int,
    accepted_cap: c_int,
    err: ?[*]u8,
    errlen: usize,
) c_int;
pub extern fn ds4_session_invalidate(s: *Session) void;
pub extern fn ds4_session_rewind(s: *Session, pos: c_int) void;
pub extern fn ds4_session_pos(s: *Session) c_int;
pub extern fn ds4_session_ctx(s: *Session) c_int;
pub extern fn ds4_engine_routed_quant_bits(e: *Engine) c_int;
pub extern fn ds4_engine_has_mtp(e: *Engine) bool;
pub extern fn ds4_engine_mtp_draft_tokens(e: *Engine) c_int;
pub extern fn ds4_session_tokens(s: *Session) *const Tokens;

pub extern fn ds4_session_payload_bytes(s: *Session) u64;
pub extern fn ds4_session_save_snapshot(s: *Session, snap: *SessionSnapshot, err: ?[*]u8, errlen: usize) c_int;
pub extern fn ds4_session_load_snapshot(s: *Session, snap: *const SessionSnapshot, err: ?[*]u8, errlen: usize) c_int;
pub extern fn ds4_session_snapshot_free(snap: *SessionSnapshot) void;
