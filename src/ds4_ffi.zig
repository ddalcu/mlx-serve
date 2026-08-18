// Zig FFI bindings for the ds4 inference engine (lib/ds4/ds4.h).
//
// 1:1 mirror of the public header; do not add behavior here. The Zig-friendly
// wrapper that owns lifetimes, errors, and Metal-kernel extraction lives in
// `src/arch/ds4.zig`. Keeping this layer mechanical means an upstream `ds4.h`
// drift shows up as a Zig compile error here rather than in the bridge.
//
// Submodule pin: lib/ds4 @ efdadd4.

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

pub const DistributedRole = enum(c_int) {
    none = 0,
    coordinator = 1,
    worker = 2,
};

pub const DistributedLayers = extern struct {
    start: u32 = 0,
    end: u32 = 0,
    has_output: bool = false,
    set: bool = false,
};

pub const DistributedOptions = extern struct {
    role: DistributedRole = .none,
    layers: DistributedLayers = .{},
    listen_host: ?[*:0]const u8 = null,
    listen_port: c_int = 0,
    coordinator_host: ?[*:0]const u8 = null,
    coordinator_port: c_int = 0,
    prefill_chunk: u32 = 0,
    prefill_window: u32 = 0,
    activation_bits: u32 = 0,
    replay_check: bool = false,
    debug: bool = false,
};

// Two-machine tensor parallelism (pin efdadd4). Never enabled by mlx-serve —
// mirrored only because ds4_engine_options embeds it by value.
pub const TpRole = enum(c_int) {
    none = 0,
    leader = 1,
    worker = 2,
};

pub const TpTransport = enum(c_int) {
    auto = 0,
    rdma = 1,
    tcp = 2,
};

pub const TpOptions = extern struct {
    role: TpRole = .none,
    requested: bool = false,
    listen_host: ?[*:0]const u8 = null,
    listen_port: c_int = 0,
    leader_host: ?[*:0]const u8 = null,
    leader_port: c_int = 0,
    transport: TpTransport = .auto,
    rdma_device: ?[*:0]const u8 = null,
    rdma_gid_index: c_int = 0,
    rdma_gid_index_set: bool = false,
    glm_token_prefill: bool = false,
    debug_hash: c_int = 0,
};

// Mirrors `ds4_engine_options` in lib/ds4/ds4.h EXACTLY (field order + types are
// the C ABI contract; a mismatch silently corrupts the struct at open time —
// the layout test at the bottom of this file pins it against the real header).
pub const EngineOptions = extern struct {
    model_path: ?[*:0]const u8 = null,
    mtp_path: ?[*:0]const u8 = null,
    backend: Backend = .metal,
    n_threads: c_int = 0,
    context_size: c_int = 0,
    prefill_chunk: u32 = 0,
    mtp_draft_tokens: c_int = 0,
    mtp_margin: f32 = 0,
    dspark_confidence_threshold: f32 = 0,
    directional_steering_file: ?[*:0]const u8 = null,
    expert_profile_path: ?[*:0]const u8 = null,
    directional_steering_attn: f32 = 0,
    directional_steering_ffn: f32 = 0,
    power_percent: c_int = 0,
    ssd_streaming_cache_experts: u32 = 0,
    ssd_streaming_cache_bytes: u64 = 0,
    ssd_streaming_full_layers: u32 = 0,
    ssd_streaming_preload_experts: u32 = 0,
    simulate_used_memory_bytes: u64 = 0,
    warm_weights: bool = false,
    quality: bool = false,
    glm_mtp: bool = false,
    glm_mtp_timing: bool = false,
    dspark: bool = false,
    dspark_strict: bool = false,
    dspark_confidence_threshold_set: bool = false,
    cuda_tensor_parallel: bool = false,
    ssd_streaming: bool = false,
    ssd_streaming_cold: bool = false,
    ssd_streaming_full_layers_set: bool = false,
    inspect_only: bool = false,
    placement_ctx_hint: c_int = 0,
    share_session_prefill_workspace: bool = false,
    first_token_test: bool = false,
    metal_graph_test: bool = false,
    load_slice: bool = false,
    load_layer_start: u32 = 0,
    load_layer_end: u32 = 0,
    load_output: bool = false,
    distributed: DistributedOptions = .{},
    tp: TpOptions = .{},
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

// Layout cross-check exports from src/ds4_layout_check.c (compiled against the
// real lib/ds4/ds4.h). Test-only; never call these on a hot path.
extern fn mlxserve_ds4_sizeof_engine_options() usize;
extern fn mlxserve_ds4_offsetof_mtp_draft_tokens() usize;
extern fn mlxserve_ds4_offsetof_ssd_streaming() usize;
extern fn mlxserve_ds4_offsetof_distributed() usize;
extern fn mlxserve_ds4_sizeof_distributed_options() usize;
extern fn mlxserve_ds4_sizeof_tokens() usize;
extern fn mlxserve_ds4_sizeof_context_memory() usize;
extern fn mlxserve_ds4_sizeof_session_snapshot() usize;

test "ds4 FFI mirror layout matches ds4.h (mid-struct-insert guard)" {
    // Upstream inserts fields mid-struct on upgrades; a stale mirror corrupts
    // ds4_engine_options at open time and generation hangs with no error.
    // sizeof catches inserts/appends; the three offsets catch reorders in the
    // head, the bool block, and the tail respectively.
    try std.testing.expectEqual(mlxserve_ds4_sizeof_engine_options(), @sizeOf(EngineOptions));
    try std.testing.expectEqual(mlxserve_ds4_offsetof_mtp_draft_tokens(), @offsetOf(EngineOptions, "mtp_draft_tokens"));
    try std.testing.expectEqual(mlxserve_ds4_offsetof_ssd_streaming(), @offsetOf(EngineOptions, "ssd_streaming"));
    try std.testing.expectEqual(mlxserve_ds4_offsetof_distributed(), @offsetOf(EngineOptions, "distributed"));
    try std.testing.expectEqual(mlxserve_ds4_sizeof_distributed_options(), @sizeOf(DistributedOptions));
    try std.testing.expectEqual(mlxserve_ds4_sizeof_tokens(), @sizeOf(Tokens));
    try std.testing.expectEqual(mlxserve_ds4_sizeof_context_memory(), @sizeOf(ContextMemory));
    try std.testing.expectEqual(mlxserve_ds4_sizeof_session_snapshot(), @sizeOf(SessionSnapshot));
}
