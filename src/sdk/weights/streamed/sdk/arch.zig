//! The `arch` kind: one erased table per arch, built at comptime from the plugin's namespace. Resolved once: `claims`
//! at discovery, `parse` at config parse, `apply_settings` at the load sites, `init` at load; the host copies `caps`
//! into its own tables there. Called once per prompt, serial step, handover and draft round, never per layer: the
//! routed experts, the quant and the reader stay inside the plugin, bound at comptime. A hook that fails at load
//! refuses the load by name; an installed lane runs directly.

const std = @import("std");
const mlx = @import("mlx_host").mlx;
const peek = @import("peek.zig");
const spec = @import("spec.zig");
const bill = @import("memory_bill.zig");
const check = @import("check.zig");
const weights_mod = @import("weights.zig");
const Weights = weights_mod.Weights;
const WeightLoader = weights_mod.WeightLoader;

const Allocator = std.mem.Allocator;

/// Facts the host copies once at load into its own tables.
pub const Caps = struct {
    /// G1: per-request decode state lives on the arch's module, not the host's KVCache: no batched decode,
    /// single-flight admission, no KVCache snapshot or rewind; the prefix cache only through `restore_prefix`.
    owns_decode_state: bool = false,
    /// The host sends the whole prompt in one forward; the arch chunks it.
    prefill_whole_prompt: bool = false,
    /// The prompt forward returns the last row's logits: no separate one-row forward.
    prefill_yields_last_logits: bool = false,
    /// The proposal's opt-in; an arch that owns its decode state is never batched.
    batches_decode: bool = false,
    /// The loader reads the resident weights past the page cache unless the model setting says otherwise.
    residents_past_page_cache: bool = false,
};

/// The parsed model's facts the host keeps on its own config.
pub const Shell = struct {
    num_experts: u32 = 0,
    num_layers: u32 = 0,
    /// The installed KV precision (0 = bf16, 8 = affine 8-bit); null when the arch has no choice.
    kv_quant_bits: ?u8 = null,
};

/// The host's generic streamed-expert facts at load (CLI flags and the preflight's sample).
pub const LoadFacts = struct {
    /// The memory in use before the model loads (`--memory-baseline-gb`, else the preflight's reading).
    memory_baseline_bytes: ?u64 = null,
    /// Decode slot rows per layer (`--expert-rows`); null = the arch's fill.
    expert_rows: ?u32 = null,
    /// Prompt slot rows per layer (a harness's); null = the fill's.
    expert_prefill_rows: ?u32 = null,
    /// The loader's page-cache setting for resident weights (`nocache_weights`; null = the arch's default).
    nocache_weights: ?bool = null,
    /// The context the load serves (the model's `ctx_size`, else the arch's default context), stamped before billing.
    effective_context: ?u64 = null,
    /// A construction-time KV precision choice (0 = bf16, 8 = affine 8-bit); null = the arch's default.
    kv_quant_bits: ?u8 = null,
    /// What a plan leaves unplanned under the GPU memory ceiling (`--wired-margin-gib` / `--wired-margin`), stamped
    /// once at load; the arch never reads the host's flag.
    wired_margin_bytes: u64,
};

/// What `init` builds the module from.
pub const LoadCtx = struct {
    gpa: Allocator,
    io: std.Io,
    stream: mlx.mlx_stream,
    /// The loaded residents (the host's weight map).
    weights: *Weights,
    /// The host's loaders, for a sidecar the model directory's index does not name.
    loader: *const WeightLoader,
    facts: LoadFacts,
    /// The GPU memory ceiling every plan fits under (an argument, never a global).
    ceiling: u64,
};

/// A request as the host sees it at its prompt, set once per request before its first forward: each arch derives
/// its own reservation from it. `prompt_tokens` counts absolute positions; `host_context` is the window the host
/// serves the model within.
pub const RequestShape = struct { prompt_tokens: u64, max_tokens: u32, host_context: u64 };

/// The prefill-to-decode handover (G2: the phase change), once per request at its first decode step, serial or
/// lane: the request's prompt length, the positions it reserved, and whether the host drives draft rounds.
pub const DecodeHandover = struct { prompt_tokens: u32, reserved_tokens: u64, native_draft: bool };

pub const Arch = struct {
    name: []const u8,
    caps: Caps,
    claims: *const fn (p: *const peek.ConfigPeek) ?peek.Priority,
    /// The arch's own config (refusals by name through `diag`); owned by the host's config, freed by `free_config`.
    parse: *const fn (gpa: Allocator, p: *const peek.ConfigPeek, diag: *peek.Diag) anyerror!*anyopaque,
    free_config: *const fn (gpa: Allocator, cfg: *anyopaque) void,
    shell: *const fn (cfg: *const anyopaque) Shell,
    /// This model's model-settings.json object (`.null` when it has none).
    apply_settings: *const fn (cfg: *anyopaque, raw: std.json.Value) void,
    /// The load preflight's requirement: the bytes the arch needs free to load at all.
    load_bytes: *const fn (gpa: Allocator, io: std.Io, cfg: *const anyopaque, facts: *const LoadFacts, ceiling: u64) anyerror!u64,
    /// The prompt admission's bytes for one request; null = the host's own estimator.
    prompt_bytes: ?*const fn (cfg: *const anyopaque, seq: u64, max_tokens: u32) u64,
    init: *const fn (load: *const LoadCtx, cfg: *const anyopaque) anyerror!*anyopaque,
    deinit: *const fn (m: *anyopaque) void,
    /// The prompt pass (a new request): the last row's logits.
    prefill: *const fn (m: *anyopaque, ids: []const u32, req: RequestShape) anyerror!mlx.mlx_array,
    /// A serial step after the prompt.
    step: *const fn (m: *anyopaque, ids: []const u32) anyerror!mlx.mlx_array,
    /// The committed length; the host mirrors its cache step from it.
    position: *const fn (m: *const anyopaque) u64,
    /// An arch that owns its decode state under the host's prefix cache: the host matched `prefix` (its own match and
    /// policy) and would not run it again; the module keeps at most that many positions of its state and returns how
    /// many (0: none, the prompt runs whole). The host then sends the rest to `prefill`, its `RequestShape.prompt_tokens`
    /// counting the kept ones. Once per request, before its prompt pass; null = the host's prefix cache stays off.
    restore_prefix: ?*const fn (m: *anyopaque, prefix: []const u32) u64,
    /// The phase change; null = the arch has none.
    handover: ?*const fn (m: *anyopaque, h: DecodeHandover) anyerror!void,
    /// The request's end (the host's finish, on the inference thread, after its last step); null = none.
    request_end: ?*const fn (m: *anyopaque) void,
    spec: spec.Spec,
    /// G4: the arch's terms of the composed bill (waves, KV by owner, prompt state, cache limits); null = none.
    /// Pure host: it may read the model's headers through `io`, never the device.
    bill: ?*const fn (gpa: Allocator, io: std.Io, req: *const bill.BillRequest) anyerror!bill.MemoryBill,
    /// The most tokens a request may generate past its prompt, from the arch's config with the model's settings
    /// applied (the host serves the billed prompts plus these); null = the plugin's `generation_headroom`.
    max_output: ?*const fn (cfg: *const anyopaque) u64 = null,
    /// A resource the arch holds once per process (an expert reader): the host claims it at the load claim, before
    /// the preflight and the weights, and releases it when the loaded model goes (or the load fails). A second load
    /// that needs it is refused by the claim's error. Null = the arch claims nothing.
    claim_process: ?*const fn () anyerror!void,
    release_process: ?*const fn () void,

    /// The table of `T`, a namespace declaring the arch (a missing or mistyped declaration is a compile error
    /// naming it): name, caps, claims, Config, parse, freeConfig, shell, applySettings, loadBytes, Module, init,
    /// deinit, prefill, step, position; optional (absent when undeclared or `{}`) promptBytes, handover, requestEnd,
    /// restorePrefix (only with `owns_decode_state`), draft_lane, bill, maxOutput, and the pair claimProcess /
    /// releaseProcess.
    pub fn of(comptime T: type) Arch {
        comptime {
            const w = "arch " ++ @typeName(T);
            check.nameDecl(w, T);
            check.valueDecl(w, T, "caps", Caps);
            if (T.caps.owns_decode_state and T.caps.batches_decode) @compileError(w ++ ": batches_decode with owns_decode_state");
            check.fnDecl(w, T, "claims", &.{*const peek.ConfigPeek}, ?peek.Priority);
            check.typeDecl(w, T, "Config");
            check.typeDecl(w, T, "Module");
            check.fnDecl(w, T, "parse", &.{ Allocator, *const peek.ConfigPeek, *peek.Diag }, *T.Config);
            check.fnDecl(w, T, "freeConfig", &.{ Allocator, *T.Config }, void);
            check.fnDecl(w, T, "shell", &.{*const T.Config}, Shell);
            check.fnDecl(w, T, "applySettings", &.{ *T.Config, std.json.Value }, void);
            check.fnDecl(w, T, "loadBytes", &.{ Allocator, std.Io, *const T.Config, *const LoadFacts, u64 }, u64);
            if (check.has(T, "promptBytes")) check.fnDecl(w, T, "promptBytes", &.{ *const T.Config, u64, u32 }, u64);
            check.fnDecl(w, T, "init", &.{ *const LoadCtx, *const T.Config }, *T.Module);
            check.fnDecl(w, T, "deinit", &.{*T.Module}, void);
            check.fnDecl(w, T, "prefill", &.{ *T.Module, []const u32, RequestShape }, mlx.mlx_array);
            check.fnDecl(w, T, "step", &.{ *T.Module, []const u32 }, mlx.mlx_array);
            check.fnDecl(w, T, "position", &.{*const T.Module}, u64);
            if (check.has(T, "handover")) check.fnDecl(w, T, "handover", &.{ *T.Module, DecodeHandover }, void);
            if (check.has(T, "requestEnd")) check.fnDecl(w, T, "requestEnd", &.{*T.Module}, void);
            if (check.has(T, "restorePrefix")) {
                if (!T.caps.owns_decode_state) @compileError(w ++ ": restorePrefix without owns_decode_state");
                check.fnDecl(w, T, "restorePrefix", &.{ *T.Module, []const u32 }, u64);
            }
            if (check.has(T, "bill")) check.fnDecl(w, T, "bill", &.{ Allocator, std.Io, *const bill.BillRequest }, bill.MemoryBill);
            if (check.has(T, "maxOutput")) check.fnDecl(w, T, "maxOutput", &.{*const T.Config}, u64);
            if (check.has(T, "claimProcess") != check.has(T, "releaseProcess")) @compileError(w ++ ": claimProcess and releaseProcess come as a pair");
            if (check.has(T, "claimProcess")) {
                check.fnDecl(w, T, "claimProcess", &.{}, void);
                check.fnDecl(w, T, "releaseProcess", &.{}, void);
            }
        }
        const W = struct {
            fn cfgOf(cfg: *anyopaque) *T.Config {
                return @ptrCast(@alignCast(cfg));
            }
            fn constCfg(cfg: *const anyopaque) *const T.Config {
                return @ptrCast(@alignCast(cfg));
            }
            fn mod(m: *anyopaque) *T.Module {
                return @ptrCast(@alignCast(m));
            }
            fn parse(gpa: Allocator, p: *const peek.ConfigPeek, diag: *peek.Diag) anyerror!*anyopaque {
                return try T.parse(gpa, p, diag);
            }
            fn freeConfig(gpa: Allocator, cfg: *anyopaque) void {
                T.freeConfig(gpa, cfgOf(cfg));
            }
            fn shell(cfg: *const anyopaque) Shell {
                return T.shell(constCfg(cfg));
            }
            fn applySettings(cfg: *anyopaque, raw: std.json.Value) void {
                T.applySettings(cfgOf(cfg), raw);
            }
            fn loadBytes(gpa: Allocator, io: std.Io, cfg: *const anyopaque, facts: *const LoadFacts, ceiling: u64) anyerror!u64 {
                return T.loadBytes(gpa, io, constCfg(cfg), facts, ceiling);
            }
            fn promptBytes(cfg: *const anyopaque, seq: u64, max_tokens: u32) u64 {
                return T.promptBytes(constCfg(cfg), seq, max_tokens);
            }
            fn init(load: *const LoadCtx, cfg: *const anyopaque) anyerror!*anyopaque {
                return try T.init(load, constCfg(cfg));
            }
            fn deinit(m: *anyopaque) void {
                T.deinit(mod(m));
            }
            fn prefill(m: *anyopaque, ids: []const u32, req: RequestShape) anyerror!mlx.mlx_array {
                return T.prefill(mod(m), ids, req);
            }
            fn step(m: *anyopaque, ids: []const u32) anyerror!mlx.mlx_array {
                return T.step(mod(m), ids);
            }
            fn position(m: *const anyopaque) u64 {
                return T.position(@ptrCast(@alignCast(m)));
            }
            fn handover(m: *anyopaque, h: DecodeHandover) anyerror!void {
                return T.handover(mod(m), h);
            }
            fn requestEnd(m: *anyopaque) void {
                T.requestEnd(mod(m));
            }
            fn restorePrefix(m: *anyopaque, prefix: []const u32) u64 {
                return T.restorePrefix(mod(m), prefix);
            }
            fn billOf(gpa: Allocator, io: std.Io, req: *const bill.BillRequest) anyerror!bill.MemoryBill {
                return T.bill(gpa, io, req);
            }
            fn maxOutput(cfg: *const anyopaque) u64 {
                return T.maxOutput(constCfg(cfg));
            }
            fn claimProcess() anyerror!void {
                return T.claimProcess();
            }
            fn releaseProcess() void {
                T.releaseProcess();
            }
        };
        return .{
            .name = T.name,
            .caps = T.caps,
            .claims = T.claims,
            .parse = W.parse,
            .free_config = W.freeConfig,
            .shell = W.shell,
            .apply_settings = W.applySettings,
            .load_bytes = W.loadBytes,
            .prompt_bytes = if (check.has(T, "promptBytes")) W.promptBytes else null,
            .init = W.init,
            .deinit = W.deinit,
            .prefill = W.prefill,
            .step = W.step,
            .position = W.position,
            .handover = if (check.has(T, "handover")) W.handover else null,
            .request_end = if (check.has(T, "requestEnd")) W.requestEnd else null,
            .restore_prefix = if (check.has(T, "restorePrefix")) W.restorePrefix else null,
            .spec = if (check.has(T, "draft_lane")) .{ .draft_lane = spec.DraftLane.of(T.Module, T.draft_lane) } else .none,
            .bill = if (check.has(T, "bill")) W.billOf else null,
            .max_output = if (check.has(T, "maxOutput")) W.maxOutput else null,
            .claim_process = if (check.has(T, "claimProcess")) W.claimProcess else null,
            .release_process = if (check.has(T, "claimProcess")) W.releaseProcess else null,
        };
    }
};

/// One loaded model's arch: its table, its parsed config and its module.
pub const ArchInstance = struct { vt: *const Arch, cfg: *anyopaque, module: *anyopaque };
