//! spec gate threshold, KvAttnMode, resolveKvAttnFused; defaultEnableMtp, dsv4DraftStages, mtpCapable,
//! admitMtpForCtx; PldDefaults; requestPld; drafter/spec request modes, pickStreamMode

const std = @import("std");
const kv_quant_mod = @import("../../sdk/blocks/attention/kv_quant.zig");
const mtp_mod = @import("../../sdk/spec/mtp/mtp.zig");
const model_mod = @import("../../sdk/config/model.zig");
const log = @import("../../sdk/util/log.zig");
const model_registry_mod = @import("../registry/registry.zig");
const mlx_engine = @import("../engines/mlx.zig");
const LoadedModel = model_registry_mod.LoadedModel;
const testing = std.testing;
const http_server = @import("../http/server.zig");

const ServerConfig = http_server.ServerConfig;
const configuredKvQuantFor = http_server.configuredKvQuantFor;
const StreamMode = @import("chat_stream.zig").StreamMode;

/// Adaptive spec-decode gate threshold. Per-request, we score the prompt's
/// 3-gram repetition density; if `score < spec_gate_threshold` AND the user
/// did not explicitly set the flag, PLD/drafter are disabled for this request.
///
/// Empirically tuned (Qwen3.5 BPE, 24-300 token prompts):
///   - "creative story" / "write an essay" prompts:                    0.000
///   - 82-token echo-heavy code rename:                                0.013
///   - 155-token JS-translation w/ many `add/sub/mul/div` repetitions: 0.128
/// The plan started from 0.15 but BPE tokenization fragments echo-heavy
/// content enough that even strong echo cases land in the 0.01–0.10 range.
/// 0.01 cleanly separates "any repetition at all" (PLD likely helps) from
/// "pure novel" (PLD overhead-only). Re-tune by sweeping `enable_pld` on/off
/// across an echo-heavy vs novel-content prompt set and picking the threshold
/// that maximizes (echo_speedup / novel_overhead).
pub const spec_gate_threshold: f32 = 0.01;

/// MTP-vs-PLD coexistence routing. When a model has BOTH a trained MTP head and
/// PLD available for a request, the DEFAULT is the MTP head — it holds ~93%+
/// per-draft on novel AND echo content and never risks PLD's runtime acceptance
/// gate degrading to plain decode. Only a DOMINANT-echo prompt (score >= this)
/// is handed to PLD, whose long n-gram drafts then reliably emit more tokens per
/// verify forward than the depth-1 head.
///
/// The bar is deliberately high. Measured Qwen3.6-27B 4-bit: at ngram-score
/// 0.055 PLD is a coin flip — a long whole-file reproduction sustains ~4.3
/// accepted/round (48 tok/s) but a shorter one collapses below the 50% runtime
/// gate and falls back to plain decode (~29 tok/s, WORSE than the head's 40).
/// The score is computed on the PROMPT and can't see which way the GENERATION
/// will go, so anything below a clearly-dominant-echo score routes to the robust
/// head (40-42 tok/s, reliable). User `enable_mtp` in the body overrides.

// Plan 05: drafter state moved to `LoadedModel` (per-model `drafter`,
// `drafter_block_size`, `drafter_path`). The previous module-level
// `default_drafter` / `default_enable_drafter` / `default_draft_block_size`
// / `global_drafter_path` singletons were removed; handlers now read
// these fields off the request's resolved `*LoadedModel` (`lm.drafter`,
// etc.). `default_enable_drafter` semantics (`drafter != null and !isMoe()`)
// is re-computed per-request as `lm.drafter != null and !config.isMoe()`.

/// Server-level configuration. Single source of truth for all process-wide
/// defaults that handlers might consult. Populated once by `serve()` from
/// its CLI args; read-only afterwards (no synchronization required, the
/// values don't change for the server's lifetime).
pub const KvAttnMode = enum { dense, fused, auto };

/// docs/kv-quant-perf.md Phase 2 — auto-mode crossover. µbench (Phase 0
/// table in the doc): the decode kernel is ~break-even with dense-mode reads
/// at 8K context and wins from ~16K on every measured geometry (48/8 and
/// 32/8 at hd 128, 16/8 at hd 256), so `auto` engages fused reads from 8K
/// prompt tokens. One constant to start (per the plan); the per-request
/// `kv_attn_mode` field and an explicit --kv-attn-mode outrank it.
pub const KV_ATTN_AUTO_CROSSOVER_TOKENS: usize = 8192;

/// Pure resolution core (unit-tested): explicit per-request choice >
/// server mode; `auto` = fused iff the request's EFFECTIVE cache scheme is
/// .affine and the prompt clears the crossover.
pub fn resolveKvAttnFusedPure(mode: KvAttnMode, explicit: ?bool, prompt_len: usize, scheme: kv_quant_mod.Scheme) bool {
    if (explicit) |e| return e;
    return switch (mode) {
        .dense => false,
        .fused => true,
        .auto => scheme == .affine and prompt_len >= KV_ATTN_AUTO_CROSSOVER_TOKENS,
    };
}

/// Wrapper reading the live server config + scheduler default scheme.
/// Exact decode (a DFlash drafter bound) reads every row through `row_attn`
/// over the dense view: the packed kernels neither take a draft tree's mask
/// nor give a serial step and a verify row the same bits.
pub fn resolveKvAttnFused(config: *const model_mod.ModelConfig, explicit: ?bool, prompt_len: usize, kv_override: ?kv_quant_mod.KVQuantConfig) bool {
    if (config.rowExactDecode()) return false;
    const scheme: kv_quant_mod.Scheme = (kv_override orelse configuredKvQuantFor(config)).scheme;
    return resolveKvAttnFusedPure(http_server.serverState().server_config.kv_attn_mode, explicit, prompt_len, scheme);
}

/// Per-request `kv_attn_mode` body field: "fused" | "dense" force the read
/// path; "auto" or an absent/unrecognized value falls back to the server
/// mode (the kv_quant field's tolerant precedent).
pub fn parseKvAttnExplicit(root: std.json.ObjectMap) ?bool {
    const v = root.get("kv_attn_mode") orelse return null;
    switch (v) {
        .string => |s| {
            if (std.mem.eql(u8, s, "fused")) return true;
            if (std.mem.eql(u8, s, "dense")) return false;
            return null;
        },
        else => return null,
    }
}

/// The per-request `enable_mtp` default, for a request that omitted the field.
/// The ONE place this policy lives — every HTTP surface calls it, so a new
/// surface can't silently ship a different default (the drafter-dispatch-hole
/// lesson: an output-equality test cannot see a spec path that never engaged).
/// A loaded head drafts, dense or MoE; `--no-mtp`, a model's `"mtp": false` or a
/// request's `enable_mtp:false` opt out. `dsv4_stages`: DeepSeek-V4's own DSpark
/// stages, loaded only on opt-in `--dspark`, which carry no qwen head.
pub fn defaultEnableMtp(mtp_loaded: bool, dsv4_stages: bool) bool {
    return mtp_loaded or dsv4_stages;
}

/// Where an MTP head drafts without `--mtp`. CUDA's multi-row matmuls make a verify round pay;
/// Vulkan has none and decodes plain.
pub fn mtpAllowedByDefault(mtp_forced: bool, metal: bool, cuda: bool) bool {
    return mtp_forced or metal or cuda;
}

/// Does this model serve DeepSeek-V4 or V4.1 with DSpark draft stages loaded?
pub fn dsv4DraftStages(lm: *LoadedModel) bool {
    return mlx_engine.api().draft_stages(lm);
}

/// Can this model run an MTP-flagged request speculatively? Either a qwen
/// sidecar/in-checkpoint head (lm.mtp) or dsv4's native DSpark stages. Every
/// per-surface `enable_mtp` conjunct must use THIS, not `lm.mtp != null` —
/// the bare conjunct silently killed the flag for dsv4 at submit while the
/// default/dispatch layers were correct (the per-surface wiring class).
pub fn mtpCapable(lm: *LoadedModel) bool {
    return lm.mtp != null or dsv4DraftStages(lm);
}

/// `--max-mtp-ctx` admission: MTP is refused past the operator's context ceiling. Called once
/// per surface after tokenization; a machine limit, so it outranks an explicit `enable_mtp:true`.
pub fn admitMtpForCtx(enable_mtp: bool, prompt_tokens: usize) bool {
    return admitMtpForCtxWith(mlx_engine.settings().max_mtp_ctx, enable_mtp, prompt_tokens);
}

fn admitMtpForCtxWith(max: u32, enable_mtp: bool, prompt_tokens: usize) bool {
    if (!enable_mtp) return false;
    if (mtp_mod.mtpCtxWithinLimit(max, prompt_tokens)) return true;
    log.info("  mtp=off (ctx {d} > --max-mtp-ctx {d})\n", .{ prompt_tokens, max });
    return false;
}

/// PLD request defaults carried as ONE value, so a `ServerConfig` builder
/// cannot thread the enable bit and forget the two lengths beside it.
///
/// That is exactly how this broke: `runHeadlessServe` hand-rolled all three
/// as `false`/`5`/`3` literals, so no `--pld*` flag reached a headless
/// request. Threading `enable` alone fixed a third of it and left
/// `--pld-draft-len` / `--pld-key-len` silently dropped — and headless is
/// the mode the Swift app ALWAYS launches (`--serve --model-dir`, no
/// `--model`), passing all three flags on every boot. Single chokepoint for
/// every ServerConfig builder; `effectiveSsmCheckpointStride` above plays
/// the same role for LoadParams.
///
/// Serve paths whose decode never routes through the PLD-capable generator
/// (media gen, ds4, llama.cpp) take `.off`: the field is dead weight there,
/// and saying so once beats five hand-written `false`s that read like a
/// decision but drift like a typo.
pub const PldDefaults = struct {
    enable: bool,
    draft_len: u32,
    key_len: u32,

    /// Embedded-engine and media-gen serve paths — PLD is unreachable.
    pub const off: PldDefaults = .{ .enable = false, .draft_len = 5, .key_len = 3 };

    /// Text-gen serve paths: the CLI's values verbatim, no reinterpretation.
    /// `--no-pld` arrives as `enable == false` and the lengths ride along
    /// unused, so flipping PLD back on later cannot resurrect stale numbers.
    pub fn fromCli(enable: bool, draft_len: u32, key_len: u32) PldDefaults {
        return .{ .enable = enable, .draft_len = draft_len, .key_len = key_len };
    }
};

/// The request's `enable_pld`, else the launch default; never on a Neural Engine model, whose
/// engine drafts on its own.
pub fn requestPld(root: std.json.ObjectMap, lm: *const LoadedModel) bool {
    if (lm.engine != null) return false; // PLD runs in the MLX generator only
    return if (root.get("enable_pld")) |v| v == .bool and v.bool else http_server.serverState().server_config.default_enable_pld;
}

/// Choose the speculative-decoding mode for a streaming request based on
/// the request flags and the model capabilities. Mirrors the dispatch in
/// `handleNonStreamingGeneration` so streaming and non-streaming pick the
/// same path for the same inputs.
///
/// Priority: drafter > PLD > regular (matches the non-streaming dispatch in
/// `handleNonStreamingGeneration`). The same per-mode disable rules (no
/// logprobs, no grammar constraint, no hybrid SSM for drafter) apply at the
/// request-entry parse site; `pickStreamMode` re-enforces them defensively
/// here so a missed gate at the parse site doesn't crash the dispatch.
/// Does this trunk's architecture rule the loaded assistant sidecar out?
///
/// The hybrid veto exists for the GEMMA cross-attention drafter, whose
/// multi-token verify was never wired for a recurrent trunk. A DFlash/DSpark
/// sidecar is a different mechanism — it reads the trunk's own layer captures
/// and rolls the conv state back from the verify pass's per-position capture
/// — so a hybrid target is exactly what LiquidAI ships DSpark FOR. ONE
/// predicate, because this gate is re-derived on four request surfaces and a
/// missed site decodes serial with everything else looking healthy.
pub fn archBlocksAssistantSidecar(has_hybrid_layers: bool, dflash_loaded: bool) bool {
    return has_hybrid_layers and !dflash_loaded;
}

/// Resolve sidecar admission before the request's logprobs, grammar and yield gates.
pub fn requestEnableDrafter(explicit: ?std.json.Value, gemma_loaded: bool, dflash_loaded: bool, is_moe: bool, has_hybrid_layers: bool) bool {
    const enabled = if (explicit) |v| v == .bool and v.bool else (gemma_loaded and !is_moe) or dflash_loaded;
    if (!enabled or (!gemma_loaded and !dflash_loaded)) return false;
    return !archBlocksAssistantSidecar(has_hybrid_layers, dflash_loaded);
}

pub const RequestSpec = struct { use_mtp: bool, use_drafter: bool, use_pld: bool };

/// Generation requests between the router and their response's end (`router.handleConnection`).
pub var text_gen_live = std.atomic.Value(u32).init(0);

/// Another request is live: one already in the engine, or one still between the router and its submit.
pub fn requestHasCompany() bool {
    if (text_gen_live.load(.monotonic) > 1) return true;
    return if (http_server.launch().global_scheduler) |sch| mlx_engine.api().has_company(sch) else false;
}

/// The ONE request-level spec dispatch, shared by every surface x stream /
/// non-stream. Priority: DFlash > MTP > gemma drafter > PLD, mirroring
/// `scheduler.specInitWiring`. A loaded DFlash sidecar is an explicit choice
/// (`--drafter` or the pack's own `drafter/`) while the MTP head ships with
/// the checkpoint, so the sidecar wins; `--no-drafter` / `enable_drafter:
/// false` hand the round back to MTP. The gemma cross-attention drafter
/// stays below MTP. logprobs, a grammar constraint or a penalty (`shaped_logits`)
/// disable every mode: verify compares raw logits.
/// `has_company` (another request live at admission) hands a DFlash round to
/// a loaded MTP head: only MTP slots batch.
/// `enable_drafter` arrives with the hybrid veto (`archBlocksAssistantSidecar`)
/// already applied at every surface's parse site.
pub fn requestSpecModes(
    enable_pld: bool,
    enable_drafter: bool,
    enable_mtp: bool,
    gemma_drafter_loaded: bool,
    dflash_loaded: bool,
    mtp_loaded: bool,
    shaped_logits: bool,
    logprobs_n: u32,
    has_company: bool,
) RequestSpec {
    const spec_ok = logprobs_n == 0 and !shaped_logits;
    const sidecar = spec_ok and enable_drafter and (gemma_drafter_loaded or dflash_loaded);
    // DFlash slots decode serial; MTP slots draft and verify as one group.
    const dflash_yields = has_company and enable_mtp and mtp_loaded;
    const use_dflash = sidecar and dflash_loaded and !dflash_yields;
    const use_mtp = !use_dflash and spec_ok and enable_mtp and mtp_loaded;
    const use_drafter = use_dflash or (!use_mtp and sidecar);
    return .{
        .use_mtp = use_mtp,
        .use_drafter = use_drafter,
        .use_pld = !use_mtp and !use_drafter and spec_ok and enable_pld,
    };
}

pub fn pickStreamMode(
    enable_pld: bool,
    enable_drafter: bool,
    enable_mtp: bool,
    gemma_drafter_loaded: bool,
    dflash_loaded: bool,
    mtp_loaded: bool,
    shaped_logits: bool,
    logprobs_n: u32,
    has_company: bool,
) StreamMode {
    const r = requestSpecModes(enable_pld, enable_drafter, enable_mtp, gemma_drafter_loaded, dflash_loaded, mtp_loaded, shaped_logits, logprobs_n, has_company);
    if (r.use_mtp) return .mtp;
    if (r.use_drafter) return .drafter;
    if (r.use_pld) return .pld;
    return .regular;
}

// ── tests ──

test "PldDefaults: CLI lengths survive alongside the enable bit" {
    // The regression: `enable` reached headless while the lengths stayed at
    // the 5/3 literals. All three travel together or the class is back.
    const cli = PldDefaults.fromCli(true, 8, 4);
    try std.testing.expect(cli.enable);
    try std.testing.expectEqual(@as(u32, 8), cli.draft_len);
    try std.testing.expectEqual(@as(u32, 4), cli.key_len);

    // --no-pld carries its lengths unchanged rather than snapping to defaults.
    const disabled = PldDefaults.fromCli(false, 8, 4);
    try std.testing.expect(!disabled.enable);
    try std.testing.expectEqual(@as(u32, 8), disabled.draft_len);

    // The non-text serve paths stay pinned to the historical literals.
    try std.testing.expect(!PldDefaults.off.enable);
    try std.testing.expectEqual(@as(u32, 5), PldDefaults.off.draft_len);
    try std.testing.expectEqual(@as(u32, 3), PldDefaults.off.key_len);
}

test "PldDefaults: ServerConfig built from it reports the CLI values" {
    // Pins the wiring, not just the struct: a ServerConfig fed from
    // `fromCli` must expose the CLI numbers on the exact fields every
    // request path and the boot banner read.
    const cfg = ServerConfig{
        .default_enable_pld = PldDefaults.fromCli(true, 8, 4).enable,
        .default_pld_draft_len = PldDefaults.fromCli(true, 8, 4).draft_len,
        .default_pld_key_len = PldDefaults.fromCli(true, 8, 4).key_len,
    };
    try std.testing.expect(cfg.default_enable_pld);
    try std.testing.expectEqual(@as(u32, 8), cfg.default_pld_draft_len);
    try std.testing.expectEqual(@as(u32, 4), cfg.default_pld_key_len);
}

test "drafter admission: defaults and explicit opt-in cover every sidecar and architecture" {
    const Case = struct {
        gemma: bool,
        dflash: bool,
        moe: bool,
        hybrid: bool,
        default: bool,
        explicit: bool,
    };
    const cases = [_]Case{
        .{ .gemma = false, .dflash = false, .moe = false, .hybrid = false, .default = false, .explicit = false },
        .{ .gemma = false, .dflash = false, .moe = true, .hybrid = false, .default = false, .explicit = false },
        .{ .gemma = false, .dflash = false, .moe = false, .hybrid = true, .default = false, .explicit = false },
        .{ .gemma = false, .dflash = false, .moe = true, .hybrid = true, .default = false, .explicit = false },
        .{ .gemma = true, .dflash = false, .moe = false, .hybrid = false, .default = true, .explicit = true },
        .{ .gemma = true, .dflash = false, .moe = true, .hybrid = false, .default = false, .explicit = true },
        .{ .gemma = true, .dflash = false, .moe = false, .hybrid = true, .default = false, .explicit = false },
        .{ .gemma = true, .dflash = false, .moe = true, .hybrid = true, .default = false, .explicit = false },
        .{ .gemma = false, .dflash = true, .moe = false, .hybrid = false, .default = true, .explicit = true },
        .{ .gemma = false, .dflash = true, .moe = true, .hybrid = false, .default = true, .explicit = true },
        .{ .gemma = false, .dflash = true, .moe = false, .hybrid = true, .default = true, .explicit = true },
        .{ .gemma = false, .dflash = true, .moe = true, .hybrid = true, .default = true, .explicit = true },
        .{ .gemma = true, .dflash = true, .moe = false, .hybrid = false, .default = true, .explicit = true },
        .{ .gemma = true, .dflash = true, .moe = true, .hybrid = false, .default = true, .explicit = true },
        .{ .gemma = true, .dflash = true, .moe = false, .hybrid = true, .default = true, .explicit = true },
        .{ .gemma = true, .dflash = true, .moe = true, .hybrid = true, .default = true, .explicit = true },
    };
    for (cases) |c| {
        const default = requestEnableDrafter(null, c.gemma, c.dflash, c.moe, c.hybrid);
        const explicit = requestEnableDrafter(.{ .bool = true }, c.gemma, c.dflash, c.moe, c.hybrid);
        try testing.expectEqual(c.default, default);
        try testing.expectEqual(c.explicit, explicit);
        try testing.expect(!requestEnableDrafter(.{ .bool = false }, c.gemma, c.dflash, c.moe, c.hybrid));
        for ([_]bool{ default, explicit }) |enabled| {
            const spec = requestSpecModes(false, enabled, false, c.gemma, c.dflash, false, false, 0, false);
            try testing.expectEqual(enabled, spec.use_drafter);
            try testing.expect(!spec.use_mtp and !spec.use_pld);
            try testing.expectEqual(if (enabled) StreamMode.drafter else .regular, pickStreamMode(false, enabled, false, c.gemma, c.dflash, false, false, 0, false));
        }
    }
}

test "drafter admission: present non-boolean overrides disable either sidecar" {
    for ([_][]const u8{ "null", "\"true\"", "1", "1.0", "[]", "{}" }) |json| {
        const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
        defer parsed.deinit();
        try testing.expect(!requestEnableDrafter(parsed.value, true, false, false, false));
        try testing.expect(!requestEnableDrafter(parsed.value, false, true, false, false));
        try testing.expect(!requestEnableDrafter(parsed.value, false, true, true, true));
    }
}

test "requestSpecModes: a loaded DFlash sidecar outranks the checkpoint's MTP head; the gemma drafter does not" {
    // dflash + mtp both loaded and enabled -> dflash rides the drafter arm.
    var r = requestSpecModes(true, true, true, false, true, true, false, 0, false);
    try std.testing.expect(r.use_drafter and !r.use_mtp and !r.use_pld);
    try std.testing.expectEqual(StreamMode.drafter, pickStreamMode(true, true, true, true, true, true, false, 0, false));
    // gemma drafter + mtp -> MTP keeps its rank.
    r = requestSpecModes(true, true, true, true, false, true, false, 0, false);
    try std.testing.expect(r.use_mtp and !r.use_drafter);
    try std.testing.expectEqual(StreamMode.mtp, pickStreamMode(true, true, true, true, false, true, false, 0, false));
    // enable_drafter:false (or the parse-site hybrid veto) hands the round back to MTP.
    r = requestSpecModes(true, false, true, false, true, true, false, 0, false);
    try std.testing.expect(r.use_mtp and !r.use_drafter);
    // logprobs / grammar disable every spec mode, dflash included.
    r = requestSpecModes(true, true, true, false, true, true, false, 3, false);
    try std.testing.expect(!r.use_mtp and !r.use_drafter and !r.use_pld);
    r = requestSpecModes(true, true, true, false, true, true, true, 0, false);
    try std.testing.expect(!r.use_mtp and !r.use_drafter and !r.use_pld);
    // No sidecar loaded -> MTP; nothing loaded -> PLD.
    r = requestSpecModes(true, true, true, false, false, true, false, 0, false);
    try std.testing.expect(r.use_mtp and !r.use_drafter);
    r = requestSpecModes(true, true, true, false, false, false, false, 0, false);
    try std.testing.expect(r.use_pld and !r.use_mtp and !r.use_drafter);
    // Company at admission: DFlash yields to a loaded, enabled MTP head, and only to that.
    r = requestSpecModes(true, true, true, false, true, true, false, 0, true);
    try std.testing.expect(r.use_mtp and !r.use_drafter and !r.use_pld);
    try std.testing.expectEqual(StreamMode.mtp, pickStreamMode(true, true, true, false, true, true, false, 0, true));
    r = requestSpecModes(true, true, true, false, true, false, false, 0, true);
    try std.testing.expect(r.use_drafter and !r.use_mtp);
    r = requestSpecModes(true, true, false, false, true, true, false, 0, true);
    try std.testing.expect(r.use_drafter and !r.use_mtp);
}

test "PLD: the request's flag, else the default; never on a model served beside MLX" {
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"enable_pld\":true}", .{});
    defer parsed.deinit();
    var lm: LoadedModel = undefined;
    lm.engine = null;
    try testing.expect(requestPld(parsed.value.object, &lm));
    lm.engine = @ptrFromInt(0x1000);
    try testing.expect(!requestPld(parsed.value.object, &lm));
}

test "resolveKvAttnFused: exact decode never reads packed KV, even when a request asks" {
    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "qwen3_5";
    cfg.row_exact_covered = true;
    cfg.dflash_bound = true;
    const kv8 = kv_quant_mod.KVQuantConfig.affine(8);
    try std.testing.expect(!resolveKvAttnFused(&cfg, true, 8192, kv8));
    cfg.dflash_bound = false;
    try std.testing.expect(resolveKvAttnFused(&cfg, true, 8192, kv8));
}

test "resolveKvAttnFusedPure: explicit > mode; auto keys on scheme + crossover" {
    const t = std.testing;
    // Explicit per-request choice outranks every mode.
    try t.expect(resolveKvAttnFusedPure(.dense, true, 0, .off));
    try t.expect(!resolveKvAttnFusedPure(.fused, false, 1 << 20, .affine));
    // Fixed modes.
    try t.expect(!resolveKvAttnFusedPure(.dense, null, 1 << 20, .affine));
    try t.expect(resolveKvAttnFusedPure(.fused, null, 0, .affine));
    // Auto: affine + prompt at/above the crossover.
    try t.expect(resolveKvAttnFusedPure(.auto, null, KV_ATTN_AUTO_CROSSOVER_TOKENS, .affine));
    try t.expect(resolveKvAttnFusedPure(.auto, null, KV_ATTN_AUTO_CROSSOVER_TOKENS + 1, .affine));
    try t.expect(!resolveKvAttnFusedPure(.auto, null, KV_ATTN_AUTO_CROSSOVER_TOKENS - 1, .affine));
    // Auto never engages off (no triples to consume).
    try t.expect(!resolveKvAttnFusedPure(.auto, null, 1 << 20, .off));
}

test "MTP drafts by default on Metal and CUDA, and only when asked elsewhere" {
    try std.testing.expect(mtpAllowedByDefault(false, true, false));
    try std.testing.expect(mtpAllowedByDefault(false, false, true));
    try std.testing.expect(!mtpAllowedByDefault(false, false, false));
    try std.testing.expect(mtpAllowedByDefault(true, false, false));
}

test "defaultEnableMtp: a loaded head drafts by default, MoE or not" {
    const t = std.testing;
    try t.expect(!defaultEnableMtp(false, false));
    try t.expect(defaultEnableMtp(true, false));
    // DSpark: dsv4's own stages, loaded without a qwen head.
    try t.expect(defaultEnableMtp(false, true));
}

test "admitMtpForCtx: the --max-mtp-ctx ceiling outranks the request's own flag" {
    try std.testing.expect(admitMtpForCtxWith(0, true, 1_000_000));
    try std.testing.expect(!admitMtpForCtxWith(0, false, 1));
    // Inclusive at the ceiling; an explicit `enable_mtp:true` loses to it.
    try std.testing.expect(admitMtpForCtxWith(4096, true, 4096));
    try std.testing.expect(!admitMtpForCtxWith(4096, true, 4097));
    // It never turns MTP on for a request that had it off for another reason.
    try std.testing.expect(!admitMtpForCtxWith(4096, false, 1));
}

test "requestHasCompany: a second generation request still before its submit is company" {
    defer text_gen_live.store(0, .monotonic);
    text_gen_live.store(1, .monotonic);
    try std.testing.expect(!requestHasCompany());
    text_gen_live.store(2, .monotonic);
    try std.testing.expect(requestHasCompany());
}
