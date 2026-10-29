//! Speculative-decode wiring: init options, sidecar gate, tick mode and block publishing.

const std = @import("std");
const generate_mod = @import("../generate.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const log = @import("../../../sdk/util/log.zig");
const ModelConfig = model_mod.ModelConfig;
const Generator = generate_mod.Generator;
const testing = std.testing;

const finishSlot = @import("../finish.zig").finishSlot;
const Scheduler = @import("../scheduler.zig").Scheduler;
const Slot = @import("../slot.zig").Slot;
const Planner = @import("mtp_groups.zig").Planner;

/// What `runPrefill` arms in a slot's Generator init options.
pub const SpecInitWiring = struct {
    use_mtp: bool,
    use_drafter: bool,
    use_dflash: bool,
    use_pld: bool,
    /// The request's spec INTENT, forwarded to the Generator chokepoint for a
    /// module-owned arch that has its OWN draft mode (dsv4 → DSpark). Rides
    /// `mtp_enabled` alongside `use_mtp`.
    native_intent: bool,
};

/// Per-site spec gating for one slot, as a pure function.
///
/// Speculative decode must be able to ROLL BACK a rejected tail, and the shell
/// rolls back what it owns: the slot's KVCache and ssm_entries. A MODULE-OWNED
/// arch uses neither — by the time a verify forward returns, its own state has
/// already absorbed every draft token, and the shell's snapshot/truncate run
/// over an empty entries array. So every shell-driven spec mode is off there.
///
/// One predicate, never hand-written `!is_dsv4` conjuncts. A second
/// module-owned arch arrived with the same `Model.state` shape and a 0-layer
/// shell cache, got no conjunct, and `--pld` drove verify forwards straight
/// through it. One predicate, one place to extend.
pub fn specInitWiring(
    owns_module_state: bool,
    module_spec_rollback: bool,
    has_native_draft: bool,
    enable_mtp: bool,
    has_mtp: bool,
    enable_drafter: bool,
    has_drafter: bool,
    has_dflash: bool,
    enable_pld: bool,
) SpecInitWiring {
    if (owns_module_state) return .{
        // A module-owned arch that CAN rewind its state across a verify runs
        // its own MTP head like any other target. PLD and the drafters
        // stay off regardless: the drafters need a bound sidecar (DflashModel
        // .bind refuses these archs anyway), and PLD's win case is
        // echo-shaped traffic the n-gram gate rarely opens on a
        // head this cheap — neither has been measured on this family, and an
        // unmeasured spec mode is worse than none. Image requests ride the
        // head too: `qwen4MtpForward` takes the slot's M-RoPE table.
        .use_mtp = module_spec_rollback and enable_mtp and has_mtp,
        .use_drafter = false,
        .use_dflash = false,
        .use_pld = false,
        .native_intent = has_native_draft and enable_mtp,
    };
    // enable_drafter is the request-level "assistant sidecar" switch for BOTH
    // sidecar kinds; the loader guarantees at most one of drafter/dflash is
    // loaded per model. Priority: dflash > MTP > gemma drafter > PLD — a
    // loaded DFlash sidecar is an explicit choice (`--drafter` or the pack's
    // own `drafter/`) while the MTP head ships with the checkpoint;
    // `--no-drafter` (or `enable_drafter:false`) hands the round back to MTP.
    const use_dflash = enable_drafter and has_dflash;
    const use_mtp = !use_dflash and enable_mtp and has_mtp;
    const use_drafter = !use_mtp and !use_dflash and enable_drafter and has_drafter;
    return .{
        .use_mtp = use_mtp,
        .use_drafter = use_drafter,
        .use_dflash = use_dflash,
        .use_pld = !use_mtp and !use_dflash and !use_drafter and enable_pld,
        .native_intent = false,
    };
}

/// Resolve the request-level switch shared by the classic external drafter
/// and DFlash. Vision stays guarded by default: placeholder ids alone do not
/// tell an external sidecar how to position image tokens. Muse and Nemotron
/// RADIO DFlash consume the vision-conditioned trunk's captures without
/// Qwen's M-RoPE table.
pub fn assistantSidecarEnabledForRequest(
    requested: bool,
    has_vision: bool,
    has_mrope: bool,
    has_dflash: bool,
    config: *const ModelConfig,
) bool {
    if (!requested) return false;
    if (!has_vision) return true;
    const native_captures = config.muse_vision or config.radio_vision;
    return has_dflash and native_captures and !has_mrope;
}

/// Which spec mode a decode tick drives for a slot.
pub const SpecTickMode = enum { dspark, mtp, dflash, drafter, pld, regular };

/// Pure decode-tick dispatch decision. The slot flags carry the REQUEST's
/// wish; the generator-side values carry what `Generator.initWithOptions`
/// actually armed (after its deepseek_v4 spec chokepoint). Every arm must
/// require BOTH: dispatching on the slot flag alone is the exact wiring that
/// put PLD verify forwards through a dsv4 trunk — mtp/drafter
/// had their generator-state conjunct (`gen.mtp != null`), model-less PLD
/// did not, so runPrefill's `use_pld=false` shaped init options while every
/// tick still called `gen.nextPld`.
///
/// DSpark (dsv4's own draft mode) wins first and rides the MTP flag alone —
/// the "model's native head" semantics: defaulted ON server-side for a
/// stage-bearing dsv4, never n-gram prompt-gated, `enable_mtp:false` opts
/// out. The chokepoint only arms it after zeroing every other spec.
pub fn specTickMode(
    slot_enable_mtp: bool,
    gen_has_mtp: bool,
    slot_enable_drafter: bool,
    gen_has_drafter: bool,
    gen_has_dflash: bool,
    slot_enable_pld: bool,
    gen_pld_enabled: bool,
    gen_dspark_enabled: bool,
) SpecTickMode {
    if (gen_dspark_enabled and slot_enable_mtp) return .dspark;
    if (slot_enable_drafter and gen_has_dflash) return .dflash;
    if (slot_enable_mtp and gen_has_mtp) return .mtp;
    if (slot_enable_drafter and gen_has_drafter) return .drafter;
    if (slot_enable_pld and gen_pld_enabled) return .pld;
    return .regular;
}

/// `specTickMode` for a slot and its generator.
pub fn slotSpecMode(slot: *const Slot, gen: *const Generator) SpecTickMode {
    return specTickMode(
        slot.enable_mtp,
        gen.mtp != null,
        slot.enable_drafter,
        gen.drafter != null,
        gen.dflash != null,
        slot.enable_pld,
        gen.pld_enabled,
        gen.dspark_enabled,
    );
}

/// Drive one Generator step (regular / PLD / drafter) and push emitted
/// tokens into the slot's output ring. Mirrors the existing
/// `StreamingTokenStream` adapter contract: 0..N tokens per call, with EOS
/// stopping the slot but NOT being emitted.
pub fn plannerOutputClock(slot: *Slot, gen: *Generator) void {
    const now = gen.timer.read();
    slot.mtp_publish_gap_ms = if (slot.mtp_publish_ns > 0 and now >= slot.mtp_publish_ns) @as(f32, @floatFromInt(now - slot.mtp_publish_ns)) / std.time.ns_per_ms else 0;
    slot.mtp_publish_ns = now;
    if (Planner.enabled() and gen.mtp_planner_owned) gen.mtp_planner_max_gap_ms = @max(gen.mtp_planner_max_gap_ms, slot.mtp_publish_gap_ms);
}

/// The prefill already sampled the first token: send it now instead of with the
/// first decode step or speculative round, which would hold it a forward longer.
/// The count stays with the decoder that emits it (its push is swallowed).
pub fn publishFirstToken(slot: *Slot) void {
    const gen = if (slot.legacy_gen) |*g| g else return;
    if (gen.done or gen.completion_tokens != 0 or slot.logprobs_n > 0 or gen.sampling.constraint != null) return;
    const t1 = gen.next_token_id;
    if (t1 == 0 or generate_mod.isEosId(t1, slot.eos_token_ids)) return;
    slot.pushTokenWithLogprob(t1, null);
    slot.early_first = t1;
}

pub fn publishSpeculativeBlock(sch: *Scheduler, slot: *Slot, gen: *Generator, tokens: []const u32) void {
    if (gen.mtp != null and tokens.len > 0) {
        plannerOutputClock(slot, gen);
    }
    // The Generator has already committed the whole block internally. Publish
    // it incrementally so streaming, cancellation, EOS and usage all observe
    // the same per-token boundary. The generator-side cap guarantees max_tokens
    // cannot land before the returned block ends.
    for (tokens, 0..) |t, i| {
        if (slot.cancelled.load(.acquire)) {
            if (gen.mtp != null) log.debug("[mtp-publish] cancelled after round: discarded={d}\n", .{tokens.len - i});
            return;
        }
        if (generate_mod.isEosId(t, slot.eos_token_ids)) {
            finishSlot(sch, slot, "stop");
            return;
        }
        slot.pushToken(t);
        slot.completion_tokens += 1;
        if (t != 0) slot.was_pad_only = false;
        if (slot.completion_tokens >= slot.max_tokens) {
            finishSlot(sch, slot, "length");
            return;
        }
    }
    std.debug.assert(slot.completion_tokens == gen.completion_tokens);
}

// ── tests ──

test "drafter admission: vision gate admits Muse DFlash without weakening Qwen M-RoPE" {
    const generic = ModelConfig{};
    const muse = ModelConfig{ .muse_vision = true };
    // Text requests keep the existing shared sidecar behavior.
    try testing.expect(assistantSidecarEnabledForRequest(true, false, false, false, &generic));
    // A request opt-out always wins.
    try testing.expect(!assistantSidecarEnabledForRequest(false, true, false, true, &muse));
    // Image requests remain guarded for classic external drafters.
    try testing.expect(!assistantSidecarEnabledForRequest(true, true, false, false, &generic));
    // Muse DFlash consumes vision-conditioned trunk captures and may engage.
    try testing.expect(assistantSidecarEnabledForRequest(true, true, false, true, &muse));
    // Qwen's explicit M-RoPE table is never admitted through this exception.
    try testing.expect(!assistantSidecarEnabledForRequest(true, true, true, true, &muse));
    try testing.expect(!assistantSidecarEnabledForRequest(true, true, true, true, &generic));
}

test "drafter admission: vision gate admits Nemotron RADIO captures for image and video" {
    for ([_]ModelConfig{
        .{ .model_type = "nemotron_h", .radio_vision = true },
        .{ .model_type = "nemotron_h", .radio_vision = true, .has_hybrid_layers = true, .num_experts = 512 },
    }) |config| {
        try testing.expect(assistantSidecarEnabledForRequest(true, true, false, true, &config));
        try testing.expect(assistantSidecarEnabledForRequest(true, false, false, true, &config));
        try testing.expect(!assistantSidecarEnabledForRequest(false, true, false, true, &config));
        try testing.expect(!assistantSidecarEnabledForRequest(true, true, true, true, &config));
        try testing.expect(!assistantSidecarEnabledForRequest(true, true, false, false, &config));
    }
    for ([_]ModelConfig{
        .{ .model_type = "nemotron_h" },
        .{ .model_type = "qwen3_5_moe", .qwen_vision = true },
        .{ .model_type = "gemma4" },
    }) |config| {
        try testing.expect(!assistantSidecarEnabledForRequest(true, true, false, true, &config));
    }
}

test "drafter admission: vision gate keeps other encoders guarded without M-RoPE" {
    for ([_]ModelConfig{
        .{ .model_type = "gemma4", .has_vision = true },
        .{ .model_type = "lfm2", .has_vision = true, .lfm2_vision = true },
        .{ .model_type = "glm5_next", .has_vision = true, .glm_vision = true },
        .{ .model_type = "qwen3_5_moe", .has_vision = true, .minicpm_vision = true },
    }) |config| {
        // No M-RoPE is necessary, but does not alone admit an unvalidated encoder.
        try testing.expect(!assistantSidecarEnabledForRequest(true, true, false, true, &config));
        try testing.expect(!assistantSidecarEnabledForRequest(true, true, false, false, &config));
        // A text-only turn still accepts either loaded sidecar kind.
        try testing.expect(assistantSidecarEnabledForRequest(true, false, false, true, &config));
        try testing.expect(assistantSidecarEnabledForRequest(true, false, false, false, &config));
        try testing.expect(!assistantSidecarEnabledForRequest(false, false, false, true, &config));
    }
}

test "specInitWiring: a module-owned arch only gets the spec modes it can roll back" {
    // Spec decode must be able to ROLL BACK a rejected tail. The shell rolls
    // back its KVCache/ssm_entries — which a module-owned arch does not use, so
    // by the time the verify returns, the module has already absorbed every
    // draft and the shell's snapshot/truncate run over an EMPTY entries array.
    // dsv4 got a hand-written `is_dsv4` conjunct on each of the three lines;
    // the next module-owned arch (0-layer shell cache, state on the module)
    // got none, so `--pld` drove verify forwards straight through it. The
    // predicate is now per-ARCH CAPABILITY (`moduleStateSpecRollback`), not ownership.
    // Args: (owns_module_state, module_spec_rollback, has_native_draft,
    //        enable_mtp, has_mtp, enable_drafter, has_drafter, has_dflash,
    //        enable_pld)

    // Plain arch: today's precedence, unchanged.
    {
        const w = specInitWiring(false, false, false, true, true, true, true, false, true);
        try testing.expect(w.use_mtp and !w.use_drafter and !w.use_dflash and !w.use_pld and !w.native_intent);
    }
    {
        const w = specInitWiring(false, false, false, true, false, true, true, false, true);
        try testing.expect(!w.use_mtp and w.use_drafter and !w.use_pld);
    }
    {
        const w = specInitWiring(false, false, false, false, false, false, false, false, true);
        try testing.expect(!w.use_mtp and !w.use_drafter and w.use_pld and !w.native_intent);
    }
    // A flag with no loaded handle never arms.
    {
        const w = specInitWiring(false, false, false, true, false, false, false, false, false);
        try testing.expect(!w.use_mtp and !w.use_drafter and !w.use_dflash and !w.use_pld);
    }

    // DFlash rides the enable_drafter switch: dflash > MTP > drafter > PLD.
    {
        const w = specInitWiring(false, false, false, false, false, true, false, true, true);
        try testing.expect(!w.use_mtp and w.use_dflash and !w.use_drafter and !w.use_pld);
    }
    // A loaded drafter outranks the checkpoint's own MTP head: the sidecar
    // is an explicit choice (`--drafter` or the in-dir `drafter/`), the head
    // ships with every pack; `--no-drafter` restores MTP by not loading it.
    {
        const w = specInitWiring(false, false, false, true, true, true, false, true, true);
        try testing.expect(w.use_dflash and !w.use_mtp and !w.use_pld);
    }
    // enable_drafter:false on the request hands the round back to MTP.
    {
        const w = specInitWiring(false, false, false, true, true, false, false, true, true);
        try testing.expect(w.use_mtp and !w.use_dflash);
    }
    // enable_drafter:false opts BOTH sidecar kinds out.
    {
        const w = specInitWiring(false, false, false, false, false, false, false, true, true);
        try testing.expect(!w.use_dflash and !w.use_drafter and w.use_pld);
    }

    // Module-owned with NO rollback and no native draft mode: everything off,
    // and no intent bit either — nothing downstream can arm a draft path.
    {
        const w = specInitWiring(true, false, false, true, true, true, true, true, true);
        try testing.expect(!w.use_mtp and !w.use_drafter and !w.use_dflash and !w.use_pld and !w.native_intent);
    }

    // Module-owned WITH rollback: its own MTP head arms; the shell
    // spec modes stay off because none has been measured on this family.
    {
        const w = specInitWiring(true, true, false, true, true, true, true, true, true);
        try testing.expect(w.use_mtp and !w.use_drafter and !w.use_dflash and !w.use_pld and !w.native_intent);
    }
    // Rollback capability alone never arms a head that is not loaded.
    {
        const w = specInitWiring(true, true, false, true, false, true, true, true, true);
        try testing.expect(!w.use_mtp and !w.use_drafter and !w.use_dflash and !w.use_pld);
    }
    // ...nor one the request opted out of.
    {
        const w = specInitWiring(true, true, false, false, true, true, true, false, true);
        try testing.expect(!w.use_mtp);
    }
    // An image request keeps the head (the qwen4 head takes the slot's
    // M-RoPE table); the drafters stay off.
    {
        const w = specInitWiring(true, true, false, true, true, true, true, true, true);
        try testing.expect(w.use_mtp and !w.use_drafter and !w.use_dflash and !w.use_pld);
    }

    // Module-owned WITH a native draft mode (dsv4/DSpark) and no rollback: the
    // shell paths stay off, but the request's MTP intent still reaches the
    // Generator chokepoint.
    {
        const w = specInitWiring(true, false, true, true, true, true, true, true, true);
        try testing.expect(!w.use_mtp and !w.use_drafter and !w.use_dflash and !w.use_pld);
        try testing.expect(w.native_intent);
    }
    // enable_mtp:false opts out of DSpark; PLD intent alone never arms it.
    {
        const w = specInitWiring(true, false, true, false, false, false, false, false, true);
        try testing.expect(!w.native_intent);
    }
}

test "specTickMode: every spec arm requires the GENERATOR's armed state, not the slot flag alone" {
    // The dsv4 PLD-corruption wiring class: runPrefill's
    // per-site guard computed use_pld=false for a deepseek_v4 slot and wired
    // it into Generator init options — but the decode tick dispatched on
    // `slot.enable_pld` alone, so every tick still called `gen.nextPld` and
    // its verify forward appended draft tokens into dsv4's module-owned
    // state with no rollback (mangled DSML, log 166348-166361). mtp/drafter
    // were saved only by their accidental generator-state conjunct
    // (`gen.mtp != null`); PLD has no model handle, so its conjunct must be
    // the generator's post-chokepoint `pld_enabled`.

    // Slot wants PLD, generator was NOT armed (chokepoint or per-site guard
    // flipped it off) → the tick must run the regular path.
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, false, false, false, true, false, false));
    // Slot wants PLD and init armed it → PLD runs.
    try testing.expectEqual(SpecTickMode.pld, specTickMode(false, false, false, false, false, true, true, false));
    // Generator armed but the slot never asked (stale generator state must
    // not resurrect spec either) → regular.
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, false, false, false, false, true, false));

    // mtp/drafter keep their existing both-sides contract.
    try testing.expectEqual(SpecTickMode.mtp, specTickMode(true, true, false, false, false, false, false, false));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(true, false, false, false, false, false, false, false));
    try testing.expectEqual(SpecTickMode.drafter, specTickMode(false, false, true, true, false, false, false, false));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, true, false, false, false, false, false));

    // Priority: MTP > drafter > PLD (the spec-dispatch rule).
    try testing.expectEqual(SpecTickMode.mtp, specTickMode(true, true, true, true, false, true, true, false));
    try testing.expectEqual(SpecTickMode.drafter, specTickMode(false, false, true, true, false, true, true, false));

    // DSpark: the generator's post-chokepoint bit AND the slot's MTP flag —
    // the "model's native head" semantics (server defaults it ON for a
    // stage-bearing dsv4; the n-gram gate never touches enable_mtp). It wins
    // over everything (a set mtp/pld generator conjunct alongside dspark is
    // unreachable by the chokepoint's construction, but priority must hold).
    try testing.expectEqual(SpecTickMode.dspark, specTickMode(true, false, false, false, false, false, false, true));
    try testing.expectEqual(SpecTickMode.dspark, specTickMode(true, true, true, true, false, true, true, true));
    // PLD/drafter intent alone never drives dspark (their flags are
    // prompt-gated — riding them made engagement depend on the n-gram gate).
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, false, false, false, true, false, true));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, true, false, false, false, false, true));
    // Generator armed but the request opted enable_mtp off → serial.
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, false, false, false, false, false, true));
    // Slot asked, generator never armed dspark → falls through as before.
    try testing.expectEqual(SpecTickMode.regular, specTickMode(true, false, false, false, false, false, false, false));

    // DFlash: slot's enable_drafter + generator's dflash handle; outranks the
    // gemma drafter AND the MTP head (a loaded sidecar is the explicit
    // choice), loses to DSpark. Generator handle alone never resurrects it,
    // and a dflash generator with the slot flag off stays regular (the
    // specTickMode both-sides contract) — or MTP when that is armed.
    try testing.expectEqual(SpecTickMode.dflash, specTickMode(false, false, true, false, true, false, false, false));
    try testing.expectEqual(SpecTickMode.dflash, specTickMode(false, false, true, true, true, false, false, false));
    try testing.expectEqual(SpecTickMode.dflash, specTickMode(true, true, true, false, true, false, false, false));
    try testing.expectEqual(SpecTickMode.mtp, specTickMode(true, true, false, false, true, false, false, false));
    try testing.expectEqual(SpecTickMode.dspark, specTickMode(true, true, true, false, true, false, false, true));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, false, false, true, false, false, false));
}
