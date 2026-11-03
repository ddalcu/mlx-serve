//! Prefix-cache commit of a finished or cancelled slot, and the decode-end checkpoint.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const transformer_mod = @import("transformer.zig");
const generate_mod = @import("generate.zig");
const prefix_cache_mod = @import("cache/prefix_cache.zig");
const plugin_driver = @import("plugin/driver.zig");
const plugin_abi = @import("../../abi/plugin.zig");
const log = @import("../../sdk/util/log.zig");
const Generator = generate_mod.Generator;
const testing = std.testing;

const publishHotCacheResidency = @import("cache/budget.zig").publishHotCacheResidency;
const Scheduler = @import("scheduler.zig").Scheduler;
const Slot = @import("slot.zig").Slot;

fn dflashContextCoversPrefix(context_len: usize, prefix_len: usize) bool {
    return context_len == prefix_len;
}

/// Only an all-pad generation poisons the prefix; a zero-token one still holds a real prefill.
fn commitDeclinesPadOnly(n_gen: usize, all_pad: bool) bool {
    return n_gen > 0 and all_pad;
}

/// Snapshot the live SSM state at the committed length so the next turn resumes after this
/// one's generated tail; prefill snapshots end SSM_SNAPSHOT_BACKOFF tokens before the prompt end.
/// Exact there because every speculative round rolls the GDN state back before it returns.
fn appendDecodeEndCheckpoint(
    slot: *Slot,
    gen: *Generator,
    cps: []transformer_mod.SSMCheckpoint,
    total_len: usize,
) []transformer_mod.SSMCheckpoint {
    const entries = slot.ssm_entries orelse return cps;
    const xfm = slot.model.transformer orelse return cps;
    const a = gen.ssm_checkpoint_alloc orelse return cps;
    const newest: ?usize = if (cps.len > 0) cps[cps.len - 1].pos else null;
    // A plain pipelined step that met its stop already forwarded the stop token and a lookahead.
    if (slot.moe_seq_offset > total_len) _ = gen.rollbackSsmTo(total_len);
    if (!decodeEndCheckpointWanted(slot.full_prompt.len, total_len, slot.moe_seq_offset, newest)) {
        if (slot.moe_seq_offset != total_len) log.debug("[hot-cache] decode-end checkpoint skipped: live position {d}, committed {d}\n", .{ slot.moe_seq_offset, total_len });
        return cps;
    }
    var cp = transformer_mod.captureSsmCheckpoint(a, entries, total_len, xfm.s) catch |err| {
        log.warn("[hot-cache] decode-end checkpoint failed: {s} — the next turn re-forwards this tail\n", .{@errorName(err)});
        return cps;
    };
    const out = a.realloc(cps, cps.len + 1) catch {
        cp.deinit(a);
        return cps;
    };
    out[out.len - 1] = cp;
    log.info("  [hot-cache] decode-end checkpoint at {d} ({d} prompt + {d} generated)\n", .{ total_len, slot.full_prompt.len, total_len - slot.full_prompt.len });
    return out;
}

/// PURE: the live state must describe exactly the committed tokens, past every prefill snapshot.
/// `live_pos` counts tokens on media slots too: the M-RoPE shift is `mrope_delta`, kept apart.
fn decodeEndCheckpointWanted(prompt_len: usize, total_len: usize, live_pos: usize, newest_cp: ?usize) bool {
    if (total_len <= prompt_len or live_pos != total_len) return false;
    if (newest_cp) |pos| return pos < total_len;
    return true;
}

/// Phase A6: commit a successfully completed slot's KV cache to the hot
/// prefix cache. Called from the inference thread BEFORE `markFinished`
/// broadcasts, so the slot is still alive (the conn thread is blocked in
/// `waitNext`). Skipped for pad-only generations, vision-bearing slots
/// (stale embeddings would be reused), and slots with no generated tokens.
pub fn commitSlotIfApplicable(sch: *Scheduler, slot: *Slot) void {
    // Phase D: per-model prefix cache — read off the slot's LoadedModel.
    const hc: *prefix_cache_mod.HotPrefixCache = if (slot.model.prefix_cache) |p| p else return;
    if (slot.error_code != null) return;
    // `finishSlot` can be reached on an EOS the failing forward itself produced, before the
    // tick wrapper reads the latch: an unread latch means the KV under this commit may be garbage.
    if (mlx.errorPending()) return;
    if (slot.model.transformer) |xfm| if (prefix_cache_mod.ModuleBinding.init(xfm)) |binding| {
        if (slot.cancelled.load(.acquire)) return;
        const gen = if (slot.legacy_gen) |*g| g else return;
        if (commitDeclinesPadOnly(gen.generated_ids.items.len, slot.was_pad_only)) return;
        const snap = plugin_driver.auxSnapshot(xfm, gen.ctx.plugin_slot) orelse return;
        var info: plugin_abi.AuxSnapshotInfo = .{};
        if (!plugin_driver.auxDescribe(xfm, snap, &info).isOk() or
            info.n == 0 or info.n != info.ids.len or info.n > slot.full_prompt.len or
            !std.mem.eql(u32, info.ids.slice(), slot.full_prompt[0..@intCast(info.n)]) or mlx.errorPending())
        {
            plugin_driver.auxFree(xfm, snap);
            return;
        }
        // The snapshot describes its own forwarded prompt boundary, excluding the generated tail.
        _ = hc.commitModule(binding, snap, info.ids.slice(), info.prefix_key, slot.media, slot.has_tools, slot.cache.config, slot.cache_key) catch |err| {
            log.warn("[hot-cache] module commit failed: {s}\n", .{@errorName(err)});
            return;
        };
        publishHotCacheResidency(sch);
        return;
    };
    const gen_ptr = if (slot.legacy_gen) |*g| g else {
        // A Generator-less slot whose cache is non-empty is a prefill the
        // client disconnected from mid-chunk-loop: initWithOptions threw
        // error.Cancelled before `slot.legacy_gen = gen` ever ran, but every
        // chunk that DID forward still lives in slot.cache. Commit that
        // forwarded prefix instead of dropping it. This arm sits ABOVE the
        // pad-only guard: `was_pad_only` starts true and only flips on the
        // first pushed token, so a slot that never pushed one is not
        // pad-POISONED, it is merely empty.
        commitCancelledPrefillSlot(slot, hc);
        publishHotCacheResidency(sch);
        return;
    };
    const n_gen = gen_ptr.generated_ids.items.len;
    if (commitDeclinesPadOnly(n_gen, slot.was_pad_only)) return;

    // Construct the full token sequence: the original prompt + everything
    // generated this turn that the cache holds — Generator forwarded each
    // emitted token into slot.cache as it was sampled, except an
    // `unforwarded_tail` the key must not claim.
    const total_len = slot.full_prompt.len + gen_ptr.generated_ids.items.len - gen_ptr.unforwarded_tail;
    const total_tokens = sch.allocator.alloc(u32, total_len) catch return;
    defer sch.allocator.free(total_tokens);
    @memcpy(total_tokens[0..slot.full_prompt.len], slot.full_prompt);
    @memcpy(total_tokens[slot.full_prompt.len..], gen_ptr.generated_ids.items[0 .. total_len - slot.full_prompt.len]);

    // Phase 1: drain any SSM checkpoints captured by the Generator's prefill
    // loop and hand them to the cache alongside the KV snapshot. For plain-
    // attn models this returns an empty slice (no allocator hit). Ownership
    // transfers to the cache via `commitWithSsm`; freeing happens on
    // eviction.
    const ssm_cps_slice = appendDecodeEndCheckpoint(slot, gen_ptr, gen_ptr.takeSsmCheckpoints(), total_len);
    const ssm_cps_opt: ?[]transformer_mod.SSMCheckpoint = if (ssm_cps_slice.len > 0) ssm_cps_slice else null;
    if (ssm_cps_slice.len == 0 and gen_ptr.ssm_checkpoint_alloc != null) {
        // Empty list — free the (zero-length) slice we got back so the
        // allocator's bookkeeping stays clean.
        gen_ptr.ssm_checkpoint_alloc.?.free(ssm_cps_slice);
    }
    // qwen4_exp: the newest checkpoint takes the slot's live QSA indexer history as a view
    // of the capacity buffer (the slot is torn down right after). A failure commits the
    // entry history-less, which a QSA arch treats as a miss.
    if (ssm_cps_opt) |cps| {
        if (transformer_mod.qsaHistoryShareEnabled() and !transformer_mod.checkpointHasQsaPooled(&cps[cps.len - 1])) {
            if (slot.ssm_entries) |ents| {
                if (slot.model.transformer) |xf| {
                    transformer_mod.handoffQsaHistoryToLatest(cps, ents, xf.s) catch |err| {
                        log.warn("[hot-cache] QSA history handoff failed: {s} — committing without it\n", .{@errorName(err)});
                    };
                }
            }
        }
    }
    // Plain DFlash rounds park their captures; they join the context before it is paired with the prefix.
    gen_ptr.dflashSettle(slot.allocator, total_len) catch |err| log.warn("[hot-cache] dflash context settle failed: {s} — not committed\n", .{@errorName(err)});
    // A runtime fallback leaves the dormant assistant context at its last
    // speculative boundary while serial decode continues growing the trunk.
    // Only pair the assistant payload with this prefix when both end at the
    // exact same absolute position; otherwise the next turn must rebuild it.
    const dflash_commit: ?prefix_cache_mod.DflashCommit = if (gen_ptr.dflash_ctx) |*dc|
        if (dflashContextCoversPrefix(dc.absLen(), total_len))
            .{ .cache = &dc.cache, .base_pos = dc.base_pos }
        else
            null
    else
        null;
    // MTP committed history: unlike dflash there is no exact-coverage
    // requirement — the restore clamps to the matched length and declines a
    // history that ends short, so a partial history (the deferred-stash lag,
    // a runtime disable) is still worth committing. What MUST hold is that
    // only COMMITTED entries are snapshotted: truncate off the speculative
    // draft tail first (offset-only, cheap).
    // The last committed position's hidden pairs with the next prompt's first token: kept, a
    // decode-end restore appends the head's missing row instead of drafting blind.
    var mtp_tail: mlx.mlx_array = .{ .ctx = null };
    defer if (mtp_tail.ctx != null) {
        _ = mlx.mlx_array_free(mtp_tail);
    };
    const mtp_commit: ?prefix_cache_mod.DflashCommit = blk: {
        const mc = if (gen_ptr.mtp_cache) |*m| m else break :blk null;
        // A released module head is another slot's to read now; the history stopped
        // growing at the switch anyway.
        if (gen_ptr.mtpModuleHeadReleased()) break :blk null;
        const committed = gen_ptr.mtpCommittedHistoryLen();
        if (committed == 0) break :blk null;
        if (gen_ptr.has_last_hidden and gen_ptr.mtp_position_base + committed + 1 == total_len) {
            mtp_tail = transformer_mod.materializedOwnedCopy(slot.model.transformer.?.s, gen_ptr.last_hidden) catch .{ .ctx = null };
        }
        mc.truncate(committed, slot.model.transformer.?.s) catch |err| {
            log.warn("[hot-cache] mtp history trim failed: {s} — not committed\n", .{@errorName(err)});
            break :blk null;
        };
        // The qwen4_exp in-checkpoint head also commits its QSA half.
        mc.activate();
        const head = mc.head();
        // Its row count IS its cache's step (`KVCache.update` never advances a non-zero
        // layer index); a snapshot whose step disagrees is refused, and says why.
        if (head) |t| {
            const hm = &t.qwen4_mtp.?;
            if (hm.cache.step != hm.seq_offset) {
                log.warn("[hot-cache] mtp head step gap (cache.step={d}, history={d} rows) — head history not committed\n", .{ hm.cache.step, hm.seq_offset });
                break :blk null;
            }
        }
        break :blk .{
            .cache = mc.kv() orelse break :blk null,
            .base_pos = gen_ptr.mtp_position_base,
            .head = if (head) |t| &t.qwen4_mtp.?.entry else null,
            .head_pos_base = if (head) |t| t.qwen4_mtp.?.pos_base else 0,
            .head_marks = if (head) |t| t.qwen4MtpMarks() else &.{},
            .tail_hidden = if (mtp_tail.ctx != null) mtp_tail else null,
        };
    };
    const finish_st = hc.commitWithMediaState(&slot.cache, total_tokens, slot.has_tools, slot.media, slot.cache_key, ssm_cps_opt, dflash_commit, mtp_commit, slot.full_prompt.len) catch |err| {
        // Ownership of the checkpoints transferred to the cache regardless of
        // the outcome — its error paths free them (#330 adjacent: freeing
        // here too was a double free, with a different allocator).
        log.warn("[hot-cache] commit failed: {s}\n", .{@errorName(err)});
        return;
    };
    publishHotCacheResidency(sch);
    // The decline paths already log their reason inside the cache; a silent
    // `_ =` here is fine — this caller has no commit-shaped log to lie about.
    _ = finish_st;
}

/// Logical committed length for a cancelled-prefill commit: the tokens
/// actually forwarded into the KV when the chunk loop aborted, clamped to
/// the prompt and gated on the floor below which an entry is LRU pollution
/// rather than saved work. Pure so the policy is unit-testable without a slot.
fn cancelledPrefillCommitLen(step: usize, prompt_len: usize) ?usize {
    const len = @min(step, prompt_len);
    if (len < prefix_cache_mod.MIN_CANCELLED_COMMIT_TOKENS) return null;
    return len;
}

/// Commit the forwarded prefix of a prefill the client disconnected from
/// (error.Cancelled out of `Generator.initWithOptions`; `legacy_gen` was
/// never assigned). The partial KV in `slot.cache` is valid — every chunk
/// that ran was a real forward — so the next request sharing this prefix
/// skips exactly those chunks. The entry key is the forwarded prefix ONLY:
/// `full_prompt` beyond `cache.step` was never forwarded and must not ride
/// the key (a key longer than its KV is the alignment-error class).
///
/// Hybrids commit iff checkpoints were salvaged (`slot.cancelled_prefill`):
/// `initWithOptions` hands its captured stride checkpoints to the slot on
/// the abort, and they restore exactly like a normal finish's. KV-only
/// hybrid entries still restore as a cold miss ("hybrid miss") while
/// occupying an LRU slot, so a salvage-less hybrid prefill is declined —
/// the cancel landed before the first stride boundary. decode-phase cancels
/// keep their checkpoints in the live Generator and commit through the
/// normal arm above.
fn commitCancelledPrefillSlot(slot: *Slot, hc: *prefix_cache_mod.HotPrefixCache) void {
    const salvage = &slot.cancelled_prefill;
    // Hybrid restore requires SSM checkpoints; a checkpoint-less hybrid
    // entry restores as a cold miss ("hybrid miss") while occupying an LRU
    // slot. Non-hybrids commit KV-only.
    if (slot.ssm_entries != null and salvage.checkpoints.len == 0) {
        log.debug("[hot-cache] cancelled prefill carried no stride checkpoints — a hybrid entry would restore as a miss; not committed\n", .{});
        return;
    }
    // The sink's `forwarded` is the authoritative length — `cache.step`
    // only advances when Generator init completes, so it reads 0 on every
    // aborted prefill.
    const len = cancelledPrefillCommitLen(salvage.forwarded, slot.full_prompt.len) orelse {
        log.debug("[hot-cache] cancelled prefill forwarded < {d} tokens — below the commit floor; not committed\n", .{prefix_cache_mod.MIN_CANCELLED_COMMIT_TOKENS});
        return;
    };
    const cps: ?[]transformer_mod.SSMCheckpoint = if (salvage.checkpoints.len > 0) salvage.checkpoints else null;
    // Ownership of the checkpoints transfers to the cache unconditionally —
    // its error paths free them (#330 adjacent) — so detach from the slot
    // BEFORE the call or Slot.deinit frees them a second time.
    slot.cancelled_prefill = .{};
    const st = hc.commitWithMediaState(&slot.cache, slot.full_prompt[0..len], slot.has_tools, slot.media, slot.cache_key, cps, null, null, len) catch |err| {
        log.warn("[hot-cache] cancelled-prefill commit failed: {s}\n", .{@errorName(err)});
        return;
    };
    // Truthful outcome only: a budget decline must never print as a commit
    // (live: "skipped oversized" + "committed N/M" were the SAME
    // event and the cache looked healthy while a 122k session re-prefilled
    // ~95k tokens per retry).
    switch (st) {
        .ok => |n| log.info("[hot-cache] committed {d}/{d} prompt tokens from a cancelled prefill\n", .{ n, slot.full_prompt.len }),
        .disk_only => |n| log.info("[disk-cache] captured {d}/{d} prompt tokens from a cancelled prefill\n", .{ n, slot.full_prompt.len }),
        .kept_resident => |n| log.info("[hot-cache] kept resident {d}-token entry; oversized candidate declined\n", .{n}),
        .declined => {},
    }
}

// ── tests ──

test "DFlash cache payload is committed only when it spans the trunk prefix" {
    try testing.expect(dflashContextCoversPrefix(128, 128));
    try testing.expect(!dflashContextCoversPrefix(96, 128));
}

test "cancelled-prefill commit length: floor, clamp, and zero" {
    // A cancelled prefill only pays its way into the LRU once the forwarded
    // prefix is past the chat-template-prologue class (~dozens of tokens);
    // below the floor the entry is pollution, not saved work.
    try testing.expect(cancelledPrefillCommitLen(0, 1000) == null);
    try testing.expect(cancelledPrefillCommitLen(1, 1000) == null);
    try testing.expect(cancelledPrefillCommitLen(255, 1000) == null);
    try testing.expectEqual(@as(?usize, 256), cancelledPrefillCommitLen(256, 1000));
    try testing.expectEqual(@as(?usize, 300), cancelledPrefillCommitLen(300, 1000));
    // cache.step can never exceed the prompt (restore clamps to matched and
    // chunks stop at the tail); clamp defensively anyway so a bad step can
    // never key tokens the KV does not hold.
    try testing.expectEqual(@as(?usize, 1000), cancelledPrefillCommitLen(5000, 1000));
    // A short prompt is not worth an entry at any step.
    try testing.expect(cancelledPrefillCommitLen(50, 50) == null);
}

test "prefix-cache commit declines a pad-only generation, never a zero-token one" {
    // Bar: a zero-token answer still commits; only an all-pad generation is declined.
    try testing.expect(!commitDeclinesPadOnly(0, true));
    try testing.expect(!commitDeclinesPadOnly(0, false));
    try testing.expect(commitDeclinesPadOnly(3, true));
    try testing.expect(!commitDeclinesPadOnly(3, false));
}

test "decode-end checkpoint: taken only where the live state is the committed prefix" {
    // Bar: a turn that generated past its prompt-end snapshot gets one at its committed end, nothing else does.
    try testing.expect(decodeEndCheckpointWanted(1000, 1200, 1200, 970));
    try testing.expect(decodeEndCheckpointWanted(1000, 1200, 1200, null));
    try testing.expect(!decodeEndCheckpointWanted(1000, 1000, 1000, 970));
    try testing.expect(!decodeEndCheckpointWanted(1000, 1200, 1201, 970));
    try testing.expect(!decodeEndCheckpointWanted(1000, 1200, 1199, 970));
    try testing.expect(!decodeEndCheckpointWanted(1000, 1200, 1200, 1200));
}
