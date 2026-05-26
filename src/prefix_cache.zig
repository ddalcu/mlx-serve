//! Plan 03 — hot prefix cache (Phase 1).
//!
//! Replaces the legacy single-slot `cached_prompt_ids` with a small bounded
//! LRU keyed by `(prompt_ids ++ generated_ids, has_tools)`. Each entry owns a
//! `KVCacheSnapshot` of the live cache at the moment the request finished —
//! refcount-shared handles point at the GPU buffers that filled positions
//! 0..len. On a new request we longest-prefix match across entries, restore
//! the best one back into `xfm.cache`, and let the existing
//! truncate-then-prefill path handle the diverged tail.
//!
//! Hybrid SSM/conv architectures (qwen3_5/qwen3_5_moe/qwen3_next/nemotron_h/
//! lfm2) are excluded in v1: their recurrent state can't be rolled back, so
//! prefix reuse must reset on any divergence anyway. Plan 03's spec calls
//! these "hot tier only" — meaning we keep the single-slot path for them.
//! `HotPrefixCache.shouldUse(config)` returns false for those archs.

const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const model_mod = @import("model.zig");
const kv_quant = @import("kv_quant.zig");
const log = @import("log.zig");

const KVCache = transformer_mod.KVCache;
const KVCacheSnapshot = transformer_mod.KVCacheSnapshot;
const SSMCacheEntry = transformer_mod.SSMCacheEntry;
const SSMCheckpoint = transformer_mod.SSMCheckpoint;
const restoreSsmCheckpoint = transformer_mod.restoreSsmCheckpoint;
const ssmCheckpointBytes = transformer_mod.ssmCheckpointBytes;

/// Result of a cache lookup. Tells the caller how many tokens of `prompt_ids`
/// are already in the live cache after a successful restore — the caller
/// then prefills only the trailing diverged tokens (`prompt_ids[matched..]`).
pub const LookupResult = struct {
    /// Tokens already in the live cache. Caller prefills `prompt_ids[matched..]`.
    matched: usize,
    /// Did the restore land on an entry whose tokens span the FULL new prompt?
    /// Then identical-re-issue logic kicks in (truncate to len-1 and re-forward
    /// the last token), matching the existing reuseKVCache behavior.
    full_match: bool,
};

const Entry = struct {
    /// `prompt_ids ++ generated_ids` from the request that produced this snapshot.
    /// Owned by the entry; freed on eviction.
    tokens: []u32,
    /// Whether the request had tools enabled (different chat template, can't
    /// share cache across).
    has_tools: bool,
    /// Snapshot of the live KVCache at end of generation. Owns refcount-shared
    /// handles to the GPU buffers backing positions 0..tokens.len.
    snapshot: KVCacheSnapshot,
    /// Monotonic counter for LRU. Higher = more recent.
    last_used: u64,
    /// Wave 1.A: KV-quant scheme active when this entry was committed. A new
    /// request with a different scheme cannot restore from this entry — the
    /// underlying buffer layout (dense bf16 vs packed uint32 triples) differs,
    /// and dequantization would have happened at commit time anyway. Filter
    /// at lookup so per-request `kv_quant` overrides never produce a hit
    /// against an entry that was committed under another scheme.
    scheme: kv_quant.Scheme,
    /// KV-resident bytes for this entry, computed at commit time (Wave 1.B).
    /// Used for `--prefix-cache-mem` memory-budget enforcement; sum across
    /// all entries == `current_kv_bytes`.
    kv_bytes: u64,
    /// Phase 1 (perf-plan): SSM/conv state snapshots taken at stride-aligned
    /// positions during prefill. Sorted by `pos` ascending; the highest `pos`
    /// is at most `tokens.len`. Null for plain-attention archs. The hot
    /// cache restore picks the largest `pos ≤ matched` and rewinds both KV
    /// and SSM to it.
    ssm_checkpoints: ?[]SSMCheckpoint = null,
    /// Bytes resident in `ssm_checkpoints` (sum across all checkpoints and
    /// layers). Folded into `kv_bytes` for the byte-budget accounting so the
    /// memory cap covers both KV and SSM state.
    ssm_bytes: u64 = 0,
};

pub const HotPrefixCache = struct {
    entries: std.ArrayList(Entry),
    max_entries: u32,
    /// Wave 1.B: total KV bytes the cache is allowed to keep resident across
    /// all entries. 0 disables the byte budget (count cap still applies).
    /// Enforced on `commit`: evict LRU entries (in addition to the count
    /// cap) until `current_kv_bytes + new_entry_bytes <= max_kv_bytes`.
    max_kv_bytes: u64,
    /// Running total of `kv_bytes` across all live entries. Updated on
    /// commit/evict/invalidate.
    current_kv_bytes: u64,
    allocator: std.mem.Allocator,
    counter: u64 = 0,
    /// Set to true once we've called `xfm.resetCache()` at least once after
    /// init. The first commit on a fresh cache must seed an empty entry so
    /// future restores have something to land on.
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator, max_entries: u32) HotPrefixCache {
        return initWithMem(allocator, max_entries, 0);
    }

    pub fn initWithMem(allocator: std.mem.Allocator, max_entries: u32, max_kv_bytes: u64) HotPrefixCache {
        return .{
            .entries = std.ArrayList(Entry).empty,
            .max_entries = if (max_entries == 0) 1 else max_entries,
            .max_kv_bytes = max_kv_bytes,
            .current_kv_bytes = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HotPrefixCache) void {
        for (self.entries.items) |*e| {
            freeEntryOwnedState(self.allocator, e);
        }
        self.entries.deinit(self.allocator);
    }

    /// Free everything an Entry owns: token buffer, KV snapshot, SSM
    /// checkpoint array. Used by `deinit`, eviction, and replace paths so
    /// they don't drift apart.
    fn freeEntryOwnedState(allocator: std.mem.Allocator, e: *Entry) void {
        allocator.free(e.tokens);
        e.snapshot.deinit();
        if (e.ssm_checkpoints) |cps| {
            for (cps) |*cp| cp.deinit(allocator);
            allocator.free(cps);
            e.ssm_checkpoints = null;
        }
    }

    /// Pure-attention + DSV4 are eligible by default. Hybrid recurrent-state
    /// archs are gated by `enable_ssm_checkpoints` (set by the scheduler
    /// when `--ssm-checkpoint-stride > 0`): with checkpoints we can rewind
    /// both KV and SSM state to a stride-aligned position; without them
    /// every divergence would force a full reset, so we keep the legacy
    /// single-slot path.
    ///
    /// Both `has_hybrid_layers` and `full_attention_interval > 0` signal
    /// the model has SSM/GatedDeltaNet layers somewhere — `has_hybrid_layers`
    /// is set explicitly by the parsers for lfm2 / nemotron_h; the qwen3_5
    /// family sets `full_attention_interval` to N to mark "every Nth layer
    /// is full attention, the rest are GatedDeltaNet". Either way the same
    /// SSM-checkpoint gate applies.
    pub fn shouldUse(
        config: *const model_mod.ModelConfig,
        enable_ssm_checkpoints: bool,
    ) bool {
        const has_ssm_layers = config.has_hybrid_layers or config.full_attention_interval > 0;
        if (has_ssm_layers and !enable_ssm_checkpoints) return false;
        return true;
    }

    fn bumpCounter(self: *HotPrefixCache) u64 {
        self.counter += 1;
        return self.counter;
    }

    /// Wave 1.B: total KV bytes held by a snapshot — sum of `size * itemsize`
    /// across every initialized entry's storage arrays. mlx-c arrays carry
    /// their shape + dtype so this is exact, not a heuristic. Quant schemes
    /// account for q, scales, biases together; future schemes (TurboQuant)
    /// add `kv_quant.snapshotBytesExtra` for per-layer rotation state.
    fn snapshotBytes(snap: *const KVCacheSnapshot) u64 {
        var total: u64 = 0;
        for (snap.entries) |e| {
            if (!e.initialized) continue;
            total += @as(u64, mlx.mlx_array_size(e.keys)) * @as(u64, mlx.mlx_array_itemsize(e.keys));
            total += @as(u64, mlx.mlx_array_size(e.values)) * @as(u64, mlx.mlx_array_itemsize(e.values));
            if (snap.config.scheme != .off) {
                total += @as(u64, mlx.mlx_array_size(e.keys_scales)) * @as(u64, mlx.mlx_array_itemsize(e.keys_scales));
                total += @as(u64, mlx.mlx_array_size(e.keys_biases)) * @as(u64, mlx.mlx_array_itemsize(e.keys_biases));
                total += @as(u64, mlx.mlx_array_size(e.values_scales)) * @as(u64, mlx.mlx_array_itemsize(e.values_scales));
                total += @as(u64, mlx.mlx_array_size(e.values_biases)) * @as(u64, mlx.mlx_array_itemsize(e.values_biases));
            }
        }
        return total;
    }

    /// Find the entry with the longest prefix shared with `prompt_ids` AND
    /// matching `(has_tools, scheme)`. Returns the entry index and shared-
    /// prefix length; null if no entry matches the key. Wave 1.A: the scheme
    /// filter exists because cross-scheme buffer layouts differ — a slot
    /// running `kv_quant=4` cannot restore from an entry committed in dense
    /// (or 8-bit) mode and vice versa.
    fn findBestMatch(self: *const HotPrefixCache, prompt_ids: []const u32, has_tools: bool, scheme: kv_quant.Scheme) ?struct { idx: usize, shared: usize } {
        var best_idx: ?usize = null;
        var best_shared: usize = 0;
        for (self.entries.items, 0..) |*e, i| {
            if (e.has_tools != has_tools) continue;
            if (e.scheme != scheme) continue;
            const max_shared = @min(e.tokens.len, prompt_ids.len);
            var shared: usize = 0;
            while (shared < max_shared and e.tokens[shared] == prompt_ids[shared]) shared += 1;
            if (shared > best_shared) {
                best_shared = shared;
                best_idx = i;
            }
        }
        if (best_idx) |idx| return .{ .idx = idx, .shared = best_shared };
        return null;
    }

    /// Try to restore a matching entry into `target_cache`. On success, returns
    /// the matched prefix length. On miss, fully resets `target_cache` and
    /// returns 0. Caller should prefill the trailing tokens after this.
    ///
    /// The `target_*` parameters generalize the legacy single-slot path
    /// (`xfm.cache`, `xfm.moe_seq_offset`, `xfm.ssm_entries`) so Phase 2
    /// per-slot caches can reuse the same restore machinery.
    ///
    pub fn lookupAndRestore(
        self: *HotPrefixCache,
        target_cache: *KVCache,
        target_moe_seq_offset: *usize,
        target_ssm_entries: ?[]SSMCacheEntry,
        s: mlx.mlx_stream,
        prompt_ids: []const u32,
        has_tools: bool,
    ) !LookupResult {
        const match = self.findBestMatch(prompt_ids, has_tools, target_cache.config.scheme);

        if (match == null) {
            try target_cache.truncate(0, s);
            if (target_ssm_entries) |entries| {
                for (entries) |*ssm| {
                    _ = mlx.mlx_array_free(ssm.conv_state);
                    _ = mlx.mlx_array_free(ssm.ssm_state);
                    ssm.conv_state = mlx.mlx_array_new();
                    ssm.ssm_state = mlx.mlx_array_new();
                    ssm.initialized = false;
                }
            }
            target_moe_seq_offset.* = 0;
            return .{ .matched = 0, .full_match = false };
        }
        const m = match.?;
        const e = &self.entries.items[m.idx];
        e.last_used = self.bumpCounter();

        try target_cache.restore(&e.snapshot);

        // Hybrid path: if the entry carries SSM checkpoints, restore the SSM
        // state at the largest stride-aligned position ≤ m.shared and clamp
        // the effective matched length to that position. KV is positionally
        // trimmable; SSM is only restorable at the snapshotted positions.
        // The two MUST stay in sync, so we rewind KV further too.
        var effective_matched: usize = m.shared;
        if (target_ssm_entries) |entries| {
            if (e.ssm_checkpoints) |cps| {
                // findHighestCheckpoint: cps is sorted ascending by pos.
                var picked: ?*SSMCheckpoint = null;
                for (cps) |*cp| {
                    if (cp.pos > m.shared) break;
                    picked = cp;
                }
                if (picked) |cp| {
                    try restoreSsmCheckpoint(entries, cp);
                    effective_matched = cp.pos;
                } else {
                    // No checkpoint at or before this prefix length — reset
                    // SSM and treat the match as zero-effective (we have to
                    // cold-prefill anyway because SSM state would be wrong).
                    for (entries) |*ssm| {
                        _ = mlx.mlx_array_free(ssm.conv_state);
                        _ = mlx.mlx_array_free(ssm.ssm_state);
                        ssm.conv_state = mlx.mlx_array_new();
                        ssm.ssm_state = mlx.mlx_array_new();
                        ssm.initialized = false;
                    }
                    effective_matched = 0;
                }
            } else {
                // Hybrid model without checkpoints (e.g., committed pre-Phase-1).
                // Reset and treat as cold prefill — we can't safely reuse.
                for (entries) |*ssm| {
                    _ = mlx.mlx_array_free(ssm.conv_state);
                    _ = mlx.mlx_array_free(ssm.ssm_state);
                    ssm.conv_state = mlx.mlx_array_new();
                    ssm.ssm_state = mlx.mlx_array_new();
                    ssm.initialized = false;
                }
                effective_matched = 0;
            }
        }
        target_moe_seq_offset.* = effective_matched;

        // Miss path (hybrid without a usable checkpoint): also reset KV.
        if (effective_matched == 0) {
            try target_cache.truncate(0, s);
            log.info("  [hot-cache] hybrid miss (no checkpoint ≤ {d} of {d}); cold prefill\n", .{ m.shared, prompt_ids.len });
            return .{ .matched = 0, .full_match = false };
        }

        const full_match = effective_matched == prompt_ids.len;
        const final_len: usize = if (full_match and effective_matched > 1) effective_matched - 1 else effective_matched;

        if (final_len < e.tokens.len) {
            try target_cache.truncate(final_len, s);
        }

        if (full_match and effective_matched > 1) {
            target_moe_seq_offset.* = effective_matched - 1;
            log.info("  [hot-cache] full reuse {d}/{d}, re-forwarding last token\n", .{ effective_matched - 1, prompt_ids.len });
            return .{ .matched = effective_matched - 1, .full_match = true };
        }

        log.info("  [hot-cache] reused {d}/{d} tokens (matched {d}; entry {d}/{d})\n", .{ effective_matched, prompt_ids.len, m.shared, m.idx + 1, self.entries.items.len });
        return .{ .matched = effective_matched, .full_match = full_match };
    }

    /// Commit the current `source_cache` state under the given key. Updates
    /// the matching entry if one exists for this exact prefix, otherwise
    /// inserts a new entry, evicting the oldest if at capacity. Snapshot is
    /// taken here (cheap — refcount-share, no data copy).
    ///
    pub fn commit(
        self: *HotPrefixCache,
        source_cache: *const KVCache,
        tokens: []const u32,
        has_tools: bool,
    ) !void {
        return self.commitWithSsm(source_cache, tokens, has_tools, null);
    }

    /// Commit with optional SSM checkpoint array (Phase 1). The caller
    /// transfers ownership of the slice — the entry frees it on eviction via
    /// the shared `freeEntryOwnedState`. Pass null on plain-attention archs;
    /// the entry stays SSM-free.
    pub fn commitWithSsm(
        self: *HotPrefixCache,
        source_cache: *const KVCache,
        tokens: []const u32,
        has_tools: bool,
        ssm_cps: ?[]SSMCheckpoint,
    ) !void {
        const scheme = source_cache.config.scheme;

        var replace_idx: ?usize = null;
        for (self.entries.items, 0..) |*e, i| {
            if (e.has_tools != has_tools) continue;
            if (e.scheme != scheme) continue;
            if (e.tokens.len <= tokens.len) {
                var shared: usize = 0;
                while (shared < e.tokens.len and e.tokens[shared] == tokens[shared]) shared += 1;
                if (shared == e.tokens.len) {
                    replace_idx = i;
                    break;
                }
            }
        }

        const new_snap = try source_cache.snapshot();
        const new_kv_bytes = snapshotBytes(&new_snap);
        var new_ssm_bytes: u64 = 0;
        if (ssm_cps) |cps| {
            for (cps) |*cp| new_ssm_bytes += ssmCheckpointBytes(cp);
        }
        const new_bytes = new_kv_bytes + new_ssm_bytes;
        const tokens_owned = self.allocator.dupe(u32, tokens) catch |err| {
            var snap = new_snap;
            snap.deinit();
            if (ssm_cps) |cps| {
                for (cps) |*cp| cp.deinit(self.allocator);
                self.allocator.free(cps);
            }
            return err;
        };

        if (replace_idx) |idx| {
            const e = &self.entries.items[idx];

            // Phase 1: SSM checkpoint inheritance on prefix-extend. The
            // replace path triggers when the new entry's tokens fully
            // extend the old's (i.e., e.tokens is a prefix of `tokens`).
            // The old SSM checkpoints were captured at positions inside
            // e.tokens, so they're still valid for the new entry — those
            // positions are a strict prefix of `tokens`. Inherit them and
            // append any new checkpoints from this turn that don't overlap.
            //
            // Without this, multi-turn flows lose checkpoints fast: turn 2's
            // prefill of the short tail captures few or no checkpoints, so
            // turn 3 has nothing to restore from even though turn 2's match
            // covered nearly the full prefix. (Reproducible by alternating
            // identical-prompt requests at ssm_checkpoint_stride > prompt_len.)
            const merged_cps: ?[]SSMCheckpoint = blk: {
                const old = e.ssm_checkpoints orelse break :blk ssm_cps;
                const new = ssm_cps orelse {
                    // Detach old so the free-below doesn't touch it; it
                    // becomes the new entry's checkpoint list as-is.
                    e.ssm_checkpoints = null;
                    break :blk old;
                };
                // Both old and new have data. Concat into a sorted-by-pos,
                // dedup-by-pos list. Allocate fresh; transfer ownership.
                var merged = std.ArrayList(SSMCheckpoint).empty;
                errdefer {
                    for (merged.items) |*c| c.deinit(self.allocator);
                    merged.deinit(self.allocator);
                }
                // Detach old + new from their containers so we can move them.
                e.ssm_checkpoints = null;
                // Walk both, picking lower pos each step; on tie, prefer the
                // new one and discard old (new is the more recently observed
                // state at that position).
                var i: usize = 0;
                var j: usize = 0;
                while (i < old.len or j < new.len) {
                    if (i >= old.len) {
                        try merged.append(self.allocator, new[j]);
                        j += 1;
                    } else if (j >= new.len) {
                        try merged.append(self.allocator, old[i]);
                        i += 1;
                    } else if (old[i].pos < new[j].pos) {
                        try merged.append(self.allocator, old[i]);
                        i += 1;
                    } else if (old[i].pos > new[j].pos) {
                        try merged.append(self.allocator, new[j]);
                        j += 1;
                    } else {
                        // Same pos: keep new, drop old.
                        var dropped = old[i];
                        dropped.deinit(self.allocator);
                        i += 1;
                        try merged.append(self.allocator, new[j]);
                        j += 1;
                    }
                }
                self.allocator.free(old);
                self.allocator.free(new);
                break :blk try merged.toOwnedSlice(self.allocator);
            };

            // Free everything the old entry owned EXCEPT the (now-detached)
            // ssm_checkpoints, which were moved into `merged_cps` above.
            self.allocator.free(e.tokens);
            e.snapshot.deinit();
            self.current_kv_bytes -|= e.kv_bytes;

            // Recompute ssm bytes from the merged list.
            var merged_ssm_bytes: u64 = 0;
            if (merged_cps) |cps| {
                for (cps) |*cp| merged_ssm_bytes += ssmCheckpointBytes(cp);
            }

            e.tokens = tokens_owned;
            e.snapshot = new_snap;
            e.has_tools = has_tools;
            e.scheme = scheme;
            e.kv_bytes = new_kv_bytes + merged_ssm_bytes;
            e.ssm_checkpoints = merged_cps;
            e.ssm_bytes = merged_ssm_bytes;
            e.last_used = self.bumpCounter();
            self.current_kv_bytes += e.kv_bytes;
            self.logResident();
            return;
        }

        while (self.entries.items.len >= self.max_entries) {
            self.evictOneLru("count cap");
        }
        if (self.max_kv_bytes > 0) {
            while (self.current_kv_bytes + new_bytes > self.max_kv_bytes and self.entries.items.len > 0) {
                self.evictOneLru("byte budget");
            }
        }

        self.entries.append(self.allocator, .{
            .tokens = tokens_owned,
            .has_tools = has_tools,
            .snapshot = new_snap,
            .last_used = self.bumpCounter(),
            .scheme = scheme,
            .kv_bytes = new_bytes,
            .ssm_checkpoints = ssm_cps,
            .ssm_bytes = new_ssm_bytes,
        }) catch |err| {
            self.allocator.free(tokens_owned);
            var snap = new_snap;
            snap.deinit();
            if (ssm_cps) |cps| {
                for (cps) |*cp| cp.deinit(self.allocator);
                self.allocator.free(cps);
            }
            return err;
        };
        self.current_kv_bytes += new_bytes;
        self.logResident();
    }

    fn evictOneLru(self: *HotPrefixCache, reason: []const u8) void {
        var lru_idx: usize = 0;
        var lru_used: u64 = std.math.maxInt(u64);
        for (self.entries.items, 0..) |*e, i| {
            if (e.last_used < lru_used) {
                lru_used = e.last_used;
                lru_idx = i;
            }
        }
        var evicted = self.entries.swapRemove(lru_idx);
        const tokens_len = evicted.tokens.len;
        const kv_mb = @as(f64, @floatFromInt(evicted.kv_bytes)) / (1024.0 * 1024.0);
        const had_ssm = evicted.ssm_checkpoints != null;
        const ssm_mb = @as(f64, @floatFromInt(evicted.ssm_bytes)) / (1024.0 * 1024.0);
        self.current_kv_bytes -|= evicted.kv_bytes;
        freeEntryOwnedState(self.allocator, &evicted);
        if (had_ssm) {
            log.info("  [hot-cache] evicted LRU entry ({s}; was {d} tokens, {d:.2} MB; ssm {d:.2} MB)\n", .{
                reason, tokens_len, kv_mb, ssm_mb,
            });
        } else {
            log.info("  [hot-cache] evicted LRU entry ({s}; was {d} tokens, {d:.2} MB)\n", .{
                reason, tokens_len, kv_mb,
            });
        }
    }

    fn logResident(self: *const HotPrefixCache) void {
        const mb = @as(f64, @floatFromInt(self.current_kv_bytes)) / (1024.0 * 1024.0);
        if (self.max_kv_bytes == 0) {
            log.info("  [hot-cache] resident={d:.2} MB ({d}/{d} entries)\n", .{ mb, self.entries.items.len, self.max_entries });
        } else {
            const cap_mb = @as(f64, @floatFromInt(self.max_kv_bytes)) / (1024.0 * 1024.0);
            log.info("  [hot-cache] resident={d:.2} / {d:.2} MB ({d}/{d} entries)\n", .{ mb, cap_mb, self.entries.items.len, self.max_entries });
        }
    }

    /// Drop all entries — forces every future request to cold-prefill. Called
    /// when the cache is suspect (pad-only generation, image-bearing prompt,
    /// tools toggle change).
    pub fn invalidateAll(self: *HotPrefixCache, reason: []const u8) void {
        if (self.entries.items.len == 0) return;
        log.info("  [hot-cache] invalidating all {d} entries: {s}\n", .{ self.entries.items.len, reason });
        for (self.entries.items) |*e| {
            freeEntryOwnedState(self.allocator, e);
        }
        self.entries.clearRetainingCapacity();
        self.current_kv_bytes = 0;
    }

    /// Drop the most recently committed entry — used after a pad-only
    /// generation: the entry we just wrote may have stale K/V from the bad
    /// generation in tail positions. Other entries from prior healthy
    /// requests remain untouched (improvement over the legacy nuke-everything).
    pub fn invalidateLatest(self: *HotPrefixCache, reason: []const u8) void {
        if (self.entries.items.len == 0) return;
        var newest_idx: usize = 0;
        var newest_used: u64 = 0;
        for (self.entries.items, 0..) |*e, i| {
            if (e.last_used >= newest_used) {
                newest_used = e.last_used;
                newest_idx = i;
            }
        }
        var evicted = self.entries.swapRemove(newest_idx);
        self.current_kv_bytes -|= evicted.kv_bytes;
        freeEntryOwnedState(self.allocator, &evicted);
        log.info("  [hot-cache] invalidated latest entry: {s}\n", .{reason});
    }

    pub fn entryCount(self: *const HotPrefixCache) usize {
        return self.entries.items.len;
    }
};

// ── Tests ──

const testing = std.testing;

test "HotPrefixCache: shouldUse gates hybrid by enable_ssm_checkpoints" {
    var cfg = model_mod.ModelConfig{};
    // Plain attention: always allowed.
    try testing.expect(HotPrefixCache.shouldUse(&cfg, false));
    try testing.expect(HotPrefixCache.shouldUse(&cfg, true));
    // Hybrid (lfm2/nemotron_h-style): only with checkpoints enabled.
    cfg.has_hybrid_layers = true;
    try testing.expect(!HotPrefixCache.shouldUse(&cfg, false));
    try testing.expect(HotPrefixCache.shouldUse(&cfg, true));
    // Qwen3.5-style full_attention_interval-marks-hybrid: same gate.
    cfg.has_hybrid_layers = false;
    cfg.full_attention_interval = 4;
    try testing.expect(!HotPrefixCache.shouldUse(&cfg, false));
    try testing.expect(HotPrefixCache.shouldUse(&cfg, true));
}

test "HotPrefixCache: init zero capacity clamps to 1" {
    var cache = HotPrefixCache.init(testing.allocator, 0);
    defer cache.deinit();
    try testing.expectEqual(@as(u32, 1), cache.max_entries);
    try testing.expectEqual(@as(usize, 0), cache.entryCount());
}

test "HotPrefixCache: findBestMatch returns longest shared prefix" {
    var cache = HotPrefixCache.init(testing.allocator, 4);
    defer cache.deinit();

    // Two synthetic entries (snapshots are no-ops on freshly-zero KVCache; we
    // never restore in this unit test, so no GPU work).
    const ids_a = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, 4, 5 });
    const ids_b = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, 9, 9, 9 });
    try cache.entries.append(testing.allocator, .{
        .tokens = ids_a,
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = transformer_mod.KVQuantConfig.dense },
        .last_used = 1,
        .scheme = .off,
        .kv_bytes = 0,
        .ssm_checkpoints = null,
        .ssm_bytes = 0,
    });
    try cache.entries.append(testing.allocator, .{
        .tokens = ids_b,
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = transformer_mod.KVQuantConfig.dense },
        .last_used = 2,
        .scheme = .off,
        .kv_bytes = 0,
        .ssm_checkpoints = null,
        .ssm_bytes = 0,
    });

    // Looking up [1,2,3,4,5,6] should match entry A (5 shared tokens).
    const lookup_ids = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const m = cache.findBestMatch(&lookup_ids, false, .off).?;
    try testing.expectEqual(@as(usize, 0), m.idx);
    try testing.expectEqual(@as(usize, 5), m.shared);

    // Looking up [1,2,3,9,9,9,7] should match entry B (6 shared).
    const lookup_ids2 = [_]u32{ 1, 2, 3, 9, 9, 9, 7 };
    const m2 = cache.findBestMatch(&lookup_ids2, false, .off).?;
    try testing.expectEqual(@as(usize, 1), m2.idx);
    try testing.expectEqual(@as(usize, 6), m2.shared);

    // has_tools mismatch returns null.
    try testing.expectEqual(@as(?@TypeOf(m), null), cache.findBestMatch(&lookup_ids, true, .off));
    // scheme mismatch returns null too — entries are .off, a query for .affine
    // cannot match (Wave 1.A: cross-scheme cache hits never happen).
    try testing.expectEqual(@as(?@TypeOf(m), null), cache.findBestMatch(&lookup_ids, false, .affine));
}
