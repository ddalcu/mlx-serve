//! References to the MTP head and its cache across the sidecar, in-checkpoint and module-owned heads.
const std = @import("std");
const modules = @import("../modules/call.zig");
const Qwen4Mtp = @import("../../../plugins/qwen4_exp/mtp.zig");
const Qwen4Trunk = @import("../../../plugins/qwen4_exp/trunk.zig").Trunk;
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const mtp_lookup = @import("mtp_lookup.zig");
const mtp_mod = @import("../../../sdk/spec/mtp/mtp.zig");
const plugin_head = @import("../plugin/head.zig");
const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const ssmFreeSpecCapture = @import("../ssm/entry.zig").ssmFreeSpecCapture;
const testing = std.testing;


/// Grammar-constrained sampling state. The caller owns `grammar`, `token_bytes`,
/// and `mask_buf`; the generator only reads them. `mask_buf.len` must equal
/// `token_bytes.bytes.len` (the tokenizer's vocab size).
/// Which MTP head a generator drives.
///
/// The controller around it is head-agnostic — the EV planner, the acceptance
/// gate, `mtpBatchedAcceptGraph`, the pre-draft and the horizon valve all work
/// in tokens and probabilities — so the split is exactly these five operations
/// and nothing else.
/// Most drafts one MTP round verifies: the head chain or a prompt-lookup run.
pub const MAX_ROUND_DRAFTS: usize = @max(mtp_mod.MAX_DEPTH, mtp_lookup.MAX_DRAFT_STRONG);

pub const MtpHeadRef = union(enum) {
    qwen: *mtp_mod.MtpModel,
    /// qwen4_exp: the head and its history live on the plugin's trunk module
    /// (`qwen4_mtp`, module-owned ⇒ single-flight); row r of the history is
    /// (pre-mixer stream at position r, token r+1), query position r+1.
    qwen4: *Qwen4Trunk,
    /// A served model's head (`SpecHooks.head`); its cache is a one-layer `KVCache` (`MtpCacheRef.qwen`).
    plugin: *plugin_head.Head,

    /// Is this head's decode state module-owned (one per model, `Qwen4Mtp.cache`) rather
    /// than per-request? Both the scheduler and the sticky-serial arm ask this.
    pub fn moduleOwned(self: MtpHeadRef) bool {
        return switch (self) {
            .qwen => false,
            // Per-request state swaps onto the module (`Qwen4MtpState`); nothing is shared.
            .qwen4, .plugin => false,
        };
    }

    pub fn makeCache(self: MtpHeadRef, allocator: std.mem.Allocator) !MtpCacheRef {
        return switch (self) {
            .qwen => |h| .{ .qwen = try h.makeCache(allocator) },
            .qwen4 => |t| blk: {
                const st = try allocator.create(Qwen4Mtp.Qwen4MtpState);
                errdefer allocator.destroy(st);
                st.* = try t.qwen4MtpStateNew();
                t.qwen4MtpActivate(st);
                break :blk .{ .qwen4 = .{ .t = t, .st = st, .allocator = allocator } };
            },
            .plugin => |h| .{ .plugin = try MtpCacheRef.PluginRef.init(allocator, h.heads) },
        };
    }

    /// The head projects only what the caller consumes: this step wants the
    /// LAST row (`want_logits`) or no logits at all, never the S-row block —
    /// the merged history step after a partial accept is `1 + accepted` rows
    /// wide and a full-block mixer + 248320-wide lm_head over it is thrown
    /// away but for one row.
    fn qwen4Step(t: *Qwen4Trunk, id_arr: mlx.mlx_array, hidden: mlx.mlx_array, rope_offset: c_int, want: mtp_mod.StepWant, mrope_ctx: ?mtp_mod.MropeContext) !mtp_mod.StepOut {
        // `.mixed` is NOT `.none`: a rerank draft still needs the mixer output,
        // which is the vector the lm_head consumes. `hidden_next` is the
        // PRE-mixer stream on this arm and would be a silent shape error there.
        const project: Qwen4Mtp.Qwen4MtpProject = switch (want) {
            .logits => .last_row,
            .mixed => .mixed_last_row,
            .none => .none,
        };
        const out = try modules.call("qwen4_mtp_forward", .{ t, hidden, id_arr, rope_offset + 1, mrope_ctx, project });
        return .{ .logits = out.logits, .hidden_next = out.stream, .rerank_x = out.mixed };
    }

    /// One head forward over L positions. `.logits` returns the LAST row only,
    /// on both arms. `.mixed` asks for the lm_head's INPUT vector instead: on
    /// the sidecar that is `hidden_next` and nothing extra is produced, on
    /// qwen4_exp it is the mixer output, returned in `rerank_x`. `host_ids` are
    /// `id_arr`'s values when it carries committed rows (null on a chained draft
    /// step).
    pub fn forward(
        self: MtpHeadRef,
        target: *Transformer,
        cache: *MtpCacheRef,
        id_arr: mlx.mlx_array,
        hidden: mlx.mlx_array,
        rope_offset: c_int,
        want: mtp_mod.StepWant,
        mrope_ctx: ?mtp_mod.MropeContext,
        host_ids: ?[]const u32,
    ) !mtp_mod.StepOut {
        return switch (self) {
            .plugin => |h| h.forward(target, &cache.plugin, id_arr, hidden, rope_offset, want, mrope_ctx, host_ids != null),
            .qwen => |h| mtp_mod.forwardWithMrope(h, target, &cache.qwen, id_arr, hidden, rope_offset, want == .logits, mrope_ctx),
            .qwen4 => |t| blk: {
                cache.activate();
                break :blk qwen4Step(t, id_arr, hidden, rope_offset, want, mrope_ctx);
            },
        };
    }

    /// One draft row past a padded history append (`HeadPlace`); qwen4 only.
    pub fn forwardPlaced(self: MtpHeadRef, cache: *MtpCacheRef, id_arr: mlx.mlx_array, hidden: mlx.mlx_array, place: *const Qwen4Mtp.HeadPlace, want: mtp_mod.StepWant) !mtp_mod.StepOut {
        const t = switch (self) {
            .qwen, .plugin => return error.MtpPlacement,
            .qwen4 => |t| t,
        };
        cache.activate();
        const project: Qwen4Mtp.Qwen4MtpProject = switch (want) {
            .logits => .last_row,
            .mixed => .mixed_last_row,
            .none => .none,
        };
        const out = try modules.call("qwen4_mtp_forward_placed", .{ t, hidden, id_arr, place, project });
        return .{ .logits = out.logits, .hidden_next = out.stream, .rerank_x = out.mixed };
    }

    /// Deepest draft the head can make: a draft past a per-depth plugin's last head re-runs
    /// that head at the same position and is never accepted.
    pub fn maxDepth(self: MtpHeadRef) u32 {
        return switch (self) {
            .plugin => |h| if (h.heads > 1) h.heads else std.math.maxInt(u32),
            .qwen, .qwen4 => std.math.maxInt(u32),
        };
    }

    /// Append committed history without projecting logits.
    pub fn appendHistory(
        self: MtpHeadRef,
        target: *Transformer,
        cache: *MtpCacheRef,
        token_ids: []const u32,
        hidden: mlx.mlx_array,
        rope_offset: c_int,
        mrope_ctx: ?mtp_mod.MropeContext,
        allocator: std.mem.Allocator,
    ) !void {
        switch (self) {
            .qwen => |h| try mtp_mod.appendHistoryWithMrope(h, target, &cache.qwen, token_ids, hidden, rope_offset, mrope_ctx),
            .plugin => |h| try h.appendHistory(target, &cache.plugin, token_ids, hidden, rope_offset, mrope_ctx, allocator),
            .qwen4 => |t| {
                const ids_i32 = try allocator.alloc(i32, token_ids.len);
                defer allocator.free(ids_i32);
                for (token_ids, 0..) |tok, i| ids_i32[i] = @intCast(tok);
                const shape = [_]c_int{@intCast(token_ids.len)};
                const id_arr = mlx.mlx_array_new_data(ids_i32.ptr, &shape, 1, .int32);
                defer _ = mlx.mlx_array_free(id_arr);
                cache.activate();
                const out = try qwen4Step(t, id_arr, hidden, rope_offset, .none, mrope_ctx);
                _ = mlx.mlx_array_free(out.hidden_next);
            },
        }
    }

    /// Draft-rerank scheme (see mtp.zig): greedy drafts pick via a coarse
    /// 2-bit readout + trunk-head top-32 re-score instead of a full
    /// draft-head projection + argmax.
    pub fn canRerankDrafts(self: MtpHeadRef) bool {
        return switch (self) {
            .qwen => |h| h.canRerankDrafts(),
            // Built lazily on the first ask: `loadQwen4Mtp` runs while the
            // Transformer is still under construction and its lm_head — the
            // very weight the coarse head is a copy of — is not assigned yet.
            .qwen4 => |t| t.qwen4DraftRerankReady(),
            .plugin => |h| h.canRerankDrafts(),
        };
    }

    /// `x` is whatever the TARGET's lm_head consumes for this arm — the
    /// sidecar's post-norm hidden, or qwen4_exp's MIXER output. Never
    /// `hidden_next` on qwen4_exp: that is the pre-mixer `[B,S,hc*H]` stream.
    pub fn draftSelect(
        self: MtpHeadRef,
        target: *Transformer,
        x: mlx.mlx_array,
        suppress_mask: ?mlx.mlx_array,
    ) !mlx.mlx_array {
        return switch (self) {
            .qwen => |h| h.draftSelect(target, x, suppress_mask),
            .qwen4 => |t| modules.call("qwen4_draft_select", .{ t, x, suppress_mask }),
            .plugin => |h| h.draftSelect(target, x, suppress_mask),
        };
    }

    /// The exact re-scored top-32 shortlist a SAMPLED draft proposes from, or
    /// null when the coarse rerank head is unavailable. `x` is the same vector
    /// `draftSelect` takes.
    pub fn draftShortlist(
        self: MtpHeadRef,
        target: *Transformer,
        x: mlx.mlx_array,
        suppress_mask: ?mlx.mlx_array,
    ) !?mtp_mod.Shortlist {
        return switch (self) {
            .qwen => |h| h.draftShortlist(target, x, suppress_mask),
            .qwen4 => |t| modules.call("qwen4_draft_shortlist", .{ t, x, suppress_mask }),
            .plugin => |h| h.draftShortlist(target, x, suppress_mask),
        };
    }

    /// Can a group on this head and `other` draft as the rows of one head forward (`forwardLanes`)?
    pub fn lanesWith(self: MtpHeadRef, other: MtpHeadRef) bool {
        return switch (self) {
            .qwen => |h| other == .qwen and other.qwen == h,
            .plugin => |h| other == .plugin and other.plugin == h and h.lanes(),
            .qwen4 => false,
        };
    }

    pub const Lane = struct { cache: *MtpCacheRef, id: mlx.mlx_array, hidden: mlx.mlx_array, rope_offset: c_int };

    /// One draft step for every lane as the rows of one head forward; the post-norm rows `[1, N, H]`.
    pub fn forwardLanes(self: MtpHeadRef, target: *Transformer, lanes: []const Lane) !mlx.mlx_array {
        if (lanes.len > mtp_mod.MAX_LANES) return error.MtpLaneCount;
        switch (self) {
            .qwen => |h| {
                var ls: [mtp_mod.MAX_LANES]mtp_mod.Lane = undefined;
                for (lanes, 0..) |l, k| ls[k] = .{ .cache = &l.cache.qwen, .id = l.id, .hidden = l.hidden, .rope_offset = l.rope_offset };
                return mtp_mod.forwardLanes(h, target, ls[0..lanes.len]);
            },
            .plugin => |h| {
                var ls: [mtp_mod.MAX_LANES]plugin_head.Head.Lane = undefined;
                for (lanes, 0..) |l, k| ls[k] = .{ .ref = &l.cache.plugin, .id = l.id, .hidden = l.hidden, .rope_offset = l.rope_offset };
                return h.forwardLanes(ls[0..lanes.len]);
            },
            .qwen4 => return error.MtpLaneCount,
        }
    }

    /// Committed rows `ids` `[n]` with `hidden` `[1, n, H]` into the history at `rope_offset`, K/V only.
    pub fn appendRows(self: MtpHeadRef, target: *Transformer, cache: *MtpCacheRef, ids: mlx.mlx_array, hidden: mlx.mlx_array, rope_offset: c_int) !void {
        switch (self) {
            .qwen => |h| try mtp_mod.appendKvOnly(h, target, &cache.qwen, ids, hidden, rope_offset, null),
            .plugin => |h| try h.appendRows(target, &cache.plugin, ids, hidden, rope_offset, null),
            .qwen4 => return error.MtpLaneCount,
        }
    }

    /// `draftSelect` over group rows `x` `[N,1,H]`: `[N,1]` int32 ids.
    pub fn draftSelectBatched(self: MtpHeadRef, target: *Transformer, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !mlx.mlx_array {
        return switch (self) {
            .qwen => |h| h.draftSelectBatched(target, x, suppress_mask),
            .qwen4 => |t| modules.call("qwen4_draft_select_batched", .{ t, x, suppress_mask }),
            .plugin => mtp_mod.fullReadoutArgmax(target.s, target, x, suppress_mask),
        };
    }

    /// `draftShortlist` per group row; false = no coarse head.
    pub fn draftShortlistsBatched(self: MtpHeadRef, target: *Transformer, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array, out: []mtp_mod.Shortlist) !bool {
        return switch (self) {
            .qwen => |h| h.draftShortlistsBatched(target, x, suppress_mask, out),
            .qwen4 => |t| modules.call("qwen4_draft_shortlists_batched", .{ t, x, suppress_mask, out }),
            .plugin => mtp_mod.exactShortlistsBatched(target.s, target, x, suppress_mask, out),
        };
    }

    /// Every G17/NAX cost surface is calibrated against one exact runtime
    /// geometry — the dense Qwen3.6/3.8-27B sidecars on the `.qwen` arm, the
    /// qwen4_exp in-checkpoint head on its own arm — so every other head
    /// plans under `generic`: an off-profile head served by a calibrated
    /// surface would plan depths the measurement never covered.
    pub fn costProfile(self: MtpHeadRef, target: *const Transformer, kv: transformer_mod.KVQuantConfig) mtp_mod.MtpCostProfile {
        return switch (self) {
            .qwen => |h| h.m5NaxCostProfile(target),
            .qwen4 => |t| @import("../../../plugins/qwen4_exp/cost_profile.zig").qwen4G17CostProfileForKv(t, kv),
            .plugin => |h| h.m5NaxCostProfile(target),
        };
    }

    /// The head's last healthy acceptance surface, or null when it has never
    /// published one. BOTH arms store it on the head object itself — the
    /// sidecar `MtpModel` and the in-checkpoint `Qwen4Mtp` — so a seed dies
    /// with the head and can never cross models.
    pub fn evSeed(self: MtpHeadRef) ?struct { accept: [mtp_mod.MAX_DEPTH]f32, m_lo: u32 } {
        return switch (self) {
            .qwen => |h| if (h.ev_seed_accept) |a| .{ .accept = a, .m_lo = h.ev_seed_m_lo } else null,
            .plugin => |h| if (h.ev_seed_accept) |a| .{ .accept = a, .m_lo = h.ev_seed_m_lo } else null,
            .qwen4 => |t| blk: {
                if (t.qwen4_mtp) |*m| {
                    if (m.ev_seed_accept) |a| break :blk .{ .accept = a, .m_lo = m.ev_seed_m_lo };
                }
                break :blk null;
            },
        };
    }

    pub fn setEvSeed(self: MtpHeadRef, accept: [mtp_mod.MAX_DEPTH]f32, m_lo: u32) void {
        switch (self) {
            .qwen => |h| {
                h.ev_seed_accept = accept;
                h.ev_seed_m_lo = m_lo;
            },
            .plugin => |h| {
                h.ev_seed_accept = accept;
                h.ev_seed_m_lo = m_lo;
            },
            .qwen4 => |t| {
                // A model served with `--no-mtp` has no head to seed.
                if (t.qwen4_mtp) |*m| {
                    m.ev_seed_accept = accept;
                    m.ev_seed_m_lo = m_lo;
                }
            },
        }
    }
};

/// The head's committed-history cache.
pub const MtpCacheRef = union(enum) {
    qwen: KVCache,
    qwen4: Qwen4Ref,
    plugin: PluginRef,

    /// A served model's head history (one KV layer and one recurrent entry per head: what a KV layer cannot hold,
    /// such as rows waiting for their tokens) and the head its next chained draft runs.
    pub const PluginRef = struct {
        kv: KVCache,
        ssm: []SSMCacheEntry,
        next_head: u32 = 0,

        pub fn init(allocator: std.mem.Allocator, heads: u32) !PluginRef {
            var cache = try KVCache.init(allocator, heads);
            errdefer cache.deinit();
            const ssm = try allocator.alloc(SSMCacheEntry, heads);
            for (ssm) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
            return .{ .kv = cache, .ssm = ssm };
        }

        pub fn deinit(r: *PluginRef) void {
            for (r.ssm) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                ssmFreeSpecCapture(e);
            }
            r.kv.allocator.free(r.ssm);
            r.kv.deinit();
        }

        pub fn appendEvalArrays(r: *const PluginRef, vec: mlx.mlx_vector_array) void {
            r.kv.appendEvalArrays(vec);
            for (r.ssm) |*e| for ([_]mlx.mlx_array{ e.conv_state, e.ssm_state, e.spec_conv_input, e.spec_state_seq }) |a| {
                if (a.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, a);
            };
        }
    };

    /// The in-checkpoint head plus this request's own half of it.
    pub const Qwen4Ref = struct { t: *Qwen4Trunk, st: *Qwen4Mtp.Qwen4MtpState, allocator: std.mem.Allocator };

    /// Install this request's state on the qwen4 head; no-op on the sidecar arm.
    pub fn activate(self: *const MtpCacheRef) void {
        switch (self.*) {
            .qwen, .plugin => {},
            .qwen4 => |r| r.t.qwen4MtpActivate(r.st),
        }
    }

    pub fn qwen4StateForBatch(self: *const MtpCacheRef) *Qwen4Mtp.Qwen4MtpState {
        const ref = self.qwen4;
        ref.t.qwen4MtpRelease(ref.st);
        return ref.st;
    }

    pub fn step(self: *const MtpCacheRef) usize {
        return switch (self.*) {
            .qwen => |*c| c.step,
            .plugin => |*r| r.kv.step,
            .qwen4 => |r| blk: {
                r.t.qwen4MtpActivate(r.st);
                break :blk r.t.qwen4_mtp.?.seq_offset;
            },
        };
    }

    /// The underlying KVCache — what the prefix cache's spec-snap machinery
    /// snapshots and restores. The qwen4 head's state is not KV-only, so a caller taking this
    /// pointer must also carry `head()`. Null when head persistence is off.
    pub fn kv(self: *MtpCacheRef) ?*KVCache {
        return switch (self.*) {
            .qwen => |*c| c,
            // Per-depth heads keep rows waiting for their tokens in the recurrent entries: never persisted, the
            // history rebuilt from the uncached tail.
            .plugin => |*r| if (r.ssm.len > 1) null else &r.kv,
            .qwen4 => |r| blk: {
                r.t.qwen4MtpActivate(r.st);
                break :blk &r.t.qwen4_mtp.?.cache;
            },
            // The heads only ever read the last window of rows, rebuilt from the uncached tail.
        };
    }

    /// The trunk owning the in-checkpoint head; null on the sidecar arm and whenever `kv()` is null.
    pub fn head(self: *MtpCacheRef) ?*Qwen4Trunk {
        return switch (self.*) {
            .qwen, .plugin => null,
            .qwen4 => |r| blk: {
                r.t.qwen4MtpActivate(r.st);
                break :blk r.t;
            },
        };
    }

    pub fn truncate(self: *MtpCacheRef, len: usize, s: mlx.mlx_stream) !void {
        switch (self.*) {
            .qwen => |*c| try c.truncate(len, s),
            .plugin => |*r| {
                try r.kv.truncate(len, s);
                r.next_head = 0;
            },
            .qwen4 => |r| {
                r.t.qwen4MtpActivate(r.st);
                try r.t.qwen4MtpTruncate(len);
            },
        }
    }

    pub fn deinit(self: *MtpCacheRef) void {
        switch (self.*) {
            .qwen => |*c| c.deinit(),
            .plugin => |*r| r.deinit(),
            .qwen4 => |r| {
                r.t.qwen4MtpRelease(r.st);
                r.st.deinit();
                r.allocator.destroy(r.st);
            },
        }
    }

    /// Append this cache's storage to a prefill eval batch, so the chunk's
    /// activation graph can be freed with the chunk instead of accumulating
    /// across the whole prompt.
    pub fn appendEvalArrays(self: *const MtpCacheRef, vec: mlx.mlx_vector_array) void {
        switch (self.*) {
            .qwen => |*c| c.appendEvalArrays(vec),
            .plugin => |*r| r.appendEvalArrays(vec),
            .qwen4 => |r| {
                r.t.qwen4MtpActivate(r.st);
                r.t.qwen4_mtp.?.cache.appendEvalArrays(vec);
            },
        }
    }
};

/// An MTP committed-history cache restored from the prefix cache, plus the
/// absolute target position its index 0 represents.
pub const MtpRestored = struct { cache: MtpCacheRef, base: usize };


test "MTP plugin head cache: each head's recurrent entry keeps what its handle sets, freed with the cache" {
    const abi = @import("../../../abi/plugin.zig");
    const cache_ops = @import("../kv/cache_ops.zig");
    var ref = try MtpCacheRef.PluginRef.init(testing.allocator, 3);
    defer ref.deinit();
    var slot: cache_ops.Slot = .{ .kv = &ref.kv, .ssm = ref.ssm };
    const v = mlx.mlx_array_new_int(7);
    defer _ = mlx.mlx_array_free(v);
    try testing.expect(cache_ops.ops.state(slot.handle(), 2, .conv).ctx == null);
    try testing.expect(cache_ops.ops.set_state(slot.handle(), 2, .conv, .{ .ctx = v.ctx }).isOk());
    try testing.expect(cache_ops.ops.set_state(slot.handle(), 2, .state_seq, .{ .ctx = v.ctx }).isOk());
    try testing.expectEqual(ref.ssm[2].conv_state.ctx, cache_ops.ops.state(slot.handle(), 2, .conv).ctx);
    try testing.expect(cache_ops.ops.state(slot.handle(), 0, .conv).ctx == null);
    try testing.expectEqual(abi.Verdict.Code.unsupported, cache_ops.ops.set_state(slot.handle(), 3, .conv, .{ .ctx = v.ctx }).code);
}
