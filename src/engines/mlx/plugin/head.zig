//! A served model's MTP head (`SpecHooks.head`) as the generator drives it (`MtpHeadRef.plugin`): the plugin runs
//! the head's layer over its own KV cache and hands back the last row's normed output; the engine keeps the readout
//! the sidecar head keeps (the draft-only low-bit head, the coarse rerank head, the full trunk-head readout), built
//! from the trunk head the shell borrowed through `trunk_weight`, so drafts select exactly as on the legacy path.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const log = @import("../../../sdk/util/log.zig");
const mtp_mod = @import("../../../sdk/spec/mtp/mtp.zig");
const cache_ops = @import("../kv/cache_ops.zig");
const FsPackDir = @import("../../../host/registry/pack_dir.zig").FsPackDir;
const Transformer = @import("../transformer.zig").Transformer;
const KVCache = @import("../kv/cache.zig").KVCache;
const PluginRef = @import("../spec/mtp_head_ref.zig").MtpCacheRef.PluginRef;
const Bound = @import("driver.zig").Bound;

const MtpModel = mtp_mod.MtpModel;
const Array = mlx.mlx_array;

/// A head step's 1-D position, or its M-RoPE angles `[2, 1, 1, rows, rope_dims]` (owned).
const Place = struct {
    pos: u32,
    angles: Array = .{ .ctx = null },

    fn deinit(p: Place) void {
        if (p.angles.ctx != null) _ = mlx.mlx_array_free(p.angles);
    }
};

pub const Head = struct {
    s: mlx.mlx_stream,
    bound: *Bound,
    hooks: *const abi.HeadHooks,
    /// The cost surface the plugin declares (`SpecFacts.cost_profile`); generic when it names none.
    profile: mtp_mod.MtpCostProfile = .generic,
    /// The head's half of the measured fingerprint (`HeadHooks.cost_fingerprint`), read when no profile is declared.
    fingerprint: ?mtp_mod.HeadFingerprint = null,
    /// Heads, one per draft depth (`SpecFacts.mtp_heads`); 1 = one head chained on its own output.
    heads: u32 = 1,
    // The readout state `MtpModel` keeps, read by its readout functions.
    draft_head: ?mtp_mod.QLinear = null,
    draft_head_bits: u32 = 0,
    draft_head_group: u32 = 0,
    rerank_coarse: ?mtp_mod.RerankCoarse = null,
    rerank_logged: bool = false,
    ev_seed_accept: ?[mtp_mod.MAX_DEPTH]f32 = null,
    ev_seed_m_lo: u32 = 1,

    pub fn deinit(self: *Head) void {
        if (self.rerank_coarse) |*rc| rc.deinit();
        if (self.draft_head) |*dh| dh.deinit();
    }

    /// The declared profile, else the one the legacy sidecar arm selects for this head on `target`.
    pub fn m5NaxCostProfile(self: *const Head, target: *const Transformer) mtp_mod.MtpCostProfile {
        if (self.profile != .generic) return self.profile;
        const draft: ?*const mtp_mod.QLinear = if (self.draft_head) |*q| q else null;
        return mtp_mod.m5NaxProfileFor(self.fingerprint, target, draft, self.draft_head_bits, self.draft_head_group);
    }

    /// Draft steps for several requests as rows of one head forward (`HeadHooks.forward_lanes`, one head only).
    pub fn lanes(self: *const Head) bool {
        return self.heads <= 1 and self.hooks.forward_lanes != null;
    }

    pub fn canRerankDrafts(self: *const Head) bool {
        return self.rerank_coarse != null;
    }

    /// One head forward: `logits = .last_row` lets leading rows append K/V only (the engine projects the last
    /// row), `.none` with `want_hidden` runs every row, `.none` without it appends history only.
    fn run(self: *Head, ref: *PluginRef, head: u32, id_arr: Array, hidden: Array, place: Place, logits: abi.Logits, want_hidden: bool) !Array {
        const rows: c_int = mlx.getShape(id_arr)[0];
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        try mlx.check(mlx.mlx_reshape(&ids, id_arr, &[_]c_int{ 1, rows }, 2, self.s));
        var slot: cache_ops.Slot = .{ .kv = &ref.kv, .ssm = ref.ssm };
        const handles = [_]abi.Handle{slot.handle()};
        const pos = [_]u32{place.pos};
        const f: abi.Forward = .{
            .mrope = .{ .ctx = place.angles.ctx },
            .mode = if (rows == 1) .decode else .prefill,
            .logits = logits,
            .want_hidden = want_hidden,
            .rows = @intCast(rows),
            .tokens = .{},
            .ids = .{ .ctx = ids.ctx },
            .positions = .of(&pos),
            .caches = .of(&handles),
            .cache_ops = &cache_ops.ops,
            .hidden_in = .{ .ctx = hidden.ctx },
            .head = head,
        };
        var out: abi.ForwardOut = .{};
        const v = self.hooks.forward(self.bound.model, &f, &out);
        if (out.logits.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.logits.ctx });
        if (!v.isOk()) {
            if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
            log.err("[plugin] {s}: MTP head forward failed: {s}\n", .{ self.bound.table.name.slice(), v.why.slice() });
            return error.PluginForwardFailed;
        }
        return .{ .ctx = out.hidden.ctx };
    }

    /// Where a step of `rows` at `rope_offset` ropes, decided per call as the legacy sidecar arm decides it: inside
    /// the prompt's M-RoPE table, the angle rows (owned); past it, the 1-D position carrying the delta.
    fn placeAt(self: *Head, target: *Transformer, rope_offset: c_int, rows: c_int, mrope_ctx: ?mtp_mod.MropeContext) !Place {
        const positions = mrope_ctx orelse return .{ .pos = @intCast(rope_offset) };
        const rel: usize = @intCast(rope_offset);
        const abs = positions.absolutePosition(rel);
        if (abs >= positions.total) return .{ .pos = @intCast(@as(i64, @intCast(abs)) + positions.delta) };
        const cs = try target.buildMropeCosSin(positions, rel, @intCast(rows));
        defer _ = mlx.mlx_array_free(cs.cos);
        defer _ = mlx.mlx_array_free(cs.sin);
        const halves = [_]Array{ cs.cos, cs.sin };
        const vec = mlx.mlx_vector_array_new_data(&halves, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var angles = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(angles);
        try mlx.check(mlx.mlx_stack_axis(&angles, vec, 0, self.s));
        return .{ .pos = @intCast(rope_offset), .angles = angles };
    }

    /// Committed rows (`opens`) open a round at head 0; each chained step runs the next head, the last one again
    /// past the end. With several heads every step reads the trunk's hidden, handed on unchanged, and the head's
    /// output is the vector the trunk head reads (`rerank_x` under `.mixed`).
    pub fn forward(self: *Head, target: *Transformer, ref: *PluginRef, id_arr: Array, hidden: Array, rope_offset: c_int, want: mtp_mod.StepWant, mrope_ctx: ?mtp_mod.MropeContext, opens: bool) !mtp_mod.StepOut {
        if (mrope_ctx != null and self.heads > 1) return error.PluginForwardUnsupported;
        const rows: c_int = mlx.getShape(id_arr)[0];
        if (mrope_ctx != null and want == .logits and rows > 1) {
            // The legacy arm's merged step: leading rows K/V only, then the last row, each roped on its own side.
            var g = ops.Graph.init(target.allocator, self.s);
            defer g.deinit();
            const hs = mlx.getShape(hidden);
            const ids_head = try g.slice(id_arr, &.{0}, &.{rows - 1});
            const hid_head = try g.slice(hidden, &.{ 0, 0, 0 }, &.{ 1, rows - 1, hs[2] });
            try self.appendRows(target, ref, ids_head, hid_head, rope_offset, mrope_ctx);
            const ids_last = try g.slice(id_arr, &.{rows - 1}, &.{rows});
            const hid_last = try g.slice(hidden, &.{ 0, rows - 1, 0 }, &.{ 1, rows, hs[2] });
            return self.forward(target, ref, ids_last, hid_last, rope_offset + rows - 1, want, mrope_ctx, opens);
        }
        const head = if (opens) 0 else @min(ref.next_head, self.heads - 1);
        // Several heads: a step that wants nothing appends history (no `want_hidden`), as MiMo's legacy arm does.
        const want_hidden = self.heads <= 1 or want != .none;
        if (want_hidden) ref.next_head = head + 1;
        const place = try self.placeAt(target, rope_offset, rows, mrope_ctx);
        defer place.deinit();
        const out = try self.run(ref, head, id_arr, hidden, place, if (want == .logits) .last_row else .none, want_hidden);
        if (out.ctx == null and want_hidden) return error.PluginForwardFailed;
        if (self.heads <= 1) {
            if (want != .logits) return .{ .logits = .{ .ctx = null }, .hidden_next = out };
            errdefer _ = mlx.mlx_array_free(out);
            return .{ .logits = try mtp_mod.targetLmHead(self, target, out, self.s), .hidden_next = out };
        }
        var trunk = mlx.mlx_array_new();
        mlx.check(mlx.mlx_array_set(&trunk, hidden)) catch |e| {
            _ = mlx.mlx_array_free(out);
            _ = mlx.mlx_array_free(trunk);
            return e;
        };
        switch (want) {
            .none => {
                if (out.ctx != null) _ = mlx.mlx_array_free(out);
                return .{ .logits = .{ .ctx = null }, .hidden_next = trunk };
            },
            .mixed => return .{ .logits = .{ .ctx = null }, .hidden_next = trunk, .rerank_x = out },
            .logits => {
                defer _ = mlx.mlx_array_free(out);
                errdefer _ = mlx.mlx_array_free(trunk);
                return .{ .logits = try mtp_mod.targetLmHead(self, target, out, self.s), .hidden_next = trunk };
            },
        }
    }

    pub fn appendHistory(self: *Head, target: *Transformer, ref: *PluginRef, token_ids: []const u32, hidden: Array, rope_offset: c_int, mrope_ctx: ?mtp_mod.MropeContext, allocator: std.mem.Allocator) !void {
        if (token_ids.len == 0) return;
        if (mrope_ctx != null and self.heads > 1) return error.PluginForwardUnsupported;
        const ids = try allocator.alloc(i32, token_ids.len);
        defer allocator.free(ids);
        for (token_ids, ids) |t, *o| o.* = @intCast(t);
        const id_arr = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(ids.len)}, 1, .int32);
        defer _ = mlx.mlx_array_free(id_arr);
        try self.appendRows(target, ref, id_arr, hidden, rope_offset, mrope_ctx);
    }

    /// Committed rows `id_arr` `[n]` with `hidden` `[1, n, H]` into the history at `rope_offset`, K/V only.
    pub fn appendRows(self: *Head, target: *Transformer, ref: *PluginRef, id_arr: Array, hidden: Array, rope_offset: c_int, mrope_ctx: ?mtp_mod.MropeContext) !void {
        ref.next_head = 0;
        const place = try self.placeAt(target, rope_offset, mlx.getShape(id_arr)[0], mrope_ctx);
        defer place.deinit();
        const out = try self.run(ref, 0, id_arr, hidden, place, .none, false);
        if (out.ctx != null) _ = mlx.mlx_array_free(out);
    }

    pub const Lane = struct { ref: *PluginRef, id: Array, hidden: Array, rope_offset: c_int };

    /// One draft step for every lane as the rows of one head forward; the normed rows `[1, n, H]`.
    pub fn forwardLanes(self: *Head, lanes_in: []const Lane) !Array {
        const n = lanes_in.len;
        if (n == 0 or n > mtp_mod.MAX_LANES) return error.MtpLaneCount;
        const forward_lanes = self.hooks.forward_lanes orelse return error.PluginForwardUnsupported;
        var flat: [mtp_mod.MAX_LANES]Array = undefined;
        var flat_n: usize = 0;
        defer for (flat[0..flat_n]) |x| {
            _ = mlx.mlx_array_free(x);
        };
        var parts: [mtp_mod.MAX_LANES]Array = undefined;
        var slots: [mtp_mod.MAX_LANES]cache_ops.Slot = undefined;
        var handles: [mtp_mod.MAX_LANES]abi.Handle = undefined;
        var pos: [mtp_mod.MAX_LANES]u32 = undefined;
        for (lanes_in, 0..) |l, k| {
            flat[k] = mlx.mlx_array_new();
            flat_n = k + 1;
            try mlx.check(mlx.mlx_reshape(&flat[k], l.id, &[_]c_int{1}, 1, self.s));
            parts[k] = l.hidden;
            slots[k] = .{ .kv = &l.ref.kv, .ssm = l.ref.ssm };
            handles[k] = slots[k].handle();
            pos[k] = @intCast(l.rope_offset);
        }
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        const idv = mlx.mlx_vector_array_new_data(&flat, n);
        defer _ = mlx.mlx_vector_array_free(idv);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, idv, 0, self.s));
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        try mlx.check(mlx.mlx_reshape(&ids, cat, &[_]c_int{ @intCast(n), 1 }, 2, self.s));
        var hidden = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(hidden);
        const hv = mlx.mlx_vector_array_new_data(&parts, n);
        defer _ = mlx.mlx_vector_array_free(hv);
        try mlx.check(mlx.mlx_concatenate_axis(&hidden, hv, 1, self.s));
        const f: abi.Forward = .{
            .mode = .batched_rows,
            .logits = .none,
            .want_hidden = true,
            .rows = 1,
            .tokens = .{},
            .ids = .{ .ctx = ids.ctx },
            .positions = .of(pos[0..n]),
            .caches = .of(handles[0..n]),
            .cache_ops = &cache_ops.ops,
            .hidden_in = .{ .ctx = hidden.ctx },
        };
        var out: abi.ForwardOut = .{};
        const v = forward_lanes(self.bound.model, &f, &out);
        if (out.logits.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.logits.ctx });
        if (!v.isOk() or out.hidden.ctx == null) {
            if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
            log.err("[plugin] {s}: MTP head lanes failed: {s}\n", .{ self.bound.table.name.slice(), v.why.slice() });
            return error.PluginForwardFailed;
        }
        return .{ .ctx = out.hidden.ctx };
    }

    pub fn draftSelect(self: *Head, target: *Transformer, x: Array, suppress_mask: ?Array) !Array {
        return MtpModel.draftSelect(self, target, x, suppress_mask);
    }

    pub fn draftShortlist(self: *Head, target: *Transformer, x: Array, suppress_mask: ?Array) !?mtp_mod.Shortlist {
        return MtpModel.draftShortlist(self, target, x, suppress_mask);
    }
};

/// The served model's head, loaded (`HeadHooks.load`) and its readout built from the shell's trunk head; null when
/// the table has none or the shell carries no trunk head (`Bound.trunk`). A failed load only turns MTP off.
pub fn load(io: std.Io, a: std.mem.Allocator, xfm: *Transformer, model_dir: []const u8) ?*Head {
    const b = xfm.plugin orelse return null;
    const hooks = (b.table.spec orelse return null).head orelse return null;
    if (!b.trunk) {
        log.warn("[plugin] {s}: the MTP head needs the trunk head (trunk_weight) — MTP off\n", .{b.table.name.slice()});
        return null;
    }
    var fs = FsPackDir.open(io, model_dir) catch return null;
    defer fs.close();
    const dir = fs.packDir();
    const parsed = &xfm.config.plugin.?.parsed;
    const v = hooks.load(b.model, &.{ .model_dir = abi.Str.of(model_dir), .dir = &dir, .parsed = parsed, .stream = .{ .ctx = xfm.s.ctx } });
    if (!v.isOk()) {
        // A pack that ships no head is not a failed load.
        if (std.mem.eql(u8, v.why.slice(), "MissingMtpWeight")) {
            log.debug("[plugin] {s}: no MTP head in this pack\n", .{b.table.name.slice()});
        } else log.warn("[plugin] {s}: MTP head failed to load ({s}) — MTP off\n", .{ b.table.name.slice(), v.why.slice() });
        return null;
    }
    const h = a.create(Head) catch return null;
    h.* = .{ .s = xfm.s, .bound = b, .hooks = hooks, .profile = std.enums.fromInt(mtp_mod.MtpCostProfile, parsed.spec.cost_profile) orelse .generic, .heads = @max(1, parsed.spec.mtp_heads) };
    if (hooks.cost_fingerprint) |f| {
        const key = f(b.model);
        if (key != 0) h.fingerprint = .{ .bits = key >> 16, .group_size = key & 0xffff };
    }
    const had_err = mlx.errorPending();
    if (h.heads > 1) {
        // Per-depth heads read out as MiMo's legacy arm does: the trunk head, shortlisted on a coarse copy.
        if (xfm.lm_head_s.ctx != null) h.rerank_coarse = mtp_mod.buildRerankCoarse(h.s, xfm, mtp_mod.DRAFT_HEAD_BITS);
    } else {
        MtpModel.buildDraftHead(h, xfm) catch |err| {
            log.warn("[mtp] draft lm_head build failed ({s}) — drafts use the trunk head\n", .{@errorName(err)});
        };
        MtpModel.maybeBuildDraftRerank(h, xfm);
    }
    mlx.dropLatchedErrorUnless(had_err);
    return h;
}

pub fn destroy(h: *Head, a: std.mem.Allocator) void {
    h.deinit();
    a.destroy(h);
}
