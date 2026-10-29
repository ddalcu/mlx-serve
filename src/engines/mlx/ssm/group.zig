//! engines/mlx/ssm/group.
//! Owns SsmMergeKind, ssm_member_gen_src, nextSsmMemberGen, SsmGroupMember, SsmPatchFields, PersistentSsmGroup, and 42 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const ssmFreeQsaState = @import("../../../plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const MAX_BATCH_ROWS = @import("../batched/layered.zig").MAX_BATCH_ROWS;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const KVCache = @import("../kv/cache.zig").KVCache;
const Transformer = @import("../transformer.zig").Transformer;
const captureSsmCheckpoint = @import("checkpoint.zig").captureSsmCheckpoint;
const restoreSsmCheckpoint = @import("checkpoint.zig").restoreSsmCheckpoint;
const SSMCacheEntry = @import("entry.zig").SSMCacheEntry;
const splitSpecCaptures = @import("entry.zig").splitSpecCaptures;
const ssmFreeSpecCapture = @import("entry.zig").ssmFreeSpecCapture;
const ssmRollbackFromCapture = @import("entry.zig").ssmRollbackFromCapture;
const ssmSnapshot = @import("entry.zig").ssmSnapshot;
const ssmSnapshotDeinit = @import("entry.zig").ssmSnapshotDeinit;

pub const SsmMergeKind = packed struct {
    linear: bool = false,
    ple: bool = false,
};

/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const SsmGroupState = struct {
    ssm_member_gen_src: std.atomic.Value(u64) = .init(1),
};
pub const ssmGroup_shared = @import("../../../sdk/util/shared.zig").Shared(SsmGroupState, "ssm_group", .{});
inline fn ssmGroup() *SsmGroupState {
    return ssmGroup_shared.get();
}

pub fn nextSsmMemberGen() u64 {
    return ssmGroup().ssm_member_gen_src.fetchAdd(1, .monotonic);
}

const SsmGroupMember = struct {
    key: usize,
    gen: u64,
    entries: []SSMCacheEntry,
};

const SsmPatchFields = packed struct {
    conv: bool = false,
    ssm: bool = false,
    aux: bool = false,
};

/// The batched tick's one owner of the merged GDN/conv/PLE state across ticks.
/// A stable group reads the current generation and writes the next; slots hold
/// (group, row, member_gen), never a slice view. Rows leave through
/// `ssmExportRow` (materialized + evaluated) before any generation is freed, and
/// `entries[].member_gen` is the ONE guard: any write outside the group bumps it
/// and `matches()` then refuses the slot. Retirement is MLX refcount: every
/// consumer holds its own handle.
pub const PersistentSsmGroup = struct {
    allocator: std.mem.Allocator = undefined,
    n: usize = 0,
    keys: []usize = &.{},
    gens: []u64 = &.{},
    member_entries: [][]SSMCacheEntry = &.{},
    layers: []SSMCacheEntry = &.{},
    kinds: []SsmMergeKind = &.{},
    live: bool = false,

    pub fn matches(self: *const PersistentSsmGroup, ctxs: []const *ForwardCtx) bool {
        if (!self.live or self.n != ctxs.len) return false;
        for (ctxs, 0..) |c, i| {
            const entries = c.ssm_entries orelse return false;
            if (self.keys[i] != @intFromPtr(entries.ptr)) return false;
            if (entries.len == 0 or self.gens[i] != entries[0].member_gen) return false;
        }
        return true;
    }

    fn clear(self: *PersistentSsmGroup) void {
        if (!self.live) return;
        ssmUnbindAll(self);
        for (self.layers) |*m| ssmFreeMergedLayer(m);
        self.allocator.free(self.layers);
        self.allocator.free(self.keys);
        self.allocator.free(self.gens);
        self.allocator.free(self.member_entries);
        self.allocator.free(self.kinds);
        self.layers = &.{};
        self.keys = &.{};
        self.gens = &.{};
        self.member_entries = &.{};
        self.kinds = &.{};
        self.n = 0;
        self.live = false;
    }

    pub fn deinit(self: *PersistentSsmGroup) void {
        self.clear();
    }
};

pub const ssmConcatAxis0 = @import("../../../mlx/array.zig").concatAxis0;

fn ssmRowView(s: mlx.mlx_stream, arr: mlx.mlx_array, i: usize) !mlx.mlx_array {
    const shape = mlx.getShape(arr);
    if (shape.len > 8) return error.SsmRankTooHigh;
    var start: [8]c_int = @splat(0);
    var stop: [8]c_int = @splat(0);
    const strides: [8]c_int = @splat(1);
    start[0] = @intCast(i);
    stop[0] = @intCast(i + 1);
    for (1..shape.len) |d| stop[d] = shape[d];
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, arr, &start, @intCast(shape.len), &stop, @intCast(shape.len), &strides, @intCast(shape.len), s));
    return out;
}

pub fn ssmFreeMergedLayer(m: *SSMCacheEntry) void {
    if (m.conv_state.ctx != null) _ = mlx.mlx_array_free(m.conv_state);
    if (m.ssm_state.ctx != null) _ = mlx.mlx_array_free(m.ssm_state);
    if (m.aux_state.ctx != null) _ = mlx.mlx_array_free(m.aux_state);
    ssmFreeSpecCapture(m);
    m.conv_state = .{ .ctx = null };
    m.ssm_state = .{ .ctx = null };
    m.aux_state = .{ .ctx = null };
    m.initialized = false;
}

pub fn ssmMergeLayer(s: mlx.mlx_stream, ctxs: []const *ForwardCtx, layer: usize, kind: SsmMergeKind) !SSMCacheEntry {
    // The qwen4 PLE layer's dilated-conv window is merged like the SSM state.
    var out: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = kind.linear };
    errdefer ssmFreeMergedLayer(&out);
    var parts: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    const n = ctxs.len;
    if (kind.linear) {
        for (ctxs, 0..) |c, i| parts[i] = c.ssm_entries.?[layer].conv_state;
        out.conv_state = try ssmConcatAxis0(s, parts[0..n]);
        for (ctxs, 0..) |c, i| parts[i] = c.ssm_entries.?[layer].ssm_state;
        out.ssm_state = try ssmConcatAxis0(s, parts[0..n]);
    }
    if (kind.ple) {
        for (ctxs, 0..) |c, i| parts[i] = c.ssm_entries.?[layer].aux_state;
        out.aux_state = try ssmConcatAxis0(s, parts[0..n]);
    }
    return out;
}

pub fn ssmSplitLayer(s: mlx.mlx_stream, m: *const SSMCacheEntry, ctxs: []const *ForwardCtx, layer: usize, kind: SsmMergeKind) !void {
    for (ctxs, 0..) |c, i| {
        var e = &c.ssm_entries.?[layer];
        if (kind.linear) {
            const conv_own = try ssmRowView(s, m.conv_state, i);
            if (e.conv_state.ctx != null) _ = mlx.mlx_array_free(e.conv_state);
            e.conv_state = conv_own;
            const ssm_own = try ssmRowView(s, m.ssm_state, i);
            if (e.ssm_state.ctx != null) _ = mlx.mlx_array_free(e.ssm_state);
            e.ssm_state = ssm_own;
            e.initialized = true;
        }
        if (kind.ple) {
            const aux_own = try ssmRowView(s, m.aux_state, i);
            ssmFreeQsaState(e);
            e.aux_state = aux_own;
        }
        try splitSpecCaptures(s, m, e, i);
    }
}

fn ssmMergeAll(alloc: std.mem.Allocator, s: mlx.mlx_stream, ctxs: []const *ForwardCtx, kinds: []const SsmMergeKind) ![]SSMCacheEntry {
    const merged = try alloc.alloc(SSMCacheEntry, kinds.len);
    var built: usize = 0;
    errdefer {
        for (merged[0..built]) |*m| ssmFreeMergedLayer(m);
        alloc.free(merged);
    }
    for (merged, 0..) |*m, l| {
        m.* = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false };
        if (!kinds[l].linear and !kinds[l].ple) {
            built = l + 1;
            continue;
        }
        m.* = try ssmMergeLayer(s, ctxs, l, kinds[l]);
        built = l + 1;
    }
    return merged;
}

pub fn ssmSplitAll(s: mlx.mlx_stream, merged: []const SSMCacheEntry, ctxs: []const *ForwardCtx, kinds: []const SsmMergeKind) !void {
    for (merged, 0..) |*m, l| {
        if (!kinds[l].linear and !kinds[l].ple) continue;
        try ssmSplitLayer(s, m, ctxs, l, kinds[l]);
    }
}

pub fn ssmBindSlotRefs(group: *PersistentSsmGroup, ctxs: []const *ForwardCtx) void {
    for (ctxs, 0..) |c, row| {
        const entries = c.ssm_entries orelse continue;
        for (entries, 0..) |*e, l| {
            e.group = group;
            e.group_row = @intCast(row);
            e.group_layer = @intCast(l);
        }
    }
}

fn ssmUnbindEntries(group: *PersistentSsmGroup, entries: []SSMCacheEntry) void {
    for (entries) |*e| {
        if (e.group == group) e.group = null;
    }
}

fn ssmUnbindAll(group: *PersistentSsmGroup) void {
    if (!group.live) return;
    for (group.member_entries) |entries| ssmUnbindEntries(group, entries);
}

fn ssmGroupAdopt(
    group: *PersistentSsmGroup,
    alloc: std.mem.Allocator,
    ctxs: []const *ForwardCtx,
    kinds: []const SsmMergeKind,
    merged: []SSMCacheEntry,
) !void {
    group.clear();
    group.allocator = alloc;
    const n = ctxs.len;
    const keys = try alloc.alloc(usize, n);
    errdefer alloc.free(keys);
    const gens = try alloc.alloc(u64, n);
    errdefer alloc.free(gens);
    const member_entries = try alloc.alloc([]SSMCacheEntry, n);
    errdefer alloc.free(member_entries);
    const kinds_copy = try alloc.dupe(SsmMergeKind, kinds);
    errdefer alloc.free(kinds_copy);
    for (ctxs, 0..) |c, i| {
        const entries = c.ssm_entries.?;
        keys[i] = @intFromPtr(entries.ptr);
        gens[i] = if (entries.len > 0) entries[0].member_gen else c.ssm_member_gen;
        member_entries[i] = entries;
        c.ssm_member_gen = gens[i];
        for (entries) |*e| e.member_gen = gens[i];
    }
    group.keys = keys;
    group.gens = gens;
    group.member_entries = member_entries;
    group.kinds = kinds_copy;
    group.layers = merged;
    group.n = n;
    group.live = true;
    for (ctxs) |c| {
        if (c.ssm_entries) |entries| {
            for (entries) |*e| e.member_gen = c.ssm_member_gen;
        }
    }
    ssmBindSlotRefs(group, ctxs);
}

pub fn ssmBumpMemberGenEntries(entries: []SSMCacheEntry) void {
    const g = nextSsmMemberGen();
    for (entries) |*e| e.member_gen = g;
}

/// A cold write to a slot's entries (prefix-cache reset) must also leave the group,
/// or a repack would resurrect the pre-reset row from the merged buffer.
pub fn ssmDetachFromGroup(entries: []SSMCacheEntry) void {
    for (entries) |*e| {
        e.group = null;
        e.group_row = 0;
        e.group_layer = 0;
    }
    ssmBumpMemberGenEntries(entries);
}

pub fn ssmEvalHandles(arrs: []const mlx.mlx_array) void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    var count: usize = 0;
    for (arrs) |a| {
        if (a.ctx == null) continue;
        _ = mlx.mlx_vector_array_append_value(vec, a);
        count += 1;
    }
    if (count > 0) _ = mlx.mlx_eval(vec);
}

/// Hands a row back to its slot as an owned, evaluated array and bumps the
/// slot's `member_gen`, so a later failed repack can never re-adopt it.
fn ssmExportRow(s: mlx.mlx_stream, group: *PersistentSsmGroup, row: usize) !void {
    const entries = group.member_entries[row];
    if (entries.len > 0 and entries[0].group == null) return;
    var exported: [3 * 256]mlx.mlx_array = undefined;
    if (group.layers.len * 3 > exported.len) return error.SsmGroupLayerCount;
    var n_exp: usize = 0;
    for (group.layers, 0..) |*m, l| {
        var e = &entries[l];
        const kind = group.kinds[l];
        if (kind.linear and m.conv_state.ctx != null) {
            const view = try ssmRowView(s, m.conv_state, row);
            defer _ = mlx.mlx_array_free(view);
            const owned = try materializedOwnedCopy(s, view);
            if (e.conv_state.ctx != null) _ = mlx.mlx_array_free(e.conv_state);
            e.conv_state = owned;
            exported[n_exp] = owned;
            n_exp += 1;
            const view_s = try ssmRowView(s, m.ssm_state, row);
            defer _ = mlx.mlx_array_free(view_s);
            const owned_s = try materializedOwnedCopy(s, view_s);
            if (e.ssm_state.ctx != null) _ = mlx.mlx_array_free(e.ssm_state);
            e.ssm_state = owned_s;
            e.initialized = true;
            exported[n_exp] = owned_s;
            n_exp += 1;
        }
        if (kind.ple and m.aux_state.ctx != null) {
            const view = try ssmRowView(s, m.aux_state, row);
            defer _ = mlx.mlx_array_free(view);
            const owned = try materializedOwnedCopy(s, view);
            if (e.aux_state.ctx != null) _ = mlx.mlx_array_free(e.aux_state);
            e.aux_state = owned;
            exported[n_exp] = owned;
            n_exp += 1;
        }
        e.group = null;
    }
    ssmBumpMemberGenEntries(entries);
    ssmEvalHandles(exported[0..n_exp]);
}

fn ssmFindOldRow(group: *const PersistentSsmGroup, key: usize, gen: u64) ?usize {
    for (group.keys, 0..) |k, i| {
        if (k == key and group.gens[i] == gen) return i;
    }
    return null;
}

fn ssmMembersFromCtxs(ctxs: []const *ForwardCtx, buf: []SsmGroupMember) ![]SsmGroupMember {
    if (ctxs.len > buf.len) return error.SsmGroupTooWide;
    for (ctxs, 0..) |c, i| {
        const entries = c.ssm_entries.?;
        const gen = if (entries.len > 0) entries[0].member_gen else c.ssm_member_gen;
        buf[i] = .{ .key = @intFromPtr(entries.ptr), .gen = gen, .entries = entries };
    }
    return buf[0..ctxs.len];
}

fn ssmSliceUpdateRow(s: mlx.mlx_stream, parent: *mlx.mlx_array, row: usize, update: mlx.mlx_array) !void {
    const sh = mlx.getShape(parent.*);
    if (sh.len > 8) return error.SsmRankTooHigh;
    var start: [8]c_int = @splat(0);
    var stop: [8]c_int = @splat(0);
    const strides: [8]c_int = @splat(1);
    start[0] = @intCast(row);
    stop[0] = @intCast(row + 1);
    for (1..sh.len) |d| stop[d] = sh[d];
    var updated = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(updated);
    try mlx.check(mlx.mlx_slice_update(&updated, parent.*, update, &start, sh.len, &stop, sh.len, &strides, sh.len, s));
    _ = mlx.mlx_array_free(parent.*);
    parent.* = updated;
}

/// Builds the next generation for a new membership: survivors keep their rows,
/// joiners are imported, leavers are exported. The old `kinds` is duplicated and
/// the replacement descriptor is allocated before anything old is freed; on a
/// failure every surviving row is exported so no stream reverts to a stale entry.
fn ssmRepackMembers(
    alloc: std.mem.Allocator,
    s: mlx.mlx_stream,
    group: *PersistentSsmGroup,
    members: []const SsmGroupMember,
    kinds: []const SsmMergeKind,
) !void {
    var i: usize = 0;
    while (i < group.n) : (i += 1) {
        var found = false;
        for (members) |m| {
            if (ssmFindOldRow(group, m.key, m.gen) == i) {
                found = true;
                break;
            }
        }
        if (!found) try ssmExportRow(s, group, i);
    }
    const kinds_local = try alloc.dupe(SsmMergeKind, kinds);
    errdefer alloc.free(kinds_local);
    const new_layers = try alloc.alloc(SSMCacheEntry, kinds_local.len);
    var built: usize = 0;
    errdefer {
        for (new_layers[0..built]) |*m| ssmFreeMergedLayer(m);
        alloc.free(new_layers);
    }
    const n = members.len;
    for (new_layers, 0..) |*m, l| {
        m.* = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = kinds_local[l].linear };
        if (!kinds_local[l].linear and !kinds_local[l].ple) {
            built = l + 1;
            continue;
        }
        var parts: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
        // conv + ssm + aux per row.
        var views: [3 * MAX_BATCH_ROWS]mlx.mlx_array = undefined;
        var n_views: usize = 0;
        defer {
            var v: usize = 0;
            while (v < n_views) : (v += 1) _ = mlx.mlx_array_free(views[v]);
        }
        if (kinds_local[l].linear) {
            for (members, 0..) |mem, j| {
                if (ssmFindOldRow(group, mem.key, mem.gen)) |old| {
                    const view = try ssmRowView(s, group.layers[l].conv_state, old);
                    views[n_views] = view;
                    n_views += 1;
                    parts[j] = view;
                } else {
                    parts[j] = mem.entries[l].conv_state;
                }
            }
            m.conv_state = try ssmConcatAxis0(s, parts[0..n]);
            for (members, 0..) |mem, j| {
                if (ssmFindOldRow(group, mem.key, mem.gen)) |old| {
                    const view = try ssmRowView(s, group.layers[l].ssm_state, old);
                    views[n_views] = view;
                    n_views += 1;
                    parts[j] = view;
                } else {
                    parts[j] = mem.entries[l].ssm_state;
                }
            }
            m.ssm_state = try ssmConcatAxis0(s, parts[0..n]);
        }
        if (kinds_local[l].ple) {
            for (members, 0..) |mem, j| {
                if (ssmFindOldRow(group, mem.key, mem.gen)) |old| {
                    const view = try ssmRowView(s, group.layers[l].aux_state, old);
                    views[n_views] = view;
                    n_views += 1;
                    parts[j] = view;
                } else {
                    parts[j] = mem.entries[l].aux_state;
                }
            }
            m.aux_state = try ssmConcatAxis0(s, parts[0..n]);
        }
        built = l + 1;
    }
    const new_keys = try alloc.alloc(usize, n);
    errdefer alloc.free(new_keys);
    const new_gens = try alloc.alloc(u64, n);
    errdefer alloc.free(new_gens);
    const new_entries = try alloc.alloc([]SSMCacheEntry, n);
    errdefer alloc.free(new_entries);
    for (members, 0..) |mem, j| {
        new_keys[j] = mem.key;
        new_gens[j] = mem.gen;
        new_entries[j] = mem.entries;
    }
    const old_layers = group.layers;
    const old_keys = group.keys;
    const old_gens = group.gens;
    const old_member_entries = group.member_entries;
    const old_kinds = group.kinds;
    ssmUnbindAll(group);
    group.layers = new_layers;
    group.keys = new_keys;
    group.gens = new_gens;
    group.member_entries = new_entries;
    group.kinds = kinds_local;
    group.n = n;
    group.live = true;
    for (old_layers) |*m| ssmFreeMergedLayer(m);
    alloc.free(old_layers);
    alloc.free(old_keys);
    alloc.free(old_gens);
    alloc.free(old_member_entries);
    alloc.free(old_kinds);
    for (members, 0..) |mem, row| {
        for (mem.entries, 0..) |*e, l| {
            e.group = group;
            e.group_row = @intCast(row);
            e.group_layer = @intCast(l);
            e.member_gen = mem.gen;
        }
    }
}

pub fn ssmGroupDropMember(group: *PersistentSsmGroup, s: mlx.mlx_stream, entries: []SSMCacheEntry) !void {
    if (!group.live) return;
    const key = @intFromPtr(entries.ptr);
    var row: ?usize = null;
    for (group.keys, 0..) |k, i| {
        if (k == key) {
            row = i;
            break;
        }
    }
    const r = row orelse return;
    ssmUnbindEntries(group, entries);
    if (group.n == 1) {
        group.clear();
        return;
    }
    var keep: [MAX_BATCH_ROWS]SsmGroupMember = undefined;
    var n_keep: usize = 0;
    for (0..group.n) |i| {
        if (i == r) continue;
        keep[n_keep] = .{ .key = group.keys[i], .gen = group.gens[i], .entries = group.member_entries[i] };
        n_keep += 1;
    }
    ssmRepackMembers(group.allocator, s, group, keep[0..n_keep], group.kinds) catch |err| {
        var export_err: ?anyerror = null;
        for (0..group.n) |i| {
            if (i == r) continue;
            ssmExportRow(s, group, i) catch |e| {
                export_err = e;
            };
        }
        ssmUnbindAll(group);
        group.clear();
        if (export_err) |e| return e;
        return err;
    };
}

pub fn ssmGroupReleaseCtx(group: *PersistentSsmGroup, s: mlx.mlx_stream, ctx: *ForwardCtx) !void {
    const entries = ctx.ssm_entries orelse return;
    if (!group.live) return;
    const key = @intFromPtr(entries.ptr);
    var row: ?usize = null;
    for (group.keys, 0..) |k, i| {
        if (k == key) {
            row = i;
            break;
        }
    }
    const r = row orelse return;
    if (group.n == 1) {
        try ssmExportRow(s, group, r);
        if (entries.len > 0) ctx.ssm_member_gen = entries[0].member_gen;
        group.clear();
        return;
    }
    var keep: [MAX_BATCH_ROWS]SsmGroupMember = undefined;
    var n_keep: usize = 0;
    for (0..group.n) |i| {
        if (i == r) continue;
        keep[n_keep] = .{ .key = group.keys[i], .gen = group.gens[i], .entries = group.member_entries[i] };
        n_keep += 1;
    }
    try ssmRepackMembers(group.allocator, s, group, keep[0..n_keep], group.kinds);
    if (entries.len > 0) ctx.ssm_member_gen = entries[0].member_gen;
}

/// The tick's bind: a stable membership returns the group's layers with no concat,
/// no slice and no allocation; a changed membership repacks once.
pub fn ssmBindTickState(
    alloc: std.mem.Allocator,
    s: mlx.mlx_stream,
    group: *PersistentSsmGroup,
    ctxs: []const *ForwardCtx,
    kinds: []const SsmMergeKind,
    persist: bool,
) !struct { merged: []SSMCacheEntry, owned: bool } {
    if (!persist) {
        return .{ .merged = try ssmMergeAll(alloc, s, ctxs, kinds), .owned = true };
    }
    if (group.live and group.matches(ctxs) and group.kinds.len == kinds.len) {
        return .{ .merged = group.layers, .owned = false };
    }
    if (group.live) {
        var buf: [MAX_BATCH_ROWS]SsmGroupMember = undefined;
        try ssmRepackMembers(alloc, s, group, try ssmMembersFromCtxs(ctxs, &buf), kinds);
        return .{ .merged = group.layers, .owned = false };
    }
    const merged = try ssmMergeAll(alloc, s, ctxs, kinds);
    errdefer {
        for (merged) |*m| ssmFreeMergedLayer(m);
        alloc.free(merged);
    }
    try ssmGroupAdopt(group, alloc, ctxs, kinds, merged);
    return .{ .merged = group.layers, .owned = false };
}

pub fn ssmPatchGroupRow(
    s: mlx.mlx_stream,
    group: *PersistentSsmGroup,
    row: usize,
    layer: usize,
    conv: mlx.mlx_array,
    ssm: mlx.mlx_array,
    aux: mlx.mlx_array,
    fields: SsmPatchFields,
) !void {
    if (!group.live or layer >= group.layers.len) return;
    const m = &group.layers[layer];
    if (fields.conv and conv.ctx != null and m.conv_state.ctx != null) {
        try ssmSliceUpdateRow(s, &m.conv_state, row, conv);
    }
    if (fields.ssm and ssm.ctx != null and m.ssm_state.ctx != null) {
        try ssmSliceUpdateRow(s, &m.ssm_state, row, ssm);
    }
    if (fields.aux and aux.ctx != null and m.aux_state.ctx != null) {
        try ssmSliceUpdateRow(s, &m.aux_state, row, aux);
    }
}

pub fn ssmMaterializeLive(
    s: mlx.mlx_stream,
    src: *const SSMCacheEntry,
    arr: mlx.mlx_array,
) !mlx.mlx_array {
    if (arr.ctx == null) return mlx.mlx_array_new();
    if (src.group) |g| {
        if (g.live) {
            const view = try ssmRowView(s, arr, src.group_row);
            defer _ = mlx.mlx_array_free(view);
            return materializedOwnedCopy(s, view);
        }
    }
    return materializedOwnedCopy(s, arr);
}

pub fn ssmLiveAux(e: *const SSMCacheEntry) mlx.mlx_array {
    if (e.group) |g| {
        if (g.live and e.group_layer < g.layers.len) {
            const a = g.layers[e.group_layer].aux_state;
            if (a.ctx != null) return a;
        }
    }
    return e.aux_state;
}

pub fn ssmGroupDrop(self: *Transformer, entries: []SSMCacheEntry) void {
    ssmGroupDropMember(&self.ssm_group, self.s, entries) catch {};
}

/// Called at the entry of EVERY non-group forward (serial decode, prefill, the
/// MTP solo arms): a member's live row goes back to its own entries before the
/// solo path reads or writes them. A no-op for a slot that is not bound.
pub fn ssmGroupRelease(self: *Transformer, ctx: *ForwardCtx) !void {
    try ssmGroupReleaseCtx(&self.ssm_group, self.s, ctx);
}

// ── tests ──

pub fn ssmBumpMemberGen(ctx: *ForwardCtx) void {
    const g = nextSsmMemberGen();
    ctx.ssm_member_gen = g;
    if (ctx.ssm_entries) |entries| {
        for (entries) |*e| e.member_gen = g;
    }
}

fn ssmTickDummyArr(s: mlx.mlx_stream, v: f32, nd: usize) !mlx.mlx_array {
    const val = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(val);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (nd == 4) {
        const shape = [_]c_int{ 1, 2, 2, 2 };
        try mlx.check(mlx.mlx_full(&out, &shape, 4, val, .float32, s));
    } else {
        const shape = [_]c_int{ 1, 2, 2 };
        try mlx.check(mlx.mlx_full(&out, &shape, 3, val, .float32, s));
    }
    return out;
}

const SsmTickSlot = struct {
    off: usize = 0,
    entries: []SSMCacheEntry,
    ctx: ForwardCtx = undefined,

    fn init(alloc: std.mem.Allocator, s: mlx.mlx_stream, kinds: []const SsmMergeKind, cache: *KVCache, seed: f32) !SsmTickSlot {
        const entries = try alloc.alloc(SSMCacheEntry, kinds.len);
        errdefer alloc.free(entries);
        for (entries, 0..) |*e, l| {
            e.* = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = kinds[l].linear };
            if (kinds[l].linear) {
                e.conv_state = try ssmTickDummyArr(s, seed + @as(f32, @floatFromInt(l)), 3);
                e.ssm_state = try ssmTickDummyArr(s, seed + 100 + @as(f32, @floatFromInt(l)), 4);
            }
            if (kinds[l].ple) {
                e.aux_state = try ssmTickDummyArr(s, seed + 200 + @as(f32, @floatFromInt(l)), 3);
            }
        }
        const gen = nextSsmMemberGen();
        for (entries) |*e| e.member_gen = gen;
        return .{
            .entries = entries,
            .ctx = .{
                .cache = cache,
                .moe_seq_offset = undefined,
                .ssm_entries = entries,
                .ssm_member_gen = gen,
                .capture_hidden = null,
                .vision_embeddings = null,
            },
        };
    }

    fn bind(self: *SsmTickSlot) void {
        self.ctx.moe_seq_offset = &self.off;
        self.ctx.ssm_entries = self.entries;
    }

    fn deinit(self: *SsmTickSlot, alloc: std.mem.Allocator) void {
        for (self.entries) |*e| {
            if (e.conv_state.ctx != null) _ = mlx.mlx_array_free(e.conv_state);
            if (e.ssm_state.ctx != null) _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
            ssmFreeSpecCapture(e);
        }
        alloc.free(self.entries);
    }
};

test "persistent group: N=4 stable ticks issue zero merge/split ops after the first" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();

    const n_linear: usize = 36;
    const n_layers: usize = n_linear + 1;
    const kinds = try alloc.alloc(SsmMergeKind, n_layers);
    defer alloc.free(kinds);
    for (kinds[0..n_linear]) |*k| k.* = .{ .linear = true, .ple = false };
    kinds[n_linear] = .{ .linear = false, .ple = true };

    var slots: [4]SsmTickSlot = undefined;
    for (&slots, 0..) |*sl, i| {
        sl.* = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        sl.bind();
    }
    defer for (&slots) |*sl| sl.deinit(alloc);

    var ctxs: [4]*ForwardCtx = undefined;
    for (&slots, 0..) |*sl, i| ctxs[i] = &sl.ctx;

    var xfm = ssmTickStubXfm(alloc, s, true);
    defer xfm.ssm_group.deinit();

    const ticks: usize = 8;
    var after_first: u64 = 0;
    var first_delta: u64 = 0;
    var t_i: usize = 0;
    while (t_i < ticks) : (t_i += 1) {
        const ops0 = mlx.op_count.load(.monotonic);
        try ssmTickBindXfm(&xfm, &ctxs, kinds);
        const delta = mlx.op_count.load(.monotonic) - ops0;
        if (t_i == 0) {
            first_delta = delta;
        } else {
            after_first += delta;
        }
    }
    try std.testing.expect(first_delta > 0);
    try std.testing.expectEqual(@as(u64, 0), after_first);
}

fn ssmTickBumpArr(s: mlx.mlx_stream, arr: mlx.mlx_array) !mlx.mlx_array {
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, arr, one, s));
    return out;
}

fn ssmTickBumpMerged(s: mlx.mlx_stream, layers: []SSMCacheEntry, kinds: []const SsmMergeKind) !void {
    for (layers, kinds) |*m, kind| {
        if (kind.linear) {
            const c = try ssmTickBumpArr(s, m.conv_state);
            _ = mlx.mlx_array_free(m.conv_state);
            m.conv_state = c;
            const st = try ssmTickBumpArr(s, m.ssm_state);
            _ = mlx.mlx_array_free(m.ssm_state);
            m.ssm_state = st;
        }
        if (kind.ple) {
            const a = try ssmTickBumpArr(s, m.aux_state);
            _ = mlx.mlx_array_free(m.aux_state);
            m.aux_state = a;
        }
    }
}

fn ssmTickBumpSlots(s: mlx.mlx_stream, ctxs: []const *ForwardCtx, kinds: []const SsmMergeKind) !void {
    for (ctxs) |c| {
        for (c.ssm_entries.?, kinds) |*e, kind| {
            if (kind.linear) {
                const cv = try ssmTickBumpArr(s, e.conv_state);
                _ = mlx.mlx_array_free(e.conv_state);
                e.conv_state = cv;
                const st = try ssmTickBumpArr(s, e.ssm_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                e.ssm_state = st;
            }
            if (kind.ple) {
                const a = try ssmTickBumpArr(s, e.aux_state);
                _ = mlx.mlx_array_free(e.aux_state);
                e.aux_state = a;
            }
        }
    }
}

fn ssmTickCompareRow(
    s: mlx.mlx_stream,
    persist_e: *const SSMCacheEntry,
    old_e: *const SSMCacheEntry,
    kind: SsmMergeKind,
) !void {
    if (kind.linear) {
        const pc = try ssmMaterializeLive(s, persist_e, if (persist_e.group) |g| g.layers[persist_e.group_layer].conv_state else persist_e.conv_state);
        defer _ = mlx.mlx_array_free(pc);
        const ps = try ssmMaterializeLive(s, persist_e, if (persist_e.group) |g| g.layers[persist_e.group_layer].ssm_state else persist_e.ssm_state);
        defer _ = mlx.mlx_array_free(ps);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pc, old_e.conv_state, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(ps, old_e.ssm_state, s));
    }
    if (kind.ple) {
        const pa = try ssmMaterializeLive(s, persist_e, ssmLiveAux(persist_e));
        defer _ = mlx.mlx_array_free(pa);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pa, old_e.aux_state, s));
    }
}

fn ssmTickKinds(alloc: std.mem.Allocator, n_linear: usize) ![]SsmMergeKind {
    const kinds = try alloc.alloc(SsmMergeKind, n_linear + 1);
    for (kinds[0..n_linear]) |*k| k.* = .{ .linear = true, .ple = false };
    kinds[n_linear] = .{ .linear = true, .ple = true };
    return kinds;
}

test "persistent group: join leave cancel capture snapshot rollback restore match the old path" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();
    const kinds = try ssmTickKinds(alloc, 2);
    defer alloc.free(kinds);

    var p_slots: [3]SsmTickSlot = undefined;
    var o_slots: [3]SsmTickSlot = undefined;
    for (0..3) |i| {
        p_slots[i] = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        p_slots[i].bind();
        o_slots[i] = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        o_slots[i].bind();
    }
    defer for (&p_slots) |*sl| sl.deinit(alloc);
    defer for (&o_slots) |*sl| sl.deinit(alloc);

    var p_ctxs2 = [_]*ForwardCtx{ &p_slots[0].ctx, &p_slots[1].ctx };
    var o_ctxs2 = [_]*ForwardCtx{ &o_slots[0].ctx, &o_slots[1].ctx };
    var p_group: PersistentSsmGroup = .{ .allocator = alloc };
    defer p_group.deinit();
    var o_group: PersistentSsmGroup = .{ .allocator = alloc };
    defer o_group.deinit();

    var t_i: usize = 0;
    while (t_i < 3) : (t_i += 1) {
        try ssmTickBind(alloc, s, &p_group, &p_ctxs2, kinds, true);
        try ssmTickBumpMerged(s, p_group.layers, kinds);
        try ssmTickBind(alloc, s, &o_group, &o_ctxs2, kinds, false);
        try ssmTickBumpSlots(s, &o_ctxs2, kinds);
    }
    for (kinds, 0..) |kind, l| {
        try ssmTickCompareRow(s, &p_slots[0].entries[l], &o_slots[0].entries[l], kind);
        try ssmTickCompareRow(s, &p_slots[1].entries[l], &o_slots[1].entries[l], kind);
    }

    var p_ctxs3 = [_]*ForwardCtx{ &p_slots[0].ctx, &p_slots[1].ctx, &p_slots[2].ctx };
    var o_ctxs3 = [_]*ForwardCtx{ &o_slots[0].ctx, &o_slots[1].ctx, &o_slots[2].ctx };
    try ssmTickBind(alloc, s, &p_group, &p_ctxs3, kinds, true);
    try ssmTickBumpMerged(s, p_group.layers, kinds);
    try ssmTickBind(alloc, s, &o_group, &o_ctxs3, kinds, false);
    try ssmTickBumpSlots(s, &o_ctxs3, kinds);
    for (kinds, 0..) |kind, l| {
        try ssmTickCompareRow(s, &p_slots[0].entries[l], &o_slots[0].entries[l], kind);
        try ssmTickCompareRow(s, &p_slots[1].entries[l], &o_slots[1].entries[l], kind);
        try ssmTickCompareRow(s, &p_slots[2].entries[l], &o_slots[2].entries[l], kind);
    }

    var cp = try captureSsmCheckpoint(alloc, p_slots[1].entries, 4, s);
    defer cp.deinit(alloc);
    try ssmTickBumpMerged(s, p_group.layers, kinds);
    var cp_old = try captureSsmCheckpoint(alloc, o_slots[1].entries, 4, s);
    defer cp_old.deinit(alloc);
    try ssmTickBumpSlots(s, &o_ctxs3, kinds);
    for (cp.layers, cp_old.layers, kinds) |*pl, *ol, kind| {
        if (kind.linear) {
            try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pl.conv_state, ol.conv_state, s));
            try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pl.ssm_state, ol.ssm_state, s));
        }
        if (kind.ple) {
            try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pl.aux_state, ol.aux_state, s));
        }
    }

    var p_leave = [_]*ForwardCtx{ &p_slots[0].ctx, &p_slots[2].ctx };
    var o_leave = [_]*ForwardCtx{ &o_slots[0].ctx, &o_slots[2].ctx };
    try ssmTickBind(alloc, s, &p_group, &p_leave, kinds, true);
    try ssmTickBumpMerged(s, p_group.layers, kinds);
    try ssmTickBind(alloc, s, &o_group, &o_leave, kinds, false);
    try ssmTickBumpSlots(s, &o_leave, kinds);
    for (kinds, 0..) |kind, l| {
        try ssmTickCompareRow(s, &p_slots[0].entries[l], &o_slots[0].entries[l], kind);
        try ssmTickCompareRow(s, &p_slots[2].entries[l], &o_slots[2].entries[l], kind);
        try ssmTickCompareRow(s, &p_slots[1].entries[l], &o_slots[1].entries[l], kind);
    }

    var snap = ssmSnapshot(&p_slots[1].entries[0]);
    defer ssmSnapshotDeinit(&snap);
    var snap_old = ssmSnapshot(&o_slots[1].entries[0]);
    defer ssmSnapshotDeinit(&snap_old);
    if (kinds[0].linear) {
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(snap.conv_state, snap_old.conv_state, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(snap.ssm_state, snap_old.ssm_state, s));
    }

    try restoreSsmCheckpoint(p_slots[2].entries, &cp);
    try restoreSsmCheckpoint(o_slots[2].entries, &cp_old);
    try ssmTickBind(alloc, s, &p_group, &p_leave, kinds, true);
    try ssmTickBind(alloc, s, &o_group, &o_leave, kinds, false);
    for (kinds, 0..) |kind, l| {
        try ssmTickCompareRow(s, &p_slots[2].entries[l], &o_slots[2].entries[l], kind);
    }

    {
        const state_len: c_int = 2;
        const verify_len: c_int = 3;
        const width: c_int = 2;
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const n: f64 = @floatFromInt((state_len + verify_len) * width);
        try mlx.check(mlx.mlx_arange(&flat, 0.0, n, 1.0, .float32, s));
        const shape = [_]c_int{ 1, state_len + verify_len, width };
        for ([_]*SSMCacheEntry{ &p_slots[0].entries[0], &o_slots[0].entries[0] }) |e| {
            if (e.spec_conv_input.ctx != null) _ = mlx.mlx_array_free(e.spec_conv_input);
            e.spec_conv_input = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&e.spec_conv_input, flat, &shape, 3, s));
        }
        try ssmRollbackFromCapture(&p_slots[0].entries[0], 1, @intCast(verify_len), s);
        try ssmRollbackFromCapture(&o_slots[0].entries[0], 1, @intCast(verify_len), s);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(p_slots[0].entries[0].conv_state, o_slots[0].entries[0].conv_state, s));
    }
}

test "persistent group: a member released from a full 16-row group owns a single-row state" {
    // The bar: after `ssmGroupReleaseCtx` the leaving slot's conv/ssm state has batch 1 and
    // equals its row of the merged group, so its own verify never sees the [N, ...] arrays.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();
    const kinds = try ssmTickKinds(alloc, 2);
    defer alloc.free(kinds);
    const n = 16;
    var slots: [n]SsmTickSlot = undefined;
    var ctxs: [n]*ForwardCtx = undefined;
    for (0..n) |i| {
        slots[i] = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        slots[i].bind();
        ctxs[i] = &slots[i].ctx;
    }
    defer for (&slots) |*sl| sl.deinit(alloc);
    var group: PersistentSsmGroup = .{ .allocator = alloc };
    defer group.deinit();
    try ssmTickBind(alloc, s, &group, &ctxs, kinds, true);
    try ssmTickBumpMerged(s, group.layers, kinds);
    try ssmTickBind(alloc, s, &group, &ctxs, kinds, true);
    try ssmTickBumpMerged(s, group.layers, kinds);
    // Row 5 leaves for a serial round of its own, the way the scheduler releases a speculating slot.
    const leaver = 5;
    const expect_conv = try ssmRowView(s, group.layers[0].conv_state, leaver);
    defer _ = mlx.mlx_array_free(expect_conv);
    const expect_ssm = try ssmRowView(s, group.layers[0].ssm_state, leaver);
    defer _ = mlx.mlx_array_free(expect_ssm);
    try ssmGroupReleaseCtx(&group, s, ctxs[leaver]);
    const e = &slots[leaver].entries[0];
    try std.testing.expectEqual(@as(c_int, 1), mlx.getShape(e.conv_state)[0]);
    try std.testing.expectEqual(@as(c_int, 1), mlx.getShape(e.ssm_state)[0]);
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(e.conv_state, expect_conv, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(e.ssm_state, expect_ssm, s));
    try std.testing.expectEqual(@as(usize, n - 1), group.n);
}

test "persistent group: merge/split op counts at N=2/4/8 after the first tick are zero" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();
    const kinds = try ssmTickKinds(alloc, 36);
    defer alloc.free(kinds);
    const ns = [_]usize{ 2, 4, 8 };
    for (ns) |n| {
        const slots = try alloc.alloc(SsmTickSlot, n);
        defer {
            for (slots) |*sl| sl.deinit(alloc);
            alloc.free(slots);
        }
        for (slots, 0..) |*sl, i| {
            sl.* = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
            sl.bind();
        }
        const ctxs = try alloc.alloc(*ForwardCtx, n);
        defer alloc.free(ctxs);
        for (slots, 0..) |*sl, i| ctxs[i] = &sl.ctx;
        var xfm = ssmTickStubXfm(alloc, s, true);
        defer xfm.ssm_group.deinit();
        var after: u64 = 0;
        var first: u64 = 0;
        var t_i: usize = 0;
        while (t_i < 8) : (t_i += 1) {
            const ops0 = mlx.op_count.load(.monotonic);
            try ssmTickBindXfm(&xfm, ctxs, kinds);
            const delta = mlx.op_count.load(.monotonic) - ops0;
            if (t_i == 0) first = delta else after += delta;
        }
        try std.testing.expect(first > 0);
        try std.testing.expectEqual(@as(u64, 0), after);
    }
}

test "persistent group: drop a member does not free kinds the adopt still reads" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();
    const kinds = try ssmTickKinds(alloc, 1);
    defer alloc.free(kinds);
    var slots: [2]SsmTickSlot = undefined;
    for (&slots, 0..) |*sl, i| {
        sl.* = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        sl.bind();
    }
    defer for (&slots) |*sl| sl.deinit(alloc);
    var ctxs = [_]*ForwardCtx{ &slots[0].ctx, &slots[1].ctx };
    var xfm: Transformer = undefined;
    xfm.rht = null;
    xfm.s = s;
    xfm.allocator = alloc;
    xfm.ssm_group = .{ .allocator = alloc };
    defer xfm.ssm_group.deinit();
    try ssmTickBind(alloc, s, &xfm.ssm_group, &ctxs, kinds, true);
    xfm.ssmGroupDrop(slots[0].entries);
    try std.testing.expect(xfm.ssm_group.live);
    try std.testing.expectEqual(@as(usize, 1), xfm.ssm_group.n);
    try std.testing.expect(xfm.ssm_group.kinds.len == kinds.len);
    try std.testing.expect(xfm.ssm_group.kinds[0].linear);
    try std.testing.expect(xfm.ssm_group.kinds[1].ple);
}

fn ssmTickStubXfm(alloc: std.mem.Allocator, s: mlx.mlx_stream, persist: bool) Transformer {
    var xfm: Transformer = undefined;
    xfm.rht = null;
    xfm.s = s;
    xfm.allocator = alloc;
    xfm.persistent_group_state_override = persist;
    xfm.ssm_group = .{ .allocator = alloc };
    return xfm;
}

fn ssmTickBindXfm(xfm: *Transformer, ctxs: []const *ForwardCtx, kinds: []const SsmMergeKind) !void {
    try xfm.ssmBindBatched(ctxs, kinds);
}

fn ssmTickBind(
    alloc: std.mem.Allocator,
    s: mlx.mlx_stream,
    group: *PersistentSsmGroup,
    ctxs: []const *ForwardCtx,
    kinds: []const SsmMergeKind,
    persist: bool,
) !void {
    const bind = try ssmBindTickState(alloc, s, group, ctxs, kinds, persist);
    if (bind.owned) {
        defer {
            for (bind.merged) |*m| ssmFreeMergedLayer(m);
            alloc.free(bind.merged);
        }
        try ssmSplitAll(s, bind.merged, ctxs, kinds);
    }
}

test "persistent group: a solo tick after N=2 batched ticks matches the old path" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();
    const kinds = try ssmTickKinds(alloc, 1);
    defer alloc.free(kinds);
    var p_slots: [2]SsmTickSlot = undefined;
    var o_slots: [2]SsmTickSlot = undefined;
    for (0..2) |i| {
        p_slots[i] = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        p_slots[i].bind();
        o_slots[i] = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        o_slots[i].bind();
    }
    defer for (&p_slots) |*sl| sl.deinit(alloc);
    defer for (&o_slots) |*sl| sl.deinit(alloc);
    var p_ctxs = [_]*ForwardCtx{ &p_slots[0].ctx, &p_slots[1].ctx };
    var o_ctxs = [_]*ForwardCtx{ &o_slots[0].ctx, &o_slots[1].ctx };
    var p_xfm = ssmTickStubXfm(alloc, s, true);
    defer p_xfm.ssm_group.deinit();
    var o_xfm = ssmTickStubXfm(alloc, s, false);
    defer o_xfm.ssm_group.deinit();
    var t_i: usize = 0;
    while (t_i < 2) : (t_i += 1) {
        try ssmTickBindXfm(&p_xfm, &p_ctxs, kinds);
        try ssmTickBumpMerged(s, p_xfm.ssm_group.layers, kinds);
        try ssmTickBindXfm(&o_xfm, &o_ctxs, kinds);
        try ssmTickBumpSlots(s, &o_ctxs, kinds);
    }
    try p_xfm.ssmGroupRelease(&p_slots[0].ctx);
    try ssmTickBumpSlots(s, p_ctxs[0..1], kinds);
    try ssmTickBumpSlots(s, o_ctxs[0..1], kinds);
    try ssmTickBindXfm(&p_xfm, &p_ctxs, kinds);
    try ssmTickBindXfm(&o_xfm, &o_ctxs, kinds);
    for (kinds, 0..) |kind, l| {
        try ssmTickCompareRow(s, &p_slots[0].entries[l], &o_slots[0].entries[l], kind);
        try ssmTickCompareRow(s, &p_slots[1].entries[l], &o_slots[1].entries[l], kind);
    }
}

test "persistent group: PLE-only rollback patches the group window" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();
    var kinds = [_]SsmMergeKind{.{ .linear = false, .ple = true }};
    var p_slots: [2]SsmTickSlot = undefined;
    var o_slots: [2]SsmTickSlot = undefined;
    for (0..2) |i| {
        p_slots[i] = try SsmTickSlot.init(alloc, s, &kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        p_slots[i].bind();
        o_slots[i] = try SsmTickSlot.init(alloc, s, &kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        o_slots[i].bind();
    }
    defer for (&p_slots) |*sl| sl.deinit(alloc);
    defer for (&o_slots) |*sl| sl.deinit(alloc);
    var p_ctxs = [_]*ForwardCtx{ &p_slots[0].ctx, &p_slots[1].ctx };
    var o_ctxs = [_]*ForwardCtx{ &o_slots[0].ctx, &o_slots[1].ctx };
    var p_xfm = ssmTickStubXfm(alloc, s, true);
    defer p_xfm.ssm_group.deinit();
    var o_xfm = ssmTickStubXfm(alloc, s, false);
    defer o_xfm.ssm_group.deinit();
    try ssmTickBindXfm(&p_xfm, &p_ctxs, &kinds);
    try ssmTickBindXfm(&o_xfm, &o_ctxs, &kinds);
    const state_len: c_int = 2;
    const verify_len: c_int = 3;
    const width: c_int = 2;
    const n: f64 = @floatFromInt((state_len + verify_len) * width);
    const shape = [_]c_int{ 1, state_len + verify_len, width };
    for ([_]*SSMCacheEntry{ &p_slots[0].entries[0], &o_slots[0].entries[0] }) |e| {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_arange(&flat, 0.0, n, 1.0, .float32, s));
        e.spec_ple_input = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&e.spec_ple_input, flat, &shape, 3, s));
        e.spec_ple_len = @intCast(state_len + verify_len);
    }
    try ssmRollbackFromCapture(&p_slots[0].entries[0], 1, @intCast(verify_len), s);
    try ssmRollbackFromCapture(&o_slots[0].entries[0], 1, @intCast(verify_len), s);
    try ssmTickCompareRow(s, &p_slots[0].entries[0], &o_slots[0].entries[0], kinds[0]);
}

test "persistent group: single-capture rollback does not write a stale sibling" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();
    var kinds = [_]SsmMergeKind{.{ .linear = true, .ple = false }};
    var p_slots: [2]SsmTickSlot = undefined;
    var o_slots: [2]SsmTickSlot = undefined;
    for (0..2) |i| {
        p_slots[i] = try SsmTickSlot.init(alloc, s, &kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        p_slots[i].bind();
        o_slots[i] = try SsmTickSlot.init(alloc, s, &kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        o_slots[i].bind();
    }
    defer for (&p_slots) |*sl| sl.deinit(alloc);
    defer for (&o_slots) |*sl| sl.deinit(alloc);
    var p_ctxs = [_]*ForwardCtx{ &p_slots[0].ctx, &p_slots[1].ctx };
    var o_ctxs = [_]*ForwardCtx{ &o_slots[0].ctx, &o_slots[1].ctx };
    var p_xfm = ssmTickStubXfm(alloc, s, true);
    defer p_xfm.ssm_group.deinit();
    var o_xfm = ssmTickStubXfm(alloc, s, false);
    defer o_xfm.ssm_group.deinit();
    try ssmTickBindXfm(&p_xfm, &p_ctxs, &kinds);
    try ssmTickBumpMerged(s, p_xfm.ssm_group.layers, &kinds);
    try ssmTickBindXfm(&o_xfm, &o_ctxs, &kinds);
    try ssmTickBumpSlots(s, &o_ctxs, &kinds);
    const t_len: c_int = 3;
    const km1: c_int = 2;
    const width: c_int = 2;
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    const n: f64 = @floatFromInt((km1 + t_len) * width);
    try mlx.check(mlx.mlx_arange(&flat, 0.0, n, 1.0, .float32, s));
    const shape = [_]c_int{ 1, km1 + t_len, width };
    for ([_]*SSMCacheEntry{ &p_slots[0].entries[0], &o_slots[0].entries[0] }) |e| {
        e.spec_conv_input = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&e.spec_conv_input, flat, &shape, 3, s));
    }
    const ssm_before = try ssmMaterializeLive(s, &p_slots[0].entries[0], p_xfm.ssm_group.layers[0].ssm_state);
    defer _ = mlx.mlx_array_free(ssm_before);
    try ssmRollbackFromCapture(&p_slots[0].entries[0], 1, @intCast(t_len), s);
    try ssmRollbackFromCapture(&o_slots[0].entries[0], 1, @intCast(t_len), s);
    const ssm_after = try ssmMaterializeLive(s, &p_slots[0].entries[0], p_xfm.ssm_group.layers[0].ssm_state);
    defer _ = mlx.mlx_array_free(ssm_after);
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(ssm_after, o_slots[0].entries[0].ssm_state, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(ssm_after, ssm_before, s));
}

test "persistent group: snapshot is an owned row and survives departure" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();
    const kinds = try ssmTickKinds(alloc, 1);
    defer alloc.free(kinds);
    var p_slots: [2]SsmTickSlot = undefined;
    var o_slots: [2]SsmTickSlot = undefined;
    for (0..2) |i| {
        p_slots[i] = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        p_slots[i].bind();
        o_slots[i] = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        o_slots[i].bind();
    }
    defer for (&p_slots) |*sl| sl.deinit(alloc);
    defer for (&o_slots) |*sl| sl.deinit(alloc);
    var p_ctxs = [_]*ForwardCtx{ &p_slots[0].ctx, &p_slots[1].ctx };
    var o_ctxs = [_]*ForwardCtx{ &o_slots[0].ctx, &o_slots[1].ctx };
    var p_xfm = ssmTickStubXfm(alloc, s, true);
    defer p_xfm.ssm_group.deinit();
    var o_xfm = ssmTickStubXfm(alloc, s, false);
    defer o_xfm.ssm_group.deinit();
    try ssmTickBindXfm(&p_xfm, &p_ctxs, kinds);
    try ssmTickBumpMerged(s, p_xfm.ssm_group.layers, kinds);
    try ssmTickBindXfm(&o_xfm, &o_ctxs, kinds);
    try ssmTickBumpSlots(s, &o_ctxs, kinds);
    var snap = ssmSnapshot(&p_slots[0].entries[0]);
    defer ssmSnapshotDeinit(&snap);
    var snap_old = ssmSnapshot(&o_slots[0].entries[0]);
    defer ssmSnapshotDeinit(&snap_old);
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(snap.conv_state, snap_old.conv_state, s));
    const row_n = mlx.mlx_array_size(snap.conv_state);
    const merged_n = mlx.mlx_array_size(p_xfm.ssm_group.layers[0].conv_state);
    try std.testing.expect(row_n * 2 == merged_n);
    var p_leave = [_]*ForwardCtx{&p_slots[1].ctx};
    try ssmTickBindXfm(&p_xfm, &p_leave, kinds);
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(snap.conv_state, snap_old.conv_state, s));
}

test "persistent group: allocator failure before a replacement descriptor keeps the old group" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();
    const kinds = try ssmTickKinds(alloc, 1);
    defer alloc.free(kinds);
    var slots: [3]SsmTickSlot = undefined;
    for (&slots, 0..) |*sl, i| {
        sl.* = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        sl.bind();
    }
    defer for (&slots) |*sl| sl.deinit(alloc);
    var ctxs2 = [_]*ForwardCtx{ &slots[0].ctx, &slots[1].ctx };
    var xfm = ssmTickStubXfm(alloc, s, true);
    defer xfm.ssm_group.deinit();
    try ssmTickBindXfm(&xfm, &ctxs2, kinds);
    try std.testing.expectEqual(@as(usize, 2), xfm.ssm_group.n);
    var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0 });
    var ctxs3 = [_]*ForwardCtx{ &slots[0].ctx, &slots[1].ctx, &slots[2].ctx };
    var buf: [MAX_BATCH_ROWS]SsmGroupMember = undefined;
    const failed = ssmRepackMembers(failing.allocator(), s, &xfm.ssm_group, try ssmMembersFromCtxs(&ctxs3, &buf), kinds);
    try std.testing.expect(failed == error.OutOfMemory);
    try std.testing.expect(xfm.ssm_group.live);
    try std.testing.expectEqual(@as(usize, 2), xfm.ssm_group.n);
}

test "persistent group: cancel keeps the row until the next membership freeze" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();
    const kinds = try ssmTickKinds(alloc, 1);
    defer alloc.free(kinds);
    var p_slots: [2]SsmTickSlot = undefined;
    var o_slots: [2]SsmTickSlot = undefined;
    for (0..2) |i| {
        p_slots[i] = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        p_slots[i].bind();
        o_slots[i] = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        o_slots[i].bind();
    }
    defer for (&p_slots) |*sl| sl.deinit(alloc);
    defer for (&o_slots) |*sl| sl.deinit(alloc);
    var p_ctxs = [_]*ForwardCtx{ &p_slots[0].ctx, &p_slots[1].ctx };
    var o_ctxs = [_]*ForwardCtx{ &o_slots[0].ctx, &o_slots[1].ctx };
    var p_xfm = ssmTickStubXfm(alloc, s, true);
    defer p_xfm.ssm_group.deinit();
    var o_xfm = ssmTickStubXfm(alloc, s, false);
    defer o_xfm.ssm_group.deinit();
    try ssmTickBindXfm(&p_xfm, &p_ctxs, kinds);
    try ssmTickBumpMerged(s, p_xfm.ssm_group.layers, kinds);
    try ssmTickBindXfm(&o_xfm, &o_ctxs, kinds);
    try ssmTickBumpSlots(s, &o_ctxs, kinds);
    try ssmTickBindXfm(&p_xfm, &p_ctxs, kinds);
    try ssmTickBumpMerged(s, p_xfm.ssm_group.layers, kinds);
    try ssmTickBindXfm(&o_xfm, &o_ctxs, kinds);
    try ssmTickBumpSlots(s, &o_ctxs, kinds);
    try std.testing.expectEqual(@as(usize, 2), p_xfm.ssm_group.n);
    var p_left = [_]*ForwardCtx{&p_slots[0].ctx};
    var o_left = [_]*ForwardCtx{&o_slots[0].ctx};
    try ssmTickBindXfm(&p_xfm, &p_left, kinds);
    try ssmTickBumpMerged(s, p_xfm.ssm_group.layers, kinds);
    try ssmTickBindXfm(&o_xfm, &o_left, kinds);
    try ssmTickBumpSlots(s, &o_left, kinds);
    try ssmTickCompareRow(s, &p_slots[0].entries[0], &o_slots[0].entries[0], kinds[0]);
    try ssmTickCompareRow(s, &p_slots[1].entries[0], &o_slots[1].entries[0], kinds[0]);
}

test "persistent group: unload then slot-address reuse does not match the dead group" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();
    const kinds = try ssmTickKinds(alloc, 1);
    defer alloc.free(kinds);
    var slots: [2]SsmTickSlot = undefined;
    for (&slots, 0..) |*sl, i| {
        sl.* = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        sl.bind();
    }
    defer for (&slots) |*sl| sl.deinit(alloc);
    var ctxs = [_]*ForwardCtx{ &slots[0].ctx, &slots[1].ctx };
    var xfm = ssmTickStubXfm(alloc, s, true);
    try ssmTickBindXfm(&xfm, &ctxs, kinds);
    xfm.ssm_group.deinit();
    xfm.ssm_group = .{ .allocator = alloc };
    defer xfm.ssm_group.deinit();
    ssmBumpMemberGen(&slots[0].ctx);
    ssmBumpMemberGen(&slots[1].ctx);
    try ssmTickBindXfm(&xfm, &ctxs, kinds);
    try std.testing.expect(xfm.ssm_group.live);
    try std.testing.expectEqual(@as(usize, 2), xfm.ssm_group.n);
    try std.testing.expect(xfm.ssm_group.matches(&ctxs));
}

fn ssmTickResetLikePrefixCache(entries: []SSMCacheEntry, s: mlx.mlx_stream, seed: f32, kinds: []const SsmMergeKind) !void {
    for (entries, kinds) |*e, kind| {
        if (e.conv_state.ctx != null) _ = mlx.mlx_array_free(e.conv_state);
        if (e.ssm_state.ctx != null) _ = mlx.mlx_array_free(e.ssm_state);
        if (e.aux_state.ctx != null) _ = mlx.mlx_array_free(e.aux_state);
        e.conv_state = .{ .ctx = null };
        e.ssm_state = .{ .ctx = null };
        e.aux_state = .{ .ctx = null };
        e.initialized = kind.linear;
        ssmFreeQsaState(e);
        e.ple_prev_valid = false;
        if (kind.linear) {
            e.conv_state = try ssmTickDummyArr(s, seed, 3);
            e.ssm_state = try ssmTickDummyArr(s, seed + 1, 4);
        }
        if (kind.ple) {
            e.aux_state = try ssmTickDummyArr(s, seed + 2, 3);
        }
    }
    ssmDetachFromGroup(entries);
}

test "persistent group: resetSsmEntries departure is not resurrected from the merged buffer" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();
    const kinds = try ssmTickKinds(alloc, 1);
    defer alloc.free(kinds);
    var p_slots: [2]SsmTickSlot = undefined;
    var o_slots: [2]SsmTickSlot = undefined;
    for (0..2) |i| {
        p_slots[i] = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        p_slots[i].bind();
        o_slots[i] = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        o_slots[i].bind();
    }
    defer for (&p_slots) |*sl| sl.deinit(alloc);
    defer for (&o_slots) |*sl| sl.deinit(alloc);
    var p_ctxs = [_]*ForwardCtx{ &p_slots[0].ctx, &p_slots[1].ctx };
    var o_ctxs = [_]*ForwardCtx{ &o_slots[0].ctx, &o_slots[1].ctx };
    var p_xfm = ssmTickStubXfm(alloc, s, true);
    defer p_xfm.ssm_group.deinit();
    var o_xfm = ssmTickStubXfm(alloc, s, false);
    defer o_xfm.ssm_group.deinit();
    try ssmTickBindXfm(&p_xfm, &p_ctxs, kinds);
    try ssmTickBumpMerged(s, p_xfm.ssm_group.layers, kinds);
    try ssmTickBindXfm(&o_xfm, &o_ctxs, kinds);
    try ssmTickBumpSlots(s, &o_ctxs, kinds);
    try ssmTickResetLikePrefixCache(p_slots[0].entries, s, 99.0, kinds);
    try ssmTickResetLikePrefixCache(o_slots[0].entries, s, 99.0, kinds);
    try ssmTickBindXfm(&p_xfm, &p_ctxs, kinds);
    try ssmTickBindXfm(&o_xfm, &o_ctxs, kinds);
    for (kinds, 0..) |kind, l| {
        try ssmTickCompareRow(s, &p_slots[0].entries[l], &o_slots[0].entries[l], kind);
    }
}

test "persistent group: export then a failed repack invalidates matches" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();
    const kinds = try ssmTickKinds(alloc, 1);
    defer alloc.free(kinds);
    var slots: [2]SsmTickSlot = undefined;
    for (&slots, 0..) |*sl, i| {
        sl.* = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        sl.bind();
    }
    defer for (&slots) |*sl| sl.deinit(alloc);
    var ctxs = [_]*ForwardCtx{ &slots[0].ctx, &slots[1].ctx };
    var xfm = ssmTickStubXfm(alloc, s, true);
    defer xfm.ssm_group.deinit();
    try ssmTickBindXfm(&xfm, &ctxs, kinds);
    try ssmExportRow(s, &xfm.ssm_group, 0);
    var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0 });
    var buf: [MAX_BATCH_ROWS]SsmGroupMember = undefined;
    _ = ssmRepackMembers(failing.allocator(), s, &xfm.ssm_group, try ssmMembersFromCtxs(&ctxs, &buf), kinds) catch {};
    try std.testing.expect(!xfm.ssm_group.matches(&ctxs));
}

test "persistent group: drop failure exports survivor rows" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();
    const kinds = try ssmTickKinds(alloc, 1);
    defer alloc.free(kinds);
    var slots: [2]SsmTickSlot = undefined;
    for (&slots, 0..) |*sl, i| {
        sl.* = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        sl.bind();
    }
    defer for (&slots) |*sl| sl.deinit(alloc);
    var ctxs = [_]*ForwardCtx{ &slots[0].ctx, &slots[1].ctx };
    var xfm = ssmTickStubXfm(alloc, s, true);
    defer xfm.ssm_group.deinit();
    try ssmTickBindXfm(&xfm, &ctxs, kinds);
    try ssmTickBumpMerged(s, xfm.ssm_group.layers, kinds);
    const want = try ssmMaterializeLive(s, &slots[1].entries[0], xfm.ssm_group.layers[0].conv_state);
    defer _ = mlx.mlx_array_free(want);
    var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 2 });
    const drop_alloc = xfm.ssm_group.allocator;
    xfm.ssm_group.allocator = failing.allocator();
    xfm.ssmGroupDrop(slots[0].entries);
    xfm.ssm_group.allocator = drop_alloc;
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(slots[1].entries[0].conv_state, want, s));
}

test "persistent group: a full-width group repacks when a member leaves" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();
    const kinds = try ssmTickKinds(alloc, 0);
    defer alloc.free(kinds);
    var slots: [MAX_BATCH_ROWS]SsmTickSlot = undefined;
    var ctxs: [MAX_BATCH_ROWS]*ForwardCtx = undefined;
    for (&slots, 0..) |*sl, i| {
        sl.* = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        sl.bind();
        ctxs[i] = &sl.ctx;
    }
    defer for (&slots) |*sl| sl.deinit(alloc);
    var xfm = ssmTickStubXfm(alloc, s, true);
    defer xfm.ssm_group.deinit();
    try ssmTickBindXfm(&xfm, &ctxs, kinds);
    try ssmTickBindXfm(&xfm, ctxs[1..], kinds);
    try std.testing.expect(xfm.ssm_group.matches(ctxs[1..]));
}

test "persistent group: forwardWith releases a bound member between batched ticks" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();
    const kinds = try ssmTickKinds(alloc, 1);
    defer alloc.free(kinds);
    var p_slots: [2]SsmTickSlot = undefined;
    var o_slots: [2]SsmTickSlot = undefined;
    for (0..2) |i| {
        p_slots[i] = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        p_slots[i].bind();
        o_slots[i] = try SsmTickSlot.init(alloc, s, kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        o_slots[i].bind();
    }
    defer for (&p_slots) |*sl| sl.deinit(alloc);
    defer for (&o_slots) |*sl| sl.deinit(alloc);
    var p_ctxs = [_]*ForwardCtx{ &p_slots[0].ctx, &p_slots[1].ctx };
    var o_ctxs = [_]*ForwardCtx{ &o_slots[0].ctx, &o_slots[1].ctx };
    var p_xfm = ssmTickStubXfm(alloc, s, true);
    defer p_xfm.ssm_group.deinit();
    var o_xfm = ssmTickStubXfm(alloc, s, false);
    defer o_xfm.ssm_group.deinit();
    var t_i: usize = 0;
    while (t_i < 2) : (t_i += 1) {
        try ssmTickBindXfm(&p_xfm, &p_ctxs, kinds);
        try ssmTickBumpMerged(s, p_xfm.ssm_group.layers, kinds);
        try ssmTickBindXfm(&o_xfm, &o_ctxs, kinds);
        try ssmTickBumpSlots(s, &o_ctxs, kinds);
    }
    p_xfm.mlx_stream_ext = null;
    p_xfm.plugin = null;
    p_xfm.moe_layers = null;
    p_xfm.emb_w = .{ .ctx = null };
    p_xfm.config = std.mem.zeroes(ModelConfig);
    const tok_shape = [_]c_int{ 1, 1 };
    var tok_id: i32 = 1;
    const tok = mlx.mlx_array_new_data(&tok_id, &tok_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(tok);
    try std.testing.expectError(error.PluginDeclinedNoLegacy, p_xfm.forwardWith(&p_slots[0].ctx, tok));
    try std.testing.expect(p_slots[0].entries[0].group == null);
    try ssmTickBumpSlots(s, p_ctxs[0..1], kinds);
    try ssmTickBumpSlots(s, o_ctxs[0..1], kinds);
    try ssmTickBindXfm(&p_xfm, &p_ctxs, kinds);
    try ssmTickBindXfm(&o_xfm, &o_ctxs, kinds);
    for (kinds, 0..) |kind, l| {
        try ssmTickCompareRow(s, &p_slots[0].entries[l], &o_slots[0].entries[l], kind);
        try ssmTickCompareRow(s, &p_slots[1].entries[l], &o_slots[1].entries[l], kind);
    }
}

test "persistent group: a row remap between bind and rollback matches the old path" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(alloc, 1);
    defer cache.deinit();
    var kinds = [_]SsmMergeKind{.{ .linear = true, .ple = false }};
    var p_slots: [2]SsmTickSlot = undefined;
    var o_slots: [2]SsmTickSlot = undefined;
    for (0..2) |i| {
        p_slots[i] = try SsmTickSlot.init(alloc, s, &kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        p_slots[i].bind();
        o_slots[i] = try SsmTickSlot.init(alloc, s, &kinds, &cache, @as(f32, @floatFromInt(i + 1)));
        o_slots[i].bind();
    }
    defer for (&p_slots) |*sl| sl.deinit(alloc);
    defer for (&o_slots) |*sl| sl.deinit(alloc);
    var p_ctxs = [_]*ForwardCtx{ &p_slots[0].ctx, &p_slots[1].ctx };
    var o_ctxs = [_]*ForwardCtx{ &o_slots[0].ctx, &o_slots[1].ctx };
    var p_xfm = ssmTickStubXfm(alloc, s, true);
    defer p_xfm.ssm_group.deinit();
    var o_xfm = ssmTickStubXfm(alloc, s, false);
    defer o_xfm.ssm_group.deinit();
    try ssmTickBindXfm(&p_xfm, &p_ctxs, &kinds);
    try ssmTickBumpMerged(s, p_xfm.ssm_group.layers, &kinds);
    try ssmTickBindXfm(&o_xfm, &o_ctxs, &kinds);
    try ssmTickBumpSlots(s, &o_ctxs, &kinds);
    p_ctxs = [_]*ForwardCtx{ &p_slots[1].ctx, &p_slots[0].ctx };
    o_ctxs = [_]*ForwardCtx{ &o_slots[1].ctx, &o_slots[0].ctx };
    try ssmTickBindXfm(&p_xfm, &p_ctxs, &kinds);
    try ssmTickBindXfm(&o_xfm, &o_ctxs, &kinds);
    const t_len: c_int = 3;
    const km1: c_int = 2;
    const width: c_int = 2;
    const n: f64 = @floatFromInt((km1 + t_len) * width);
    const shape = [_]c_int{ 1, km1 + t_len, width };
    for ([_]*SSMCacheEntry{ &p_slots[0].entries[0], &o_slots[0].entries[0] }) |e| {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_arange(&flat, 0.0, n, 1.0, .float32, s));
        e.spec_conv_input = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&e.spec_conv_input, flat, &shape, 3, s));
    }
    try ssmRollbackFromCapture(&p_slots[0].entries[0], 1, @intCast(t_len), s);
    try ssmRollbackFromCapture(&o_slots[0].entries[0], 1, @intCast(t_len), s);
    try ssmTickCompareRow(s, &p_slots[0].entries[0], &o_slots[0].entries[0], kinds[0]);
    try ssmTickCompareRow(s, &p_slots[1].entries[0], &o_slots[1].entries[0], kinds[0]);
}

/// Concatenate every slot's `[1, …]` GDN state for one layer into a single
/// `[N, …]` entry. Both halves are concatenated on axis 0 — the batch axis
/// the recurrence kernel already indexes with `b_idx`.
pub fn mergeSsmAcrossSlots(self: *Transformer, ctxs: []const *ForwardCtx, layer: usize) !SSMCacheEntry {
    const lw = &self.moe_layers.?[layer];
    return ssmMergeLayer(self.s, ctxs, layer, .{ .linear = lw.is_linear, .ple = lw.ple != null });
}

/// Inverse of `mergeSsmAcrossSlots`: hand row i of the advanced state back
/// to slot i, replacing what it held.
///
/// The rows are handed over as SLICE VIEWS, not copies. mlx arrays are
/// refcounted, so each view keeps the merged parent alive and the slot
/// owns a perfectly valid state array — while a copy would make
/// was the single most expensive thing on the batched path: the GDN state
/// is ~2 MB per layer per slot, so materializing it here cost a full extra
/// read+write of the whole trunk state (hundreds of MB) every step, for
/// nothing. Next tick's merge concatenates the views straight back.
///
/// Safe because the one consumer that must outlive the parent — the
/// prefix-cache checkpoint (`captureSsmCheckpoint`) — already takes
/// `materializedOwnedCopy` + a batched eval of whatever it is handed, so
/// no view escapes into storage. Do NOT read that guarantee off the
/// similarly-named `ssmSnapshot`: the PLD-rollback snapshot deliberately
/// shares the handle (transient, per-decode-step), so a snapshot taken
/// while these views are live pins the whole merged [N, …] parent until
/// the rollback window closes. Bounded, but it is the same retention
/// shape as the 3.4x hot-cache under-count.
pub fn splitSsmToSlots(self: *Transformer, m: *const SSMCacheEntry, ctxs: []const *ForwardCtx, layer: usize) !void {
    const lw = &self.moe_layers.?[layer];
    try ssmSplitLayer(self.s, m, ctxs, layer, .{ .linear = lw.is_linear, .ple = lw.ple != null });
}

pub fn ssmBindBatched(self: *Transformer, ctxs: []const *ForwardCtx, kinds: []const SsmMergeKind) !void {
    const persist = self.persistent_group_state_override orelse true;
    const bind = try ssmBindTickState(self.allocator, self.s, &self.ssm_group, ctxs, kinds, persist);
    if (bind.owned) {
        defer {
            for (bind.merged) |*m| ssmFreeMergedLayer(m);
            self.allocator.free(bind.merged);
        }
        try ssmSplitAll(self.s, bind.merged, ctxs, kinds);
    }
}
