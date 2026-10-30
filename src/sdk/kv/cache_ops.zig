//! `CacheOps` over a `KVCache` and one recurrent entry per layer: a pure adapter that makes the same calls the legacy
//! forward paths make directly. A `Handle` points at a `Slot`. `Entry` is the engine's `SSMCacheEntry`, or `Entry`
//! below for a plugin that drives a trunk itself (`bases/trunk.zig`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const abi = @import("../../abi/plugin.zig");
const KVCache = @import("cache.zig").KVCache;
const DenseKVView = @import("../blocks/attention/kv_view.zig").DenseKVView;
const MAX_BATCH_ROWS = @import("../bases/run.zig").MAX_BATCH_ROWS;
const Prework = @import("../blocks/gdn/decode.zig").Prework;

/// One layer's recurrent state, as the engine's `SSMCacheEntry` holds it for every arch but qwen4_exp.
pub const Entry = struct {
    conv_state: mlx.mlx_array = .{ .ctx = null },
    ssm_state: mlx.mlx_array = .{ .ctx = null },
    initialized: bool = false,
    spec_state_seq: mlx.mlx_array = .{ .ctx = null },
    spec_conv_input: mlx.mlx_array = .{ .ctx = null },
    spec_state_in: mlx.mlx_array = .{ .ctx = null },
    spec_prework: Prework = .{},
};

pub fn For(comptime E: type) type {
    return struct {
        /// What a `Handle` points at: one slot's KV cache and its recurrent entries (one per layer, or none).
        pub const Slot = struct {
            kv: *KVCache,
            ssm: []E = &.{},
            /// `KVCache.appendFromStacked`'s one-quantize arm for a stacked append (the Transformer's
            /// `batched_kv_append_override orelse true`).
            batched_append: bool = true,
            /// The request's `ForwardCtx.kv_attn_fused`, handed to attention on every view.
            packed_reads: bool = false,
            /// Set on every slot of a batched tick whose engine keeps the batch's recurrent state merged across ticks
            /// (one entry per layer, rows in slot order): `state_rows` / `set_state_rows` read and write it whole.
            merged: []E = &.{},

            pub fn handle(self: *Slot) abi.Handle {
                return @ptrCast(self);
            }
        };

        pub const ops: abi.CacheOps = .{
            .len = len,
            .append = append,
            .append_ring = appendRing,
            .view = view,
            .release = release,
            .append_rows = appendRows,
            .state = state,
            .set_state = setState,
            .state_rows = stateRows,
            .set_state_rows = setStateRows,
            .sdk_slots = true,
        };

        const Rows = @import("../util/state_rows.zig").Rows(state, setState);

        fn slotOf(h: abi.Handle) *Slot {
            return @ptrCast(@alignCast(h.?));
        }

        fn mx(a: abi.Array) mlx.mlx_array {
            return .{ .ctx = a.ctx };
        }

        fn ab(a: mlx.mlx_array) abi.Array {
            return .{ .ctx = a.ctx };
        }

        fn failed(e: anyerror) abi.Verdict {
            return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
        }

        fn toAbi(v: DenseKVView, packed_reads: bool) abi.KvView {
            return .{
                .k = ab(v.k),
                .v = ab(v.v),
                .owned = v.owned,
                .packed_reads = packed_reads,
                .bits = if (v.has_quant_triple) v.bits else 0,
                .group_size = v.group_size,
                .capacity = @intCast(v.kv_cap),
                .k_q = ab(v.k_triple_q),
                .k_scales = ab(v.k_triple_scales),
                .k_biases = ab(v.k_triple_biases),
                .v_q = ab(v.v_triple_q),
                .v_scales = ab(v.v_triple_scales),
                .v_biases = ab(v.v_triple_biases),
            };
        }

        fn len(h: abi.Handle, layer: u32) callconv(.c) u64 {
            return slotOf(h).kv.seqLen(layer);
        }

        fn append(h: abi.Handle, layer: u32, k: abi.Array, v: abi.Array, keep: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
            const sl = slotOf(h);
            const r = sl.kv.update(layer, mx(k), mx(v), .{ .ctx = s.ctx }, keep) catch |e| return failed(e);
            out.* = toAbi(r, sl.packed_reads);
            return .ok;
        }

        fn appendRing(h: abi.Handle, layer: u32, k: abi.Array, v: abi.Array, keep: u32, window: u32, ring: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
            const sl = slotOf(h);
            const r = sl.kv.updateSliding(layer, mx(k), mx(v), .{ .ctx = s.ctx }, keep, window, ring) catch |e| return failed(e);
            out.* = toAbi(r, sl.packed_reads);
            return .ok;
        }

        fn view(h: abi.Handle, layer: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
            const sl = slotOf(h);
            const r = sl.kv.denseView(layer, .{ .ctx = s.ctx }) catch |e| return failed(e);
            out.* = toAbi(r, sl.packed_reads);
            return .ok;
        }

        fn release(v: *abi.KvView) callconv(.c) void {
            if (v.owned) {
                _ = mlx.mlx_array_free(mx(v.k));
                _ = mlx.mlx_array_free(mx(v.v));
            }
            v.* = .{};
        }

        fn appendRows(caches: [*]const abi.Handle, n: u32, layer: u32, k: abi.Array, v: abi.Array, keep: u32, s: abi.Stream) callconv(.c) abi.Verdict {
            if (n == 0) return .ok;
            if (n > MAX_BATCH_ROWS) return abi.Verdict.refuse(.too_large, "more batched rows than MAX_BATCH_ROWS");
            var kvs: [MAX_BATCH_ROWS]*KVCache = undefined;
            for (caches[0..n], 0..) |h, i| kvs[i] = slotOf(h).kv;
            KVCache.appendFromStacked(mx(k), mx(v), kvs[0..n], layer, .{ .ctx = s.ctx }, keep, slotOf(caches[0]).batched_append) catch |e| return failed(e);
            return .ok;
        }

        fn field(e: *E, which: abi.StateSlot) *mlx.mlx_array {
            return switch (which) {
                .conv => &e.conv_state,
                .ssm => &e.ssm_state,
                .conv_input => &e.spec_conv_input,
                .state_seq => &e.spec_state_seq,
                .state_in => &e.spec_state_in,
                .prework_k => &e.spec_prework.k,
                .prework_v => &e.spec_prework.v,
                .prework_gb => &e.spec_prework.gb,
            };
        }

        fn state(h: abi.Handle, layer: u32, which: abi.StateSlot) callconv(.c) abi.Array {
            const sl = slotOf(h);
            if (layer >= sl.ssm.len) return .{};
            const e = &sl.ssm[layer];
            if (!e.initialized) return .{};
            return ab(field(e, which).*);
        }

        fn put(e: *E, which: abi.StateSlot, value: abi.Array) void {
            const dst = field(e, which);
            var r: mlx.mlx_array = .{ .ctx = null };
            if (value.ctx != null) {
                r = mlx.mlx_array_new();
                _ = mlx.mlx_array_set(&r, mx(value));
            }
            if (dst.ctx != null) _ = mlx.mlx_array_free(dst.*);
            dst.* = r;
            if (which == .conv or which == .ssm) e.initialized = true;
        }

        fn setState(h: abi.Handle, layer: u32, which: abi.StateSlot, value: abi.Array) callconv(.c) abi.Verdict {
            const sl = slotOf(h);
            if (layer >= sl.ssm.len) return abi.Verdict.refuse(.unsupported, "layer has no recurrent state");
            put(&sl.ssm[layer], which, value);
            return .ok;
        }

        /// The merged entry's array whole (a new reference) when the batch is merged, else the rows concatenated.
        fn stateRows(caches: [*]const abi.Handle, n: u32, layer: u32, which: abi.StateSlot, s: abi.Stream) callconv(.c) abi.Array {
            const sl = slotOf(caches[0]);
            if (layer >= sl.merged.len) return Rows.stateRows(caches, n, layer, which, s);
            const e = &sl.merged[layer];
            const a = field(e, which).*;
            if (!e.initialized or a.ctx == null) return .{};
            var r = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&r, a);
            return .{ .ctx = r.ctx };
        }

        fn setStateRows(caches: [*]const abi.Handle, n: u32, layer: u32, which: abi.StateSlot, value: abi.Array, s: abi.Stream) callconv(.c) abi.Verdict {
            const sl = slotOf(caches[0]);
            if (layer >= sl.merged.len) return Rows.setStateRows(caches, n, layer, which, value, s);
            put(&sl.merged[layer], which, value);
            return .ok;
        }
    };
}
