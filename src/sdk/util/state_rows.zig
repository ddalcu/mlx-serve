//! `CacheOps.state_rows` / `set_state_rows` over a cache's per-slot `state` / `set_state`: the rows concatenated on
//! the batch axis, and split back one cache at a time. For a `CacheOps` that keeps no merged group.

const mlx = @import("../../mlx/ffi.zig");
const abi = @import("../../abi/plugin.zig");

/// Caches one call takes (the engine's `MAX_BATCH_ROWS`).
pub const MAX_ROWS = 32;

/// The batch axis of `slot`'s arrays: the per-row state sequence is `[S, B, ...]`.
pub fn axis(slot: abi.StateSlot) c_int {
    return if (slot == .state_seq) 1 else 0;
}

pub fn Rows(
    comptime get: fn (abi.Handle, u32, abi.StateSlot) callconv(.c) abi.Array,
    comptime set: fn (abi.Handle, u32, abi.StateSlot, abi.Array) callconv(.c) abi.Verdict,
) type {
    return struct {
        pub fn stateRows(caches: [*]const abi.Handle, n: u32, layer: u32, slot: abi.StateSlot, s: abi.Stream) callconv(.c) abi.Array {
            if (n == 0 or n > MAX_ROWS) return .{};
            var parts: [MAX_ROWS]mlx.mlx_array = undefined;
            for (caches[0..n], 0..) |h, i| {
                const a = get(h, layer, slot);
                if (a.ctx == null) return .{};
                parts[i] = .{ .ctx = a.ctx };
            }
            const v = mlx.mlx_vector_array_new_data(&parts, n);
            defer _ = mlx.mlx_vector_array_free(v);
            var out = mlx.mlx_array_new();
            if (mlx.mlx_concatenate_axis(&out, v, axis(slot), .{ .ctx = s.ctx }) != 0) {
                _ = mlx.mlx_array_free(out);
                return .{};
            }
            return .{ .ctx = out.ctx };
        }

        pub fn setStateRows(caches: [*]const abi.Handle, n: u32, layer: u32, slot: abi.StateSlot, value: abi.Array, s: abi.Stream) callconv(.c) abi.Verdict {
            const x: mlx.mlx_array = .{ .ctx = value.ctx };
            const sh = mlx.getShape(x);
            if (sh.len > 8) return abi.Verdict.refuse(.failed, "state rank above 8");
            const ax: usize = @intCast(axis(slot));
            var start: [8]c_int = @splat(0);
            var stop: [8]c_int = @splat(0);
            const strides: [8]c_int = @splat(1);
            for (sh, 0..) |d, i| stop[i] = d;
            for (caches[0..n], 0..) |h, i| {
                start[ax] = @intCast(i);
                stop[ax] = @intCast(i + 1);
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                if (mlx.mlx_slice(&row, x, &start, sh.len, &stop, sh.len, &strides, sh.len, .{ .ctx = s.ctx }) != 0)
                    return abi.Verdict.refuse(.failed, "state row slice");
                const v = set(h, layer, slot, .{ .ctx = row.ctx });
                if (!v.isOk()) return v;
            }
            return .ok;
        }
    };
}
