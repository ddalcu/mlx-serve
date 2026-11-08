//! A wide prompt layer's attention projections' large outputs (its attention norm's rows, `wq_b`'s heads, their roped tail: [L, 5120],
//! [L, 64, 512] and [L, 64, 64] at L rows, 1.4 GB at 16K) kept past the layer's teardown by an extra reference and handed
//! to MLX's buffer cache at the next wide layer's start, so that layer's same arrays take those buffers back instead of
//! new Metal allocations.
//!
//! Without it the layer's scope frees them at its end. MLX recycles a freed buffer only while its cache is under the limit, and
//! at a wide layer's end the cache already holds the routed call's freed transients, so the projections' buffers go back
//! to Metal and the next layer allocates them anew when it encodes them (~15 ms of host a GB). The attention stage is
//! host-bound (177 ms of host a layer for 158 ms of GPU), and the encode of `wq_b`'s 1.07 GB output takes 19.2 ms of
//! host a regular wide layer with the GPU idle 5.1 ms of it.
//!
//! So a wide layer with another wide one after it in the pass (`layerEnd`, at its end) takes a reference to each of those
//! arrays of at least `min_bytes` (one per buffer: `wq_b`'s heads are a view of its output). The next wide layer, at its
//! start (`release`), lifts MLX's cache limit, frees the references (each buffer goes to the cache, the newest there),
//! and puts the limit back at once: MLX's next allocation trims the cache to the limit from its oldest buffers, and the
//! layer's same-shaped arrays (the same shapes every wide layer) take these (MLX hands out a cached buffer within two
//! pages of the size asked). The same kernels over the same inputs: only which buffer an output lands in moves, so the
//! same words (a text gate; MLX's cache hands out used buffers already).
//!
//! Memory: the kept arrays were alive through the layer's routed call either way (its scope held them until its end), so
//! from the teardown to the next layer's start they add only to the tail's in-flight buffers, which the routed call's own
//! encode already held beside them. At the next layer's start the cache is over its limit by their bytes until that
//! layer's first allocation trims it (the layer's graph build, no allocations, the footprint's low point), and the limit
//! itself never moves past an instant: no other freed buffer is cached beyond the limit.
//!
//! Measured (a1 b1 b2 a2): cold 16K +0.63%, new text +1.12%, the texts the same bytes, the bank rows and the guard's
//! peak unchanged.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");

const A = mlx.mlx_array;

/// The smallest array kept: a fresh allocation of it costs the host ~1 ms (MLX's new Metal buffer, ~15 ms a GB).
pub const min_bytes: u64 = 64 << 20;
/// Arrays a layer keeps at most (its attention norm's rows, `wq_b`'s heads, their roped tail).
pub const max_kept = 4;

/// The references a layer keeps (`layerEnd`) until the next wide layer's `release`.
/// This file's process state, one copy across `-Dunits` units.
const State = struct {
    kept: [max_kept]A = undefined,
    n_kept: usize = 0,
};
pub const shared_state = @import("../../../sdk/util/shared.zig").Shared(State, "dsv41_transient_keep", .{});
pub inline fn ps() *State {
    return shared_state.get();
}

fn bytesOf(x: A) u64 {
    return @as(u64, mlx.mlx_array_size(x)) * @as(u64, mlx.mlx_array_itemsize(x));
}

/// The buffer's data pointer (0 unevaluated).
fn dataOf(x: A) usize {
    const p = mlx.mlx_array_data_uint8(x) orelse return 0;
    return @intFromPtr(p);
}

/// A wide layer's end (its arrays evaluated by the routed call, still held by its scope). `keep` (the next layer of the
/// pass wide too): a reference to each of `xs` of at least `min_bytes`, one per buffer, for that layer's `release`.
pub fn layerEnd(xs: []const A, keep: bool) void {
    drop();
    if (!keep) return;
    for (xs) |x| {
        if (x.ctx == null or ps().n_kept == max_kept or bytesOf(x) < min_bytes) continue;
        const d = dataOf(x);
        // One reference a buffer (a view of a kept array's buffer at its start is the same buffer).
        if (d != 0 and (for (ps().kept[0..ps().n_kept]) |k| {
            if (dataOf(k) == d) break true;
        } else false)) continue;
        var h = mlx.mlx_array_new();
        if (mlx.mlx_array_set(&h, x) != 0) {
            _ = mlx.mlx_array_free(h);
            continue;
        }
        ps().kept[ps().n_kept] = h;
        ps().n_kept += 1;
    }
}

/// The next wide layer's start: the kept buffers to MLX's cache (the limit lifted while they are freed, then put back),
/// for this layer's arrays of their sizes.
pub fn release() void {
    if (ps().n_kept == 0) return;
    var prev: usize = 0;
    const lifted = mlx.mlx_set_cache_limit(&prev, std.math.maxInt(usize)) == 0;
    // `wq_b`'s heads (the largest, kept first) freed last: the newest in the cache, the last a trim would take.
    var i = ps().n_kept;
    while (i > 0) {
        i -= 1;
        _ = mlx.mlx_array_free(ps().kept[i]);
    }
    if (lifted) {
        var tmp: usize = 0;
        _ = mlx.mlx_set_cache_limit(&tmp, prev);
    }
    ps().n_kept = 0;
}

/// A pass's end (or a kept set no layer took): the references let go as the layer's scope would have.
pub fn drop() void {
    for (ps().kept[0..ps().n_kept]) |k| _ = mlx.mlx_array_free(k);
    ps().n_kept = 0;
}
