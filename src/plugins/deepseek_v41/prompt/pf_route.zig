//! A prompt call's routed pairs sorted by expert as a counting sort (`MLX_SERVE_DSV41_PF_MOE_EPI`): the SDK's prefill
//! route pack (histogram, scan, scatter over 256-pair blocks, `verify_kernels.moePrefillRoutePack`) gives each pair's
//! place, the sorted experts and their rows in three launches, and one scatter puts the routing weights in that order.
//! `moePrefill` otherwise makes them with two argsorts (a merge sort of about a dozen narrow launches each), a divide
//! and two takes. Ties keep pair order, as MLX's argsort does, so the outputs are the same words.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const vk = @import("../../../sdk/blocks/moe/verify_kernels.zig");
const dsv41 = @import("../deepseek_v41.zig");

const A = mlx.mlx_array;

/// The pairs in expert order: `inv[p]` is pair p's place, `rows` each place's token row, `sidx` its expert, `w` its
/// f32 routing weight.
pub const Sorted = struct { inv: A, rows: A, sidx: A, w: A };

var engaged_logged: std.atomic.Value(bool) = .init(false);

/// `flat` `[n]` uint32 expert ids of `n / k` rows (`k` a row), `wts` `[n]` f32, ids under `experts`; null where the
/// switch is off or the pack declines (decode widths, more than 1024 experts).
pub fn sorted(sc: *dsv41.Scope, flat: A, wts: A, k: usize, experts: usize) !?Sorted {
    if (!env.on(.MLX_SERVE_DSV41_PF_MOE_EPI)) return null;
    return try pack(sc, flat, wts, k, experts);
}

/// `sorted` with the switch unread.
pub fn pack(sc: *dsv41.Scope, flat: A, wts: A, k: usize, experts: usize) !?Sorted {
    var p = (try vk.moePrefillRoutePack(sc.s, flat, @intCast(k), @intCast(experts))) orelse return null;
    const inv = sc.keep(p.inverse) catch |e| {
        p.inverse = .{};
        p.deinit();
        return e;
    };
    p.inverse = .{};
    const sidx = sc.keep(p.sorted) catch |e| {
        p.sorted = .{};
        p.deinit();
        return e;
    };
    p.sorted = .{};
    const rows = try sc.keep(p.lhs);
    const n: c_int = @intCast(mlx.mlx_array_size(flat));
    // Each pair's weight to its place: `inv` is a permutation, so every place is written once.
    var o = mlx.mlx_array_new();
    const w = try sc.res(mlx.mlx_put_along_axis(&o, try sc.zeros(&.{n}, .float32), inv, wts, 0, sc.s), &o);
    if (!engaged_logged.swap(true, .monotonic)) log.info("[dsv41] MLX_SERVE_DSV41_PF_MOE_EPI engaged (pairs {d}, experts {d})\n", .{ n, experts });
    return .{ .inv = inv, .rows = rows, .sidx = sidx, .w = w };
}
