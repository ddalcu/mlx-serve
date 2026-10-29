//! mlx/dtype.
//! Owns DtypeCastCache, constTableAs.

const std = @import("std");
const mlx = @import("ffi.zig");

/// One narrowed copy of a load-time constant table, keyed by dtype.
pub const DtypeCastCache = struct {
    arr: mlx.mlx_array = .{ .ctx = null },
    dtype: mlx.mlx_dtype = .float32,

    pub fn deinit(self: *DtypeCastCache) void {
        if (self.arr.ctx != null) _ = mlx.mlx_array_free(self.arr);
        self.arr = .{ .ctx = null };
    }
};

/// Hand back `table` in the dtype the activation it scales is in, building and
/// caching one narrowed copy on first use. Returns a BORROWED array — `cache`
/// owns it.
///
/// A constant table built f32 at load and multiplied straight into a bf16
/// activation promotes the PRODUCT to f32 (mlx's promotion rule). Where that
/// product feeds the residual stream, every projection and expert weight for
/// the rest of the forward is upcast on read — a ~3x token cost that shows up
/// in no op count, no kernel choice, and no pre-loop dtype log. Same class as
/// the MageFlow `scalarLike` rule.
pub fn constTableAs(table: mlx.mlx_array, want: mlx.mlx_dtype, cache: *DtypeCastCache, s: mlx.mlx_stream) !mlx.mlx_array {
    if (mlx.mlx_array_dtype(table) == want) return table;
    if (cache.arr.ctx != null and cache.dtype == want) return cache.arr;
    cache.deinit();
    var cast = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cast);
    try mlx.check(mlx.mlx_astype(&cast, table, want, s));
    try mlx.check(mlx.mlx_array_eval(cast));
    cache.arr = cast;
    cache.dtype = want;
    return cast;
}

// ── tests ──

// ── Constant-table dtype class guard ─────────────────────────────────────────
// A constant lookup table built at load in f32 and multiplied into a bf16
// activation PROMOTES the product to f32 (mlx's promotion rule). On Laguna that
// product is post-RoPE q/k, so the f32 travelled through SDPA, o_proj and the
// residual — every projection and expert weight for the REST of the forward was
// upcast on read, at 3x the token cost, with nothing in the op count, the
// kernel choice or a pre-loop dtype log to show for it.

test "constTableAs hands a load-time f32 table back in the activation dtype" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const vals = [_]f32{ 1.3465735, 1.3465735, 1.0, 1.0 };
    const shape = [_]c_int{4};
    const table = mlx.mlx_array_new_data(&vals, &shape, 1, .float32);
    defer _ = mlx.mlx_array_free(table);

    var cache: DtypeCastCache = .{};
    defer cache.deinit();

    // Same dtype in, same ARRAY out — no copy, no cache entry.
    const same = try constTableAs(table, .float32, &cache, s);
    try std.testing.expect(same.ctx == table.ctx);
    try std.testing.expect(cache.arr.ctx == null);

    // Narrower activation: the table comes back narrowed, and is cached.
    const bf = try constTableAs(table, .bfloat16, &cache, s);
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(bf));
    const again = try constTableAs(table, .bfloat16, &cache, s);
    try std.testing.expect(again.ctx == bf.ctx); // cached, not rebuilt
}

test "a constant table must not widen the activation it scales" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const mvals = [_]f32{ 1.3465735, 1.3465735, 1.0, 1.0 };
    const mshape = [_]c_int{4};
    const table = mlx.mlx_array_new_data(&mvals, &mshape, 1, .float32);
    defer _ = mlx.mlx_array_free(table);

    // A bf16 activation, exactly as post-RoPE q/k arrives.
    const avals = [_]f32{ 0.5, 0.25, 0.125, 0.0625 };
    const ashape = [_]c_int{4};
    const a32 = mlx.mlx_array_new_data(&avals, &ashape, 1, .float32);
    defer _ = mlx.mlx_array_free(a32);
    var act = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act);
    try mlx.check(mlx.mlx_astype(&act, a32, .bfloat16, s));

    // The raw f32 table is what the bug shipped: the product widens.
    var widened = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(widened);
    try mlx.check(mlx.mlx_multiply(&widened, act, table, s));
    try std.testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(widened));

    // Routed through constTableAs it stays in the activation's dtype.
    var cache: DtypeCastCache = .{};
    defer cache.deinit();
    const matched = try constTableAs(table, mlx.mlx_array_dtype(act), &cache, s);
    var kept = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kept);
    try mlx.check(mlx.mlx_multiply(&kept, act, matched, s));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(kept));
}

test "dtypeStep reports a residual change once, in the direction it moved" {
    // Pure decision layer of [dtype-trace]. The mscale bug presented as a
    // UNIFORM 3x with an identical op count, so the only cheap way to find the
    // next one is to watch the residual itself — and a watcher that reports
    // every layer is unreadable while one that reports only the endpoints
    // can't say WHERE it happened.
    try std.testing.expectEqual(mlx.DtypeStep.quiet, mlx.dtypeStep(.bfloat16, .bfloat16));
    try std.testing.expectEqual(mlx.DtypeStep.widened, mlx.dtypeStep(.bfloat16, .float32));
    try std.testing.expectEqual(mlx.DtypeStep.narrowed, mlx.dtypeStep(.float32, .bfloat16));
    // Equal width but a different dtype still changes kernel selection.
    try std.testing.expectEqual(mlx.DtypeStep.changed, mlx.dtypeStep(.bfloat16, .float16));

    // Walk a stack that widens at layer 7 and stays wide: exactly one report,
    // naming layer 7. A per-layer dump would emit 40 lines; endpoints alone
    // would say "it widened" without saying where.
    const stack = [_]mlx.mlx_dtype{
        .bfloat16, .bfloat16, .bfloat16, .bfloat16, .bfloat16, .bfloat16, .bfloat16,
        .float32,  .float32,  .float32,  .float32,
    };
    var prev = stack[0];
    var reports: usize = 0;
    var first_at: usize = 0;
    for (stack[1..], 1..) |cur, idx| {
        if (mlx.dtypeStep(prev, cur) != .quiet) {
            if (reports == 0) first_at = idx;
            reports += 1;
        }
        prev = cur;
    }
    try std.testing.expectEqual(@as(usize, 1), reports);
    try std.testing.expectEqual(@as(usize, 7), first_at);
}

test "dtypeTraceArm latches one trace per forward path" {
    // The trace must fire on the FIRST forward of each path and then cost
    // nothing — a per-layer FFI dtype read on every token would be a real
    // (small) tax, and repeated identical lines would bury the one that
    // matters. Distinct paths latch independently so a vision tower and its
    // text trunk both get traced in the same boot.
    mlx.resetDtypeTraceForTest();
    try std.testing.expect(mlx.dtypeTraceArm("moe"));
    try std.testing.expect(!mlx.dtypeTraceArm("moe"));
    try std.testing.expect(mlx.dtypeTraceArm("vision"));
    try std.testing.expect(!mlx.dtypeTraceArm("vision"));
    try std.testing.expect(!mlx.dtypeTraceArm("moe"));
}

