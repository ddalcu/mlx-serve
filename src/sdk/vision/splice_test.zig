//! `splice.zig`'s tests, out of its import graph: a plugin unit that reaches `splice.zig` never compiles them.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;

// ── Chunked vision splice (issue #197) ───────────────────────────────────────
// A vision prompt prefills in chunks; the splice's row lookup is a cumsum over
// the CURRENT forward's placeholder mask, so each chunk passes the rows earlier
// chunks consumed as `row_offset` — a restarted scatter silently reuses the
// first image rows in every chunk (coherent output, wrong image content).

test "spliceVisionRows: two chunks with row offsets equal the whole-prompt splice" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    const hidden: c_int = 4;
    const seq: c_int = 6;
    const img_id: u32 = 99;
    // token ids: [7, 99, 99, 8, 99, 9] — placeholders at 1, 2 (chunk A) and 4
    // (chunk B), so the boundary lands INSIDE the image row sequence.
    const ids = [_]i32{ 7, 99, 99, 8, 99, 9 };
    // h rows: row i filled with i+1 (bf16-exact values).
    var h_data: [6 * 4]f32 = undefined;
    for (0..6) |r| for (0..4) |c| {
        h_data[r * 4 + c] = @floatFromInt(r + 1);
    };
    // vision rows: row k filled with 100+k — distinct per row, so a restarted
    // scatter (row 0 reused in chunk B) produces different bytes.
    var ve_data: [3 * 4]f32 = undefined;
    for (0..3) |r| for (0..4) |c| {
        ve_data[r * 4 + c] = @floatFromInt(100 + r);
    };
    const ve_shape = [_]c_int{ 1, 3, hidden };
    const ve = mlx.mlx_array_new_data(&ve_data, &ve_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(ve);

    const readRow = struct {
        fn f(arr: mlx.mlx_array, row: usize, out: *[4]f32, st: mlx.mlx_stream) !void {
            var f32arr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32arr);
            try mlx.check(mlx.mlx_astype(&f32arr, arr, .float32, st));
            try mlx.check(mlx.mlx_array_eval(f32arr));
            const ptr = mlx.mlx_array_data_float32(f32arr).?;
            for (0..4) |c| out[c] = ptr[row * 4 + c];
        }
    }.f;

    // Whole-prompt splice (the reference: what the unchunked path computes).
    var full_rows: [6][4]f32 = undefined;
    {
        const h_shape = [_]c_int{ 1, seq, hidden };
        const h = mlx.mlx_array_new_data(&h_data, &h_shape, 3, .float32);
        const ids_shape = [_]c_int{ 1, seq };
        const tid = mlx.mlx_array_new_data(&ids, &ids_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tid);
        const spliced = try Transformer.spliceVisionRows(h, tid, ve, img_id, 0, 0, 0, s);
        defer _ = mlx.mlx_array_free(spliced);
        for (0..6) |r| try readRow(spliced, r, &full_rows[r], s);
    }

    // Chunk A = tokens [0..3), offset 0; chunk B = tokens [3..6), offset 2
    // (chunk A consumed 2 placeholder rows).
    const chunks = [_]struct { start: usize, len: c_int, off: usize }{
        .{ .start = 0, .len = 3, .off = 0 },
        .{ .start = 3, .len = 3, .off = 2 },
    };
    for (chunks) |ck| {
        const h_shape = [_]c_int{ 1, ck.len, hidden };
        const h = mlx.mlx_array_new_data(h_data[ck.start * 4 ..].ptr, &h_shape, 3, .float32);
        const ids_shape = [_]c_int{ 1, ck.len };
        const tid = mlx.mlx_array_new_data(ids[ck.start..].ptr, &ids_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tid);
        const spliced = try Transformer.spliceVisionRows(h, tid, ve, img_id, 0, 0, ck.off, s);
        defer _ = mlx.mlx_array_free(spliced);
        for (0..@intCast(ck.len)) |r| {
            var got: [4]f32 = undefined;
            try readRow(spliced, r, &got, s);
            for (0..4) |c| try std.testing.expectEqual(full_rows[ck.start + r][c], got[c]);
        }
    }
}

test "spliceVisionRows: video placeholders are scattered like image rows (Qwen video_token_id)" {
    // Live (qwen4_exp, 4-frame clip): 690 <|video_pad|> tokens were
    // inserted and the tower encoded 690 rows, but the mask keyed on image |
    // audio ids only, so no video row ever reached the prompt and the model
    // described a clip it never saw. Same hole on qwen3_5 (its e2e bar of
    // >= 2 of 4 generic subjects was passing on hallucination).
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const hidden: c_int = 4;
    const img_id: u32 = 99;
    const vid_id: u32 = 77;
    // [99, 7, 77, 77] — one image row, text, two video rows from ONE source.
    const ids = [_]i32{ 99, 7, 77, 77 };
    var h_data: [16]f32 = @splat(0);
    var ve_data: [3 * 4]f32 = undefined;
    for (0..3) |r| for (0..4) |c| {
        ve_data[r * 4 + c] = @floatFromInt(50 + r);
    };
    const ve_shape = [_]c_int{ 1, 3, hidden };
    const ve = mlx.mlx_array_new_data(&ve_data, &ve_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(ve);
    const h_shape = [_]c_int{ 1, 4, hidden };
    const h = mlx.mlx_array_new_data(&h_data, &h_shape, 3, .float32);
    const ids_shape = [_]c_int{ 1, 4 };
    const tid = mlx.mlx_array_new_data(&ids, &ids_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(tid);
    const spliced = try Transformer.spliceVisionRows(h, tid, ve, img_id, 0, vid_id, 0, s);
    defer _ = mlx.mlx_array_free(spliced);
    var f32arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32arr);
    try mlx.check(mlx.mlx_astype(&f32arr, spliced, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f32arr));
    const ptr = mlx.mlx_array_data_float32(f32arr).?;
    try std.testing.expectEqual(@as(f32, 50), ptr[0]); // image row 0
    try std.testing.expectEqual(@as(f32, 0), ptr[4]); // text untouched
    try std.testing.expectEqual(@as(f32, 51), ptr[8]); // video row 1
    try std.testing.expectEqual(@as(f32, 52), ptr[12]); // video row 2
}

test "spliceVisionRows: audio placeholders share the row stream in prompt order" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const hidden: c_int = 4;
    const img_id: u32 = 99;
    const aud_id: u32 = 88;
    // [99, 7, 88] — image row then audio row from ONE concatenated source.
    const ids = [_]i32{ 99, 7, 88 };
    var h_data: [12]f32 = @splat(0);
    var ve_data: [2 * 4]f32 = undefined;
    for (0..2) |r| for (0..4) |c| {
        ve_data[r * 4 + c] = @floatFromInt(50 + r);
    };
    const ve_shape = [_]c_int{ 1, 2, hidden };
    const ve = mlx.mlx_array_new_data(&ve_data, &ve_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(ve);
    const h_shape = [_]c_int{ 1, 3, hidden };
    const h = mlx.mlx_array_new_data(&h_data, &h_shape, 3, .float32);
    const ids_shape = [_]c_int{ 1, 3 };
    const tid = mlx.mlx_array_new_data(&ids, &ids_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(tid);
    const spliced = try Transformer.spliceVisionRows(h, tid, ve, img_id, aud_id, 0, 0, s);
    defer _ = mlx.mlx_array_free(spliced);
    var f32arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32arr);
    try mlx.check(mlx.mlx_astype(&f32arr, spliced, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f32arr));
    const ptr = mlx.mlx_array_data_float32(f32arr).?;
    try std.testing.expectEqual(@as(f32, 50), ptr[0]); // image row 0
    try std.testing.expectEqual(@as(f32, 0), ptr[4]); // text untouched
    try std.testing.expectEqual(@as(f32, 51), ptr[8]); // audio row 1
}

test "spliceVisionRows: an unset placeholder id (0) never claims token 0" {
    // An audio-only model has no image token: token 0 (Qwen's "!") stays text.
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const hidden: c_int = 4;
    const ids = [_]i32{ 0, 88, 7 };
    var h_data: [12]f32 = @splat(0);
    var ve_data: [4]f32 = @splat(50);
    const ve = mlx.mlx_array_new_data(&ve_data, &[_]c_int{ 1, 1, hidden }, 3, .float32);
    defer _ = mlx.mlx_array_free(ve);
    const h = mlx.mlx_array_new_data(&h_data, &[_]c_int{ 1, 3, hidden }, 3, .float32);
    const tid = mlx.mlx_array_new_data(&ids, &[_]c_int{ 1, 3 }, 2, .int32);
    defer _ = mlx.mlx_array_free(tid);
    const spliced = try Transformer.spliceVisionRows(h, tid, ve, 0, 88, 0, 0, s);
    defer _ = mlx.mlx_array_free(spliced);
    var f32arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32arr);
    try mlx.check(mlx.mlx_astype(&f32arr, spliced, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f32arr));
    const ptr = mlx.mlx_array_data_float32(f32arr).?;
    try std.testing.expectEqual(@as(f32, 0), ptr[0]); // token 0 untouched
    try std.testing.expectEqual(@as(f32, 50), ptr[4]); // audio row
}

test "spliceVisionRows: the residual keeps its own dtype (an f16 stream stays f16)" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    for ([_]mlx.mlx_dtype{ .float16, .bfloat16 }) |dt| {
        var h_data: [8]f32 = @splat(1);
        var ve_data: [4]f32 = @splat(50);
        const ve = mlx.mlx_array_new_data(&ve_data, &[_]c_int{ 1, 1, 4 }, 3, .float32);
        defer _ = mlx.mlx_array_free(ve);
        const h32 = mlx.mlx_array_new_data(&h_data, &[_]c_int{ 1, 2, 4 }, 3, .float32);
        defer _ = mlx.mlx_array_free(h32);
        var h = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&h, h32, dt, s));
        const ids = [_]i32{ 9, 250 };
        const tid = mlx.mlx_array_new_data(&ids, &[_]c_int{ 1, 2 }, 2, .int32);
        defer _ = mlx.mlx_array_free(tid);
        const spliced = try Transformer.spliceVisionRows(h, tid, ve, 250, 0, 0, 0, s);
        defer _ = mlx.mlx_array_free(spliced);
        try std.testing.expectEqual(dt, mlx.mlx_array_dtype(spliced));
    }
}
