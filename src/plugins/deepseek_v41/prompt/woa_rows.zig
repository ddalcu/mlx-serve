//! A wide prompt layer's grouped `wo_a` (an EXL3 pack's, turboexl3's `linearGrouped`) over runs of its attention spans'
//! rows, not span by span.
//!
//! Span by span, the model's `wideLayer` would run `wo_a` on each 256-row span's grouped heads as the span's
//! attention ends. At prefill rows turboexl3's grouped linear runs `linear` on each of the 8 slices, and `linear` decodes the slice's whole
//! trellis to W_hat and multiplies at M = the span's rows. A 16K-row layer so decodes every slice 64 times and runs 512
//! GEMMs at M = 256, each slice's prepare and finish on its own launches.
//!
//! Here the spans' grouped heads are joined in runs of `rows` rows (`spansPer` spans), one grouped linear a run: each slice
//! decoded once a run, its GEMM at M = the run's rows. The prepare (row by row), the decode (the same trellis) and the
//! finish (row by row) give the same words; only MLX's GEMM is handed another M, which rounds a row's sums in another
//! order: on the full pack every layer's runs differ from the span-by-span words in about 2% of them, by at most one bf16
//! ulp (7.8e-3; w-pio-20261101-0624), a reorder-level change.
//!
//! A run holds its spans' grouped heads, their join and the grouped linear's copy of them (about 0.8 GB at 4096 rows),
//! within the prompt bill's term for the chunk's grouped heads (`prefillBytes`: the heads for every row of the chunk).

const std = @import("std");

/// Rows a run joins: a machine constant (where the M5 Max's matrix units run the slices' GEMMs at full rate), never a
/// prompt's length.
pub const rows: usize = 4096;

/// Spans a run joins at `span` rows a span: `rows` over it, at least one.
pub fn spansPer(span: usize) usize {
    return @max(1, rows / @max(span, 1));
}

test "woa_rows runs: 16 spans of 256 rows, one span past the run's rows, never zero" {
    try std.testing.expectEqual(@as(usize, 16), spansPer(256));
    try std.testing.expectEqual(@as(usize, 32), spansPer(128));
    try std.testing.expectEqual(@as(usize, 1), spansPer(rows));
    try std.testing.expectEqual(@as(usize, 1), spansPer(2 * rows));
    try std.testing.expectEqual(rows, spansPer(1));
    try std.testing.expectEqual(rows, spansPer(0));
}
