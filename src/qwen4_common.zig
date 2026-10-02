const std = @import("std");
pub const MAX_HEADS = 32;
pub const MAX_NGRAM_SIZE = 8;
pub var live_warm_bytes = std.atomic.Value(u64).init(0);
pub var live_warm_total = std.atomic.Value(u64).init(0);
