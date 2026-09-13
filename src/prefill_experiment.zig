//! Opt-in same-process research screening; never changes request contents.
const std = @import("std");
const log = @import("log.zig");
threadlocal var cursor: usize = 0;
threadlocal var arm: ?u8 = null;
fn digit(c: u8) ?u8 {
    return switch (c) {
        '0'...'9' => c - '0',
        'a'...'f' => c - 'a' + 10,
        'A'...'F' => c - 'A' + 10,
        else => null,
    };
}

pub fn begin(tokens: usize) void {
    arm = null;
    if (tokens < 2048) return;
    const raw = std.c.getenv("QWEN4_PREFILL_ARM_SEQUENCE") orelse return;
    const sequence = std.mem.sliceTo(raw, 0);
    if (cursor >= sequence.len) return;
    for (sequence) |c| if (digit(c) == null) return;
    arm = digit(sequence[cursor]).?;
    log.info("[prefill-experiment] request={d} arm={d} group={} ple={} hc_upmix={} moe_mpp={} same_process=true\n", .{ cursor, arm.?, arm.? & 9 != 0, arm.? & 2 != 0, arm.? & 4 != 0, arm.? & 8 != 0 });
    cursor += 1;
}

pub fn group() ?bool {
    return if (arm) |v| v & 9 != 0 else null;
}
pub fn ple() ?bool {
    return if (arm) |v| v & 2 != 0 else null;
}

pub fn hcUpmix() ?bool {
    return if (arm) |v| v & 4 != 0 else null;
}
pub fn moeMpp() ?bool {
    return if (arm) |v| v & 8 != 0 else null;
}
