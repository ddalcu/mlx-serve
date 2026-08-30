const std = @import("std");
const builtin = @import("builtin");
const log = @import("log.zig");

const is_macos = builtin.os.tag == .macos;

// ABI declarations from IOPMLib.h and CoreFoundation.

extern "c" fn IOPMAssertionCreateWithName(
    assertion_type: ?*anyopaque,
    level: u32,
    name: ?*anyopaque,
    out_id: *u32,
) i32;
extern "c" fn IOPMAssertionRelease(id: u32) i32;
extern "c" fn CFStringCreateWithCString(alloc: ?*anyopaque, s: [*:0]const u8, encoding: u32) ?*anyopaque;
extern "c" fn CFRelease(cf: ?*anyopaque) void;

const utf8_encoding: u32 = 0x08000100;
const level_on: u32 = 255;
// PreventSystemSleep is deprecated; this still permits display sleep.
const assertion_type: [*:0]const u8 = "PreventUserIdleSystemSleep";
const assertion_name: [*:0]const u8 = "mlx-serve is generating";

var enabled = true;
var held = false;
var assertion_id: u32 = 0;
// Stop retrying after powerd rejects the assertion.
var broken = false;

pub const Action = enum { none, acquire, release };

/// Returns the required state transition.
pub fn actionFor(want: bool, enabled_: bool, held_: bool, broken_: bool) Action {
    if (want == held_) return .none;
    if (!want) return .release;
    if (!enabled_ or broken_) return .none;
    return .acquire;
}

pub fn setEnabled(on: bool) void {
    enabled = on;
    if (!on) release();
}

pub fn isEnabled() bool {
    return enabled;
}

/// Inference-thread only.
pub fn setActive(want: bool) void {
    if (comptime !is_macos) return;
    switch (actionFor(want, enabled, held, broken)) {
        .none => return,
        .acquire => acquire(),
        .release => release(),
    }
}

/// Idempotent.
pub fn release() void {
    if (!held) return;
    held = false;
    if (comptime is_macos) _ = IOPMAssertionRelease(assertion_id);
    assertion_id = 0;
    log.debug("[sleep] idle-sleep assertion released\n", .{});
}

fn acquire() void {
    if (comptime !is_macos) return;
    const typ = CFStringCreateWithCString(null, assertion_type, utf8_encoding) orelse return;
    defer CFRelease(typ);
    const name = CFStringCreateWithCString(null, assertion_name, utf8_encoding) orelse return;
    defer CFRelease(name);
    var id: u32 = 0;
    if (IOPMAssertionCreateWithName(typ, level_on, name, &id) != 0) {
        broken = true;
        log.warn("[sleep] idle-sleep prevention unavailable; continuing without it\n", .{});
        return;
    }
    assertion_id = id;
    held = true;
    log.debug("[sleep] idle-sleep assertion held\n", .{});
}

const testing = std.testing;

test "sleep-inhibit transitions: edges only, never while disabled" {
    try testing.expectEqual(Action.none, actionFor(false, true, false, false));
    try testing.expectEqual(Action.none, actionFor(true, true, true, false));
    try testing.expectEqual(Action.acquire, actionFor(true, true, false, false));
    try testing.expectEqual(Action.release, actionFor(false, true, true, false));
    try testing.expectEqual(Action.none, actionFor(true, false, false, false));
    try testing.expectEqual(Action.release, actionFor(false, false, true, false));
    try testing.expectEqual(Action.none, actionFor(true, true, false, true));
}
