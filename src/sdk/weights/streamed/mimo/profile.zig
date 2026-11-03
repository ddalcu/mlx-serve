//! Diagnostic builds only. No clock, counter, or branch in served builds.
const std = @import("std");
const sdk = @import("sdk");
pub const enabled = sdk.plugin_profile;
pub const Bucket = @import("../sdk_ext.zig").profile.PrefillBucket;
pub const Stamp = if (enabled) u64 else void;
const n = @typeInfo(Bucket).@"enum".field_names.len;
var ns: [n]u64 = @splat(0);
var calls: u64 = 0;
pub inline fn now() Stamp {
    if (comptime !enabled) return {};
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
pub inline fn charge(bucket: Bucket, start: Stamp) void {
    if (comptime enabled) ns[@backingInt(bucket)] += now() - start;
}
pub inline fn call() void {
    if (comptime enabled) calls += 1;
}
pub fn reset() void {
    if (comptime !enabled) return;
    ns = @splat(0);
    calls = 0;
}
/// SDK counters are cumulative; the caller supplies a settled phase delta.
pub fn reportReadBytes(phase: []const u8, record_bytes: u64, read_bytes: u64) void {
    reportReadBytesAs("MIMO", phase, record_bytes, read_bytes);
}

/// `reportReadBytes` under an arch's tag (`<tag>_PROFILE phase=...`: glm5_next's `GLM5_NEXT`).
pub fn reportReadBytesAs(comptime tag: []const u8, phase: []const u8, record_bytes: u64, read_bytes: u64) void {
    if (comptime !enabled) return;
    if (calls == 0) return;
    std.log.info(tag ++ "_PROFILE phase={s} calls={d} records={d} bytes={d} barrier_s={d:.6} route_s={d:.6} read_wait_s={d:.6} encode_s={d:.6} drain_s={d:.6} join_s={d:.6}", .{
        phase,             calls,           read_bytes / record_bytes, read_bytes,
        seconds(.barrier), seconds(.route), seconds(.read_wait),       seconds(.encode),
        seconds(.drain),   seconds(.join),
    });
}
fn seconds(bucket: Bucket) f64 {
    return @as(f64, @floatFromInt(ns[@backingInt(bucket)])) / std.time.ns_per_s;
}
