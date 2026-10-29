//! The registrations of a contract run (`Set`): what an area's `register` fills. Its own file so an area's contract
//! reaches the harness without the registry, which imports every area.

const std = @import("std");
const spec = @import("spec.zig");
const gates = @import("gates.zig");

/// The registrations of every area: each kernel's test instance (owned here) and its hooks.
pub const Set = struct {
    a: std.mem.Allocator,
    kernels: std.ArrayList(gates.Kernel) = .empty,
    frees: std.ArrayList(*const fn (std.mem.Allocator, *anyopaque) void) = .empty,

    pub fn init(a: std.mem.Allocator) Set {
        return .{ .a = a };
    }

    pub fn deinit(set: *Set) void {
        for (set.kernels.items, set.frees.items) |k, free| free(set.a, k.ctx);
        set.kernels.deinit(set.a);
        set.frees.deinit(set.a);
    }

    /// Registers `value` (a test type with `prepare`, `run`, `count`, `reference` and a `rig`) under `s`; a planted
    /// control names the gate it must fail.
    pub fn add(set: *Set, comptime T: type, value: T, s: spec.Spec, control: ?spec.Gate) !void {
        const H = struct {
            const vt = gates.hooks(T);
            fn free(a: std.mem.Allocator, ctx: *anyopaque) void {
                const p: *T = @ptrCast(@alignCast(ctx));
                p.deinit();
                a.destroy(p);
            }
        };
        const p = try set.a.create(T);
        errdefer set.a.destroy(p);
        p.* = value;
        try set.frees.ensureUnusedCapacity(set.a, 1);
        try set.kernels.append(set.a, .{ .spec = s, .ctx = p, .vt = &H.vt, .control = control });
        set.frees.appendAssumeCapacity(H.free);
    }
};
