//! Process-wide state under `-Dunits`. A plugin unit (`src/unit.zig`) compiles its own copy of whatever code it
//! reaches, globals included, so state the host sets or reads (CLI choices, memory accounting, the cluster, the MLX
//! error latch, the log sink) would split into one copy per unit. `Shared` keeps the storage in the host and hands a
//! unit the host's address, by an exported C function per name. Outside `-Dunits` it is a plain global.
//! `scripts/unit_globals.py` fails on a duplicated global that is neither shared nor listed as safe to duplicate.

/// True when compiled into a plugin unit: its root declares `mlx_serve_unit`.
pub const in_unit = @hasDecl(@import("root"), "mlx_serve_unit");

pub fn Shared(comptime T: type, comptime name: []const u8, comptime init: T) type {
    return struct {
        var storage: T = init;
        var cached: ?*T = null;
        const sym = "mlx_serve_shared_" ++ name;

        fn addr() callconv(.c) *anyopaque {
            return &storage;
        }

        comptime {
            if (!in_unit) @export(&addr, .{ .name = sym });
        }

        pub inline fn get() *T {
            if (!in_unit) return &storage;
            if (cached) |p| return p;
            const f = @extern(*const fn () callconv(.c) *anyopaque, .{ .name = sym });
            const p: *T = @ptrCast(@alignCast(f()));
            cached = p;
            return p;
        }
    };
}

/// A `threadlocal` the host and every unit share: the host holds the storage, and a unit takes the CURRENT thread's
/// slot from the host on every access (each thread's address differs, so nothing is cached).
pub fn SharedThreadLocal(comptime T: type, comptime name: []const u8, comptime init: T) type {
    return struct {
        threadlocal var storage: T = init;
        const sym = "mlx_serve_shared_tls_" ++ name;

        fn addr() callconv(.c) *anyopaque {
            return @ptrCast(&storage);
        }

        comptime {
            if (!in_unit) @export(&addr, .{ .name = sym });
        }

        pub inline fn get() *T {
            if (!in_unit) return &storage;
            const f = @extern(*const fn () callconv(.c) *anyopaque, .{ .name = sym });
            return @ptrCast(@alignCast(f()));
        }
    };
}

test "SharedThreadLocal: each thread starts at init and keeps its own value" {
    const S = SharedThreadLocal(u32, "test_shared_tls", 3);
    S.get().* = 9;
    var other: u32 = 0;
    const t = try @import("std").Thread.spawn(.{}, struct {
        fn run(out: *u32) void {
            out.* = S.get().*;
        }
    }.run, .{&other});
    t.join();
    try @import("std").testing.expectEqual(@as(u32, 3), other);
    try @import("std").testing.expectEqual(@as(u32, 9), S.get().*);
}

test "Shared: outside a unit the storage is the module's own and starts at init" {
    const S = Shared(u32, "test_shared_counter", 7);
    try @import("std").testing.expectEqual(@as(u32, 7), S.get().*);
    S.get().* += 1;
    try @import("std").testing.expectEqual(@as(u32, 8), S.get().*);
}
