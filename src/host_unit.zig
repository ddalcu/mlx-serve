//! The host as an object of its own under `-Dunits`, compiled in parallel with the plugin units: `main.zig` without a
//! `main` decl at the root (std's start code would export a second C `main`), plus `mlx_serve_host_main`, which
//! `src/exe.zig` calls. Builds the `Init` std's start code would have (std.start `callMain`).

const std = @import("std");
const main_mod = @import("main.zig");

// What the absorbed modules read from their host root (`mlx_host`), as `main.zig` declares them.
pub const env = main_mod.env;
pub const mlx = main_mod.mlx;
pub const io_util = main_mod.io_util;
pub const shared = main_mod.shared;
pub const mtp_acceptance = main_mod.mtp_acceptance;
pub const exl3 = main_mod.exl3;
pub const dsa_nax = main_mod.dsa_nax;
pub const fp8_block128_gemv = main_mod.fp8_block128_gemv;
pub const log = main_mod.log;
pub const plugin_profile = main_mod.plugin_profile;

comptime {
    @export(&hostMain, .{ .name = "mlx_serve_host_main" });
    // The host owns every `Shared`'s storage; the engine and plugin units link to it.
    _ = @import("shared_all.zig");
}

/// Marks the host unit: the MLX engine's objects are opaque here (`abi/mlx_handles.zig`), reached through its table.
pub const mlx_serve_host_unit = true;

fn hostMain(argc: c_int, argv: [*][*:0]u8, envp: [*:null]?[*:0]u8) callconv(.c) u8 {
    var env_count: usize = 0;
    while (envp[env_count] != null) : (env_count += 1) {}
    const args: std.process.Args.Vector = argv[0..@intCast(argc)];
    const environ: std.process.Environ.Block = .{ .slice = envp[0..env_count :null] };
    if (std.Options.debug_threaded_io) |t| {
        if (@sizeOf(std.Io.Threaded.Argv0) != 0) t.argv0.value = argv[0];
        t.environ = .{ .process_environ = .{ .block = environ } };
        t.environ_initialized = environ.isEmpty();
    }
    const gpa = std.heap.c_allocator;
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    var threaded: std.Io.Threaded = .init(gpa, .{ .argv0 = .init(.{ .vector = args }), .environ = .{ .block = environ } });
    defer threaded.deinit();
    var environ_map = std.process.Environ.createMap(.{ .block = environ }, gpa) catch |e| std.process.fatal("failed to parse environment variables: {t}", .{e});
    defer environ_map.deinit();
    const preopens = std.process.Preopens.init(arena.allocator()) catch |e| std.process.fatal("failed to init preopens: {t}", .{e});
    main_mod.main(.{
        .minimal = .{ .args = .{ .vector = args }, .environ = .{ .block = environ } },
        .arena = &arena,
        .gpa = gpa,
        .io = threaded.io(),
        .environ_map = &environ_map,
        .preopens = preopens,
    }) catch |e| {
        std.log.err("{t}", .{e});
        if (@errorReturnTrace()) |trace| std.debug.dumpErrorReturnTrace(trace);
        return 1;
    };
    return 0;
}
