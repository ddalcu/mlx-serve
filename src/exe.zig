//! The executable's root under `-Dunits`: the host (`main.zig`) is an object of its own, compiled in parallel with
//! the plugin units, and exports `mlx_serve_host_main`; this hands it the process's arguments and environment.

const std = @import("std");
const builtin = @import("builtin");

extern fn mlx_serve_host_main(argc: c_int, argv: [*]const [*:0]const u8, envp: [*:null]const ?[*:0]const u8) callconv(.c) u8;

const windows = struct {
    extern fn mlx_serve_host_main(command_line: [*]const u16, len: usize) callconv(.c) u8;
};

pub fn main(init: std.process.Init.Minimal) u8 {
    if (comptime builtin.os.tag == .windows) {
        const command_line = init.args.vector;
        return windows.mlx_serve_host_main(command_line.ptr, command_line.len);
    }
    const argv = init.args.vector;
    return mlx_serve_host_main(@intCast(argv.len), argv.ptr, init.environ.block.slice.ptr);
}
