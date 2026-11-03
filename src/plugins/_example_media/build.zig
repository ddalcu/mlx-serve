//! Builds this plugin with the mlx-serve plugin kit: `zig build` writes zig-out/lib/lib_example_media.dylib, `zig build test`
//! runs its MLX-free tests; `mlx-serve plugin check` runs the conformance suite on the dylib.

pub fn build(b: *@import("std").Build) void {
    _ = @import("mlx_serve_kit").plugin(b, .{ .name = "_example_media" });
}
