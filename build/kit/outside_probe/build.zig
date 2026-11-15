//! Builds the outside probe from the staged kit: `zig build` writes zig-out/lib/lib_outside_probe.dylib.

pub fn build(b: *@import("std").Build) void {
    _ = @import("mlx_serve_kit").plugin(b, .{ .name = "_outside_probe" });
}
