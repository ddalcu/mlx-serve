//! Builds `check.zig` against the kit: `zig build` compiles every public function of `kit.blocks`.

pub fn build(b: *@import("std").Build) void {
    @import("mlx_serve_kit").blocksCheck(b);
}
