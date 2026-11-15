//! Compiles every public function of `kit.blocks`, `kit.bases`, `kit.bill`, `kit.outside_model`, `kit.testing`,
//! `kit.mlx_ops`, `kit.mlx_array`, `kit.log`, `kit.mlx_scope`, `kit.system`, `kit.mlx_device`, `kit.foreign` and
//! `kit.streamed` into an object, and every MLX header `mlx_steel_sources` embeds, so a block that no longer builds
//! from the staged kit (a file, header or module the kit lacks, a format a block still imports) fails here, not in a
//! plugin author's build. Nothing links or runs it.

const std = @import("std");
const kit = @import("mlx_serve");
const steel = @import("mlx_steel_sources");

/// Public helpers of the blocks' own tests that sit in non-test files and call `std.testing`: not part of the surface.
const test_helpers = [_][]const u8{ "attn256RandBf16", "attn256RandBf16Scaled", "errVsTruth", "expectBitEqual", "gdnParityRand" };

fn skipped(name: []const u8) bool {
    for (test_helpers) |h| if (std.mem.eql(u8, name, h)) return true;
    return false;
}

/// Down to the files of each area (depth 2): their public functions. Deeper decls are what the files import.
fn touch(comptime T: type, comptime depth: u8) void {
    @setEvalBranchQuota(2_000_000);
    inline for (comptime std.meta.declarations(T)) |name| {
        if (comptime skipped(name)) continue;
        const F = @TypeOf(@field(T, name));
        if (F == type) {
            const U = @field(T, name);
            if (depth < 2 and @typeInfo(U) == .@"struct") touch(U, depth + 1);
        } else if (@typeInfo(F) == .@"fn") {
            _ = &@field(T, name);
        }
    }
}

comptime {
    touch(kit.blocks, 0);
    touch(kit.bases.blocks, 1);
    touch(kit.bases.run, 2);
    touch(kit.bases.kda_hybrid, 1);
    touch(kit.bill, 1);
    touch(kit.outside_model, 2);
    touch(kit.testing.kernel_contract, 1);
    touch(kit.mlx_ops, 2);
    touch(kit.mlx_array, 2);
    touch(kit.log, 2);
    touch(kit.mlx_scope, 2);
    touch(kit.system, 2);
    touch(kit.mlx_device, 2);
    touch(kit.foreign, 2);
    touch(kit.streamed, 1);
    for (std.meta.declarations(steel)) |name| _ = @field(steel, name);
}
