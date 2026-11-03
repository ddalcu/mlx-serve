//! Compiles every public function of `kit.blocks` and `kit.foreign` into an object, so a block that no longer builds from
//! the staged kit (a file or module the kit lacks, a format a block still imports) fails here, not in a plugin author's
//! build. Nothing links or runs it.

const std = @import("std");
const kit = @import("mlx_serve");

/// Public helpers of the blocks' own tests that sit in non-test files and call `std.testing`: not part of the surface.
const test_helpers = [_][]const u8{ "attn256RandBf16", "attn256RandBf16Scaled", "errVsTruth", "gdnParityRand" };

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
    touch(kit.foreign, 2);
}
