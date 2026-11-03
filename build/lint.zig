//! The checks that compile no Zig: the house-rule lint, its own tests, and every plugin's import closure.

const std = @import("std");

/// `lint`, `lint-test`, `unit-closure`.
pub fn add(b: *std.Build) void {
    // `zig build lint`: the house rules over src/ (scripts/lint.py). Report-only unless -Dlint-strict, which fails on
    // the rules listed in scripts/lint_enforced.txt. `--all`: run the script directly.
    const lint = b.addSystemCommand(&.{ "python3", "scripts/lint.py" });
    lint.setCwd(b.path("."));
    lint.has_side_effects = true;
    if (b.option(bool, "lint-strict", "zig build lint: fail on violations of the enforced rules") orelse false) lint.addArg("--strict");
    if (b.option([]const u8, "lint-json", "zig build lint: also write the JSON summary to this path")) |p| lint.addArgs(&.{ "--json", p });
    b.step("lint", "House-rule lint over src/ (report-only unless -Dlint-strict)").dependOn(&lint.step);
    const lint_test = b.addSystemCommand(&.{ "python3", "scripts/lint_test.py" });
    lint_test.setCwd(b.path("."));
    lint_test.has_side_effects = true;
    b.step("lint-test", "Run the lint's own tests").dependOn(&lint_test.step);
    // `zig build unit-closure`: no plugin's import closure leaves src/sdk, src/mlx, src/abi, src/kernels, env.zig and
    // the plugin folders (scripts/unit_closure.py), so a plugin edit rebuilds its -Dunits group alone. The plugins
    // named here still reach the engine; each leaves the list once it no longer does.
    const closure = b.addSystemCommand(&.{ "python3", "scripts/unit_closure.py", "--except", "clef,qwen4_exp" });
    closure.setCwd(b.path("."));
    closure.has_side_effects = true;
    b.step("unit-closure", "Check every plugin's import closure stays in the SDK and the plugin folders").dependOn(&closure.step);
}
