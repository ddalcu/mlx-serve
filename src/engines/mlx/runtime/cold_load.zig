//! Cold-load launch flags, drafter directory resolution and the vision opt-out.

const std = @import("std");
const testing = std.testing;

const scheduler_mod = @import("../scheduler.zig");

/// Which drafter directory a COLD load should use.
///
/// `--no-drafter` is a policy — it silences every model, including one whose
/// own dir ships a sidecar `dflash.resolveInDirDrafter` would otherwise find.
/// `--drafter <path>`, by contrast, names a sidecar for the checkpoint it was
/// passed beside: handing it to whatever model is swapped in next would load a
/// mismatched assistant, so it applies only when the entry being loaded IS the
/// launch model (which happens on a reload after eviction). Every other model
/// is served by the in-dir probe.
pub fn coldLoadDrafterDir(
    no_drafter: bool,
    primary_model_dir: []const u8,
    drafter_dir: []const u8,
    entry_path: []const u8,
) []const u8 {
    if (no_drafter) return "";
    if (drafter_dir.len == 0) return "";
    if (!std.mem.eql(u8, primary_model_dir, entry_path)) return "";
    return drafter_dir;
}

/// The drafter a load uses: "" = none, null = the in-dir probe decides.
/// `--no-drafter`/`--drafter` win over the per-model `drafter` setting.
pub fn drafterFor(no_drafter: bool, explicit: []const u8, setting: ?[]const u8) ?[]const u8 {
    if (no_drafter) return "";
    if (explicit.len > 0) return explicit;
    const s = setting orelse return null;
    if (std.mem.eql(u8, s, "off")) return "";
    if (std.mem.eql(u8, s, "auto")) return null;
    return s;
}

/// Should a cold load bring up the checkpoint's vision tower?
pub fn coldLoadVision(has_vision: bool) bool {
    return has_vision and !scheduler_mod.no_vision_global;
}

// ── tests ──

test "coldLoadDrafterDir: --no-drafter wins, an explicit --drafter belongs to its OWN model" {
    // `--drafter <path>` names a sidecar for the checkpoint it was passed
    // with; handing it to whatever model gets swapped in next would load a
    // mismatched assistant. Other models are served by the in-dir probe.
    // Reloading the launch model AFTER an eviction must still get it back.
    try testing.expectEqualStrings("/d", coldLoadDrafterDir(false, "/m", "/d", "/m"));
    try testing.expectEqualStrings("", coldLoadDrafterDir(false, "/m", "/d", "/other"));
    // --no-drafter is a policy, not a path: it silences every model, including
    // one whose own dir ships a sidecar the in-dir probe would find.
    try testing.expectEqualStrings("", coldLoadDrafterDir(true, "/m", "/d", "/m"));
    try testing.expectEqualStrings("", coldLoadDrafterDir(true, "/m", "/d", "/other"));
    // No --drafter at launch: nothing to carry, the in-dir probe decides.
    try testing.expectEqualStrings("", coldLoadDrafterDir(false, "/m", "", "/m"));
}

test "drafterFor: launch flags win, then the per-model setting, then the in-dir probe" {
    try testing.expectEqualStrings("", drafterFor(true, "/d", "/s").?);
    try testing.expectEqualStrings("/d", drafterFor(false, "/d", "off").?);
    try testing.expectEqualStrings("", drafterFor(false, "", "off").?);
    try testing.expectEqualStrings("/s", drafterFor(false, "", "/s").?);
    try testing.expectEqual(@as(?[]const u8, null), drafterFor(false, "", "auto"));
    try testing.expectEqual(@as(?[]const u8, null), drafterFor(false, "", null));
}

test "coldLoadVision honors the process-wide vision opt-out" {
    scheduler_mod.no_vision_global = false;
    try std.testing.expect(coldLoadVision(true));
    try std.testing.expect(!coldLoadVision(false));
    scheduler_mod.no_vision_global = true;
    defer scheduler_mod.no_vision_global = false;
    try std.testing.expect(!coldLoadVision(true));
}
