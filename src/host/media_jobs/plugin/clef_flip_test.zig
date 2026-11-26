//! Clef's flip test against its legacy HTTP golden (kept separate from `flip_test.zig` so `flip_test.zig` reaches
//! only pure-SDK media plugins).

const harness = @import("flip_harness.zig");

test "flip: clef answers its legacy arm's bytes" {
    try harness.expectDecisions(harness.Fixture(@import("../../../plugins/clef/plugin.zig"), @embedFile("../../../plugins/clef/fixtures/legacy_http.golden")), &.{"{\"truncate\":1}"});
}
