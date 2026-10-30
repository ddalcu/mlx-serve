//! Root of `zig build preview-test`: the hermetic preview graph, rooted at src/ so its tests can embed src/fixtures/.
test {
    _ = @import("sdk/media/preview.zig");
}
