// Test root — imports all modules to run their embedded tests.
// Run with: zig build test

test {
    _ = @import("log.zig");
    _ = @import("chat.zig");
    _ = @import("server.zig");
    _ = @import("model.zig");
    _ = @import("generate.zig");
    _ = @import("transformer.zig");
}
