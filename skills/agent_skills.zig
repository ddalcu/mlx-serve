//! The mlx-serve agent skill, embedded so `mlx-serve launch` can install it.
//! The app bundles the same folder (app/build.sh).
pub const name = "mlx-serve";

pub const files = [_]struct { name: []const u8, bytes: []const u8 }{
    .{ .name = "SKILL.md", .bytes = @embedFile("mlx-serve/SKILL.md") },
    .{ .name = "chat.md", .bytes = @embedFile("mlx-serve/chat.md") },
    .{ .name = "decisions.md", .bytes = @embedFile("mlx-serve/decisions.md") },
    .{ .name = "media.md", .bytes = @embedFile("mlx-serve/media.md") },
};
