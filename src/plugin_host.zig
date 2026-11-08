//! The host surface a plugin's own test build imports as `mlx_host`: the one
//! list. The served build (main.zig) and the host test build (tests.zig) pass
//! their root, which re-exports each name; `plugin_host_check.zig` fails their
//! build on a missing one.
pub const mlx = @import("mlx/ffi.zig");
pub const log = @import("sdk/util/log.zig");
pub const io_util = @import("sdk/util/io_util.zig");
/// The process environment (`src/env.zig`): every variable a plugin reads is registered there.
pub const env = @import("env.zig");
