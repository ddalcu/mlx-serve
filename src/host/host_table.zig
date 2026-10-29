//! The process's `Host` table as the host published it at startup (`plugin_host_impl.processHost`), for the engine's
//! model loads (`LoadCtx.host`). Its own file, so a unit that reads it never compiles the table's builder.

const abi = @import("../abi/host.zig");

/// A struct: `Shared` hands its storage out as `*anyopaque`, which a pointer does not coerce to.
pub const Published = struct { host: ?*const abi.Host = null };

/// Process-wide under `-Dunits` (`sdk/util/shared.zig`); set once by the host.
pub const published = @import("../sdk/util/shared.zig").Shared(Published, "plugin_host_published", .{});

/// The process's table, or null where no server published one (tests, commands that serve no model).
pub fn get() ?*const abi.Host {
    return published.get().host;
}
