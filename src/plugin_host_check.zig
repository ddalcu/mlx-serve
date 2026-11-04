//! A host root that a plugin imports as `mlx_host` (the served build's main.zig, the host test build's tests.zig)
//! re-exports every name of the host surface, `plugin_host.zig`: the one list.
const surface = @import("plugin_host.zig");

/// Fails the build on the first name of the surface that `Root` does not re-export (pub, the same file or module).
pub fn reexports(comptime Root: type) void {
    inline for (@typeInfo(surface).@"struct".decl_names) |name| {
        if (!@hasDecl(Root, name)) @compileError(@typeName(Root) ++ " does not re-export plugin_host." ++ name);
        if (@field(Root, name) != @field(surface, name)) @compileError(@typeName(Root) ++ "." ++ name ++ " is not plugin_host." ++ name);
    }
}
