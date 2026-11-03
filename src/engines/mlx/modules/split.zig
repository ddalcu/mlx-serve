//! Whether this compile unit reaches the module archs (the streaming glue, `call.zig`'s functions) through the symbols
//! `src/modules_unit.zig` exports instead of compiling them in: the MLX engine and the plugin units under `-Dunits`.

const root = @import("root");

pub const active = @import("build_options").units and !@import("builtin").is_test and
    @hasDecl(root, "mlx_serve_unit") and !@hasDecl(root, "mlx_serve_modules_unit");
