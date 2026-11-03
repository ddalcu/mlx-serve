//! The MLX engine as its own compile unit under `-Dunits`, compiled in parallel with the host and the plugin units: it
//! exports its table (`mlx_serve_mlx_api`, `engines/mlx/api.zig`) and links to the host's `Shared` state.

comptime {
    _ = @import("engines/mlx/api.zig");
}

/// Marks a unit: `Shared` state resolves to the host's copy (sdk/util/shared.zig).
pub const mlx_serve_unit = true;

// What the absorbed modules read from their host root (`mlx_host`), as `main.zig` declares them.
pub const env = @import("env.zig");
pub const mlx = @import("mlx/ffi.zig");
pub const io_util = @import("sdk/util/io_util.zig");
pub const shared = @import("sdk/util/shared.zig");
pub const mtp_acceptance = @import("sdk/spec/mtp_acceptance.zig");
pub const exl3 = @import("sdk/quant/exl3/format.zig");
pub const dsa_nax = @import("plugins/glm5_next/dsa_nax.zig");
pub const fp8_block128_gemv = @import("sdk/quant/fp8_block128_gemv.zig");
pub const log = @import("sdk/util/log.zig");
pub const plugin_profile: bool = @import("build_options").plugin_profile;
