//! The module archs as a compile unit of their own under `-Dunits`, beside the MLX engine, which calls them by symbol
//! (`engines/mlx/modules/split.zig`): the streaming packs' glue (GLM-5.3 on mlx-stream: its arch, the expert streamer,
//! the EXL3 lanes) and the functions in `engines/mlx/modules/call.zig` (DeepSeek's DSpark rounds and V4.1's tower,
//! the cluster worker, Sushi's EXL3 experts, qwen4_exp's MTP head and grouped verify).

comptime {
    _ = @import("engines/mlx/modules/stream_export.zig");
    @import("engines/mlx/modules/call.zig").exportAll();
}

/// Marks this unit: it compiles the module archs itself rather than calling their tables.
pub const mlx_serve_modules_unit = true;

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
