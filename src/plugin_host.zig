//! The host surface a plugin's own test build imports as `mlx_host`: the one
//! list. The served build (main.zig) and the host test build (tests.zig) pass
//! their root, which re-exports each name; `plugin_host_check.zig` fails their
//! build on a missing one.
pub const mlx = @import("mlx/ffi.zig");
pub const log = @import("sdk/util/log.zig");
pub const io_util = @import("sdk/util/io_util.zig");
pub const shared = @import("sdk/util/shared.zig");
/// The process environment (`src/env.zig`): every variable a plugin reads is registered there.
pub const env = @import("env.zig");
pub const mtp_acceptance = @import("sdk/spec/mtp_acceptance.zig");
/// The EXL3 quant format (`src/sdk/quant/exl3/format.zig`): a plugin's EXL3 bank multiplies through it.
pub const exl3 = @import("sdk/quant/exl3/format.zig");
pub const dsa_nax = @import("plugins/glm5_next/dsa_nax.zig");
/// `fp8_block128`'s GEMV plan and host arithmetic (`sdk/quant/fp8_block128_gemv.zig`).
pub const fp8_block128_gemv = @import("sdk/quant/fp8_block128_gemv.zig");
