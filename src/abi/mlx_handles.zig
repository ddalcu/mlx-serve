//! The MLX engine's objects as the host holds them: stored, passed back and null-checked, never opened. The host's
//! own compile unit (`-Dunits`, root `src/host_unit.zig` declares `mlx_serve_host_unit`) sees opaque types, so a host
//! edit that reaches into one fails to build there; every other build (the engine unit, a single-unit build, the
//! tests) sees the engine's real types. Every handle is a pointee: a field holding one is a pointer, so the layout is
//! the same on both sides.

const host_only = @hasDecl(@import("root"), "mlx_serve_host_unit");

pub const Transformer = if (host_only) opaque {} else @import("../engines/mlx/transformer.zig").Transformer;
pub const VisionEncoder = if (host_only) opaque {} else @import("../engines/mlx/vision_encoder.zig").VisionEncoder;
pub const DrafterModel = if (host_only) opaque {} else @import("../plugins/gemma4/assistant_drafter.zig").DrafterModel;
pub const DflashModel = if (host_only) opaque {} else @import("../engines/mlx/spec/dflash/dflash.zig").DflashModel;
pub const MtpHeadRef = if (host_only) opaque {} else @import("../engines/mlx/spec/mtp_head_ref.zig").MtpHeadRef;
pub const HotPrefixCache = if (host_only) opaque {} else @import("../engines/mlx/cache/prefix_cache.zig").HotPrefixCache;
pub const Slot = if (host_only) opaque {} else @import("../engines/mlx/slot.zig").Slot;
