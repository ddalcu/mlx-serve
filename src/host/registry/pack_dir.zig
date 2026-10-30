//! The pack folder a plugin reads lives in the SDK (`sdk/util/pack_dir.zig`).

const pack_dir = @import("../../sdk/util/pack_dir.zig");
pub const none = pack_dir.none;
pub const FsPackDir = pack_dir.FsPackDir;
