//! The KDA-hybrid base (GLM-5.3-Flash, Ling 3.0) as the plugin kit ships it (`kit.bases.kda_hybrid`): the family
//! config, the loaded trunk, the KDA layer, the slots, the table glue, the MoE layers and the tiny fixture packs.
//! `builtin_formats.zig` (the trunk's `Hooks.exl3` for Sushi packs) is not named here: it imports the `sushi_exl3`
//! module, which the kit does not ship, so only the server's own plugins reach it.

pub const config = @import("config.zig");
pub const trunk = @import("trunk.zig");
pub const kda = @import("kda.zig");
pub const slots = @import("slots.zig");
pub const host = @import("host.zig");
pub const moe = @import("moe.zig");
pub const fixture = @import("fixture.zig");
