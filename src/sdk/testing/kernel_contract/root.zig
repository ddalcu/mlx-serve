//! The kernel contract (how to register: REGISTERING.md): the spec a kernel registers, gates 1-4 (`gates`), gate 5
//! (`limits`), the Metal launch and rig (`metal`), and the registry an area adds its kernels to (`Set`).

pub const spec = @import("spec.zig");
pub const gates = @import("gates.zig");
pub const limits = @import("limits.zig");
pub const metal = @import("metal.zig");
pub const host = @import("host.zig");
pub const Set = @import("set.zig").Set;
