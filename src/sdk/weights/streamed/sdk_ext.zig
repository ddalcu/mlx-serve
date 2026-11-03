//! What mlx-stream adds beside the host's `sdk`: the seams only this plugin consumes, kept here rather than in the host
//! (a kind's interface lands in the host with its second consumer). The expert source (G6: the record layout, the read
//! pool, the slot cache, the residency policy and the stream), the kernel registry (G5), the EXL3 / gather quant
//! contract and its comptime tables, the KV lanes of the module-owned decode state, the profile probes and a host-side
//! strided copy. Everything here reaches the host through `sdk` only.

const std = @import("std");

/// G6: the expert source's record layout, source contract, slot types and the process's one reader.
pub const expert = @import("sdk_ext/expert.zig");
/// G5: the kernel registry (routes, sets, the trace backend).
pub const kernels = @import("sdk_ext/kernels.zig");
/// Every MLX custom kernel's launch through the plugin's launch cache (no per-call source rebuild and hash).
pub const metal_launch = @import("sdk_ext/metal_launch.zig");
/// C2: the routed-expert quant contract and the generic gather quant (`GatherQmm` through `FromGatherMatmul`).
pub const quant = @import("sdk_ext/quant.zig");
/// G7: profile probes, injected by the arch's backend type (`of(Backend)`); off everywhere else.
pub const profile = @import("sdk_ext/profile.zig");
/// A strided copy out of an evaluated array's buffer.
pub const ops = @import("sdk_ext/ops.zig");
/// The draft lanes' acceptance: the exact and typical rules, the sampled path, the seeded draws (every arch's lane).
pub const acceptance = @import("sdk_ext/acceptance.zig");
/// The serial sampler: the plugin's position-keyed token for every row it hands the host (one stream with the lanes').
pub const sampler = @import("sdk_ext/sampler.zig");
/// The KV seam: lane storage, routes and bounded caps for a module-owned decode state.
pub const kv = @import("sdk_ext/kv.zig");
/// `kv`'s conformance helpers (CPU lane).
pub const kv_testing = @import("sdk_ext/kv_testing.zig");

const kinds = @import("sdk_ext/kinds.zig");
/// The quant and expert-source contracts' comptime tables (what `exl3_quant.zig` and `exl3_source.zig` declare).
pub const Quant = kinds.Quant;
pub const ExpertSource = kinds.ExpertSource;
pub const ExpertCaps = kinds.ExpertCaps;

test {
    std.testing.refAllDecls(@This());
    _ = kinds;
}
