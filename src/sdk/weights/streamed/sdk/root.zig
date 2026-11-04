//! The plugin SDK (docs/plugins.md): the one module a plugin imports. The small SDK: what a module-owned arch needs
//! from the host (G1 module-owned decode state, G2 the phase change, G3 spec decode, G4 phase bills, the process
//! claim, the weight loader and the memory ledgers) as optional declarations of the kinds below. A seam only one
//! plugin consumes (an expert source, a kernel registry, a quant contract) stays in that plugin until a second
//! consumer exists. The registry (src/plugins.zig) builds each kind's table once, at compile time, and the host
//! resolves a model's tables once at load: a hook runs per request, step or round, never per layer.
//!
//! `KVCache`, `ForwardCtx` and `Linear` (an arch over the host's cache) land with their first consumer: the
//! archs registered so far own their decode state (G1).

const std = @import("std");

/// The SDK's version. A plugin built against another major is refused (`negotiate`); a newer minor on either
/// side is compatible (newer hooks are optional). 2.0: a draft lane's `round` takes the request's `SamplingParams`.
/// 2.1: an arch's optional `maxOutput` (the generation a request may take past its prompt).
pub const api: Version = .{ .major = 2, .minor = 1 };

/// Compile the registered plugins' profile probes in: the host's `zig build -Dplugin-profile=true`, which also compiles
/// the probes' C shims (csrc/dsv41_*: the command-buffer timeline the profile links) and publishes the switch as the
/// host root's `pub const plugin_profile`. Off in every served build, and with a host that has no such switch.
pub const plugin_profile: bool = if (@hasDecl(@import("mlx_host"), "plugin_profile")) @import("mlx_host").plugin_profile else false;

pub const mlx = @import("mlx_host").mlx;
pub const log = @import("mlx_host").log;
pub const io_util = @import("io_util.zig");

const plugin = @import("plugin.zig");
pub const Version = plugin.Version;
pub const Plugin = plugin.Plugin;
pub const Provides = plugin.Provides;
pub const Host = plugin.Host;
pub const NegotiationError = plugin.NegotiationError;
pub const negotiate = plugin.negotiate;
/// What this process checks every plugin against: this SDK and the MLX it links, as MLX reports itself (`v` + its
/// version). One MLX per process, so an MLX bump in the host fails every plugin tested on the old one. `buf` holds
/// the version text.
pub fn host(buf: []u8) !Host {
    var s = mlx.mlx_string_new();
    defer _ = mlx.mlx_string_free(s);
    try mlx.check(mlx.mlx_version(&s));
    return .{ .api = api, .mlx = try std.fmt.bufPrint(buf, "v{s}", .{std.mem.span(mlx.mlx_string_data(s))}) };
}

const peek = @import("peek.zig");
pub const Priority = peek.Priority;
pub const Diag = peek.Diag;
pub const ConfigPeek = peek.ConfigPeek;
pub const GroupPeek = peek.GroupPeek;
pub const LayerPeek = peek.LayerPeek;
pub const Segment = peek.Segment;

const arch = @import("arch.zig");
pub const Arch = arch.Arch;
pub const ArchInstance = arch.ArchInstance;
pub const Caps = arch.Caps;
pub const Shell = arch.Shell;
pub const LoadFacts = arch.LoadFacts;
pub const LoadCtx = arch.LoadCtx;
pub const RequestShape = arch.RequestShape;
pub const DecodeHandover = arch.DecodeHandover;

const spec = @import("spec.zig");
pub const Spec = spec.Spec;
pub const DraftLane = spec.DraftLane;
pub const SamplingParams = spec.SamplingParams;
pub const ArmRequest = spec.ArmRequest;
pub const DraftArm = spec.DraftArm;
pub const DraftRound = spec.DraftRound;
pub const DraftStats = spec.DraftStats;

const memory_bill = @import("memory_bill.zig");
pub const MemoryBill = memory_bill.MemoryBill;
pub const BillRequest = memory_bill.BillRequest;
pub const Rows = memory_bill.Rows;
pub const fill = memory_bill.fill;
pub const admit = memory_bill.admit;
pub const checkMeasured = memory_bill.checkMeasured;
pub const checkRows = memory_bill.checkRows;
pub const checkConstruction = memory_bill.checkConstruction;

pub const lifecycle = @import("lifecycle.zig");
pub const PhaseObserver = lifecycle.PhaseObserver;

const kinds = @import("kinds.zig");
pub const Source = kinds.Source;
pub const Engine = kinds.Engine;

/// Process and box memory readings (the kernel's ledgers) that bills and construction checks compare against.
pub const memory = @import("memory.zig");
const weights = @import("weights.zig");
pub const Weights = weights.Weights;
pub const LoadOpts = weights.LoadOpts;
pub const WeightLoader = weights.WeightLoader;
/// The plugin's own loaders (every shard the index names; past the page cache on request).
pub const loader = weights.loader;
pub const QuantMode = @import("quant_mode.zig").QuantMode;

/// The comptime interface checks the kinds run on a plugin's namespaces; a plugin's own contracts reuse them.
pub const check = @import("check.zig");

/// The MTP acceptance modes the host serves (`Mode`, `DEFAULT_TYPICAL_DELTA`, `typicalThreshold`).
pub const acceptance = @import("mlx_host").mtp_acceptance;

/// Conformance (docs/plugins.md): every check declares its lane.
pub const testing = @import("testing.zig");

test {
    std.testing.refAllDecls(@This());
    _ = plugin;
    _ = peek;
    _ = arch;
    _ = spec;
    _ = memory_bill;
    _ = lifecycle;
    _ = kinds;
    _ = testing;
}
