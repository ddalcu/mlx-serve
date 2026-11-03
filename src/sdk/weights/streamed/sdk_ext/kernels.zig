//! G5, the kernel registry: a plugin that brings Metal texts declares one manifest that pins every text by its
//! sha256, and the host builds and binds the set once per load. The machinery is generic over the plugin's registry
//! `R` (its kernels, texts, manifest pin, bound launcher and self-check plan); a second plugin registers its own `R`
//! and gets its own set and pin. No arm adds a construction-time device self-check.

/// A kernel set's pin: what conformance and `/props` report, next to the binary's own sha256 the window pins.
pub const Pin = struct {
    /// sha256 (hex) of the manifest, which pins every text by its own sha256: moving the directory keeps it,
    /// changing a text does not.
    manifest_sha256: []const u8,
};

/// The kernel set over registry `R`: `Set`, its `Device`, the consumers' partition checks.
pub const KernelSet = @import("kernel_set.zig").KernelSet;
/// A load's kernel set as a quant's accept receives it, erased with its registry's pin.
pub const SetRef = @import("kernel_set.zig").SetRef;
/// One load's kernel set over registry `R`.
pub fn Set(comptime R: type) type {
    return KernelSet(R).Set;
}
/// The route machinery over registry `R`: statics, per-M launch tables, bound-input checks, named refusals.
pub const Routes = @import("kernel_routes.zig").Routes;
/// The routes' host test backend over registry `R` (test code only).
pub const Trace = @import("kernel_trace.zig").KernelTrace;
