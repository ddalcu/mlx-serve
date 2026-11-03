//! engines/mlx/dispatch.
//! Owns defaultCtx, module_owned_state_fields, module_shared_readonly_fields, ownsModuleDecodeState, sharesModuleReadonlyState, moduleSpecWiring, and 7 more.

const std = @import("std");
const mlx_stream_mod = @import("modules/stream.zig");
const distributed_mod = @import("cluster/distributed.zig");
const mlx = @import("../../mlx/ffi.zig");
const abi = @import("../../abi/plugin.zig");
const ForwardCtx = @import("forward_ctx.zig").ForwardCtx;
const testing = @import("kv/cache.zig").testing;
const forwardMlxStream = @import("modules/mlx_stream.zig").forwardMlxStream;
const Transformer = @import("transformer.zig").Transformer;
const plugin_driver = @import("plugin/driver.zig");
const intree = @import("plugin/intree.zig");

/// Default forward context, routing through the Transformer's own state.
/// Used by the single-slot legacy path and by Phase-2 prefill on a slot
/// that has had its KVCache temporarily swapped onto the Transformer.
pub fn defaultCtx(self: *Transformer) ForwardCtx {
    return .{
        .cache = &self.cache,
        .moe_seq_offset = &self.moe_seq_offset,
        .ssm_entries = self.ssm_entries,
        .capture_hidden = self.capture_hidden,
        .vision_embeddings = self.vision_embeddings,
        .plugin_slot = ForwardCtx.shell_slot,
    };
}

/// Archs whose per-request decode state lives on their OWN module — one
/// instance per MODEL, not one per slot — instead of in this shell's
/// KVCache/ssm_entries. Every member is by construction single-flight at
/// admission (`scheduler.modelExclusiveDecode`): the state is rebuilt at
/// `cache.step == 0` and advanced by each tick, so a second interleaved
/// slot deinits and rebuilds the live request's state and both then append
/// to the ONE state. Add a new arm here the moment its pointer field is
/// added above, or the arch serves two clients one mangled stream.
pub const module_owned_state_fields = [_][]const u8{"mlx_stream_ext"};

/// Module pointer fields that hold READ-ONLY per-model state (qwen4: the trunk module with the
/// n-gram hash + mmapped table). Every per-request thing lives on the
/// slot's `SSMCacheEntry` (GDN pair, PLE window + tokens, QSA keys +
/// pooled blocks), so slots interleave AND batch freely. The one
/// module-owned piece on this arch is the MTP head's cache, and the
/// scheduler makes the SLOT that uses it exclusive (`slotExclusiveDecode`).
pub const module_shared_readonly_fields = [_][]const u8{};

/// A served model's caps; null on the legacy path.
fn servedCaps(self: *const Transformer) ?abi.Caps {
    if (self.plugin == null) return null;
    return self.config.plugin.?.parsed.caps;
}

pub fn ownsModuleDecodeState(self: *const Transformer) bool {
    if (servedCaps(self)) |c| if (c.owns_decode_state) return true;
    inline for (module_owned_state_fields) |f| {
        if (@field(self, f) != null) return true;
    }
    // A pipeline rank 0's worker holds the later layers' per-request state.
    if (intree.trunk(self)) |t| if (t.dist_peer != null) return true;
    return false;
}

pub fn sharesModuleReadonlyState(self: *const Transformer) bool {
    if (servedCaps(self)) |c| if (c.shared_readonly_state) return true;
    inline for (module_shared_readonly_fields) |f| {
        if (@field(self, f) != null) return true;
    }
    return false;
}

/// Spec wiring class: module archs run only their OWN MTP head
/// (`moduleStateSpecRollback`), PLD/drafters stay off — unmeasured there.
/// Covers both the owned and the shared-readonly lists so a shared arch
/// does not fall onto the generic wiring and pick up PLD by default.
pub fn moduleSpecWiring(self: *const Transformer) bool {
    return self.ownsModuleDecodeState() or self.sharesModuleReadonlyState();
}

/// Can this arch's MODULE-owned decode state be rolled back across a
/// speculative verify?
///
/// This is what `ownsModuleDecodeState` actually stood in for: the reason
/// module-owned archs had every spec mode forced off is that a rejected
/// draft tail cannot be rewound, not the ownership itself. An arch earns
/// a true here by shipping per-position state capture plus a rewind,
/// pinned against fresh forwards at every accepted count. dsv4 cannot:
/// its rings and compressed caches have no per-position capture, and it
/// has DSpark for drafting anyway.
/// mlx-stream runs its own prompt pass: the generator hands it the whole prompt in one forward.
pub fn prefillsWholePrompt(self: *const Transformer) bool {
    if (servedCaps(self)) |c| if (c.whole_prompt_prefill) return true;
    return self.mlx_stream_ext != null;
}

/// DeepSeek's own DSpark draft stages (V4 or V4.1) loaded on this model.
pub fn dsparkStages(self: *const Transformer) usize {
    if (intree.dsv4(self)) |m| return m.n_mtp;
    if (intree.dsv41(self)) |m| return m.n_mtp;
    if (plugin_driver.nativeStages(self) > 0) return plugin_driver.nativeStages(self);
    if (self.mlx_stream_ext) |m| return @intFromBool(mlx_stream_mod.blockSize(m) > 0);
    return 0;
}

pub fn moduleStateSpecRollback(self: *const Transformer) bool {
    if (servedCaps(self)) |c| if (c.spec_rollback) return true;
    // qwen4_exp: GDN per-position capture + PLE window/token capture +
    // positional QSA key histories (`ssmRollbackFromCapture`), so the MTP
    // head's verify can rewind. dsv4 cannot.
    return false;
}

/// Is this arch's MTP head the checkpoint's OWN trained layer AND measured
/// no-worse-than-serial across the whole context ladder on this family?
///
/// `defaultEnableMtp`'s `is_moe` caution is about a BOLTED-ON sidecar
/// paying expert routing in a verify it was never designed around. It does
/// not describe a head that ships inside the checkpoint — dsv4's DSpark
/// stages already get the same exemption through `dsv4_stages`.
///
/// The bar for adding an arch here is a MEASUREMENT, not the head's
/// provenance: no losing cell on the context ladder against the same
/// model's serial decode, on prompt text that is not trivially draftable.
pub fn forward(self: *Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
    var ctx = self.defaultCtx();
    return self.forwardWith(&ctx, token_ids);
}

pub fn forwardWith(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
    if (ctx.batch_slots == null) try self.ssmGroupRelease(ctx);
    if (self.plugin) |b| return plugin_driver.forward(self, b, ctx, token_ids);
    if (self.mlx_stream_ext) |m| return forwardMlxStream(self, ctx, token_ids, m);
    return error.PluginDeclinedNoLegacy;
}

/// Does `forwardWith` honor `ForwardCtx.capture_layers`? A served plugin captures layers; a drafter also reads the
/// trunk's embedding and head (`Bound.drafters`). `DflashModel.bind` gates on this predicate.
pub fn supportsLayerCapture(self: *const Transformer) bool {
    const b = self.plugin orelse return false;
    return b.drafters;
}

// ── tests ──

test "ownsModuleDecodeState covers every module-owned arch" {
    // The bug this pins: a SECOND module-owned arch arrived and
    // `scheduler.modelExclusiveDecode` kept keying on `dsv4` alone, so two
    // concurrent requests shared ONE module state — the dsv4
    // clobber, verbatim. The predicate reads a NAMED LIST so it cannot
    // drift from itself; this test pins the list against the struct.
    var t: Transformer = undefined;
    t.plugin = null;
    t.rht = null;
    t.mlx_stream_ext = null;
    try testing.expect(!t.ownsModuleDecodeState());

    var fake_ext: mlx_stream_mod.Model = undefined;
    t.mlx_stream_ext = &fake_ext;
    try testing.expect(t.ownsModuleDecodeState());
}

test "every optional pointer on Transformer is declared module-owned, shared read-only or engine-owned" {
    // A third arch declared like the module-owned ones is exclusive by construction only if it lands in a list:
    // forgetting it is silent until two clients arrive at once.
    const engine_owned = [_][]const u8{ "ane_prefill", "plugin", "rht", "int8_cache", "capture_hidden", "spec_tree" };
    const info = @typeInfo(Transformer).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        const ti = @typeInfo(T);
        if (ti == .optional and @typeInfo(ti.optional.child) == .pointer and @typeInfo(ti.optional.child).pointer.size == .one) {
            var listed = false;
            for (module_owned_state_fields ++ module_shared_readonly_fields ++ engine_owned) |l| {
                if (std.mem.eql(u8, l, name)) listed = true;
            }
            if (!listed) std.debug.print("Transformer.{s}: an optional pointer in no list\n", .{name});
            try testing.expect(listed);
        }
    }
}
