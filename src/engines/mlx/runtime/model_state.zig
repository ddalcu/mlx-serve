//! A model's MLX state on its registry entry: freed on the GPU thread at unload and at teardown.

const std = @import("std");
const registry = @import("../../../host/registry/registry.zig");
const distributed = @import("../cluster/distributed.zig");
const LoadedModel = registry.LoadedModel;

/// Frees the entry's MLX objects (MTP head, drafters, vision tower, transformer, weights, prefix cache), in
/// dependency order. Null fields are skipped.
pub fn freeMlx(self: *LoadedModel) void {
    // Cache snapshots call into their plugin binding, which the transformer owns.
    if (self.prefix_cache) |hc| {
        hc.deinit();
        self.allocator.destroy(hc);
        self.prefix_cache = null;
    }

    if (self.mtp) |h| {
        // The Qwen sidecar and the plugin heads are separately allocated; an in-trunk head is owned by the
        // Transformer and freed with it.
        switch (h.*) {
            .qwen => |q| {
                q.deinit();
                self.allocator.destroy(q);
            },
            .qwen4 => {},
            .plugin => |m| {
                m.deinit();
                self.allocator.destroy(m);
            },
        }
        self.allocator.destroy(h);
        self.mtp = null;
    }
    if (self.drafter) |d| {
        d.deinit();
        self.allocator.destroy(d);
        self.drafter = null;
    }
    if (self.dflash) |d| {
        d.deinit();
        self.allocator.destroy(d);
        self.dflash = null;
    }
    if (self.vision_encoder) |v| {
        v.deinit();
        self.allocator.destroy(v);
        self.vision_encoder = null;
    }
    if (self.transformer) |x| {
        x.deinit();
        self.allocator.destroy(x);
        self.transformer = null;
    }
    if (self.weights) |w| {
        w.deinit();
        self.allocator.destroy(w);
        self.weights = null;
    }
}

/// Free only the mlx-allocating state (weights/transformer/vision/
/// drafter/prefix_cache), leaving CPU-only fields (config/tokenizer/
/// chat_config) AND the discovery stub (id/path/bytes_on_disk) intact.
/// Used by eviction so the registry keeps the entry around as
/// `.unloaded` for later listing/reload, AND by `Scheduler.deinit` so
/// mlx frees happen on the inference thread.
pub fn unloadResident(self: *LoadedModel) void {
    // A Thunderbolt pair's model unloads on both Macs; its next load plans again.
    if (distributed.serves(self.path)) distributed.release();
    self.closeEngine();
    if (self.media_plugin) |e| {
        e.deinit();
        self.media_plugin = null;
    }
    self.gen_busy = false;
    freeMlx(self);
    if (self.drafter_path.len > 0) {
        self.allocator.free(self.drafter_path);
        self.drafter_path = "";
    }
    self.drafter_block_size = 0;
    if (self.bytes_resident > 0) registry.generations().last_unload.record(registry.monotonicMs(), self.bytes_resident);
    self.bytes_resident = 0;
    // The prefill-chunk pin was resolved from LIVE memory at load, so it
    // can be stale-narrow (pinned while a since-evicted model was
    // resident). Widening a RESIDENT model's pin would let a prefill run
    // wider than a bill the guard already issued, so the re-resolve rides
    // the reload: clear it here and the next load's `server.pinPrefillChunk`
    // re-pins from then-current memory. `pinned_context` stays — clients
    // budget against it for the whole session.
    if (self.config) |c| c.pinned_prefill_chunk = 0;
    self.state = .unloaded;
}
