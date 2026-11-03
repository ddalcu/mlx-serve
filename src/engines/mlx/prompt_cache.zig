//! engines/mlx/prompt_cache.
//! Owns PrefillCache, resetCache, tryRestoreCache, savePromptCache.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ssmFreeQsaState = @import("../../plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
const KVCacheEntry = @import("kv/cache.zig").KVCacheEntry;
const freeKVEntry = @import("kv/cache.zig").freeKVEntry;
const newEmptyKVEntry = @import("kv/cache.zig").newEmptyKVEntry;
const SSMCacheEntry = @import("ssm/entry.zig").SSMCacheEntry;
const ssmFreeSpecCapture = @import("ssm/entry.zig").ssmFreeSpecCapture;
const Transformer = @import("transformer.zig").Transformer;

// ── Prompt Cache (snapshot of KV + SSM state for prefix reuse) ──

pub const PrefillCache = struct {
    tokens: []u32,
    kv_entries: []KVCacheEntry,
    offsets: []usize,
    kv_step: usize,
    ssm_entries: ?[]SSMCacheEntry,
    moe_seq_offset: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *PrefillCache) void {
        self.allocator.free(self.tokens);
        for (self.kv_entries) |*e| {
            freeKVEntry(e);
        }
        self.allocator.free(self.kv_entries);
        self.allocator.free(self.offsets);
        if (self.ssm_entries) |entries| {
            for (entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                ssmFreeQsaState(e);
            }
            self.allocator.free(entries);
        }
    }
};

/// Reset all caches for a new request (KV cache + SSM state for MoE).
pub fn resetCache(self: *Transformer) !void {
    const prev_config = self.cache.config;
    try self.cache.reinit(self.config.num_hidden_layers, prev_config);
    if (self.ssm_entries) |entries| {
        for (entries) |*e| {
            ssmFreeSpecCapture(e);
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
            e.conv_state = mlx.mlx_array_new();
            e.ssm_state = mlx.mlx_array_new();
            e.ple_prev_valid = false;
            e.initialized = false;
        }
    }
    self.moe_seq_offset = 0;
}

/// Try to restore state from prompt cache if the cached tokens are an exact
/// prefix of new_ids. Returns the number of matched (restored) tokens, or 0
/// if the cache missed and a full reset was performed.
pub fn tryRestoreCache(self: *Transformer, new_ids: []const u32) !usize {
    const pc = self.prompt_cache orelse {
        try self.resetCache();
        return 0;
    };

    const match_limit = @min(pc.tokens.len, new_ids.len);
    var matched: usize = 0;
    while (matched < match_limit) : (matched += 1) {
        if (pc.tokens[matched] != new_ids[matched]) break;
    }

    if (matched < pc.tokens.len or matched >= new_ids.len) {
        try self.resetCache();
        return 0;
    }

    // Full prefix match with tokens remaining — restore cached state.
    const prev_config = self.cache.config;
    try self.cache.reinit(self.config.num_hidden_layers, prev_config);
    self.cache.step = pc.kv_step;
    for (pc.kv_entries, 0..) |src, i| {
        if (src.initialized) {
            try mlx.check(mlx.mlx_array_set(&self.cache.entries[i].keys, src.keys));
            try mlx.check(mlx.mlx_array_set(&self.cache.entries[i].values, src.values));
            if (prev_config.scheme == .affine) {
                try mlx.check(mlx.mlx_array_set(&self.cache.entries[i].keys_scales, src.keys_scales));
                try mlx.check(mlx.mlx_array_set(&self.cache.entries[i].keys_biases, src.keys_biases));
                try mlx.check(mlx.mlx_array_set(&self.cache.entries[i].values_scales, src.values_scales));
                try mlx.check(mlx.mlx_array_set(&self.cache.entries[i].values_biases, src.values_biases));
            }
            self.cache.entries[i].initialized = true;
            self.cache.entries[i].offset = pc.offsets[i];
            // *_view fields left as mlx_array_new() — recreated on next update()
        }
    }

    if (pc.ssm_entries) |ssm_src| {
        if (self.ssm_entries) |ssm_dst| {
            for (ssm_src, ssm_dst) |src, *dst| {
                _ = mlx.mlx_array_free(dst.conv_state);
                _ = mlx.mlx_array_free(dst.ssm_state);
                dst.conv_state = mlx.mlx_array_new();
                dst.ssm_state = mlx.mlx_array_new();
                dst.initialized = src.initialized;
                // Per-field null guard — LFM2 gated_conv layers fill only
                // conv_state, never ssm_state, even after initialization.
                if (src.conv_state.ctx != null) {
                    try mlx.check(mlx.mlx_array_set(&dst.conv_state, src.conv_state));
                }
                if (src.ssm_state.ctx != null) {
                    try mlx.check(mlx.mlx_array_set(&dst.ssm_state, src.ssm_state));
                }
            }
        }
    }

    self.moe_seq_offset = pc.moe_seq_offset;
    return matched;
}

/// Snapshot the current KV cache + SSM state so the next request can reuse
/// them if its prompt starts with the same token prefix.
pub fn savePromptCache(self: *Transformer, prompt_ids: []const u32) void {
    if (self.prompt_cache) |*pc| pc.deinit();
    self.prompt_cache = null;

    const tokens = self.allocator.dupe(u32, prompt_ids) catch return;
    const num_layers = self.cache.entries.len;
    const kv = self.allocator.alloc(KVCacheEntry, num_layers) catch {
        self.allocator.free(tokens);
        return;
    };
    const offsets = self.allocator.alloc(usize, num_layers) catch {
        self.allocator.free(tokens);
        self.allocator.free(kv);
        return;
    };
    const scheme = self.cache.config.scheme;
    for (self.cache.entries, kv, 0..) |src, *dst, i| {
        dst.* = newEmptyKVEntry();
        dst.offset = src.offset;
        if (src.initialized) {
            _ = mlx.mlx_array_set(&dst.keys, src.keys);
            _ = mlx.mlx_array_set(&dst.values, src.values);
            if (scheme == .affine) {
                _ = mlx.mlx_array_set(&dst.keys_scales, src.keys_scales);
                _ = mlx.mlx_array_set(&dst.keys_biases, src.keys_biases);
                _ = mlx.mlx_array_set(&dst.values_scales, src.values_scales);
                _ = mlx.mlx_array_set(&dst.values_biases, src.values_biases);
            }
        }
        dst.initialized = src.initialized;
        offsets[i] = src.offset;
    }

    var ssm: ?[]SSMCacheEntry = null;
    if (self.ssm_entries) |entries| {
        const ssm_copy = self.allocator.alloc(SSMCacheEntry, entries.len) catch return;
        for (entries, ssm_copy) |src, *dst| {
            dst.conv_state = mlx.mlx_array_new();
            dst.ssm_state = mlx.mlx_array_new();
            dst.initialized = src.initialized;
            // Per-field null guard — LFM2 gated_conv layers fill only
            // conv_state, never ssm_state, even when `initialized==true`.
            if (src.conv_state.ctx != null) {
                _ = mlx.mlx_array_set(&dst.conv_state, src.conv_state);
            }
            if (src.ssm_state.ctx != null) {
                _ = mlx.mlx_array_set(&dst.ssm_state, src.ssm_state);
            }
        }
        ssm = ssm_copy;
    }

    self.prompt_cache = .{
        .tokens = tokens,
        .kv_entries = kv,
        .offsets = offsets,
        .kv_step = self.cache.step,
        .ssm_entries = ssm,
        .moe_seq_offset = self.moe_seq_offset,
        .allocator = self.allocator,
    };
}
