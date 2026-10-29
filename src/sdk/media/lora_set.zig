//! An image backend's attached LoRA stack, reconciled with each request's `set_loras`: an empty list detaches, the
//! same paths and scales in the same order reuse what is attached, anything else reloads every adapter. Adapters
//! are summed at forward time (`lora.Slot`), never merged into the base weight.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const lora = @import("../weights/lora.zig");
const log = @import("../util/log.zig");

const Verdict = abi.Verdict;

pub const LoraSet = struct {
    /// Owns the adapter arrays the backend's slots point at: it lives until the next detach.
    stack: ?lora.Stack = null,
    matched: u32 = 0,

    /// `dit` declares `attachLora(*const lora.Stack) u32` (the attachment count) and `detachLora()`.
    pub fn set(self: *LoraSet, a: std.mem.Allocator, loras: []const abi.Lora, arch: lora.Arch, dit: anytype) Verdict {
        if (loras.len == 0) {
            self.clear(dit);
            return .ok;
        }
        if (loras.len > lora.MAX_LORAS) return Verdict.refuse(.unsupported, "too many LoRA adapters requested");
        var paths: [lora.MAX_LORAS][]const u8 = undefined;
        var scales: [lora.MAX_LORAS]f32 = undefined;
        for (loras, 0..) |l, i| {
            paths[i] = l.path.slice();
            scales[i] = l.scale;
        }
        if (self.stack) |*st| if (st.matches(paths[0..loras.len], scales[0..loras.len])) return .ok;
        self.clear(dit);
        var stack: lora.Stack = .{ .allocator = a };
        for (paths[0..loras.len], scales[0..loras.len]) |p, sc| {
            const dup = a.dupe(u8, p) catch {
                stack.deinit();
                return Verdict.refuse(.failed, "out of memory");
            };
            const file = lora.loadFile(a, p, arch) catch |e| {
                a.free(dup);
                stack.deinit();
                return switch (e) {
                    error.BadLoraPath => Verdict.refuse(.unsupported, "'lora_path'/'lora_paths' must be absolute path(s) to .safetensors file(s)"),
                    else => Verdict.refuse(.unsupported, "failed to load a LoRA file"),
                };
            };
            stack.files[stack.count] = file;
            stack.paths[stack.count] = dup;
            stack.scales[stack.count] = sc;
            stack.count += 1;
        }
        const matched = dit.attachLora(&stack);
        if (matched == 0) {
            stack.deinit();
            return Verdict.refuse(.unsupported, "LoRA(s) have no modules matching this model's DiT — wrong LoRA for this architecture?");
        }
        if (matched != stack.entryCount())
            log.warn("[image] lora: matched {d} module-attachment(s) across {d} adapter(s); {d} of {d} adapter module(s) target modules this model does not have\n", .{ matched, stack.count, stack.entryCount() - matched, stack.entryCount() })
        else
            log.info("[image] lora: matched {d} module-attachment(s) across {d} adapter(s)\n", .{ matched, stack.count });
        self.stack = stack;
        self.matched = matched;
        return .ok;
    }

    pub fn clear(self: *LoraSet, dit: anytype) void {
        dit.detachLora();
        if (self.stack) |*st| st.deinit();
        self.stack = null;
        self.matched = 0;
    }
};
