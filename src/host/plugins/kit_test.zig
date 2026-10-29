//! The kit's proof: the example plugins, built from the staged kit by `zig build test` (build/kit.zig hands their
//! folder over as `MLX_SERVE_KIT_PLUGINS`), are installed into a scratch plugins folder, loaded by the loader, and pass
//! the conformance suite; the model example's KV cache is read and written across the dylib boundary.

const std = @import("std");
const env = @import("../../env.zig");
const loader = @import("loader.zig");
const cli = @import("cli.zig");
const conformance = @import("../../sdk/testing/conformance.zig");
const test_cache = @import("../../sdk/testing/cache_ops.zig");
const sdk_dylib = @import("../../sdk/dylib.zig");
const dylib_abi = @import("../../abi/dylib.zig");
const model_abi = @import("../../abi/plugin.zig");
const examples = [_][]const u8{ "_example", "_example_media", "_example_quant" };

const testing = std.testing;

/// The scratch plugins folder with every example installed, their receipts, and the loaded set.
const Installed = struct {
    tmp: testing.TmpDir,
    arena: *std.heap.ArenaAllocator,
    receipts: [examples.len]loader.Receipt = undefined,
    set: loader.Set = .{},

    fn init() !Installed {
        const built = std.mem.span(env.get(.MLX_SERVE_KIT_PLUGINS) orelse return error.SkipZigTest);
        const arena = try testing.allocator.create(std.heap.ArenaAllocator);
        arena.* = .init(testing.allocator);
        var self: Installed = .{ .tmp = testing.tmpDir(.{}), .arena = arena };
        errdefer self.deinit();
        const a = arena.allocator();
        const dir = try self.path("plugins");
        for (examples, &self.receipts) |ex, *r| {
            const src = try std.fmt.allocPrint(a, "{s}/{s}/lib/lib{s}.dylib", .{ built, ex, ex });
            switch (try cli.install(testing.io, a, src, dir, &yes)) {
                .ok => |got| r.* = got,
                .refused => |why| {
                    std.debug.print("install {s}: {s}\n", .{ ex, why });
                    return error.Refused;
                },
            }
        }
        self.set = try loader.loadDir(testing.io, testing.allocator, dir, .{ .enabled = true }, @import("mlx_stamp").bytes);
        return self;
    }
    fn yes(_: std.Io, _: cli.Pending) bool {
        return true;
    }
    /// `rel` under the scratch folder (in the arena).
    fn path(self: *Installed, rel: []const u8) ![]const u8 {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const root = buf[0..try self.tmp.dir.realPath(testing.io, &buf)];
        return std.fs.path.join(self.arena.allocator(), &.{ root, rel });
    }
    fn deinit(self: *Installed) void {
        self.set.deinit(testing.allocator);
        self.tmp.cleanup();
        self.arena.deinit();
        testing.allocator.destroy(self.arena);
    }
};

test "kit examples: built from the kit, installed with receipts, loaded, negotiated, and conformant" {
    var inst = try Installed.init();
    defer inst.deinit();
    for (inst.set.refused.items) |r| std.debug.print("refused {s}: {s}\n", .{ r.path, r.why });
    try testing.expectEqual(@as(usize, 0), inst.set.refused.items.len);
    try testing.expectEqual(examples.len, inst.set.plugins.items.len);
    for (inst.set.plugins.items, examples) |p, ex| {
        try testing.expectEqualStrings(ex, p.name());
        try cli.checkPlugin(testing.io, testing.allocator, p);
    }
    try testing.expectEqual(dylib_abi.Kind.media, inst.set.plugins.items[1].manifest.kind);
    // The receipt Settings lists: the plugin's own version and the file's signer beside its hash.
    for (inst.receipts, inst.set.plugins.items) |r, p| {
        try testing.expectEqualStrings("0.1.0", r.version);
        if (p.manifest.kind != .quant) try testing.expectEqualStrings(p.name(), r.model_types[0]);
        try testing.expectEqualStrings(&p.sha256, r.sha256);
        try testing.expect(r.signature.len > 0);
    }
    try testing.expect(inst.set.entries.items[0].model != null and inst.set.entries.items[2].quant != null);
}

test "kit examples: a changed install is refused by its receipt; a disabled one is not opened" {
    var inst = try Installed.init();
    defer inst.deinit();
    const dir = try inst.path("plugins");
    // Another plugin's bytes under this name: the hash no longer matches the receipt.
    var plugins = try inst.tmp.dir.openDir(testing.io, "plugins", .{});
    defer plugins.close(testing.io);
    const other = try plugins.readFileAlloc(testing.io, "_example_quant.dylib", testing.allocator, .limited(1 << 30));
    defer testing.allocator.free(other);
    // A new file renamed over it: this process has the old one mapped.
    try plugins.writeFile(testing.io, .{ .sub_path = "swap.tmp", .data = other });
    try plugins.rename("swap.tmp", plugins, "_example_media.dylib", testing.io);
    var set = try loader.loadDir(testing.io, testing.allocator, dir, .{ .enabled = true, .disabled = &.{"_example"} }, @import("mlx_stamp").bytes);
    defer set.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), set.plugins.items.len);
    try testing.expectEqualStrings("_example_quant", set.plugins.items[0].name());
    try testing.expectEqual(@as(usize, 1), set.refused.items.len);
    try testing.expectEqualStrings("changed since it was installed (hash differs from its receipt)", set.refused.items[0].why);
    // Built against another MLX: refused by negotiation.
    var stale = try loader.loadDir(testing.io, testing.allocator, dir, .{ .enabled = true, .disabled = &.{"_example_media"} }, "mlx=another");
    defer stale.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), stale.plugins.items.len);
    try testing.expectEqualStrings("plugin was built against another MLX: rebuild it with this release's kit", stale.refused.items[0].why);
}

/// The last row's logits of `tokens` fed as `chunks` forwards (prefill, then decode steps) on one fresh cache.
fn lastLogits(pack: *conformance.Loaded, tokens: []const u32, first_chunk: usize) ![]f32 {
    var cache = try test_cache.TestCache.init(testing.allocator, pack.parsed.shell.num_hidden_layers);
    defer cache.deinit();
    const h = [_]model_abi.Handle{cache.handle()};
    var at: usize = 0;
    var logits: []f32 = &.{};
    while (at < tokens.len) {
        const n = if (at == 0) first_chunk else 1;
        var out: model_abi.ForwardOut = .{};
        const pos = [_]u32{@intCast(at)};
        const v = pack.table.forward(pack.model, &.{ .mode = if (n > 1) .prefill else .decode, .rows = @intCast(n), .tokens = .of(tokens[at..][0..n]), .slots = .of(&[_]u32{0}), .positions = .of(&pos), .caches = .of(&h), .cache_ops = &test_cache.ops }, &out);
        if (!v.isOk()) return error.ForwardRefused;
        testing.allocator.free(logits);
        logits = try conformance.consumeF32(testing.allocator, out.logits);
        at += n;
    }
    return logits;
}

test "kit examples: the model plugin's cache crosses the boundary (prefill then decode equals one prefill)" {
    var inst = try Installed.init();
    defer inst.deinit();
    const p = inst.set.plugins.items[0];
    const m = p.manifest;
    var f: dylib_abi.Fixture = .{};
    try testing.expect(m.fixture(&f).isOk());
    defer m.fixture_free(&f);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try conformance.Loaded.open(testing.io, arena.allocator(), p.name(), @ptrCast(@alignCast(m.table)), try sdk_dylib.filesOf(arena.allocator(), &f));
    defer pack.close();
    const tokens = [_]u32{ 1, 5, 9, 3, 30, 2 };
    const whole = try lastLogits(&pack, &tokens, tokens.len);
    defer testing.allocator.free(whole);
    const stepped = try lastLogits(&pack, &tokens, 2);
    defer testing.allocator.free(stepped);
    try testing.expectEqual(@as(usize, 260), whole.len);
    for (whole, stepped) |x, y| try testing.expectApproxEqAbs(x, y, 1e-4);
}

test "kit examples: the model plugin's fixture written as a pack is one it claims and parses" {
    var inst = try Installed.init();
    defer inst.deinit();
    const a = inst.arena.allocator();
    const p = inst.set.plugins.items[0];
    const pack = try inst.path("pack");
    try cli.writeFixture(testing.io, a, p, pack);
    for ([_][]const u8{ "config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors" }) |f|
        try inst.tmp.dir.access(testing.io, try std.fs.path.join(a, &.{ "pack", f }), .{});
    try testing.expectEqualStrings("_example, 2 layers, hidden 16, vocab 260 (by_type)", try cli.checkPack(testing.io, a, p, pack));
    // The media example claims nothing of a model pack.
    try testing.expectError(error.NotClaimed, cli.checkPack(testing.io, a, inst.set.plugins.items[1], pack));
}

test "kit examples: the model plugin's fixture pack tokenizes and renders its chat template as the server will" {
    var inst = try Installed.init();
    defer inst.deinit();
    const a = inst.arena.allocator();
    const pack = try inst.path("pack");
    try cli.writeFixture(testing.io, a, inst.set.plugins.items[0], pack);
    var tok = try @import("../../sdk/tokenizer/load.zig").loadTokenizer(testing.io, a, pack);
    defer tok.deinit();
    try testing.expectEqualSlices(u32, &.{ 258, 'h', 'i', 256 }, try tok.encode(a, "<|user|>hi<|end|>"));
    const chat_config = try @import("../chat/template_load.zig").loadChatConfig(testing.io, a, pack);
    const Message = @import("../chat/types.zig").Message;
    const prompt = try @import("../chat/render.zig").renderChatTemplate(a, &[_]Message{.{ .role = "user", .content = "hi" }}, &chat_config, null, null, false, null, false);
    try testing.expectEqualStrings("<|user|>hi<|end|><|assistant|>", prompt);
}
