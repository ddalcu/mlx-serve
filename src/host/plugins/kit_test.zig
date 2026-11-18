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
/// In the loader's order (file names sorted).
const examples = [_][]const u8{ "_example", "_example_engine", "_example_media", "_example_quant" };

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
    try testing.expectEqual(dylib_abi.Kind.media, inst.set.plugins.items[2].manifest.kind);
    // The receipt Settings lists: the plugin's own version and the file's signer beside its hash.
    for (inst.receipts, inst.set.plugins.items) |r, p| {
        try testing.expectEqualStrings("0.1.0", r.version);
        if (p.manifest.kind == .model or p.manifest.kind == .media) try testing.expectEqualStrings(p.name(), r.model_types[0]);
        try testing.expectEqualStrings(&p.sha256, r.sha256);
        try testing.expect(r.signature.len > 0);
    }
    try testing.expect(inst.set.entries.items[0].model != null and inst.set.entries.items[3].quant != null);
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
    var set = try loader.loadDir(testing.io, testing.allocator, dir, .{ .enabled = true, .disabled = &.{ "_example", "_example_engine" } }, @import("mlx_stamp").bytes);
    defer set.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), set.plugins.items.len);
    try testing.expectEqualStrings("_example_quant", set.plugins.items[0].name());
    try testing.expectEqual(@as(usize, 1), set.refused.items.len);
    try testing.expectEqualStrings("changed since it was installed (hash differs from its receipt)", set.refused.items[0].why);
    // Built against another MLX: refused by negotiation.
    var stale = try loader.loadDir(testing.io, testing.allocator, dir, .{ .enabled = true, .disabled = &.{"_example_media"} }, "mlx=another");
    defer stale.deinit(testing.allocator);
    // The engine never calls MLX: another MLX leaves it loading.
    try testing.expectEqual(@as(usize, 1), stale.plugins.items.len);
    try testing.expectEqualStrings("_example_engine", stale.plugins.items[0].name());
    try testing.expectEqualStrings("plugin was built against another MLX: rebuild it with this release's kit", stale.refused.items[0].why);
    // A rebuild fixes that one, not the changed hash.
    try testing.expect(stale.refused.items[0].rebuild and !set.refused.items[0].rebuild);
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
    try testing.expectError(error.NotClaimed, cli.checkPack(testing.io, a, inst.set.plugins.items[2], pack));
}

test "kit examples: a plugin with no fixture is told apart from one failing the suite, and its pack still checks" {
    var inst = try Installed.init();
    defer inst.deinit();
    const a = inst.arena.allocator();
    const p = inst.set.plugins.items[0];
    const pack = try inst.path("pack");
    try cli.writeFixture(testing.io, a, p, pack);
    // The model example as a plugin whose packs have no tiny form: `plugin check --model` checks the pack alone.
    var m = p.manifest.*;
    m.fixture = &noFixture;
    var bare = p;
    bare.manifest = &m;
    try testing.expectError(error.NoFixture, cli.checkPlugin(testing.io, testing.allocator, bare));
    try testing.expectEqualStrings("_example, 2 layers, hidden 16, vocab 260 (by_type)", try cli.checkPack(testing.io, a, bare, pack));
}

fn noFixture(_: *dylib_abi.Fixture) callconv(.c) dylib_abi.Verdict {
    return dylib_abi.Verdict.refuse(.unsupported, "no tiny form");
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

test "kit blocks: every block file of an exposed area is in kit.blocks, or is a test, contract or CUDA file" {
    inline for (comptime std.meta.declarations(@import("../../kit.zig").blocks)) |area| try expectAreaExposed(area);
}

/// `area` is a folder of `src/sdk/blocks/` when it has one (`embedding` is a file, not an area).
fn expectAreaExposed(comptime area: []const u8) !void {
    const exposed = @field(@import("../../kit.zig").blocks, area);
    var dir = std.Io.Dir.cwd().openDir(testing.io, "src/sdk/blocks/" ++ area, .{ .iterate = true }) catch return;
    defer dir.close(testing.io);
    var it = dir.iterate();
    while (try it.next(testing.io)) |e| {
        if (e.kind != .file or !std.mem.endsWith(u8, e.name, ".zig")) continue;
        const stem = e.name[0 .. e.name.len - 4];
        if (std.mem.endsWith(u8, stem, "_test") or std.mem.startsWith(u8, stem, "contract") or std.mem.eql(u8, stem, "testkit")) continue;
        if (std.mem.indexOf(u8, stem, "_cuda") != null or std.mem.indexOf(u8, stem, "ubench") != null) continue;
        var listed = false;
        inline for (comptime std.meta.declarations(exposed)) |d| listed = listed or std.mem.eql(u8, d, stem);
        if (!listed) std.debug.print("src/sdk/blocks/{s}/{s} is not in kit.blocks.{s} (src/kit.zig)\n", .{ area, e.name, area });
        try testing.expect(listed);
    }
}

test "kit: the module names the sdk files the Ling and GLM runners import, each as that file" {
    // (the name a plugin reaches, its path from `kit`, the file the runner imports: the KDA-hybrid base's trunk, kda.zig
    // and moe.zig, and the base itself, for both runners; bailing_hybrid's plugin.zig, mla.zig and table_forward.zig for
    // Ling; mlx_scope (glm5_next.zig), system (GLM's per-core lever) and mlx_device (the NAX router's naxAvailable and
    // tf32Matmul) for GLM's outside plugin; the kernel contract for a plugin's own kernel gates)
    const want = .{
        .{ "bases.run", .{ "bases", "run" }, @import("../../sdk/bases/run.zig") },
        .{ "bases.blocks.compiled", .{ "bases", "blocks", "compiled" }, @import("../../sdk/bases/blocks/compiled.zig") },
        .{ "bases.blocks.conv1d", .{ "bases", "blocks", "conv1d" }, @import("../../sdk/bases/blocks/conv1d.zig") },
        .{ "bases.blocks.gdn", .{ "bases", "blocks", "gdn" }, @import("../../sdk/bases/blocks/gdn.zig") },
        .{ "bases.blocks.moe", .{ "bases", "blocks", "moe" }, @import("../../sdk/bases/blocks/moe.zig") },
        .{ "bases.blocks.state", .{ "bases", "blocks", "state" }, @import("../../sdk/bases/blocks/state.zig") },
        .{ "bill.prefill", .{ "bill", "prefill" }, @import("../../sdk/bill/prefill.zig") },
        .{ "outside_model", .{"outside_model"}, @import("../../sdk/quant/outside_model.zig") },
        .{ "blocks.kda.gate", .{ "blocks", "kda", "gate" }, @import("../../sdk/blocks/kda/gate.zig") },
        .{ "blocks.kda.recurrence", .{ "blocks", "kda", "recurrence" }, @import("../../sdk/blocks/kda/recurrence.zig") },
        .{ "testing.kernel_contract", .{ "testing", "kernel_contract" }, @import("../../sdk/testing/kernel_contract/root.zig") },
        .{ "log", .{"log"}, @import("../../sdk/util/log.zig") },
        .{ "mlx_ops", .{"mlx_ops"}, @import("../../mlx/ops.zig") },
        .{ "mlx_array", .{"mlx_array"}, @import("../../mlx/array.zig") },
        .{ "mlx_scope", .{"mlx_scope"}, @import("../../mlx/scope.zig") },
        .{ "system", .{"system"}, @import("../../sdk/util/system.zig") },
        .{ "mlx_device", .{"mlx_device"}, @import("../../mlx/device.zig") },
        .{ "bases.kda_hybrid.config", .{ "bases", "kda_hybrid", "config" }, @import("../../sdk/bases/kda_hybrid/config.zig") },
        .{ "bases.kda_hybrid.trunk", .{ "bases", "kda_hybrid", "trunk" }, @import("../../sdk/bases/kda_hybrid/trunk.zig") },
        .{ "bases.kda_hybrid.kda", .{ "bases", "kda_hybrid", "kda" }, @import("../../sdk/bases/kda_hybrid/kda.zig") },
        .{ "bases.kda_hybrid.slots", .{ "bases", "kda_hybrid", "slots" }, @import("../../sdk/bases/kda_hybrid/slots.zig") },
        .{ "bases.kda_hybrid.host", .{ "bases", "kda_hybrid", "host" }, @import("../../sdk/bases/kda_hybrid/host.zig") },
        .{ "bases.kda_hybrid.moe", .{ "bases", "kda_hybrid", "moe" }, @import("../../sdk/bases/kda_hybrid/moe.zig") },
        .{ "bases.kda_hybrid.fixture", .{ "bases", "kda_hybrid", "fixture" }, @import("../../sdk/bases/kda_hybrid/fixture.zig") },
    };
    var missing: usize = 0;
    inline for (want) |w| {
        const ok = if (comptime named(w[1])) |T| T == w[2] else false;
        if (!ok) {
            std.debug.print("kit.{s} does not name the file the runners import\n", .{w[0]});
            missing += 1;
        }
    }
    try testing.expectEqual(@as(usize, 0), missing);
}

/// The declaration a plugin reaches as `kit.<path[0]>.<path[1]>...`, where each step is a type; null where the kit has
/// no such name.
fn named(comptime path: anytype) ?type {
    comptime var T: type = @import("../../kit.zig");
    inline for (path) |d| {
        if (!@hasDecl(T, d) or @TypeOf(@field(T, d)) != type) return null;
        T = @field(T, d);
    }
    return T;
}

test "kit examples: the engine example serves the model it names, passes the engine contract and claims no pack" {
    var inst = try Installed.init();
    defer inst.deinit();
    const a = inst.arena.allocator();
    const p = &inst.set.plugins.items[1];
    try testing.expectEqual(dylib_abi.Kind.engine, p.manifest.kind);
    try testing.expectEqualStrings("_example_engine", inst.receipts[1].models[0]);
    try testing.expectEqual(@as(?*const loader.Plugin, p), loader.engineIn(inst.set.plugins.items, "_example_engine"));
    try testing.expect(loader.engineIn(inst.set.plugins.items, "_example") == null);
    const e = inst.set.entries.items[1];
    try testing.expect(e.model == null and e.media == null and e.quant == null);
    const pack = try inst.path("engine_pack");
    try cli.writeFixture(testing.io, a, p.*, pack);
    try testing.expectEqualStrings("it probes the pack (the engine serves the model folders it names)", try cli.checkPack(testing.io, a, p.*, pack));
    const cfg = try inst.tmp.dir.readFileAlloc(testing.io, "engine_pack/config.json", a, .limited(1 << 16));
    try @import("../../abi/contract_test.zig").run(@ptrCast(@alignCast(p.manifest.table)), .{ .model_dir = pack, .model_type = "", .config_json = cfg, .prompt = &.{ 258, 'h', 'i', 256, 259 } });
}


/// The host's own quant table for the outside probe: it claims a block naming "_outside_toy" and counts its opens and
/// closes; it loads nothing.
const OutsideToy = struct {
    const quant = @import("../../abi/quant.zig");
    var opens: u32 = 0;
    var closes: u32 = 0;
    fn claims(block: quant.Str, _: quant.Str) callconv(.c) quant.Claim {
        return if (std.mem.indexOf(u8, block.slice(), "_outside_toy") != null) .by_type else .no;
    }
    fn validate(_: quant.Str, _: quant.Str, _: *const quant.PackDir) callconv(.c) quant.Verdict {
        return .ok;
    }
    fn bill(_: quant.Str, _: quant.Str, stored: u64) callconv(.c) u64 {
        return stored;
    }
    fn open(_: *quant.LoadCtx) callconv(.c) quant.Verdict {
        opens += 1;
        return .ok;
    }
    fn close(_: *quant.LoadCtx) callconv(.c) void {
        closes += 1;
    }
    fn load(_: *const quant.LoadCtx, _: quant.Str, _: u32, _: *quant.Weight) callconv(.c) quant.Verdict {
        return quant.Verdict.refuse(.unsupported, "_outside_toy");
    }
    fn release(_: *quant.Weight) callconv(.c) void {}
    fn matmul(_: *const quant.Weight, _: quant.Array, _: quant.Stream, _: *quant.Array) callconv(.c) quant.Verdict {
        return quant.Verdict.refuse(.unsupported, "_outside_toy");
    }
    fn moe(_: *const quant.Weight, _: quant.Array, _: quant.Array, _: quant.Array, _: bool, _: quant.Stream, _: *quant.Array) callconv(.c) quant.Verdict {
        return quant.Verdict.refuse(.unsupported, "_outside_toy");
    }
    const table: quant.QuantTable = .{ .name = .of("_outside_toy"), .claims = &claims, .validate = &validate, .bill = &bill, .open = &open, .close = &close, .load_tensor = &load, .load_bank = &load, .release = &release, .matmul = &matmul, .moe = &moe };
};

test "kit outside quant: a kit-built model dylib, its own quant tables empty, opens its pack through the host's quant_claimant; with the member null nothing claims it" {
    const built = std.mem.span(env.get(.MLX_SERVE_KIT_PLUGINS) orelse return error.SkipZigTest);
    const outside = @import("../../sdk/quant/outside.zig");
    const host_impl = @import("../plugin_host_impl.zig");
    const FsPackDir = @import("../../sdk/util/pack_dir.zig").FsPackDir;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(testing.io, &root_buf)];

    // The probe, installed and loaded as any outside model plugin.
    const plugins_dir = try std.fs.path.join(a, &.{ root, "plugins" });
    const src = try std.fmt.allocPrint(a, "{s}/_outside_probe/lib/lib_outside_probe.dylib", .{built});
    switch (try cli.install(testing.io, a, src, plugins_dir, &Installed.yes)) {
        .ok => {},
        .refused => |why| {
            std.debug.print("install _outside_probe: {s}\n", .{why});
            return error.Refused;
        },
    }
    var set = try loader.loadDir(testing.io, testing.allocator, plugins_dir, .{ .enabled = true }, @import("mlx_stamp").bytes);
    defer set.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), set.plugins.items.len);
    const model = set.plugins.items[0].entry().model.?;

    // A pack the toy claims, and the toy registered with the host (the probe's own list stays empty).
    try tmp.dir.createDirPath(testing.io, "pack");
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "pack/config.json", .data = "{\"model_type\":\"_outside_probe\",\"quantization_config\":{\"quant_method\":\"_outside_toy\"}}" });
    const pack = try std.fs.path.join(a, &.{ root, "pack" });
    var fs = try FsPackDir.open(testing.io, pack);
    defer fs.close();
    const pd = fs.packDir();
    const saved = outside.tables.get().*;
    defer outside.tables.get().* = saved;
    outside.tables.get().* = .{};
    outside.register(&OutsideToy.table);
    OutsideToy.opens = 0;
    OutsideToy.closes = 0;

    var b: host_impl.Builtin = undefined;
    b.init(.{ .allocator = testing.allocator });
    defer b.deinit();
    var parsed: model_abi.Parsed = .{};
    var ctx: model_abi.LoadCtx = .{ .model_dir = .of(pack), .dir = &pd, .parsed = &parsed, .host = b.host() };
    var v: model_abi.Verdict = .{};
    const h = model.load(&ctx, &v) orelse {
        std.debug.print("_outside_probe load: {s}\n", .{v.why.slice()});
        return error.ProbeRefused;
    };
    model.unload(h);
    try testing.expectEqual(@as(u32, 1), OutsideToy.opens);
    try testing.expectEqual(@as(u32, 1), OutsideToy.closes);

    // The same host with its member null: the probe's own empty list is all it has.
    var bare = b.table;
    bare.quant_claimant = null;
    ctx.host = &bare;
    v = .{};
    try testing.expect(model.load(&ctx, &v) == null);
    try testing.expectEqualStrings("_outside_probe: no outside quant table claims the pack", v.why.slice());
    try testing.expectEqual(@as(u32, 1), OutsideToy.opens);
}
