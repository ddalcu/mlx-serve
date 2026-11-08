//! The example engine plugin, built from the kit as a dylib: a whole engine of its own (`kit.abi.engine`) for the
//! models it names. It never calls MLX and runs its own thread: the server hands it requests and renders its tokens,
//! as it does for the in-tree ds4 (raw Metal) and llama.cpp. This one streams `config.reply`; a real engine runs its
//! runtime in `round`. `mlx-serve plugin new <name> --engine` copies this folder.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.engine;
pub const config = @import("config.zig");

/// Shown at install and in Settings.
pub const version = "0.1.0";

pub const plugin = kit.sdk.Plugin{ .name = "_example_engine", .kind = .engine, .models = &config.models, .options = &.{
    .{ .name = config.reply_option, .kind = .text, .default = config.reply, .help = "What it answers, one byte a token" },
} };

pub const table = abi.Engine{
    .probe = &probe,
    .load_bytes = &loadBytes,
    .open = &open,
    .close = &close,
    .submit = &submit,
    .cancel = &cancel,
    .step = null,
    .status = &status,
    .memory = &memory,
    .report_json = null,
    .trim = null,
    .media_rows = null,
    .embed = null,
    .decide = null,
    .tokenize = null,
    .token_bytes = null,
    .chat_template = null,
    .render = null,
};

const gpa = std.heap.c_allocator;
const io = std.Io.Threaded.global_single_threaded.io();
/// Plain sampling only: a request asking for more is refused by name (`abi.refusal`).
const caps: abi.Caps = .{};
const max_concurrent = 4;

const Req = struct {
    id: u64,
    prompt_len: u32,
    max_tokens: u32,
    /// It names an EOS: the reply ends after one pass, as a model's would.
    stops: bool,
    /// The engine's `State.reply`.
    reply: []const u8,
    sink: abi.Sink,
    sent: u32 = 0,
    prefilled: std.atomic.Value(bool) = .init(false),
    cancelled: std.atomic.Value(bool) = .init(false),
    finished: bool = false,

    fn emit(r: *Req, ev: abi.Event) abi.Action {
        return r.sink.event(r.sink.ctx, r.id, &ev);
    }
    fn finish(r: *Req, reason: @FieldType(abi.Event, "reason")) void {
        _ = r.emit(.{ .kind = .finished, .reason = reason });
        r.finished = true;
    }
};

const State = struct {
    mu: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    reqs: std.ArrayList(*Req) = .empty,
    closing: bool = false,
    worker: std.Thread = undefined,
    /// `config.reply`, or its option's value when the host set one: read once, at open.
    reply: []const u8 = config.reply,
};

fn probe(_: abi.Str, config_json: abi.Str, out: *abi.Verdict) callconv(.c) bool {
    if (std.mem.indexOf(u8, abi.items(u8, config_json), "\"" ++ config.model_type ++ "\"") != null) return true;
    out.* = .{ .code = .unsupported, .why = abi.span(u8, "config.json names another model_type than " ++ config.model_type) };
    return false;
}

fn loadBytes(_: *const abi.Open, bytes: *u64) callconv(.c) abi.Verdict {
    bytes.* = 1 << 20;
    return .{};
}

fn open(_: *const abi.Open, info: *abi.Info, out: *abi.Verdict) callconv(.c) ?*anyopaque {
    const st = gpa.create(State) catch {
        out.* = .{ .code = .failed, .why = abi.span(u8, "out of memory") };
        return null;
    };
    st.* = .{};
    const want = config.replyFor(if (std.c.getenv(config.reply_option)) |v| std.mem.span(v) else null);
    if (want.ptr != config.reply.ptr) st.reply = gpa.dupe(u8, want) catch config.reply;
    st.worker = std.Thread.spawn(.{}, work, .{st}) catch {
        freeReply(st);
        gpa.destroy(st);
        out.* = .{ .code = .failed, .why = abi.span(u8, "could not start the engine thread") };
        return null;
    };
    info.* = .{
        .name = abi.span(u8, plugin.name),
        .version = abi.span(u8, version),
        .caps = caps,
        .threading = .own_thread,
        .context_window = 4096,
        .max_concurrent = max_concurrent,
        .embedding_dims = 0,
        .eos = .{},
        .startup = abi.span(u8, "example engine: no model, no MLX, its own thread"),
    };
    return st;
}

/// Cancels every request; the worker finishes each before it exits.
fn close(e: *anyopaque) callconv(.c) void {
    const st: *State = @ptrCast(@alignCast(e));
    st.mu.lockUncancelable(io);
    st.closing = true;
    for (st.reqs.items) |r| r.cancelled.store(true, .release);
    st.cond.broadcast(io);
    st.mu.unlock(io);
    st.worker.join();
    st.reqs.deinit(gpa);
    freeReply(st);
    gpa.destroy(st);
}

fn freeReply(st: *State) void {
    if (st.reply.ptr != config.reply.ptr) gpa.free(st.reply);
}

fn submit(e: *anyopaque, id: u64, r: *const abi.Request, sink: abi.Sink) callconv(.c) abi.Verdict {
    const st: *State = @ptrCast(@alignCast(e));
    if (abi.refusal(caps, abi.needs(r))) |v| return v;
    const req = gpa.create(Req) catch return .{ .code = .failed, .why = abi.span(u8, "out of memory") };
    req.* = .{ .id = id, .prompt_len = @intCast(r.prompt.len), .max_tokens = r.max_tokens, .stops = r.eos.len > 0, .reply = st.reply, .sink = sink };
    st.mu.lockUncancelable(io);
    defer st.mu.unlock(io);
    if (st.reqs.items.len >= max_concurrent or st.closing) {
        gpa.destroy(req);
        return .{ .code = .busy, .why = abi.span(u8, "the example engine runs 4 requests at most") };
    }
    st.reqs.append(gpa, req) catch {
        gpa.destroy(req);
        return .{ .code = .failed, .why = abi.span(u8, "out of memory") };
    };
    st.cond.broadcast(io);
    return .{};
}

fn cancel(e: *anyopaque, id: u64) callconv(.c) void {
    const st: *State = @ptrCast(@alignCast(e));
    st.mu.lockUncancelable(io);
    defer st.mu.unlock(io);
    for (st.reqs.items) |r| if (r.id == id) r.cancelled.store(true, .release);
}

fn status(e: *anyopaque, out: *abi.Status) callconv(.c) void {
    const st: *State = @ptrCast(@alignCast(e));
    st.mu.lockUncancelable(io);
    defer st.mu.unlock(io);
    var running: u32 = 0;
    for (st.reqs.items) |r| running += @intFromBool(r.prefilled.load(.acquire));
    out.* = .{ .running = running, .waiting = @as(u32, @intCast(st.reqs.items.len)) - running };
}

fn memory(_: *anyopaque, out: *abi.Memory) callconv(.c) void {
    out.* = .{};
}

/// The engine's thread: a round for every live request, until `close` and none is left. The lock is never held
/// across a sink call, so the host's `status` and `memory` answer from inside one.
fn work(st: *State) void {
    var live: [max_concurrent]*Req = undefined;
    while (true) {
        st.mu.lockUncancelable(io);
        while (st.reqs.items.len == 0 and !st.closing) st.cond.waitUncancelable(io, &st.mu);
        const n = st.reqs.items.len;
        @memcpy(live[0..n], st.reqs.items);
        st.mu.unlock(io);
        if (n == 0) return;
        for (live[0..n]) |r| round(r);
        st.mu.lockUncancelable(io);
        var i: usize = 0;
        while (i < st.reqs.items.len) {
            if (st.reqs.items[i].finished) gpa.destroy(st.reqs.swapRemove(i)) else i += 1;
        }
        st.mu.unlock(io);
    }
}

/// One step of `r`: its prefill, else its next token, else its end.
fn round(r: *Req) void {
    if (r.cancelled.load(.acquire)) return r.finish(.cancelled);
    if (!r.prefilled.load(.acquire)) {
        r.prefilled.store(true, .release);
        _ = r.emit(.{ .kind = .prefilled, .prompt_tokens = r.prompt_len });
        return;
    }
    if (r.sent == r.max_tokens) return r.finish(.length);
    if (r.stops and r.sent == r.reply.len) return r.finish(.stop);
    const tok = [_]u32{config.token(r.reply, r.sent)};
    r.sent += 1;
    if (r.emit(.{ .kind = .tokens, .tokens = abi.span(u32, &tok) }) == .stop) r.finish(.stop);
}

/// The tiny pack `plugin fixture` writes and the conformance suite opens: a config.json its probe reads and the byte
/// tokenizer with a chat template, so the server tokenizes and renders for it.
pub fn conformanceFixture(a: std.mem.Allocator) !kit.sdk.fixture.Fixture {
    const fx = kit.sdk.fixture;
    const specials = [_]kit.sdk.byte_tokenizer.Special{
        .{ .id = 256, .content = "<|end|>" },  .{ .id = 257, .content = "<|system|>" },
        .{ .id = 258, .content = "<|user|>" }, .{ .id = 259, .content = "<|assistant|>" },
    };
    return .{ .files = try a.dupe(fx.File, &.{
        .{ .name = "config.json", .bytes = "{\"model_type\":\"" ++ config.model_type ++ "\",\"vocab_size\":260,\"eos_token_id\":256}" },
        .{ .name = "tokenizer.json", .bytes = try kit.sdk.byte_tokenizer.json(a, &specials) },
        .{ .name = "tokenizer_config.json", .bytes =
        \\{"eos_token":"<|end|>","chat_template":"{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"}
        },
    }) };
}
