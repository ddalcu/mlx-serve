//! The process environment, in one place. Every variable the code reads is registered in `vars` (kind, type,
//! default, owning folder, the release that first shipped it, one line of doc) and read through the accessors
//! below, which refuse an unregistered name at compile time. Nothing else calls getenv/setenv or walks environ
//! (`zig build lint`, rule env-read).
//!
//! When a value is read: `public` and `tuning` variables once, by `init` at startup (before it, as in tests and
//! the iOS library, on every call); `kill` and `fixture` variables on every call, so a call site that latches its
//! answer decides how often. A `diag` is no variable of its own: it is a token of `MLX_SERVE_DIAG` (`STEP_TRACE`
//! is `step-trace`; `token=value` gives a value), read when `MLX_SERVE_DIAG` is.

const std = @import("std");
const builtin = @import("builtin");

const is_windows = builtin.os.tag == .windows;

pub const Kind = enum {
    /// Picks one of two arms of a path (a default-on path's `=0`, or an opt-in arm); retires two releases after
    /// `since`, with its losing arm.
    kill,
    /// Traces, profiles, dumps, probes, µbenches and fault injection; never changes what is served. Armed only
    /// through `MLX_SERVE_DIAG`, so its name carries no `MLX_SERVE_` prefix.
    diag,
    /// A number or plan the code should measure instead; becomes a constant or a per-machine table, two releases
    /// after `since`.
    tuning,
    /// A user-facing or system variable; no expiry.
    public,
    /// A test input: fixture paths and env-gated tests; unset, the test skips.
    fixture,
};

pub const Type = enum { flag, int, float, text };

pub const Var = struct {
    kind: Kind,
    type: Type,
    /// What unset means: a value, a named constant, or the arm in words.
    default: []const u8,
    /// The folder under src/ that reads it.
    owner: []const u8,
    /// The release that first shipped it (`-` for the system's own variables).
    since: []const u8,
    doc: []const u8,
};

/// The registration of `name`; a name missing from `vars` is a compile error.
pub fn info(comptime name: @EnumLiteral()) Var {
    const n = @tagName(name);
    if (!@hasDecl(vars, n)) @compileError("env: " ++ n ++ " is not registered in src/env.zig");
    return @field(vars, n);
}

/// The raw value of a registered variable, or null when unset.
pub fn get(comptime name: @EnumLiteral()) ?[*:0]const u8 {
    return named(@tagName(name));
}

/// `get` for a name built at compile time (a plugin's `tag ++ "_ROUTE_TRACE"`, a helper's comptime parameter).
pub fn named(comptime n: []const u8) ?[*:0]const u8 {
    if (!@hasDecl(vars, n)) @compileError("env: " ++ n ++ " is not registered in src/env.zig");
    const z = comptime std.fmt.comptimePrint("{s}", .{n});
    switch (comptime @field(vars, n).kind) {
        .public, .tuning => if (snapshot_taken) return Slot(n).value,
        .diag => return diagFind(span(.MLX_SERVE_DIAG) orelse "", comptime diagToken(n), &DiagValue(n).buf),
        .kill, .fixture => {},
    }
    return std.c.getenv(z);
}

pub fn span(comptime name: @EnumLiteral()) ?[:0]const u8 {
    return if (get(name)) |v| std.mem.span(v) else null;
}

/// Set and not starting with `0`: `FOO=0` is off, `FOO=` and `FOO=yes` are on.
pub fn on(comptime name: @EnumLiteral()) bool {
    comptime expectType(name, .flag);
    return onValue(get(name));
}

/// Set at all, `FOO=0` included.
pub fn present(comptime name: @EnumLiteral()) bool {
    return get(name) != null;
}

/// Exactly `1`.
pub fn one(comptime name: @EnumLiteral()) bool {
    comptime expectType(name, .flag);
    const v = get(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// Base 10; `default` when unset, empty or unparseable.
pub fn int(comptime name: @EnumLiteral(), comptime T: type, default: T) T {
    comptime expectType(name, .int);
    return intValue(T, get(name), default);
}

/// A finite, non-negative number; `default` otherwise.
pub fn float(comptime name: @EnumLiteral(), default: f32) f32 {
    comptime expectType(name, .float);
    return floatValue(get(name), default);
}

/// `on`, asked once: the first answer is latched in `cache`.
pub fn onCached(cache: *?bool, comptime name: @EnumLiteral()) bool {
    if (cache.*) |v| return v;
    const v = on(name);
    cache.* = v;
    return v;
}

/// A variable whose name is chosen at run time (`--api-key-env`, a provider's key variable, a plugin's lever
/// through the Host table); read on every call, registered or not.
pub fn byName(name: [*:0]const u8) ?[*:0]const u8 {
    return std.c.getenv(name);
}

/// The process environment as the C runtime holds it, for a child process that inherits it (`posix_spawn`).
pub fn block() [*:null]?[*:0]u8 {
    return std.c.environ;
}

/// Whether any diag token starting with `prefix` is armed and not `0` (the qwen4 profilers, `qwen4-profile-`).
pub fn diagAny(comptime prefix: []const u8) bool {
    var it = std.mem.splitScalar(u8, span(.MLX_SERVE_DIAG) orelse "", ',');
    while (it.next()) |item| {
        if (!std.mem.startsWith(u8, item, prefix)) continue;
        const eq = std.mem.indexOfScalar(u8, item, '=') orelse return true;
        if (!std.mem.startsWith(u8, item[eq + 1 ..], "0")) return true;
    }
    return false;
}

/// Reads every `public` and `tuning` variable (`MLX_SERVE_DIAG` among them) once; `main` calls it first.
pub fn init() void {
    @setEvalBranchQuota(10_000);
    inline for (comptime std.meta.declarations(vars)) |n| {
        switch (comptime @field(vars, n).kind) {
            .public, .tuning => Slot(n).value = std.c.getenv(n),
            else => {},
        }
    }
    snapshot_taken = true;
}

/// Sets a registered variable for this process; `get` answers with it from then on.
pub fn set(comptime name: @EnumLiteral(), value: [*:0]const u8) void {
    _ = comptime info(name);
    _ = setForeign(@tagName(name), value);
    if (snapshot_taken) init();
}

/// Sets a variable another library reads (MLX's distributed rank, ds4's kernel sources); false when the C
/// runtime refuses.
pub fn setForeign(name: [*:0]const u8, value: [*:0]const u8) bool {
    return (if (is_windows) c._putenv_s(name, value) else c.setenv(name, value, 1)) == 0;
}

pub fn unset(comptime name: @EnumLiteral()) void {
    _ = comptime info(name);
    _ = if (is_windows) c._putenv_s(@tagName(name), "") else c.unsetenv(@tagName(name));
    if (snapshot_taken) init();
}

/// Test seam: `name` holds `value` (null: unset) until `restore` puts back what was there.
pub fn override(comptime name: @EnumLiteral(), value: ?[*:0]const u8) Override(name) {
    var o: Override(name) = .{};
    if (get(name)) |v| {
        const s = std.mem.span(v);
        o.len = @min(s.len, o.buf.len - 1);
        @memcpy(o.buf[0..o.len], s[0..o.len]);
        o.buf[o.len] = 0;
        o.had = true;
    }
    if (value) |v| set(name, v) else unset(name);
    return o;
}

pub fn Override(comptime name: @EnumLiteral()) type {
    return struct {
        buf: [512]u8 = undefined,
        len: usize = 0,
        had: bool = false,

        pub fn restore(self: *const @This()) void {
            if (self.had) set(name, self.buf[0..self.len :0].ptr) else unset(name);
        }
    };
}

pub fn onValue(raw: ?[*:0]const u8) bool {
    const v = raw orelse return false;
    return v[0] != '0';
}

pub fn intValue(comptime T: type, raw: ?[*:0]const u8, default: T) T {
    const s = std.mem.span(raw orelse return default);
    if (s.len == 0) return default;
    return std.fmt.parseInt(T, s, 10) catch default;
}

pub fn floatValue(raw: ?[*:0]const u8, default: f32) f32 {
    const s = std.mem.span(raw orelse return default);
    if (s.len == 0) return default;
    const v = std.fmt.parseFloat(f32, s) catch return default;
    return if (std.math.isFinite(v) and v >= 0) v else default;
}

const c = struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    extern "c" fn _putenv_s(name: [*:0]const u8, value: [*:0]const u8) c_int;
};

var snapshot_taken = false;

/// One process-lifetime slot per snapshotted variable, keyed by its registration (a decl's address is unique).
fn Slot(comptime n: []const u8) type {
    return SlotOf(&@field(vars, n));
}

fn SlotOf(comptime v: *const Var) type {
    return struct {
        const of = v;
        var value: ?[*:0]const u8 = null;
    };
}

fn expectType(comptime name: @EnumLiteral(), comptime t: Type) void {
    if (info(name).type != t) @compileError("env: " ++ @tagName(name) ++ " is registered as ." ++ @tagName(info(name).type));
}

/// `STEP_TRACE` -> `step-trace`.
fn diagToken(comptime n: []const u8) []const u8 {
    var out: [n.len]u8 = undefined;
    for (n, 0..) |ch, i| out[i] = if (ch == '_') '-' else std.ascii.toLower(ch);
    const final = out;
    return &final;
}

/// A diag's value, NUL-terminated for `get`: one buffer per registration.
fn DiagValue(comptime n: []const u8) type {
    return struct {
        const of = &@field(vars, n);
        var buf: [256]u8 = undefined;
    };
}

/// What the comma list gives `token`: `1` for the bare token, the text after `=` for `token=value` (copied into
/// `out`), null when absent.
fn diagFind(list: []const u8, token: []const u8, out: []u8) ?[*:0]const u8 {
    var it = std.mem.splitScalar(u8, list, ',');
    while (it.next()) |item| {
        if (std.mem.eql(u8, item, token)) return "1";
        if (item.len > token.len and item[token.len] == '=' and std.mem.startsWith(u8, item, token)) {
            const v = item[token.len + 1 ..];
            const len = @min(v.len, out.len - 1);
            @memcpy(out[0..len], v[0..len]);
            out[len] = 0;
            return out[0..len :0].ptr;
        }
    }
    return null;
}

/// Every environment variable the code reads, one line each; sorted by kind, then name.
pub const vars = struct {
    // ── public ──
    pub const CUDA_HOME: Var = .{ .kind = .public, .type = .text, .default = "/usr/local/cuda or /opt/cuda when present", .owner = "mlx", .since = "26.10.2", .doc = "CUDA toolkit root for MLX's JIT on Linux; exported when unset" };
    pub const CUDA_PATH: Var = .{ .kind = .public, .type = .text, .default = "unset", .owner = "mlx", .since = "26.10.2", .doc = "Windows CUDA toolkit: its bin goes first on PATH, and it counts as CUDA_HOME" };
    pub const HF_TOKEN: Var = .{ .kind = .public, .type = .text, .default = "unset", .owner = "host/cli", .since = "26.7.1", .doc = "Hugging Face token sent by `pull`" };
    pub const HOME: Var = .{ .kind = .public, .type = .text, .default = "/tmp where a caller needs one", .owner = "host", .since = "-", .doc = "the user's home: ~/.mlx-serve (models, settings, logs, caches)" };
    pub const HY3D_PAINT_DIR: Var = .{ .kind = .public, .type = .text, .default = "the pack's paint/ dir", .owner = "host/media_jobs", .since = "26.7.2", .doc = "Hunyuan3D paint-stage model dir override (absolute, with a config.json)" };
    pub const LTX_AUDIO_DIR: Var = .{ .kind = .public, .type = .text, .default = "the model dir", .owner = "plugins/ltx", .since = "26.7.1", .doc = "dir holding the LTX audio VAE and vocoder files" };
    pub const LTX_GEMMA_DIR: Var = .{ .kind = .public, .type = .text, .default = "the Gemma encoder under ~/.mlx-serve/models", .owner = "plugins/ltx", .since = "26.6.13", .doc = "LTX text encoder (Gemma) dir, absolute" };
    pub const MLXSERVE_DEVICE: Var = .{ .kind = .public, .type = .text, .default = "the GPU", .owner = "mlx", .since = "26.7.2", .doc = "iOS: `cpu` forces the CPU device" };
    pub const MLXSERVE_LOWMEM: Var = .{ .kind = .public, .type = .flag, .default = "by total RAM", .owner = "host/media_jobs", .since = "26.7.2", .doc = "low-memory media mode (1 on, 0 off)" };
    pub const MLX_ENABLE_TF32: Var = .{ .kind = .public, .type = .flag, .default = "on", .owner = "plugins/qwen4_exp/qsa", .since = "26.9.3", .doc = "MLX's own TF32 switch; the QSA score follows it" };
    pub const MLX_PTX_CACHE_DIR: Var = .{ .kind = .public, .type = .text, .default = "$TMPDIR/mlx/<mlx stamp>/ptx", .owner = "mlx", .since = "26.10.3", .doc = "MLX's CUDA PTX cache; exported per MLX build when unset" };
    pub const MLX_SERVE_ANE_CACHE_CAP_GB: Var = .{ .kind = .public, .type = .int, .default = "40", .owner = "engines/ane", .since = "26.10.3", .doc = "ANE compile cache cap, GB (read by lib/ane)" };
    pub const MLX_SERVE_ANE_CACHE_DIR: Var = .{ .kind = .public, .type = .text, .default = "~/.mlx-serve/ane-cache", .owner = "engines/ane", .since = "26.10.3", .doc = "ANE compile cache root (read by lib/ane)" };
    pub const MLX_SERVE_CACHE_LIMIT: Var = .{ .kind = .public, .type = .text, .default = "by RAM", .owner = "mlx", .since = "26.7.12", .doc = "MLX allocator cache cap" };
    pub const MLX_SERVE_DIAG: Var = .{ .kind = .public, .type = .text, .default = "unset", .owner = ".", .since = "-", .doc = "comma list of diag tokens (`step-trace`, `decode-graph-dump=/path`): arms each diag variable by name" };
    pub const MLX_SERVE_DSV4_DSPARK: Var = .{ .kind = .public, .type = .flag, .default = "on (drafting; --dspark sets 1 to load the stages)", .owner = "engines/mlx/spec", .since = "26.7.12", .doc = "DeepSeek-V4 DSpark drafting (=0 off)" };
    pub const MLX_SERVE_GPU_CEILING_MB: Var = .{ .kind = .public, .type = .int, .default = "0 (the machine's)", .owner = "host/registry", .since = "26.9.5", .doc = "GPU memory ceiling for the admission guards, MiB" };
    pub const MLX_SERVE_KERNEL_DIR: Var = .{ .kind = .public, .type = .text, .default = "unset", .owner = "mlx", .since = "unreleased", .doc = "dir of .metal files that replace the built-in kernel sources by name" };
    pub const MLX_SERVE_KIT: Var = .{ .kind = .public, .type = .text, .default = "~/.mlx-serve/kit", .owner = "host/plugins", .since = "unreleased", .doc = "the plugin kit dir `mlx-serve plugin` builds against" };
    pub const MLX_SERVE_LAYA_BATCH_WINDOW_US: Var = .{ .kind = .public, .type = .int, .default = "0", .owner = "host/media_jobs", .since = "26.9.6", .doc = "Laya decision batch window, microseconds" };
    pub const MLX_SERVE_LAYA_MAX_INPUT_TOKENS: Var = .{ .kind = .public, .type = .int, .default = "DecisionLimits default", .owner = "host/media_jobs", .since = "26.9.6", .doc = "Laya input token limit" };
    pub const MLX_SERVE_LAYA_MAX_QUESTIONS: Var = .{ .kind = .public, .type = .int, .default = "DecisionLimits default", .owner = "host/media_jobs", .since = "26.9.6", .doc = "Laya questions per request limit" };
    pub const MLX_SERVE_RAW_DUMP_FILE: Var = .{ .kind = .public, .type = .text, .default = "unset", .owner = "host/api", .since = "26.7.2", .doc = "path: append every tools request's full pre-parse text (debug log level)" };
    pub const MLX_SERVE_ROUND_COST_PERSIST: Var = .{ .kind = .public, .type = .flag, .default = "off", .owner = "engines/mlx/spec", .since = "26.8.10", .doc = "=1: persist the measured round-cost table" };
    pub const MLX_SERVE_SSD_WRITE_THROUGH: Var = .{ .kind = .public, .type = .flag, .default = "writeThroughEnabledFromEnv(null)", .owner = "engines/mlx", .since = "26.9.2", .doc = "SSD tier write-through at commit" };
    pub const MLX_SERVE_WIRED: Var = .{ .kind = .public, .type = .text, .default = "the default policy", .owner = "mlx", .since = "26.7.12", .doc = "Metal wired-memory policy" };
    pub const PATH: Var = .{ .kind = .public, .type = .text, .default = "empty", .owner = "mlx", .since = "-", .doc = "executable search path; Windows CUDA prepends its bin" };
    pub const SHELL: Var = .{ .kind = .public, .type = .text, .default = "bash or zsh by platform", .owner = "host/launch", .since = "-", .doc = "login shell for `launch`" };
    pub const TEMP: Var = .{ .kind = .public, .type = .text, .default = "/tmp", .owner = "mlx", .since = "-", .doc = "Windows temp dir (the PTX cache root)" };
    pub const TMPDIR: Var = .{ .kind = .public, .type = .text, .default = "/tmp", .owner = "mlx", .since = "-", .doc = "temp dir (the PTX cache root)" };
    pub const USERPROFILE: Var = .{ .kind = .public, .type = .text, .default = "unset", .owner = "sdk/util", .since = "-", .doc = "Windows: the HOME when HOME is unset" };
    pub const XDG_CONFIG_HOME: Var = .{ .kind = .public, .type = .text, .default = "~/.config", .owner = "host/launch", .since = "26.9.3", .doc = "opencode CLI config root for `launch`" };

    // ── kill switches and opt-in arms ──
    pub const MINIMAX_H3_GLUE_FUSED: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "plugins/minimax_h3", .since = "26.10.2", .doc = "H3 fused glue kernels (=0: the op chain)" };
    pub const MINIMAX_H3_VSA: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "plugins/minimax_h3", .since = "26.10.3", .doc = "H3 trained VSA sparse attention (=0: dense attention, no gate branch)" };
    pub const MLX_SERVE_CUDA_QMV_TC: Var = .{ .kind = .kill, .type = .flag, .default = "on where measured (sm_120)", .owner = "sdk/blocks/qmm", .since = "26.10.3", .doc = "CUDA tensor-core 4-bit GEMV for 2..8 rows" };
    pub const MLX_SERVE_DSV41_ATTN_F16: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "DeepSeek-V4.1 f16 attention (=0: f32)" };
    pub const MLX_SERVE_DSV41_CED: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "DeepSeek-V4.1 CED prefill chunks (=0: full forward)" };
    pub const MLX_SERVE_DSV41_HEAD8: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "DeepSeek-V4.1 dense LM head requantized to 8-bit (=0: as shipped)" };
    pub const MLX_SERVE_GDN_STATE_F32: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "engines/mlx", .since = "26.10.2", .doc = "f32 GDN recurrent state (=0: bf16)" };
    pub const MLX_SERVE_H3_RESIDENT: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "plugins/minimax_h3", .since = "26.10.2", .doc = "H3 DiT resident between requests" };

    // ── diagnostics ──
    pub const ANE_DRAFT_DEBUG: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/ane", .since = "26.10.3", .doc = "keeps the ANE drafter program's intermediate tensors for the parity test" };
    pub const ANE_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/ane", .since = "26.10.3", .doc = "ANE engine per-call timings in the log" };
    pub const DECODE_FWD_UBENCH: Var = .{ .kind = .diag, .type = .int, .default = "0", .owner = "engines/mlx", .since = "26.7.12", .doc = "time N decode-width forwards after load (tests/fwd_ubench.sh)" };
    pub const DECODE_FWD_UBENCH_KV: Var = .{ .kind = .diag, .type = .int, .default = "0", .owner = "engines/mlx", .since = "26.8.11", .doc = "KV tokens prefilled before the forward ubench" };
    pub const DECODE_FWD_UBENCH_S: Var = .{ .kind = .diag, .type = .int, .default = "1", .owner = "engines/mlx", .since = "26.8.11", .doc = "rows per forward in the ubench" };
    pub const DECODE_FWD_UBENCH_TREE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/mlx", .since = "26.10.1", .doc = "16-row ubench forwards as a spec tree" };
    pub const DECODE_GRAPH_DUMP: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "engines/mlx", .since = "26.9.6", .doc = "path: print one decode forward's lazy graph" };
    pub const DECODE_PROFILE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/mlx/diag", .since = "26.7.10", .doc = "per-layer decode profile (syncs every layer)" };
    pub const DFLASH_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/mlx/spec/mtp", .since = "26.8.5", .doc = "DFlash per-round trace (exactly 1)" };
    pub const DISPATCH_PROBE: Var = .{ .kind = .diag, .type = .int, .default = "0", .owner = "engines/mlx/diag", .since = "26.7.12", .doc = "count the dispatches of N decode steps" };
    pub const DSPARK_PROFILE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/deepseek_v4", .since = "26.7.12", .doc = "DeepSeek-V4 DSpark per-round profile" };
    pub const DSPARK_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/deepseek_v4", .since = "26.7.12", .doc = "DeepSeek-V4 DSpark draft trace" };
    pub const DSV41_KERNELS_RECEIPT: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel self-check: receipt path" };
    pub const DSV41_KERNEL_DECODE2_RECEIPT: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel decode2 gate: per-output receipt path" };
    pub const DSV41_KERNEL_DRAFT_RECEIPT: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel draft gate: per-output receipt path" };
    pub const DSV41_KERNEL_OPS_RECEIPT: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel ops gate: per-output receipt path" };
    pub const DSV41_KERNEL_PREFILL2_RECEIPT: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel prefill2 gate: per-output receipt path" };
    pub const DSV41_KERNEL_PREFILL_RECEIPT: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel prefill gate: per-output receipt path" };
    pub const DSV41_SELFCHECK_REPORT: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel self-check: log every result" };
    pub const DSV4_SKIP_MOE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/deepseek_v4", .since = "26.8.1", .doc = "set: DeepSeek-V4 decode skips the MoE (timing only, garbage output)" };
    pub const DSV4_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/deepseek_v4", .since = "26.8.1", .doc = "DeepSeek-V4 decode trace (exactly 1)" };
    pub const DUMP_RESTORE: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "engines/mlx/cache", .since = "26.9.3", .doc = "dir: dump restored prefix-cache state" };
    pub const FORCE_GPU_FAMILY_FALLBACK: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "mlx", .since = "26.7.10", .doc = "=1: report no NAX (the pre-NAX GPU-family fallback)" };
    pub const GDN_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/blocks/gdn", .since = "26.7.7", .doc = "GDN µbench: sequential kernel vs bare qmm at 27B shapes (attribution)" };
    pub const GLM53_ROUTES: Var = .{ .kind = .diag, .type = .text, .default = "off", .owner = "plugins/glm_moe_dsa/stream", .since = "26.10.3", .doc = "GLM-5.3 profile builds: write each request's decode routes under this dir" };
    pub const GLM53_WAVE_PROBE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/glm_moe_dsa/stream", .since = "26.10.3", .doc = "GLM-5.3: log each prompt layer's MLX peak by stage" };
    pub const GLM5_NEXT_PREFILL_TIMELINE: Var = .{ .kind = .diag, .type = .text, .default = "off", .owner = "sdk/weights/streamed/mimo", .since = "26.10.3", .doc = "glm5_next profile builds: the prefill timeline" };
    pub const GLM5_NEXT_ROUTE_TRACE: Var = .{ .kind = .diag, .type = .text, .default = "off", .owner = "sdk/weights/streamed/glm5n", .since = "26.10.3", .doc = "glm5_next profile builds: route trace file prefix" };
    pub const GQMM_UBENCH: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "sdk/blocks/moe", .since = "26.10.2", .doc = "gather-QMM NAX ubench (`sweep`: every tile config)" };
    pub const H3_SPLIT_SHARE: Var = .{ .kind = .diag, .type = .float, .default = "the calibrated cut", .owner = "plugins/minimax_h3", .since = "26.10.3", .doc = "H3 cluster: share of rows kept on this Mac (0..1)" };
    pub const KVQ_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/mlx/cache", .since = "26.7.12", .doc = "kv-quant decode attention ubench" };
    pub const LAGUNA_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/laguna", .since = "26.7.12", .doc = "Laguna-XS decode µbench: bf16 attention projections" };
    pub const LAYER_CAP: Var = .{ .kind = .diag, .type = .int, .default = "0 (all layers)", .owner = "engines/mlx/diag", .since = "26.7.12", .doc = "run only the first N layers (timing only, garbage output; DeepSeek-V4 too)" };
    pub const LTX_AUDIO_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/ltx", .since = "26.7.1", .doc = "LTX audio decoder shape trace on stderr" };
    pub const MIMO_DIAG: Var = .{ .kind = .diag, .type = .text, .default = "off", .owner = "sdk/weights/streamed/mimo", .since = "26.10.3", .doc = "mimo profile builds: stream source diagnostics (off | missfree)" };
    pub const MIMO_PREFILL_TIMELINE: Var = .{ .kind = .diag, .type = .text, .default = "off", .owner = "sdk/weights/streamed/mimo", .since = "26.10.3", .doc = "mimo profile builds: the prefill timeline" };
    pub const MIMO_ROUTE_TRACE: Var = .{ .kind = .diag, .type = .text, .default = "off", .owner = "sdk/weights/streamed/mimo", .since = "26.10.3", .doc = "mimo profile builds: route trace file prefix" };
    pub const MINIMAX_H3_ABLATE: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "H3 ablation arms" };
    pub const MLX_FAULT_CHUNK: Var = .{ .kind = .diag, .type = .int, .default = "0", .owner = "mlx", .since = "26.9.2", .doc = "inject a Metal OOM at the n-th prefill checkpoint" };
    pub const MLX_FAULT_KIND: Var = .{ .kind = .diag, .type = .text, .default = "the OOM", .owner = "mlx", .since = "26.10.3", .doc = "injected fault kind (`cuda_lost`)" };
    pub const MLX_FAULT_STEP: Var = .{ .kind = .diag, .type = .int, .default = "0", .owner = "mlx", .since = "26.9.2", .doc = "inject a Metal OOM at the n-th decode checkpoint" };
    pub const MOE_GATHER_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/blocks/moe", .since = "26.7.10", .doc = "trace in the MoE gather ubench" };
    pub const MOE_GATHER_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/blocks/moe", .since = "26.7.10", .doc = "MoE decode gather µbench" };
    pub const MOE_WSUM_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/blocks/moe", .since = "26.10.1", .doc = "MoE prefill expert reduction ubench" };
    pub const MTP_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/mlx/spec", .since = "26.7.7", .doc = "MTP per-round trace (exactly 1)" };
    pub const MTP_TRACE_SYNC: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/mlx/spec/mtp", .since = "26.8.11", .doc = "drain the GPU before each traced MTP verify" };
    pub const MUSIC3_COST_PROBE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/music3", .since = "26.8.6", .doc = "Music3 depth decoder cost laps" };
    pub const PREFILL_QMM_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/blocks/linear", .since = "26.7.10", .doc = "prefill qmm µbench: stock qmm vs dequant+GEMM at 27B prefill shapes" };
    pub const PREFILL_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/mlx/generate", .since = "26.5.6", .doc = "prefill phase trace at info (exactly 1)" };
    pub const QWEN4_DEBUG_SCORES: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/qwen4_exp/qsa", .since = "26.8.11", .doc = "set: qwen4 QSA indexer scores on stderr" };
    pub const QWEN4_DUMP_QSA_BLOCKS: Var = .{ .kind = .diag, .type = .text, .default = "off", .owner = "plugins/qwen4_exp/qsa", .since = "26.9.3", .doc = "dump the QSA selected blocks" };
    pub const QWEN4_GDN_CHECK: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/blocks/gdn", .since = "26.8.11", .doc = "check the fused GDN arm against the op chain per layer" };
    pub const QWEN4_NO_POOLED: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/qwen4_exp/qsa", .since = "26.8.11", .doc = "set: fail when the QSA pooled-key cache is missing" };
    pub const QWEN4_OPCOUNT: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/qwen4_exp", .since = "26.8.11", .doc = "qwen4 forward op counts" };
    pub const QWEN4_PROFILE_ATTN: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/qwen4_exp", .since = "26.10.1", .doc = "qwen4 attention profile" };
    pub const QWEN4_PROFILE_FWD: Var = .{ .kind = .diag, .type = .text, .default = "off", .owner = "plugins/qwen4_exp", .since = "26.8.11", .doc = "qwen4 forward profile (1: verify widths, all: every width)" };
    pub const QWEN4_PROFILE_QSA: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/qwen4_exp/qsa", .since = "26.9.1", .doc = "qwen4 QSA profile" };
    pub const QWEN4_STANDIN: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "sdk/diag", .since = "26.8.11", .doc = "qwen4 stand-in arms, comma list (gdn, ...)" };
    pub const QWEN4_STREAM_F32: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/qwen4_exp", .since = "26.8.11", .doc = "set: qwen4 residual stream in f32 (the math oracle)" };
    pub const STEP_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/mlx/generate", .since = "26.10.2", .doc = "per-step decode trace" };
    pub const SUSHI_DECODE_TICK_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.1", .doc = "sushi EXL3 meter: decode tick ubench" };
    pub const SUSHI_EXL3_CHAIN_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.3", .doc = "sushi EXL3 meter: exl3 chain ubench" };
    pub const SUSHI_EXL3_CODEBOOK_AB: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.1", .doc = "sushi EXL3 meter: exl3 codebook ab" };
    pub const SUSHI_EXL3_GEMM_ARMS: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.2", .doc = "sushi EXL3 meter: exl3 gemm arms" };
    pub const SUSHI_EXL3_GEMM_COUNTS: Var = .{ .kind = .diag, .type = .text, .default = "a synthetic layer", .owner = "sdk/quant/exl3/sushi", .since = "26.10.2", .doc = "sushi EXL3 GEMM meter: per-expert row counts file" };
    pub const SUSHI_EXL3_LAYER_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.1", .doc = "sushi EXL3 meter: exl3 layer ubench" };
    pub const SUSHI_EXL3_LOWK_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.3", .doc = "sushi EXL3 meter: exl3 lowk ubench" };
    pub const SUSHI_EXL3_RG_PREFILL_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "unreleased", .doc = "sushi EXL3 meter: exl3 rg prefill ubench" };
    pub const SUSHI_EXL3_RG_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "unreleased", .doc = "sushi EXL3 meter: exl3 rg ubench" };
    pub const SUSHI_EXL3_SWIGLU_MAXABS: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.1", .doc = "sushi EXL3 meter: exl3 swiglu maxabs" };
    pub const SUSHI_EXL3_UNION_HIST: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.1", .doc = "sushi EXL3 meter: exl3 union hist" };
    pub const SUSHI_EXL3_WIDE_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "unreleased", .doc = "sushi EXL3 meter: exl3 wide ubench" };
    pub const SUSHI_GLM_T1_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.3", .doc = "sushi EXL3 meter: glm t1 ubench" };
    pub const TTS_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "Qwen3-TTS first frames' codes on stderr" };
    pub const VERIFY_WIDTH_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/blocks/linear", .since = "26.8.5", .doc = "spec-verify width sweep µbench" };
    pub const VQMM_FN_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/blocks/linear", .since = "26.10.2", .doc = "verifyQmm Flash Next shapes µbench: stock vs plain vs NAX per width" };
    pub const VQMM_MIXED_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/blocks/linear", .since = "26.7.10", .doc = "verifyQmm mixed-width NAX µbench" };
    pub const VQMM_PARITY_DEBUG: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/blocks/linear", .since = "26.8.10", .doc = "test input read by `expectVerifyQmmNoWorseThanStock`" };
    pub const VQMM_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/blocks/linear", .since = "26.7.7", .doc = "verifyQmm µbench: kernel vs stock per 27B shape" };

    // ── test inputs ──
    pub const ACESTEP_COND: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: condition encoder matches reference" };
    pub const ACESTEP_COVER_COND: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: cover DiT velocity matches reference" };
    pub const ACESTEP_COVER_E2E_LATENTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: cover e2e denoise matches reference" };
    pub const ACESTEP_COVER_E2E_NOISE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: cover e2e denoise matches reference" };
    pub const ACESTEP_COVER_E2E_TEXT_HIDDEN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: cover e2e denoise matches reference" };
    pub const ACESTEP_COVER_NOISE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: cover DiT velocity matches reference" };
    pub const ACESTEP_COVER_SRC: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: cover e2e denoise matches reference" };
    pub const ACESTEP_COVER_V1: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: cover DiT velocity matches reference" };
    pub const ACESTEP_DETOK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: FSQ detokenizer matches reference (cos + scale)" };
    pub const ACESTEP_DIT_NOISE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: DiT velocity matches reference" };
    pub const ACESTEP_DIT_V1: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: DiT velocity matches reference" };
    pub const ACESTEP_E2E_LATENTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: e2e denoise matches reference" };
    pub const ACESTEP_LYRIC_EMBEDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: condition encoder matches reference" };
    pub const ACESTEP_LYRIC_LEN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: condition encoder matches reference" };
    pub const ACESTEP_REFCOND: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: condition encoder with reference audio matches reference" };
    pub const ACESTEP_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/acestep", .since = "26.7.2", .doc = "ACE-Step pack for the live engine tests" };
    pub const ACESTEP_TEXT_HIDDEN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: text encoder hidden states match reference" };
    pub const ACESTEP_TEXT_IDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: prompt tokenization matches reference ids" };
    pub const ACESTEP_TEXT_LEN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: condition encoder matches reference" };
    pub const ACESTEP_TOK_HINTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: FSQ tokenizer hints + indices match reference" };
    pub const ACESTEP_TOK_INDICES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: FSQ tokenizer hints + indices match reference" };
    pub const ACESTEP_TOK_Z: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: FSQ tokenizer hints + indices match reference" };
    pub const ACESTEP_VAEDEC_LAT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: VAE decode matches reference" };
    pub const ACESTEP_VAEDEC_WAV: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: VAE decode matches reference" };
    pub const ACESTEP_VAEENC_AUDIO: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: condition encoder with reference audio matches reference" };
    pub const ACESTEP_VAEENC_MEAN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: VAE encode mean matches reference" };
    pub const ANE_CORES_27B: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: the shared prefill cores for the ANE runner (ANE_CORES_DIR, ANE_CORES_CTXS)" };
    pub const ANE_CORES_CTXS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: the shared prefill cores for the ANE runner (ANE_CORES_DIR, ANE_CORES_CTXS)" };
    pub const ANE_CORES_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: the shared prefill cores for the ANE runner (ANE_CORES_DIR, ANE_CORES_CTXS)" };
    pub const ANE_CORES_NO_GDN: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: the shared prefill cores for the ANE runner (ANE_CORES_DIR, ANE_CORES_CTXS)" };
    pub const ANE_DRAFT_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: the drafter program for the ANE runner (ANE_DRAFT_DIR, ANE_DRAFT_PACK = a drafter dir)" };
    pub const ANE_DRAFT_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: the drafter program for the ANE runner (ANE_DRAFT_DIR, ANE_DRAFT_PACK = a drafter dir)" };
    pub const ANE_DUMP_CTXS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: one lane program for the ANE runner (ANE_DUMP_DIR, ANE_DUMP_PACK, ANE_DUMP_FIRST/LAYERS…" };
    pub const ANE_DUMP_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: one lane program for the ANE runner (ANE_DUMP_DIR, ANE_DUMP_PACK, ANE_DUMP_FIRST/LAYERS…" };
    pub const ANE_DUMP_FIRST: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: one lane program for the ANE runner (ANE_DUMP_DIR, ANE_DUMP_PACK, ANE_DUMP_FIRST/LAYERS…" };
    pub const ANE_DUMP_LANE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: one lane program for the ANE runner (ANE_DUMP_DIR, ANE_DUMP_PACK, ANE_DUMP_FIRST/LAYERS…" };
    pub const ANE_DUMP_LANES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: one lane program for the ANE runner (ANE_DUMP_DIR, ANE_DUMP_PACK, ANE_DUMP_FIRST/LAYERS…" };
    pub const ANE_DUMP_LAYERS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: one lane program for the ANE runner (ANE_DUMP_DIR, ANE_DUMP_PACK, ANE_DUMP_FIRST/LAYERS…" };
    pub const ANE_DUMP_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: one lane program for the ANE runner (ANE_DUMP_DIR, ANE_DUMP_PACK, ANE_DUMP_FIRST/LAYERS…" };
    pub const ANE_HEAD_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: head variants for the ANE runner (ANE_HEAD_DIR, ANE_HEAD_PARTS, ANE_HEAD_NOT)" };
    pub const ANE_HEAD_NOT: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: head variants for the ANE runner (ANE_HEAD_DIR, ANE_HEAD_PARTS, ANE_HEAD_NOT)" };
    pub const ANE_HEAD_PARTS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: head variants for the ANE runner (ANE_HEAD_DIR, ANE_HEAD_PARTS, ANE_HEAD_NOT)" };
    pub const ANE_TEST_LANES: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "engines/ane", .since = "26.10.3", .doc = "test input read by `testModel`" };
    pub const ANE_TEST_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "test input read by `testModel`" };
    pub const CLEF_HTTP_URL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/clef", .since = "26.10.2", .doc = "clef HTTP: discovery, both APIs, images, invalid inputs, isolation and unload (CLEF_HTTP_URL)" };
    pub const CLEF_TEST_EXPECTED: Var = .{ .kind = .fixture, .type = .text, .default = "tests/fixtures/clef/expected.json", .owner = "plugins/clef", .since = "26.10.2", .doc = "Clef HTTP oracle" };
    pub const CLEF_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/clef", .since = "26.10.2", .doc = "clef: live joint head agrees with the published MLX loader" };
    pub const CLEF_TEST_REFERENCE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/clef", .since = "26.10.2", .doc = "clef: live joint head agrees with the published MLX loader" };
    pub const CLEF_TEST_VISION_REFERENCE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/clef", .since = "26.10.2", .doc = "clef: live joint head agrees with the published MLX loader" };
    pub const DFLASH2_ASSISTANT_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx/spec/dflash", .since = "26.8.10", .doc = "dflash2 fixture parity: conv block forward + greedy selector path vs z-lab model_mlx" };
    pub const DFLASH2_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx/spec/dflash", .since = "26.8.10", .doc = "dflash2 fixture parity: conv block forward + greedy selector path vs z-lab model_mlx" };
    pub const DFLASH_ASSISTANT_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx/spec/dflash", .since = "26.8.5", .doc = "dflash fixture parity: encoder + two draft rounds vs transformers reference" };
    pub const DFLASH_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx/spec/dflash", .since = "26.8.5", .doc = "dflash fixture parity: encoder + two draft rounds vs transformers reference" };
    pub const DSV41_BANK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/weights/streamed", .since = "26.10.2", .doc = "DeepSeek-V4.1 EXL3 streamed bank dir for the bank-mode tests" };
    pub const DSV41_BENCH_ATTN: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 bench: one-token decode attention as a dependent chain (DSV41_BENCH_ATTN)" };
    pub const DSV41_BENCH_HC: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 bench: one hyper-connection stage as a dependent chain (DSV41_BENCH_HC = rows)" };
    pub const DSV41_BENCH_MOEK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 bench: one-token FFN as a dependent chain, composed vs kernels (DSV41_BENCH_MOEK)" };
    pub const DSV41_DEC_UBENCH: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the ubenches skip", .owner = "plugins/deepseek_v41", .since = "26.11.1", .doc = "dsv41 decode kernels: each kernel's ubench vs its byte floor and the chain it replaces" };
    pub const DSV41_KERNELS_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel ops gate on the GPU" };
    pub const DSV41_KERNEL_DECODE2_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel decode2 gate: the reference runtime's fixture dir" };
    pub const DSV41_KERNEL_DRAFT_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel draft gate: the reference runtime's fixture dir" };
    pub const DSV41_KERNEL_OPS_FAMILIES: Var = .{ .kind = .fixture, .type = .text, .default = "every family", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel ops gate: the families to replay" };
    pub const DSV41_KERNEL_OPS_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel ops gate: the reference runtime's fixture dir" };
    pub const DSV41_KERNEL_PREFILL2_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel prefill2 gate: the reference runtime's fixture dir" };
    pub const DSV41_KERNEL_PREFILL_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel prefill gate: the reference runtime's fixture dir" };
    pub const DSV41_PHASE0B_MLX: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "sdk/weights/streamed", .since = "26.10.2", .doc = "streamed-bank MLX device tests (a GPU-lock-held run)" };
    pub const DSV41_PHASE0_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/weights/streamed/expert", .since = "26.10.2", .doc = "dsv41 oracle fixture from the reference runtime" };
    pub const DSV41_PHASE1_ROUTE_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/weights/streamed", .since = "26.10.2", .doc = "dsv41 oracle fixture from the reference runtime" };
    pub const DSV41_PHASE2_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/weights/streamed", .since = "26.10.2", .doc = "dsv41 oracle fixture from the reference runtime" };
    pub const DSV41_PHASE4A_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/weights/streamed/expert", .since = "26.10.2", .doc = "dsv41 oracle fixture from the reference runtime" };
    pub const DSV41_PPL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.2", .doc = "dsv41 perplexity: non-overlapping windows of a token corpus through extend (DSV41_PPL)" };
    pub const DSV41_PPL_HEAD8: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 perplexity: non-overlapping windows of a token corpus through extend (DSV41_PPL)" };
    pub const DSV41_PPL_TEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.2", .doc = "dsv41 perplexity: non-overlapping windows of a token corpus through extend (DSV41_PPL)" };
    pub const DSV41_PPL_WINDOWS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/deepseek_v41", .since = "26.10.2", .doc = "dsv41 perplexity: non-overlapping windows of a token corpus through extend (DSV41_PPL)" };
    pub const DSV41_ROUND: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 DSpark round: draft, verify and finish timed apart, greedy (DSV41_ROUND)" };
    pub const DSV41_ROUND_OFFSET: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 DSpark round: draft, verify and finish timed apart, greedy (DSV41_ROUND)" };
    pub const DSV41_ROUND_TEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 DSpark round: draft, verify and finish timed apart, greedy (DSV41_ROUND)" };
    pub const DSV41_SELFCHECK_DEVICE: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 pre-ship gate: the kernels' self-check plan on the device" };
    pub const DSV41_STEP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 decode step: serial one-token steps as the server pipelines them (DSV41_STEP)" };
    pub const DSV41_STEP_CTX: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 decode step: serial one-token steps as the server pipelines them (DSV41_STEP)" };
    pub const DSV41_TINY: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.2", .doc = "dsv41 fixture: prefill, chunked prefill, spans and decode match DeepSeek's reference (DSV41_TINY)" };
    pub const DSV41_VERIFY_ROWS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 verify width: time of a verify block per draft count (DSV41_VERIFY_ROWS)" };
    pub const DSV41_VISION_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "test input read by `fixtureEnv`" };
    pub const DSV41_VISION_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 vision: tower, aligner and span match the reference" };
    pub const DSV4_MINI: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx/spec", .since = "26.7.12", .doc = "dsv4: nextPld on a chokepoint-disabled generator stays serial (DSV4_MINI)" };
    pub const DSV4_REAL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v4", .since = "26.7.12", .doc = "dsv4: forwardPrefill on the REAL mirror continues 'The capital of France is' (DSV4_REAL)" };
    pub const DUPES_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/weights/streamed", .since = "unreleased", .doc = "dupes oracle: the receipt path a pack run writes" };
    pub const DUPES_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/weights/streamed", .since = "unreleased", .doc = "dupes oracle: the pack a run loads through the engine (resident or streamed)" };
    pub const DUPES_PROMPT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/weights/streamed", .since = "unreleased", .doc = "dupes oracle: a file of prompt ids (whitespace or commas)" };
    pub const DUPES_RECEIPTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/weights/streamed", .since = "unreleased", .doc = "dupes oracle: two receipt paths, comma-separated, to compare" };
    pub const DUPES_STEPS: Var = .{ .kind = .fixture, .type = .int, .default = "32", .owner = "sdk/weights/streamed", .since = "unreleased", .doc = "dupes oracle: greedy steps a pack run takes after the prompt" };
    pub const FLINT_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flint", .since = "26.10.3", .doc = "flint fixture: one-shot and chunked prefill + decode vs rec_mlx's model (FLINT_MODEL, FLINT_FIXTURE)" };
    pub const FLINT_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flint", .since = "26.10.3", .doc = "flint fixture: one-shot and chunked prefill + decode vs rec_mlx's model (FLINT_MODEL, FLINT_FIXTURE)" };
    pub const FLUX_DECODED: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux VAE reproduces decoded image" };
    pub const FLUX_IDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux end-to-end pipeline matches reference image" };
    pub const FLUX_IMGIDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux DiT reproduces noise_step0" };
    pub const FLUX_LATENTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux DiT reproduces noise_step0" };
    pub const FLUX_MASK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux end-to-end pipeline matches reference image" };
    pub const FLUX_NOISE0: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux DiT reproduces noise_step0" };
    pub const FLUX_PACKED: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux VAE reproduces decoded image" };
    pub const FLUX_PE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux DiT reproduces noise_step0" };
    pub const FLUX_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux DiT reproduces noise_step0" };
    pub const FLUX_TXTIDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux DiT reproduces noise_step0" };
    pub const GLB_FIXTURE_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/mesh", .since = "26.7.2", .doc = "glb: write swift fixture" };
    pub const GLB_TEXTURED_FIXTURE_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/mesh", .since = "26.7.2", .doc = "glb: write textured swift fixture" };
    pub const GLM53_EXL3_BENCH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm_moe_dsa/glm53_exl3", .since = "26.10.3", .doc = "glm53 EXL3 bench: the pack dir" };
    pub const GLM53_EXL3_TINY_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm_moe_dsa/glm53_exl3", .since = "26.10.3", .doc = "glm53 EXL3: write the tiny fixture's pack to this dir" };
    pub const GLM5_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm5_next", .since = "26.10.2", .doc = "glm5_next fixture: one-shot and chunked prefill + decode vs mlx-vlm glm5_next (GLM5_MODEL…" };
    pub const GLM5_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm5_next", .since = "26.10.2", .doc = "glm5_next fixture: one-shot and chunked prefill + decode vs mlx-vlm glm5_next (GLM5_MODEL…" };
    pub const GLM5_MTP_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm5_next", .since = "26.10.2", .doc = "glm5_next MTP fixture: the pack's mtp.0 layer drafts as mlx-vlm's drafter, one-shot and row by row…" };
    pub const GLM5_REAL_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm5_next", .since = "26.10.2", .doc = "glm5_next verify windows of 2..16 rows track the same rows decoded one at a time (GLM5_REAL_MODEL)" };
    pub const GLM5_VISION_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm5_next", .since = "26.10.2", .doc = "test input read by `fixtureEnv`" };
    pub const GLM5_VISION_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm5_next", .since = "26.10.2", .doc = "glm vision: tower and merger match the reference" };
    pub const H3_VSA_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.10.3", .doc = "h3 vsa: the gather arm equals FastVideo's reference on its own fixture" };
    pub const H3_VSA_UBENCH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.10.3", .doc = "h3 vsa kernel: ubench against dense attention at a 720p clip's packed shape" };
    pub const HY3DP_DINO_IN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint oracle: DINO features + ImageProjModel context match reference" };
    pub const HY3DP_DINO_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint oracle: DINO features + ImageProjModel context match reference" };
    pub const HY3DP_DINO_PROJ: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint oracle: DINO features + ImageProjModel context match reference" };
    pub const HY3DP_FWD_SMOKE: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet: full forward runs end-to-end (shape + finite)" };
    pub const HY3DP_REF_DOWN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: dual-UNet ref cache matches reference" };
    pub const HY3DP_REF_IN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: dual-UNet ref cache matches reference" };
    pub const HY3DP_REF_MID: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: dual-UNet ref cache matches reference" };
    pub const HY3DP_REF_UP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: dual-UNet ref cache matches reference" };
    pub const HY3DP_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint oracle: VAE encode matches reference" };
    pub const HY3DP_UNET_LATENTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: full 2.5D step matches reference" };
    pub const HY3DP_UNET_NORMAL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: full 2.5D step matches reference" };
    pub const HY3DP_UNET_POSITION: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: full 2.5D step matches reference" };
    pub const HY3DP_UNET_POSMAPS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: full 2.5D step matches reference" };
    pub const HY3DP_UNET_V: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: full 2.5D step matches reference" };
    pub const HY3DP_VAE_DEC_IN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint oracle: VAE decode matches reference" };
    pub const HY3DP_VAE_DEC_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint oracle: VAE decode matches reference" };
    pub const HY3DP_VAE_ENC_IN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint oracle: VAE encode matches reference" };
    pub const HY3DP_VAE_ENC_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint oracle: VAE encode matches reference" };
    pub const HY3D_DENOISE_CTX: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: denoise loop matches reference" };
    pub const HY3D_DENOISE_GUIDANCE: Var = .{ .kind = .fixture, .type = .float, .default = "the test's own", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: denoise loop matches reference" };
    pub const HY3D_DENOISE_INIT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: denoise loop matches reference" };
    pub const HY3D_DENOISE_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: denoise loop matches reference" };
    pub const HY3D_DENOISE_STEPS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: denoise loop matches reference" };
    pub const HY3D_DINO_IN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: DINO features match reference" };
    pub const HY3D_DINO_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: DINO features match reference" };
    pub const HY3D_E2E_GRID: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: e2e SDF grid matches reference" };
    pub const HY3D_E2E_MIN: Var = .{ .kind = .fixture, .type = .float, .default = "the test's own", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: e2e SDF grid matches reference" };
    pub const HY3D_E2E_NVERT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: e2e SDF grid matches reference" };
    pub const HY3D_E2E_RES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: e2e SDF grid matches reference" };
    pub const HY3D_SDF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: latent to SDF matches reference" };
    pub const HY3D_SDF_LAT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: latent to SDF matches reference" };
    pub const HY3D_SDF_PTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: latent to SDF matches reference" };
    pub const HY3D_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: DINO features match reference" };
    pub const HY3D_VEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: DiT velocity matches reference" };
    pub const HY3D_VEL_CTX: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: DiT velocity matches reference" };
    pub const HY3D_VEL_LAT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: DiT velocity matches reference" };
    pub const HY3D_VEL_SIGMA: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: DiT velocity matches reference" };
    pub const HY3_MODEL_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "host/chat", .since = "26.7.8", .doc = "renderChatTemplate: REAL Hy3 chat_template.jinja renders without fallback (HY3_MODEL_DIR)" };
    pub const INKLING_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/inkling", .since = "26.7.12", .doc = "test input read by `loadInklingFixtures`" };
    pub const KEV_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/kev", .since = "26.10.1", .doc = "kev: oracle against kev's MLX backend (KEV_TEST_MODEL + KEV_FIXTURES)" };
    pub const KEV_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/kev", .since = "26.10.1", .doc = "kev: oracle against kev's MLX backend (KEV_TEST_MODEL + KEV_FIXTURES)" };
    pub const KOKORO_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/kokoro", .since = "26.7.12", .doc = "Kokoro reference fixtures dir" };
    pub const KOKORO_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/kokoro", .since = "26.7.12", .doc = "Kokoro pack for the live tests" };
    pub const KREA_DECODED: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea VAE reproduces decoded image" };
    pub const KREA_E2E: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea end-to-end pipeline matches reference image" };
    pub const KREA_E2E_H: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea end-to-end pipeline matches reference image" };
    pub const KREA_E2E_SEED: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea end-to-end pipeline matches reference image" };
    pub const KREA_E2E_STEPS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea end-to-end pipeline matches reference image" };
    pub const KREA_E2E_W: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea end-to-end pipeline matches reference image" };
    pub const KREA_IDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea text encoder reproduces hidden states" };
    pub const KREA_MASK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea text encoder reproduces hidden states" };
    pub const KREA_PE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea text encoder reproduces hidden states" };
    pub const KREA_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea VAE encoder round-trips through the decoder" };
    pub const KREA_TE_MIN: Var = .{ .kind = .fixture, .type = .float, .default = "the test's own", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea text encoder reproduces hidden states" };
    pub const KREA_VAE_LATENT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea VAE reproduces decoded image" };
    pub const KREA_VAE_LATH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea VAE reproduces decoded image" };
    pub const KREA_VEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea DiT reproduces velocity" };
    pub const KREA_VEL_CTX: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea DiT reproduces velocity" };
    pub const KREA_VEL_IMG: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea DiT reproduces velocity" };
    pub const KREA_VEL_POS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea DiT reproduces velocity" };
    pub const KREA_VEL_VALID: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea DiT reproduces velocity" };
    pub const LAGUNA_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/blocks/rope", .since = "26.7.10", .doc = "laguna yarn parity vs modeling_laguna.py (LAGUNA_FIXTURES)" };
    pub const LAYA_BENCH: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "plugins/laya", .since = "26.9.6", .doc = "laya: forward latency breakdown (LAYA_BENCH=1)" };
    pub const LAYA_BENCH_GAP_MS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/laya", .since = "26.9.6", .doc = "laya: forward latency breakdown (LAYA_BENCH=1)" };
    pub const LAYA_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "host/tokenizer", .since = "26.9.6", .doc = "byte_fallback: real Laya tokenizer matches HF `tokenizers` on out-of-vocab scripts (LAYA_TEST_MODEL)" };
    pub const LAYA_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "host/tokenizer", .since = "26.9.6", .doc = "byte_fallback: real Laya tokenizer matches HF `tokenizers` on out-of-vocab scripts (LAYA_TEST_MODEL)" };
    pub const LFM2_VISION_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/lfm2", .since = "26.8.6", .doc = "lfm2 vision live: tower parity vs the executed reference" };
    pub const LFM2_VISION_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/lfm2", .since = "26.8.6", .doc = "lfm2 vision live: tower parity vs the executed reference" };
    pub const LLAMA_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/llama", .since = "26.5.7", .doc = "test input read by `testModelPath`" };
    pub const LOGPROBS_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx/grammar", .since = "26.9.6", .doc = "constrained generation returns one logprob entry per token, paired with that token" };
    pub const LTX_ADA_EMB: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT adaLN conditioning reproduces AdaLayerNormSingle" };
    pub const LTX_ADA_PARAMS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT adaLN conditioning reproduces AdaLayerNormSingle" };
    pub const LTX_AUDIO_ENC_LATENT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "audio VAE encoder matches the reference latent (cos)" };
    pub const LTX_AUDIO_ENC_MIN: Var = .{ .kind = .fixture, .type = .float, .default = "the test's own", .owner = "plugins/ltx", .since = "26.7.1", .doc = "audio VAE encoder matches the reference latent (cos)" };
    pub const LTX_AUDIO_ENC_WAVE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "audio VAE encoder matches the reference latent (cos)" };
    pub const LTX_AUDIO_LATENT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "audio VAE decode matches the reference mel (cos)" };
    pub const LTX_AUDIO_MEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "audio VAE decode matches the reference mel (cos)" };
    pub const LTX_AUDIO_MEL_MIN: Var = .{ .kind = .fixture, .type = .float, .default = "the test's own", .owner = "plugins/ltx", .since = "26.7.1", .doc = "audio VAE decode matches the reference mel (cos)" };
    pub const LTX_AUDIO_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/ltx", .since = "26.7.1", .doc = "LTX audio VAE + vocoder dir for the live tests" };
    pub const LTX_AUDIO_WAVE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "vocoder matches the reference waveform (cos)" };
    pub const LTX_AUDIO_WAVE_MIN: Var = .{ .kind = .fixture, .type = .float, .default = "the test's own", .owner = "plugins/ltx", .since = "26.7.1", .doc = "vocoder matches the reference waveform (cos)" };
    pub const LTX_B0_A2VG: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_AAP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_AH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_APP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_ATEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_AVA: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_AVV: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_OUTA: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_OUTV: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_V2AG: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_VAP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_VH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_VPP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_VTEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_CONNT_AIN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx connectorTransform reproduces Embeddings1DConnector" };
    pub const LTX_CONNT_AOUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx connectorTransform reproduces Embeddings1DConnector" };
    pub const LTX_CONNT_VIN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx connectorTransform reproduces Embeddings1DConnector" };
    pub const LTX_CONNT_VOUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx connectorTransform reproduces Embeddings1DConnector" };
    pub const LTX_CONN_AUDIO: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx connectorProject reproduces TextEmbeddingProjection" };
    pub const LTX_CONN_STACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx connectorProject reproduces TextEmbeddingProjection" };
    pub const LTX_CONN_VIDEO: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx connectorProject reproduces TextEmbeddingProjection" };
    pub const LTX_CONVIN_X: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx decoderConv3d matches reference conv_in" };
    pub const LTX_CONVIN_Y: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx decoderConv3d matches reference conv_in" };
    pub const LTX_DIFFVAE_CONTEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae parity: every stage reproduces the reference decoder" };
    pub const LTX_DIFFVAE_LATENT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae parity: every stage reproduces the reference decoder" };
    pub const LTX_DIFFVAE_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae decode runs the whole ladder on the shipped checkpoint" };
    pub const LTX_DIFFVAE_PEAK: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae decode peak stays inside the budget the residency bill promises" };
    pub const LTX_DIFFVAE_PIXELS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae parity: every stage reproduces the reference decoder" };
    pub const LTX_DIFFVAE_STAGE0: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae parity: every stage reproduces the reference decoder" };
    pub const LTX_DIFFVAE_STAGE1: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae parity: every stage reproduces the reference decoder" };
    pub const LTX_DIFFVAE_STAGE2: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae parity: every stage reproduces the reference decoder" };
    pub const LTX_DIFFVAE_VPRED: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae parity: every stage reproduces the reference decoder" };
    pub const LTX_DIFFVAE_XT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae parity: every stage reproduces the reference decoder" };
    pub const LTX_ENC_H: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/ltx", .since = "26.7.1", .doc = "ltx VAE encoder reproduces reference VideoEncoder" };
    pub const LTX_ENC_LATENT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "ltx VAE encoder reproduces reference VideoEncoder" };
    pub const LTX_ENC_PIXELS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "ltx VAE encoder reproduces reference VideoEncoder" };
    pub const LTX_ENC_W: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/ltx", .since = "26.7.1", .doc = "ltx VAE encoder reproduces reference VideoEncoder" };
    pub const LTX_FRAMES_LATENT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT latent-to-frames reproduces reference decode" };
    pub const LTX_FRAMES_REF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT latent-to-frames reproduces reference decode" };
    pub const LTX_FWD_ALAT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditForward reproduces LTXModel velocity" };
    pub const LTX_FWD_ATEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditForward reproduces LTXModel velocity" };
    pub const LTX_FWD_AVEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditForward reproduces LTXModel velocity" };
    pub const LTX_FWD_VLAT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditForward reproduces LTXModel velocity" };
    pub const LTX_FWD_VTEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditForward reproduces LTXModel velocity" };
    pub const LTX_FWD_VVEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditForward reproduces LTXModel velocity" };
    pub const LTX_GEMMA_IDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx gemmaCapture reproduces reference hidden states" };
    pub const LTX_GEMMA_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx gemmaCapture reproduces reference hidden states" };
    pub const LTX_GEMMA_S0: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx gemmaCapture reproduces reference hidden states" };
    pub const LTX_GEMMA_S1: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx gemmaCapture reproduces reference hidden states" };
    pub const LTX_GEMMA_S24: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx gemmaCapture reproduces reference hidden states" };
    pub const LTX_GEMMA_S48: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx gemmaCapture reproduces reference hidden states" };
    pub const LTX_NSA_PARAMS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditNormedSA reproduces compute_video_normed_sa" };
    pub const LTX_NSA_REF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditNormedSA reproduces compute_video_normed_sa" };
    pub const LTX_NSA_VH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditNormedSA reproduces compute_video_normed_sa" };
    pub const LTX_POS_A: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_POS_V: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_ROPE_ACOS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_ROPE_ASIN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_ROPE_AXCOS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_ROPE_AXSIN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_ROPE_VCOS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_ROPE_VSIN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_ROPE_VXCOS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_ROPE_VXSIN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_SAMP_COND_A: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditSampleCfg reproduces guided denoise loop" };
    pub const LTX_SAMP_COND_V: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditSampleCfg reproduces guided denoise loop" };
    pub const LTX_SAMP_FINAL_A: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditSampleCfg reproduces guided denoise loop" };
    pub const LTX_SAMP_FINAL_V: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditSampleCfg reproduces guided denoise loop" };
    pub const LTX_SAMP_NEG_A: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditSampleCfg reproduces guided denoise loop" };
    pub const LTX_SAMP_NEG_V: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditSampleCfg reproduces guided denoise loop" };
    pub const LTX_SAMP_NOISE_A: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditSampleCfg reproduces guided denoise loop" };
    pub const LTX_SAMP_NOISE_V: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditSampleCfg reproduces guided denoise loop" };
    pub const LTX_SAMP_SIGMAS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT dynamicShiftSchedule matches reference" };
    pub const LTX_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx loader: connector + vae_decoder key map" };
    pub const LTX_TEXT_AUDIO: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT encodeTextLtx reproduces PromptEncoder" };
    pub const LTX_TEXT_IDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT encodeTextLtx reproduces PromptEncoder" };
    pub const LTX_TEXT_VIDEO: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT encodeTextLtx reproduces PromptEncoder" };
    pub const LTX_UP_LATENT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "ltx latent upsampler reproduces reference LatentUpsampler" };
    pub const LTX_UP_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "ltx latent upsampler reproduces reference LatentUpsampler" };
    pub const LTX_VAE_DECODED: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx vaeDecode reproduces reference VideoDecoder" };
    pub const LTX_VAE_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.9.1", .doc = "ltx preview: the Latent2RGB preview resembles the decoded frame" };
    pub const LTX_VAE_LATENT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx vaeDecode reproduces reference VideoDecoder" };
    pub const MAGEFLOW_DIT_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow DiT forward parity (env-gated)" };
    pub const MAGEFLOW_DIT_MASKED_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow DiT masked parity (env-gated)" };
    pub const MAGEFLOW_E2E_BF16_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow end-to-end DiT-loop parity bf16 (env-gated)" };
    pub const MAGEFLOW_EDIT_E2E_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow edit loop parity bf16 (env-gated)" };
    pub const MAGEFLOW_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow VAE decode parity (env-gated)" };
    pub const MAGEFLOW_TE_EDIT_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow TE edit parity (env-gated)" };
    pub const MAGEFLOW_TE_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow TE conditioning parity (env-gated)" };
    pub const MAGEFLOW_VAE_ENC_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow VAE encode parity (env-gated)" };
    pub const MAGEFLOW_VAE_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow VAE decode parity (env-gated)" };
    pub const MAGEFLOW_VIT_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow ViT parity (env-gated)" };
    pub const MIMO_Q4_BANK_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/weights/streamed/mimo", .since = "26.10.3", .doc = "mimo q4 bank: fixture path" };
    pub const MIMO_Q4_SDK_STREAM_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "sdk/weights/streamed/mimo", .since = "26.10.3", .doc = "mimo device test: q4 sdk stream gpu" };
    pub const MIMO_ROUTE_TRACE_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "a temp dir", .owner = "sdk/weights/streamed/mimo", .since = "26.10.3", .doc = "mimo route trace policy fixture: output prefix" };
    pub const MIMO_ROUTE_TRACE_FIXTURE_SPLIT: Var = .{ .kind = .fixture, .type = .text, .default = "a temp dir", .owner = "sdk/weights/streamed/mimo", .since = "26.10.3", .doc = "mimo route trace policy fixture: output prefix" };
    pub const MIMO_V2_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mimo_v2", .since = "26.10.2", .doc = "mimo_v2 batched decode: a slot in a two-slot tick tracks the fp32 reference as well as serial…" };
    pub const MIMO_V2_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mimo_v2", .since = "26.10.2", .doc = "mimo_v2 sliding ring: a chunked prefill past the kept rows, then decode, matches the full-buffer…" };
    pub const MIMO_V2_MTP_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mimo_v2", .since = "26.10.2", .doc = "mimo mtp heads track the torch rendering of the MiMo-V2 MTP layer across rounds, drafts and…" };
    pub const MINICPM5_MODEL_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "host/chat", .since = "26.9.1", .doc = "renderChatTemplate: REAL MiniCPM5 chat_template.jinja multi-turn tool history renders without…" };
    pub const MINICPM_VISION_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_5", .since = "26.10.3", .doc = "test input read by `fixtureEnv`" };
    pub const MINICPM_VISION_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_5", .since = "26.10.3", .doc = "minicpm vision: tower stages and merger match the reference" };
    pub const MINIMAX_H3_AUDIO_ENC_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.3", .doc = "minimax h3 audio live: encoder parity vs the executed reference" };
    pub const MINIMAX_H3_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 live: generates a clip" };
    pub const MINIMAX_H3_FRAMES: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 live: generates a clip" };
    pub const MINIMAX_H3_GEMM_UBENCH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.3", .doc = "minimax h3 live: projection gemm ubench" };
    pub const MINIMAX_H3_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 audio live: encoder parity vs the executed reference" };
    pub const MINIMAX_H3_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 live: generates a clip" };
    pub const MINIMAX_H3_PROMPT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 live: generates a clip" };
    pub const MINIMAX_H3_SDPA_UBENCH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 live: sdpa shape ubench" };
    pub const MINIMAX_H3_SIZE: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 live: generates a clip" };
    pub const MINIMAX_H3_SPLIT_OVERFLOW: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.10.2", .doc = "minimax h3 live: the SwiGLU fallback reads every row of a [S, 28672] fc1 output past 2^31" };
    pub const MINIMAX_H3_STEPS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 live: generates a clip" };
    pub const MINIMAX_H3_VAE_ENC_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 vae live: encoder parity vs the executed reference" };
    pub const MINIMAX_H3_VIT_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.3", .doc = "minimax h3 live: vision tower parity vs the executed reference" };
    pub const MINIMAX_H3_WIDTH: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 live: generates a clip" };
    pub const MLX_CUDA_QMV_MAX_ROWS: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "mlx", .since = "26.10.3", .doc = "fp quantized matmul on GB10: 8 rows equal each row alone, bit for bit, and no worse than the tiled…" };
    pub const MLX_SERVE_ANE_FORCE: Var = .{ .kind = .fixture, .type = .text, .default = "unset", .owner = "engines/mlx", .since = "26.8.10", .doc = "allows --ane-prefill on NAX GPUs, where it measured a loss" };
    pub const MLX_SERVE_FORCE_BATCHED: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "engines/mlx", .since = "26.5.6", .doc = "=1: batched decode even with one slot" };
    pub const MLX_SERVE_GGUF_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/quant/gguf", .since = "26.10.1", .doc = "real GGUF: stands in for a model dir, and --engine turns it off (set MLX_SERVE_GGUF_TEST_MODEL)" };
    pub const MLX_SERVE_GOLDEN_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/testing", .since = "unreleased", .doc = "golden-output dir for the golden tests" };
    pub const MLX_SERVE_KERNEL_CONTRACT_OFFLINE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/testing/kernel_contract", .since = "unreleased", .doc = "kernel contract offline: each record's M1/M2 limit is what the toolchain compiles" };
    pub const MLX_SERVE_KIT_PLUGINS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "host/plugins", .since = "unreleased", .doc = "dir of kit-built plugin dylibs the loader test opens" };
    pub const MLX_SERVE_TEST_ANE_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "unreleased", .doc = "test input read by `realPack`" };
    pub const MLX_SERVE_TEST_DS4_GGUF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ds4", .since = "unreleased", .doc = "test input read by `realGguf`" };
    pub const MLX_SERVE_TEST_LLAMA_GGUF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/llama", .since = "unreleased", .doc = "test input read by `realGguf`" };
    pub const MLX_SERVE_TEST_SHARD: Var = .{ .kind = .fixture, .type = .text, .default = "unset: every test", .owner = "build/test_runner.zig", .since = "unreleased", .doc = "k/n: this test process runs every n-th test from k; a `sharded:` test splits its own work the same way" };
    pub const MOE_UBENCH: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "sdk/blocks/moe", .since = "26.10.2", .doc = "moe gather kernels: microbench (MOE_UBENCH=1)" };
    pub const MOSS_IMAGE_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/moss_image", .since = "26.10.3", .doc = "MOSS-Image: the prompt is the release template over one trimmed user turn, thinking on" };
    pub const MOSS_IMAGE_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/moss_image", .since = "26.10.3", .doc = "MOSS-Image: the prompt is the release template over one trimmed user turn, thinking on" };
    pub const MTP_LMHEAD_TEST_FILE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/blocks/linear", .since = "26.9.3", .doc = "verify vocabulary projections preserve solo logits across request widths (MTP_LMHEAD_TEST_FILE)" };
    pub const MTP_REPLAY_REPORT: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "engines/mlx/spec", .since = "26.9.5", .doc = "print the MTP depth-policy replay report" };
    pub const MTP_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx/spec/mtp", .since = "26.9.3", .doc = "plain to prime resumes the same MTP history and next draft as legacy hidden capture" };
    pub const MUSE_VISION_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/muse_glimmer", .since = "26.8.5", .doc = "muse vision live: tower parity vs the executed reference" };
    pub const MUSE_VISION_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/muse_glimmer", .since = "26.8.5", .doc = "muse vision live: tower parity vs the executed reference" };
    pub const MUSIC3_COST_FRAMES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/music3", .since = "26.8.6", .doc = "music3 frame cost probe (env-gated, informational)" };
    pub const MUSIC3_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/music3", .since = "26.8.6", .doc = "Music3 reference fixtures dir" };
    pub const MUSIC3_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/music3", .since = "26.8.6", .doc = "Music3 pack for the live tests" };
    pub const MUSIC3_WAV_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/music3", .since = "26.8.6", .doc = "music3 wav dump harness (env-gated, fixed seed)" };
    pub const NEMOTRON_DIARIZATION_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/nemotron_diarization", .since = "26.10.3", .doc = "nemotron diarization: features, speaker activity and the cache match the references" };
    pub const NEMOTRON_DIARIZATION_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/nemotron_diarization", .since = "26.10.3", .doc = "nemotron diarization: features, speaker activity and the cache match the references" };
    pub const PIXAL3D_DINOV3: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d oracle: crop, DINOv3 tokens, NAF and the projection reproduce every stage's conditioning…" };
    pub const PIXAL3D_FACES: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d end to end: a cutout (PIXAL3D_IMAGE) or posed views (PIXAL3D_VIEWS) become a textured GLB…" };
    pub const PIXAL3D_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "Pixal3D reference fixtures dir" };
    pub const PIXAL3D_FOV_DEG: Var = .{ .kind = .fixture, .type = .float, .default = "the test's own", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d end to end: a cutout (PIXAL3D_IMAGE) or posed views (PIXAL3D_VIEWS) become a textured GLB…" };
    pub const PIXAL3D_IMAGE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d oracle: crop, DINOv3 tokens, NAF and the projection reproduce every stage's conditioning…" };
    pub const PIXAL3D_NAF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "naf oracle: the upsampled map at sampled pixels matches the reference (PIXAL3D_FIXTURES…" };
    pub const PIXAL3D_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d end to end: a cutout (PIXAL3D_IMAGE) or posed views (PIXAL3D_VIEWS) become a textured GLB…" };
    pub const PIXAL3D_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d end to end: a cutout (PIXAL3D_IMAGE) or posed views (PIXAL3D_VIEWS) become a textured GLB…" };
    pub const PIXAL3D_RES: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d end to end: a cutout (PIXAL3D_IMAGE) or posed views (PIXAL3D_VIEWS) become a textured GLB…" };
    pub const PIXAL3D_STEPS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d end to end: a cutout (PIXAL3D_IMAGE) or posed views (PIXAL3D_VIEWS) become a textured GLB…" };
    pub const PIXAL3D_TEX: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d end to end: a cutout (PIXAL3D_IMAGE) or posed views (PIXAL3D_VIEWS) become a textured GLB…" };
    pub const PIXAL3D_VIEWS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d end to end: a cutout (PIXAL3D_IMAGE) or posed views (PIXAL3D_VIEWS) become a textured GLB…" };
    pub const PIXAL3D_WEIGHTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "test input read by `ditOracle`" };
    pub const QV_GH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/vision", .since = "26.6.12", .doc = "qwen vision parity vs reference embeddings (QWEN_VISION_TEST_MODEL)" };
    pub const QV_GW: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/vision", .since = "26.6.12", .doc = "qwen vision parity vs reference embeddings (QWEN_VISION_TEST_MODEL)" };
    pub const QWEN3_ALIGNER_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_asr", .since = "26.10.3", .doc = "qwen3-aligner: the prompt, the classes and the times match the reference" };
    pub const QWEN3_ALIGNER_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_asr", .since = "26.10.3", .doc = "qwen3-aligner: the prompt, the classes and the times match the reference" };
    pub const QWEN3_ASR_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_asr", .since = "26.10.3", .doc = "qwen3-asr: features and audio tower match the reference" };
    pub const QWEN3_ASR_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_asr", .since = "26.10.3", .doc = "qwen3-asr: features and audio tower match the reference" };
    pub const QWEN4_EMBEDDED_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx", .since = "26.10.2", .doc = "local oQ Qwen4 metadata defers convention until loaded weights" };
    pub const QWEN4_EXTERNAL_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx", .since = "26.10.2", .doc = "local legacy Qwen4 metadata keeps external folded default" };
    pub const QWEN4_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen4_exp", .since = "26.8.11", .doc = "qwen4 batched decode: one forwardMoeBatchedDecode tick == two serial ticks (GDN + PLE history + QSA…" };
    pub const QWEN4_FIXTURE_BF16: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen4_exp", .since = "26.8.11", .doc = "qwen4 fixture bf16: residual streams track a bf16 reference (QWEN4_FIXTURE_BF16)" };
    pub const QWEN4_FIXTURE_ROWS: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "sdk/testing", .since = "26.8.11", .doc = "print per-row parity numbers" };
    pub const QWEN4_FIXTURE_VISION: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen4_exp", .since = "26.8.11", .doc = "qwen4 fixture vision: tower parity, pre-tile splice + M-RoPE prefill/decode, QSA mask at 3-D block…" };
    pub const QWEN4_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen4_exp", .since = "26.8.11", .doc = "row-axis verify: every row of a group verify is byte-identical to its solo verify" };
    pub const QWEN4_YARN_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/vision", .since = "26.9.1", .doc = "qwen4 yarn parity vs HF and vLLM (QWEN4_YARN_FIXTURES)" };
    pub const QWEN_IMAGE_E2E_GUIDANCE: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/qwen_image", .since = "26.9.5", .doc = "QwenImage e2e on a real pack (env-gated)" };
    pub const QWEN_IMAGE_E2E_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen_image", .since = "26.9.5", .doc = "QwenImage load leaves the DiT and VAE resident (env-gated)" };
    pub const QWEN_IMAGE_E2E_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen_image", .since = "26.9.5", .doc = "QwenImage e2e on a real pack (env-gated)" };
    pub const QWEN_IMAGE_E2E_SIZE: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/qwen_image", .since = "26.9.5", .doc = "QwenImage e2e on a real pack (env-gated)" };
    pub const QWEN_IMAGE_E2E_STEPS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/qwen_image", .since = "26.9.5", .doc = "QwenImage e2e on a real pack (env-gated)" };
    pub const QWEN_IMAGE_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen_image", .since = "26.9.5", .doc = "QwenImage edit expandImagePads expansion counts" };
    pub const QWEN_IMAGE_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen_image", .since = "26.9.5", .doc = "QwenImage edit expandImagePads expansion counts" };
    pub const QWEN_MID_SYSTEM_MODEL_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "host/chat", .since = "26.10.2", .doc = "mid-system: real Qwen pack loaded through ChatConfig preserves tool-round prefixes" };
    pub const QWEN_PREPROCESS_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/vision", .since = "26.7.11", .doc = "qwen preprocessing parity vs reference pixel_values (QWEN_PREPROCESS_FIXTURE)" };
    pub const QWEN_VISION_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/vision", .since = "26.6.12", .doc = "qwen vision parity vs reference embeddings (QWEN_VISION_TEST_MODEL)" };
    pub const QWEN_VISION_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/vision", .since = "26.6.12", .doc = "qwen vision parity vs reference embeddings (QWEN_VISION_TEST_MODEL)" };
    pub const RADIO_VISION_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/nemotron_h", .since = "26.10.3", .doc = "radio full checkpoint: image/video embeddings and EVS match the official fp32 oracle" };
    pub const RADIO_VISION_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/nemotron_h", .since = "26.10.3", .doc = "radio full checkpoint: image/video embeddings and EVS match the official fp32 oracle" };
    pub const REPLAY_DUMP_SOFT: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "host/chat", .since = "26.7.5", .doc = "tool traffic replay: captured agent traffic survives parse + schema coercion" };
    pub const SA3_CROSS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "sa3 oracle: prompt tokens, T5Gemma rows and conditioning match the reference" };
    pub const SA3_DIT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "sa3 oracle: one DiT velocity, the seeded 8-step sample and the SAME-S decode match the reference" };
    pub const SA3_GLOBAL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "sa3 oracle: prompt tokens, T5Gemma rows and conditioning match the reference" };
    pub const SA3_LAT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "sa3 oracle: one DiT velocity, the seeded 8-step sample and the SAME-S decode match the reference" };
    pub const SA3_PATCH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "sa3 oracle: one DiT velocity, the seeded 8-step sample and the SAME-S decode match the reference" };
    pub const SA3_T5: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "sa3 oracle: prompt tokens, T5Gemma rows and conditioning match the reference" };
    pub const SA3_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "test input read by `testEngine`" };
    pub const SA3_TOK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "sa3 oracle: prompt tokens, T5Gemma rows and conditioning match the reference" };
    pub const SA3_X: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "sa3 oracle: one DiT velocity, the seeded 8-step sample and the SAME-S decode match the reference" };
    pub const SINGLE_MTP_TEST_CONTEXT: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/qwen4_exp", .since = "26.9.3", .doc = "single-stream verify optimizations preserve native outputs and captures (QWEN4_TEST_MODEL)" };
    pub const SUSHI_EXL3_K3_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "src/sdk/quant/exl3/sushi/fixtures/exl3_k3_linear.safetensors", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "sushi K3 mul1 fixture for mlx-stream's EXL3 parity" };
    pub const TOKENIZER_PARITY_CASES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "host/tokenizer", .since = "26.10.2", .doc = "tokenizer parity: encode() matches HF `tokenizers` ids (TOKENIZER_PARITY_MODEL)" };
    pub const TOKENIZER_PARITY_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "host/tokenizer", .since = "26.10.2", .doc = "tokenizer parity: encode() matches HF `tokenizers` ids (TOKENIZER_PARITY_MODEL)" };
    pub const TRELLIS2_DUMP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_FACES: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "TRELLIS.2 reference fixtures dir" };
    pub const TRELLIS2_IMAGE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_LOAD: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_RES: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_SS_DEC: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 oracle: the sparse-structure decoder matches the reference logits (TRELLIS2_SS_DEC…" };
    pub const TRELLIS2_STEPS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_TEX: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_WEIGHTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "TRELLIS.2 weights dir" };
    pub const TTS_FRAMES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "talker+predictor reproduce greedy oracle codes" };
    pub const TTS_INPUT_IDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "talker+predictor reproduce greedy oracle codes" };
    pub const TTS_PARITY_8BIT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.7.2", .doc = "8-bit checkpoint parity vs bf16 (greedy, cloned voice)" };
    pub const TTS_PARITY_BF16: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.7.2", .doc = "8-bit checkpoint parity vs bf16 (greedy, cloned voice)" };
    pub const TTS_QUANT_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.7.2", .doc = "8-bit quantized checkpoint loads and produces non-silent audio" };
    pub const TTS_REF_AUDIO: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "codec decoder reproduces reference audio" };
    pub const TTS_REF_CODES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "talker+predictor reproduce greedy oracle codes" };
    pub const TTS_SPK_EMB: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "speaker encoder (ECAPA-TDNN) matches the Python oracle" };
    pub const TTS_SPK_MEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "speaker encoder (ECAPA-TDNN) matches the Python oracle" };
    pub const TTS_SPK_REF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "speaker encoder (ECAPA-TDNN) matches the Python oracle" };
    pub const TTS_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "generateCodes yields at frame boundaries and stops on cancel" };
    pub const TTS_WAV_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "end-to-end native TTS produces non-silent audio" };
    pub const YUE2_AR_LOGITS0: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 AR path: prefill and a cached decode step match the reference logits" };
    pub const YUE2_AR_LOGITS1: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 AR path: prefill and a cached decode step match the reference logits" };
    pub const YUE2_AR_NEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 AR path: prefill and a cached decode step match the reference logits" };
    pub const YUE2_AR_PREFIX: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 AR path: prefill and a cached decode step match the reference logits" };
    pub const YUE2_NAR_AR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 NAR path: velocity and the seeded 4-step solve match the reference" };
    pub const YUE2_NAR_CODEC: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 NAR path: velocity and the seeded 4-step solve match the reference" };
    pub const YUE2_NAR_LAT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 NAR path: velocity and the seeded 4-step solve match the reference" };
    pub const YUE2_NAR_PREFIX_LEN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 NAR path: velocity and the seeded 4-step solve match the reference" };
    pub const YUE2_NAR_STATE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 NAR path: velocity and the seeded 4-step solve match the reference" };
    pub const YUE2_NAR_V: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 NAR path: velocity and the seeded 4-step solve match the reference" };
    pub const YUE2_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "test input read by `testEngine`" };
    pub const YUE2_TOK_IDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 tokenizer: the tiktoken ranks encode like the reference" };
    pub const YUE2_TOK_TEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 tokenizer: the tiktoken ranks encode like the reference" };
    pub const YUE2_VAE_AUDIO: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 VAE: two tiles of random latents decode like the reference" };
    pub const YUE2_VAE_LAT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 VAE: two tiles of random latents decode like the reference" };
    pub const _GPU_WINDOW_LOCKED: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the GPU test refuses", .owner = "plugins", .since = "26.10.2", .doc = "set by the GPU window orchestrator: the GPU-lock-held child may run device tests" };
};

// ── tests ──

const testing = std.testing;

test "value readers: on is set-and-not-0, int and float fall back on empty or garbage" {
    try testing.expect(!onValue(null));
    try testing.expect(!onValue("0"));
    try testing.expect(!onValue("0x"));
    try testing.expect(onValue(""));
    try testing.expect(onValue("1"));
    try testing.expect(onValue("on"));
    try testing.expectEqual(@as(usize, 7), intValue(usize, null, 7));
    try testing.expectEqual(@as(usize, 7), intValue(usize, "", 7));
    try testing.expectEqual(@as(usize, 7), intValue(usize, "x", 7));
    try testing.expectEqual(@as(usize, 42), intValue(usize, "42", 7));
    try testing.expectEqual(@as(f32, 0.5), floatValue("x", 0.5));
    try testing.expectEqual(@as(f32, 0.5), floatValue("-1", 0.5));
    try testing.expectEqual(@as(f32, 0.5), floatValue("inf", 0.5));
    try testing.expectEqual(@as(f32, 2.5), floatValue("2.5", 0.5));
}

test "MLX_SERVE_DIAG: a bare token reads 1, token=value reads the value, a prefix matches nothing" {
    const list = "step-trace,decode-graph-dump=/tmp/g,empty=";
    var buf: [16]u8 = undefined;
    try testing.expectEqualStrings("1", std.mem.span(diagFind(list, "step-trace", &buf).?));
    try testing.expectEqualStrings("/tmp/g", std.mem.span(diagFind(list, "decode-graph-dump", &buf).?));
    try testing.expectEqualStrings("", std.mem.span(diagFind(list, "empty", &buf).?));
    try testing.expect(diagFind(list, "step", &buf) == null);
    try testing.expect(diagFind(list, "decode-graph", &buf) == null);
    try testing.expect(diagFind("", "step-trace", &buf) == null);
    try testing.expectEqualStrings("qwen4-gdn-check", comptime diagToken("QWEN4_GDN_CHECK"));
}

test "registry: diag tokens are unique and unprefixed, and every entry carries an owner, a release and a doc" {
    comptime {
        @setEvalBranchQuota(1_000_000);
        const names = std.meta.declarations(vars);
        for (names, 0..) |a, i| {
            const v = @field(vars, a);
            if (v.owner.len == 0 or v.since.len == 0 or v.doc.len == 0) @compileError(a ++ ": owner, since and doc are required");
            if (v.kind != .diag) continue;
            if (std.mem.startsWith(u8, a, "MLX_SERVE_")) @compileError(a ++ ": a diag is a token of MLX_SERVE_DIAG, not a variable");
            for (names[i + 1 ..]) |b| {
                if (@field(vars, b).kind == .diag and std.mem.eql(u8, diagToken(a), diagToken(b)))
                    @compileError(a ++ " and " ++ b ++ " share a diag token");
            }
        }
    }
}

test "init snapshots public variables, set refreshes the snapshot, MLX_SERVE_DIAG arms a diag by token" {
    const dump = override(.MLX_SERVE_RAW_DUMP_FILE, "/a");
    const diag = override(.MLX_SERVE_DIAG, "step-trace,qwen4-profile-fwd=0,layer-cap=3");
    defer {
        diag.restore();
        dump.restore();
        snapshot_taken = false;
    }
    try testing.expect(on(.STEP_TRACE)); // before init the list is read on every call
    init();
    try testing.expectEqualStrings("/a", span(.MLX_SERVE_RAW_DUMP_FILE).?);
    set(.MLX_SERVE_RAW_DUMP_FILE, "/b");
    try testing.expectEqualStrings("/b", span(.MLX_SERVE_RAW_DUMP_FILE).?);
    try testing.expect(on(.STEP_TRACE));
    try testing.expectEqual(@as(u32, 3), int(.LAYER_CAP, u32, 0));
    try testing.expect(!diagAny("qwen4-profile-"));
    set(.MLX_SERVE_DIAG, "qwen4-profile-attn");
    try testing.expect(!on(.STEP_TRACE) and diagAny("qwen4-profile-"));
}

test "override restores what was there, set or unset" {
    {
        const o = override(.MLX_SERVE_DSV4_DSPARK, "0");
        try testing.expectEqualStrings("0", span(.MLX_SERVE_DSV4_DSPARK).?);
        const inner = override(.MLX_SERVE_DSV4_DSPARK, null);
        try testing.expect(get(.MLX_SERVE_DSV4_DSPARK) == null);
        inner.restore();
        try testing.expectEqualStrings("0", span(.MLX_SERVE_DSV4_DSPARK).?);
        o.restore();
    }
}
