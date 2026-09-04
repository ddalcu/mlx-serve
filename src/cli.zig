//! CLI subcommands — `mlx-serve run|pull|list <model>` — Ollama-grade
//! ergonomics for the terminal.
//!
//!   mlx-serve run gemma4        # download if missing, serve, drop into a REPL
//!   mlx-serve pull qwen3.6      # download only
//!   mlx-serve list              # what's on disk
//!
//! Short names resolve through a curated alias table (mirroring the MLX
//! Core app catalog in ChatModels.swift); anything containing '/' is
//! treated as a HuggingFace repo id directly ("org/repo", with optional
//! "hf.co/" prefix and ":tag" suffix). Downloads land in
//! `~/.mlx-serve/models/<org>/<repo>` — the single source of truth shared
//! with the app's DownloadManager and the server's media-dep resolution.
//!
//! Transport is the system `curl` (always present on macOS): rock-solid
//! TLS/redirect/HTTP2 handling, `-C -` resume, `--create-dirs`, and a free
//! progress bar on the CLI path. Pure helpers (alias resolution, tree-JSON
//! parsing, file filtering, REPL body/line codecs) are hermetically tested
//! here; only the thin curl/spawn wrappers need a live network.

const std = @import("std");
const build_options = @import("build_options");
const ollama = @import("ollama.zig");
const model_discovery = @import("model_discovery.zig");
const log = @import("log.zig");

// ── Unparsed-argument reporting ─────────────────────────────────────────

/// Why main.zig's flag loop could not consume an argument.
///
/// The loop matches every flag by EXACT name and reads its value from the
/// NEXT argv slot, and it used to end with no else branch at all — so a
/// misspelled flag, or the `--flag=value` shape it never accepted, was
/// dropped in silence. That is the worst possible outcome for a launcher: the
/// flag parses as far as the user can tell, `--help` documents it, and the
/// server boots clean while ignoring what was asked for. Rejecting loudly is
/// the whole point; the variants exist only to make the message actionable.
pub const ArgReject = enum {
    /// `--model=/path` — value welded to the flag name.
    equals_form,
    /// A flag in the LAST argv slot, so its value never arrived.
    missing_value,
    /// Not a flag we know.
    unknown,

    /// Trailing advice for the error message. Never empty.
    pub fn hint(self: ArgReject) []const u8 {
        return switch (self) {
            .equals_form => "flags take their value as a separate argument (--model <path>, not --model=<path>)",
            .missing_value => "this flag expects a value after it, or it is misspelled",
            .unknown => "see --help for the flag list",
        };
    }
};

/// Classify an argument the flag loop fell through on. `is_last` is true when
/// it occupied the final argv slot — the only way a known value-taking flag
/// can reach the else branch (every such arm is guarded on `i + 1 < args.len`).
pub fn classifyUnparsedArg(arg: []const u8, is_last: bool) ArgReject {
    const is_flag = std.mem.startsWith(u8, arg, "-");
    if (is_flag and std.mem.indexOfScalar(u8, arg, '=') != null) return .equals_form;
    if (is_flag and is_last) return .missing_value;
    return .unknown;
}

// ── Alias table ─────────────────────────────────────────────────────────

pub const Alias = struct {
    /// Short name before the ':', e.g. "gemma4".
    name: []const u8,
    /// Tag after the ':'; empty = selectable only by full name:tag.
    tag: []const u8,
    repo: []const u8,
    /// Picked when the user gives the bare name with no tag.
    is_default: bool = false,
    /// Non-empty: restrict the download to this single .gguf artifact.
    gguf_file: []const u8 = "",
    /// Non-empty: restrict the download to this SUBDIR of the repo, and strip
    /// the prefix so it lands as a flat model dir. For repos that ship several
    /// variants side by side (e.g. SceneWorks/illustrious-xl-v2-mlx holds
    /// `bf16/`, `q4/`, `q8/`, each a full diffusers repo). Pair with `dest_name`
    /// so the variants don't collide on disk.
    subdir: []const u8 = "",
    /// Non-empty: the on-disk `<org>/<name>` to store under, overriding the repo
    /// id. Lets `illustrious:q4` and `illustrious:q8` (same repo, different
    /// subdir) land in distinct dirs that discovery finds as separate models.
    dest_name: []const u8 = "",
    /// Non-empty: pull exactly this ONE root file and nothing else — the
    /// single-file LDM distribution `sdxl_single_file` reads. Twins the app's
    /// `ImageModelPreset.singleFileCheckpoint`, and is a REQUIREMENT rather
    /// than an optimisation on a repo that ships a diffusers folder beside the
    /// checkpoint (NoobAI): a downloaded `model_index.json` sends
    /// `Engine.loadAuto` down the folder path, and NoobAI V-Pred's own
    /// `scheduler/scheduler_config.json` declares `"prediction_type":
    /// "epsilon"` — wrong for it. Only the checkpoint carries the `v_pred` /
    /// `ztsnr` marker tensors that say otherwise.
    single_file: []const u8 = "",
    /// The repo requires accepting a licence on HuggingFace before ANY file is
    /// readable, token or not. Only changes the failure MESSAGE: a manifest
    /// fetch that 403s otherwise reads as a typo or a network problem, and
    /// "set HF_TOKEN" is unhelpful advice to someone who already has one.
    gated: bool = false,

    /// The alias' download shape, as `pullRepo` reads it.
    pub fn resolve(self: Alias) Resolved {
        return .{
            .repo = self.repo,
            .gguf_file = self.gguf_file,
            .subdir = self.subdir,
            .dest_name = self.dest_name,
            .single_file = self.single_file,
            .gated = self.gated,
        };
    }
};

/// Mirrors the app catalog (`gemmaModelOptions` in ChatModels.swift) for chat
/// models. The SDXL image-model entries are deliberately WIDER than the
/// app's own `ImageModelPreset.all` — the picker curates a short list (base,
/// Turbo, one Illustrious quant) to keep the Image pane from turning into a
/// checkpoint shelf, while every SDXL variant the app knows how to load stays
/// one `pull` away here. Bare-name defaults pick the 4-bit build that fits
/// the widest range of Macs for that family.
pub const aliases = [_]Alias{
    .{ .name = "gemma4", .tag = "e2b", .repo = "mlx-community/gemma-4-e2b-it-4bit" },
    .{ .name = "gemma4", .tag = "e2b-8bit", .repo = "mlx-community/gemma-4-e2b-it-8bit" },
    .{ .name = "gemma4", .tag = "e4b", .repo = "mlx-community/gemma-4-e4b-it-4bit", .is_default = true },
    .{ .name = "gemma4", .tag = "e4b-8bit", .repo = "mlx-community/gemma-4-e4b-it-8bit" },
    .{ .name = "gemma4", .tag = "12b", .repo = "mlx-community/gemma-4-12b-it-4bit" },
    .{ .name = "gemma4", .tag = "12b-8bit", .repo = "mlx-community/gemma-4-12b-it-8bit" },
    .{ .name = "gemma4", .tag = "26b", .repo = "mlx-community/gemma-4-26b-a4b-it-4bit" },
    .{ .name = "gemma4", .tag = "26b-8bit", .repo = "mlx-community/gemma-4-26b-a4b-it-8bit" },
    .{ .name = "gemma4", .tag = "31b", .repo = "mlx-community/gemma-4-31b-it-4bit" },
    .{ .name = "gemma4", .tag = "31b-8bit", .repo = "mlx-community/gemma-4-31b-it-8bit" },
    .{ .name = "gemma3", .tag = "12b", .repo = "mlx-community/gemma-3-12b-it-4bit", .is_default = true },
    // Qwen 3.6 27B ships an MTP sidecar the server auto-loads for
    // multi-token speculative decode — the best default experience.
    .{ .name = "qwen3.6", .tag = "27b", .repo = "ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve", .is_default = true },
    .{ .name = "qwen3.5", .tag = "0.8b", .repo = "mlx-community/Qwen3.5-0.8B-MLX-4bit", .is_default = true },
    .{ .name = "deepseek-v4", .tag = "flash", .repo = "antirez/deepseek-v4-gguf", .is_default = true, .gguf_file = "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf" },
    // Tencent Hunyuan 3 (hy_v3, 295B-A21B MoE) — mixed 2/3-bit experts +
    // 8-bit attention/router/shared, MTP layer included. ~110 GB on disk;
    // needs a 128 GB Mac.
    .{ .name = "hy3", .tag = "295b", .repo = "mlx-community/Hy3-oQ2e", .is_default = true },
    // OpenAI gpt-oss. The MXFP4-Q8 conversions keep the native mxfp4 expert
    // banks (what the model was released in) and put attention/embeddings at
    // affine 8-bit: ~12 GB for the 20B, ~63 GB for the 120B.
    .{ .name = "gpt-oss", .tag = "20b", .repo = "mlx-community/gpt-oss-20b-MXFP4-Q8", .is_default = true },
    .{ .name = "gpt-oss", .tag = "120b", .repo = "mlx-community/gpt-oss-120b-MXFP4-Q8" },
    .{ .name = "bge-small", .tag = "en", .repo = "mlx-community/bge-small-en-v1.5-8bit", .is_default = true },
    // ── SDXL image models (media gen) ──
    // Stable Diffusion XL 1.0 — the full base model, real CFG. Same backend
    // serves every entry below it.
    .{ .name = "sdxl", .tag = "base", .repo = "stabilityai/stable-diffusion-xl-base-1.0", .is_default = true },
    // SDXL Turbo — adversarially distilled, guidance-free, 1-4 steps.
    .{ .name = "sdxl", .tag = "turbo", .repo = "stabilityai/sdxl-turbo" },
    // Pony Diffusion V6 XL — a single-file LDM checkpoint (Civitai), served by
    // `sdxl_single_file`. The repo also carries a standalone `sdxl_vae`
    // checkpoint; `single_file` names the one we want rather than landing both
    // and leaving the loader's header markers to tell them apart.
    .{ .name = "pony", .tag = "v6", .repo = "LyliaEngine/Pony_Diffusion_V6_XL", .is_default = true, .single_file = "ponyDiffusionV6XL_v6StartWithThisOne.safetensors" },
    // NoobAI-XL — single-file LDM checkpoints, same `sdxl_single_file` path.
    // `v1.1` is epsilon-prediction; `vpred` ships the v-prediction +
    // zero-terminal-SNR marker tensors `sdxl_single_file` reads at load. Both
    // repos ALSO carry a complete diffusers folder, so `single_file` is what
    // keeps `model_index.json` off disk — see `Alias.single_file` for why
    // taking that folder would silently sample V-Pred on an epsilon ladder.
    .{ .name = "noobai", .tag = "v1.1", .repo = "Laxhar/noobai-XL-1.1", .is_default = true, .single_file = "NoobAI-XL-v1.1.safetensors" },
    .{ .name = "noobai", .tag = "vpred", .repo = "Laxhar/noobai-XL-Vpred-1.0", .single_file = "NoobAI-XL-Vpred-v1.0.safetensors" },
    // Illustrious XL v2 — one repo, three diffusers variants in subfolders. Each
    // `subdir` is pulled flat into its own dest so they coexist as separate
    // models. q4/q8 need the SDXL affine-quant path (`sdxl_nn` QLinear); bf16 is
    // dense. Bare `illustrious` = q4 (widest Mac fit), matching the 4-bit-default
    // convention above.
    .{ .name = "illustrious", .tag = "q4", .repo = "SceneWorks/illustrious-xl-v2-mlx", .is_default = true, .subdir = "q4", .dest_name = "SceneWorks/illustrious-xl-v2-q4" },
    .{ .name = "illustrious", .tag = "q8", .repo = "SceneWorks/illustrious-xl-v2-mlx", .subdir = "q8", .dest_name = "SceneWorks/illustrious-xl-v2-q8" },
    .{ .name = "illustrious", .tag = "bf16", .repo = "SceneWorks/illustrious-xl-v2-mlx", .subdir = "bf16", .dest_name = "SceneWorks/illustrious-xl-v2-bf16" },
    // Stable Diffusion 1.5 — the original, non-XL checkpoint. Served by the
    // `sd1` backend (`sd1_pipeline.zig`), which reuses SDXL's UNet/VAE/CLIP-L
    // building blocks at SD 1.5's own config (single 768-wide tower, no
    // micro-conditioning). Diffusers-folder repo only for now — a Civitai
    // single-file SD 1.5 checkpoint is not yet convertible (`sd1_pipeline.zig`'s
    // header).
    .{ .name = "sd1", .tag = "v1.5", .repo = "stable-diffusion-v1-5/stable-diffusion-v1-5", .is_default = true },
    // SD-Turbo — an SD 2.1 distill, NOT an SD 1.5 one, but the same
    // `StableDiffusionPipeline` shape the `sd1` backend serves: one text
    // tower (OpenCLIP-H, 1024-wide — `sdxl.CLIP_H_CONFIG`, chosen from the
    // checkpoint's own `text_encoder/config.json` at load, never assumed
    // from the alias). Guidance-free, 1-4 steps (`timestep_spacing:
    // "trailing"` in its own scheduler config).
    .{ .name = "sd1", .tag = "turbo", .repo = "stabilityai/sd-turbo" },
    // ── Stable Diffusion 3.5 (media gen) ──
    // A different family from everything above it: flow matching, an MMDiT
    // instead of a UNet, a 16-channel VAE, and THREE text encoders (CLIP-L,
    // CLIP-G and T5-XXL). Served by `sd3_pipeline.zig`; the schedule and
    // geometry live in `sd3.zig`.
    //
    // Every stability SD 3.5 repo is licence-GATED, so `gated` is set on all of
    // them — a pull without an accepting token 403s at the manifest, and the
    // message has to say what to actually do about it.
    //
    // The T5 tower alone is 9.5 GB and is shared by all three, so bare `sd3.5`
    // is MEDIUM (~16 GB) rather than the 4-bit-default convention above: on this
    // family the smallest complete download is a different checkpoint, not a
    // narrower quantization of the same one.
    .{ .name = "sd3.5", .tag = "medium", .repo = "stabilityai/stable-diffusion-3.5-medium", .is_default = true, .gated = true },
    .{ .name = "sd3.5", .tag = "large", .repo = "stabilityai/stable-diffusion-3.5-large", .gated = true },
    // Large-Turbo: the same transformer, VAE, towers and declared scheduler as
    // Large — 4 steps at guidance 1 is the entire difference (`sd3.TURBO_CONFIG`).
    .{ .name = "sd3.5", .tag = "large-turbo", .repo = "stabilityai/stable-diffusion-3.5-large-turbo", .gated = true },
};

pub const Resolved = struct {
    repo: []const u8,
    gguf_file: []const u8 = "",
    subdir: []const u8 = "",
    dest_name: []const u8 = "",
    /// See `Alias.single_file`.
    single_file: []const u8 = "",
    /// See `Alias.gated`.
    gated: bool = false,

    /// The `<org>/<name>` to store under — `dest_name` when set, else the repo.
    pub fn destRepo(self: Resolved) []const u8 {
        return if (self.dest_name.len > 0) self.dest_name else self.repo;
    }
};

/// Short name / repo ref → HF repo id. Accepts:
///   "gemma4" / "gemma4:12b"           (alias table)
///   "org/repo" / "org/repo:tag"       (direct, tag stripped)
///   "hf.co/org/repo", "huggingface.co/org/repo"
/// Returns null for unknown alias-shaped names (no '/').
pub fn resolveShortName(name: []const u8) ?Resolved {
    var n = name;
    for ([_][]const u8{ "hf.co/", "huggingface.co/", "https://huggingface.co/" }) |prefix| {
        if (std.ascii.startsWithIgnoreCase(n, prefix)) {
            n = n[prefix.len..];
            break;
        }
    }
    n = ollama.stripTag(n);
    if (n.len == 0) return null;
    if (std.mem.indexOfScalar(u8, n, '/') != null) {
        // Direct org/repo reference.
        return .{ .repo = n };
    }
    // Alias lookup: "name" or "name:tag" (tag was stripped above — redo the
    // split on the ORIGINAL string so alias tags still work).
    var base = name;
    var tag: []const u8 = "";
    if (std.mem.lastIndexOfScalar(u8, name, ':')) |ci| {
        base = name[0..ci];
        tag = name[ci + 1 ..];
    }
    if (std.mem.eql(u8, tag, "latest")) tag = "";
    for (aliases) |a| {
        if (!std.ascii.eqlIgnoreCase(a.name, base)) continue;
        if (tag.len == 0) {
            if (a.is_default) return a.resolve();
        } else if (std.ascii.eqlIgnoreCase(a.tag, tag)) {
            return a.resolve();
        }
    }
    return null;
}

/// `~/.mlx-serve/models/<org>/<repo>` — the single models root shared with
/// the app's DownloadManager.
pub fn modelDestPath(allocator: std.mem.Allocator, home: []const u8, repo: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/.mlx-serve/models/{s}", .{ home, repo });
}

pub fn modelsRootPath(allocator: std.mem.Allocator, home: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/.mlx-serve/models", .{home});
}

/// Inputs to "which models should this boot discover?" — see
/// `shouldDefaultModelsRoot`.
pub const RootDefaulting = struct {
    /// Invoked as `mlx-serve serve` / `mlx-serve run <model>` rather than flags.
    subcommand: bool,
    /// The process will serve HTTP (subcommand, or the `--serve` flag).
    serve_mode: bool,
    /// `--model <path>` (or `run <model>`) named a specific checkpoint.
    has_explicit_model: bool,
};

/// Should an unspecified `--model-dir` fall back to `~/.mlx-serve/models`?
///
/// That path is already the single source of truth everywhere else — `pull`
/// writes there, `list` reads there, the app's DownloadManager and both
/// resolvers agree on it — so a server told to serve, but given neither a model
/// nor a directory, has exactly one sensible place to look. Without this a bare
/// `mlx-serve --serve` discovered nothing and answered 503 to everything.
///
/// The one case that must NOT default: `--model <path> --serve`, which asked
/// for one specific model. Registering everything else on disk beside it is a
/// different server than the one requested.
pub fn shouldDefaultModelsRoot(in: RootDefaulting) bool {
    if (!in.serve_mode) return false;
    if (in.subcommand) return true; // `serve` / `run` always populate the picker
    return !in.has_explicit_model;
}

fn homeDir() []const u8 {
    return std.mem.span(std.c.getenv("HOME") orelse return "/tmp");
}

// ── HF tree listing ─────────────────────────────────────────────────────

pub const RepoFile = struct {
    path: []u8,
    size: u64,
};

pub fn freeRepoFiles(allocator: std.mem.Allocator, files: []RepoFile) void {
    for (files) |f| allocator.free(f.path);
    allocator.free(files);
}

/// Parse the HF `/api/models/<repo>/tree/main?recursive=true` JSON array.
/// LFS entries report the real artifact size under `lfs.size`.
pub fn parseTreeJson(allocator: std.mem.Allocator, json: []const u8) ![]RepoFile {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json, .{}) catch return error.InvalidTree;
    defer parsed.deinit();
    if (parsed.value != .array) return error.InvalidTree;

    var files = std.ArrayList(RepoFile).empty;
    errdefer {
        for (files.items) |f| allocator.free(f.path);
        files.deinit(allocator);
    }
    for (parsed.value.array.items) |item| {
        if (item != .object) continue;
        const obj = item.object;
        const t = obj.get("type") orelse continue;
        if (t != .string or !std.mem.eql(u8, t.string, "file")) continue;
        const p = obj.get("path") orelse continue;
        if (p != .string) continue;
        var size: u64 = 0;
        if (obj.get("lfs")) |lfs| {
            if (lfs == .object) {
                if (lfs.object.get("size")) |s| {
                    if (s == .integer and s.integer > 0) size = @intCast(s.integer);
                }
            }
        }
        if (size == 0) {
            if (obj.get("size")) |s| {
                if (s == .integer and s.integer > 0) size = @intCast(s.integer);
            }
        }
        try files.append(allocator, .{ .path = try allocator.dupe(u8, p.string), .size = size });
    }
    return files.toOwnedSlice(allocator);
}

/// Chat-default file selection (mirrors the app's `FileSelection.chatDefault`):
/// top-level files + the `mtp/` spec-decode sidecar; repo housekeeping and
/// demo assets are skipped.
/// `pytorch_model.bin` / `pytorch_model-0000N-of-0000M.bin` — the HF torch
/// weights that sit beside the safetensors copy. Shared rule with the app's
/// `DownloadManager.selectNeededFiles`; keep them in sync.
pub fn isTorchShadowBin(path: []const u8) bool {
    if (!std.ascii.endsWithIgnoreCase(path, ".bin")) return false;
    const base = if (std.mem.lastIndexOfScalar(u8, path, '/')) |i| path[i + 1 ..] else path;
    return std.ascii.startsWithIgnoreCase(base, "pytorch_model") or
        std.ascii.startsWithIgnoreCase(base, "rust_model") or
        std.ascii.startsWithIgnoreCase(base, "tf_model");
}

pub fn shouldDownload(path: []const u8) bool {
    if (path.len == 0 or path[0] == '.') return false;
    if (std.mem.indexOfScalar(u8, path, '/')) |_| {
        return std.mem.startsWith(u8, path, "mtp/");
    }
    const skip_exact = [_][]const u8{ "README.md", "LICENSE", "LICENSE.txt", "USE_POLICY.md" };
    for (skip_exact) |s| {
        if (std.ascii.eqlIgnoreCase(path, s)) return false;
    }
    // Torch/flax shadow weights are a second copy of the same model in a format
    // the server never reads — a doubled download. `.bin` itself stays allowed:
    // qwen4_exp's `ngram_table.bin` is an engine-read sidecar (mmapped at serve
    // time), and dropping it is what made app-downloaded packs fail to load.
    const skip_ext = [_][]const u8{ ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".md", ".pth", ".h5", ".msgpack", ".ckpt" };
    for (skip_ext) |ext| {
        if (path.len > ext.len and std.ascii.eqlIgnoreCase(path[path.len - ext.len ..], ext)) return false;
    }
    if (isTorchShadowBin(path)) return false;
    return true;
}

// ── Pull ────────────────────────────────────────────────────────────────

pub const Reporter = struct {
    impl: *anyopaque,
    /// One human-readable status line per call (no trailing newline).
    reportFn: *const fn (impl: *anyopaque, line: []const u8) void,

    pub fn say(self: Reporter, comptime fmt: []const u8, args: anytype) void {
        var buf: [512]u8 = undefined;
        const line = std.fmt.bufPrint(&buf, fmt, args) catch return;
        self.reportFn(self.impl, line);
    }
};

/// Appends the shared curl argv prefix; returns the owned Authorization
/// header string when HF_TOKEN is set (caller frees).
fn curlBaseArgs(list: *std.ArrayList([]const u8), allocator: std.mem.Allocator) !?[]u8 {
    try list.appendSlice(allocator, &.{ "curl", "-fL", "--retry", "3", "--retry-delay", "2" });
    if (std.c.getenv("HF_TOKEN")) |tok| {
        const header = try std.fmt.allocPrint(allocator, "Authorization: Bearer {s}", .{std.mem.span(tok)});
        try list.appendSlice(allocator, &.{ "-H", header });
        return header;
    }
    return null;
}

/// GET a small HTTPS document (the tree listing) via curl; returns stdout.
fn curlFetch(allocator: std.mem.Allocator, io: std.Io, url: []const u8) ![]u8 {
    var argv = std.ArrayList([]const u8).empty;
    defer argv.deinit(allocator);
    const header_storage = try curlBaseArgs(&argv, allocator);
    defer if (header_storage) |h| allocator.free(h);
    try argv.appendSlice(allocator, &.{ "-s", url });
    const result = std.process.run(allocator, io, .{
        .argv = argv.items,
        .stdout_limit = .limited(64 * 1024 * 1024),
    }) catch return error.FetchFailed;
    defer allocator.free(result.stderr);
    errdefer allocator.free(result.stdout);
    switch (result.term) {
        // Quiet on failure — callers report (the REPL health poll EXPECTS
        // failures while the server boots).
        .exited => |code| if (code != 0) return error.FetchFailed,
        else => return error.FetchFailed,
    }
    return result.stdout;
}

/// Download one file to `<dest_dir>/<file>` via curl (`-C -` resume onto a
/// .partial, atomic rename on success). `show_progress` inherits stderr so
/// the terminal gets curl's progress bar; the server-side pull passes false.
fn curlDownload(allocator: std.mem.Allocator, io: std.Io, url: []const u8, dest_path: []const u8, show_progress: bool) !void {
    const partial = try std.fmt.allocPrint(allocator, "{s}.partial", .{dest_path});
    defer allocator.free(partial);

    var argv = std.ArrayList([]const u8).empty;
    defer argv.deinit(allocator);
    const header_storage = try curlBaseArgs(&argv, allocator);
    defer if (header_storage) |h| allocator.free(h);
    try argv.appendSlice(allocator, &.{ "--create-dirs", "-C", "-", "-o", partial });
    if (show_progress) {
        try argv.append(allocator, "--progress-bar");
    } else {
        try argv.append(allocator, "-sS");
    }
    try argv.append(allocator, url);

    var child = std.process.spawn(io, .{
        .argv = argv.items,
        .stdin = .ignore,
        .stdout = .ignore,
        .stderr = if (show_progress) .inherit else .ignore,
    }) catch return error.DownloadFailed;
    const term = child.wait(io) catch return error.DownloadFailed;
    switch (term) {
        .exited => |code| if (code != 0) return error.DownloadFailed,
        else => return error.DownloadFailed,
    }
    std.Io.Dir.renameAbsolute(partial, dest_path, io) catch return error.DownloadFailed;
}

fn fileSizeAt(io: std.Io, dir_path: []const u8, rel: []const u8) ?u64 {
    if (dir_path.len == 0 or !std.fs.path.isAbsolute(dir_path)) return null;
    var dir = std.Io.Dir.openDirAbsolute(io, dir_path, .{}) catch return null;
    defer dir.close(io);
    const st = dir.statFile(io, rel, .{}) catch return null;
    if (st.kind != .file) return null;
    return st.size;
}

/// True when the model directory already holds a COMPLETE, loadable
/// checkpoint. "config.json exists" is NOT enough: an interrupted `pull`
/// (Ctrl-C mid-weights) leaves config.json + *.partial, and treating that
/// as present skipped the resume and fed a weightless dir to the loader
/// (live SIGSEGV — see tests/test_partial_download.sh). Complete means: no
/// .partial leftovers anywhere (top level or one subdir deep, e.g.
/// mtp/weights.safetensors.partial), plus config.json AND at least one
/// .safetensors for MLX dirs — or any .gguf, which is self-contained.
pub fn modelPresent(io: std.Io, dir_path: []const u8) bool {
    if (dir_path.len == 0 or !std.fs.path.isAbsolute(dir_path)) return false;
    var dir = std.Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true }) catch return false;
    defer dir.close(io);
    return modelPresentInDir(io, dir);
}

fn modelPresentInDir(io: std.Io, dir: std.Io.Dir) bool {
    var has_config = false;
    var has_safetensors = false;
    var has_gguf = false;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        switch (entry.kind) {
            .file => {
                if (std.mem.endsWith(u8, entry.name, ".partial")) return false;
                if (std.mem.eql(u8, entry.name, "config.json")) has_config = true;
                if (std.mem.endsWith(u8, entry.name, ".safetensors")) has_safetensors = true;
                if (std.mem.endsWith(u8, entry.name, ".gguf")) has_gguf = true;
            },
            .directory => {
                // One level deep is enough for the pull layouts (mtp/ is the
                // only subdir the chat-default selection downloads into).
                var sub = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
                defer sub.close(io);
                var sit = sub.iterate();
                while (sit.next(io) catch null) |se| {
                    if (se.kind == .file and std.mem.endsWith(u8, se.name, ".partial")) return false;
                }
            },
            else => {},
        }
    }
    if (has_gguf) return true;
    return has_config and has_safetensors;
}

/// Download `resolved.repo` into `dest_dir`. Skips files already complete
/// on disk (size match), resumes partials, reports per-file progress.
pub fn pullRepo(allocator: std.mem.Allocator, io: std.Io, resolved: Resolved, dest_dir: []const u8, reporter: Reporter, show_progress: bool) !void {
    // The Mac App Store build has no `curl` (the sandboxed helper can't reach
    // it) and must not download to arbitrary paths — the Swift app owns model
    // downloads via URLSession into the container.
    if (build_options.mas) {
        reporter.say("model pull is unavailable in this build", .{});
        return error.PullUnavailable;
    }
    reporter.say("pulling manifest for {s}", .{resolved.repo});
    const tree_url = try std.fmt.allocPrint(allocator, "https://huggingface.co/api/models/{s}/tree/main?recursive=true", .{resolved.repo});
    defer allocator.free(tree_url);
    const tree_json = curlFetch(allocator, io, tree_url) catch {
        if (resolved.gated) {
            // A licence gate 403s the LISTING, so this is the first place it can
            // be reported, and "set HF_TOKEN" is unhelpful to someone who has
            // one — acceptance is a separate, per-repo click.
            reporter.say("error: {s} is gated — accept its licence at https://huggingface.co/{s} while signed in, then set HF_TOKEN to a token from that account", .{ resolved.repo, resolved.repo });
        } else {
            reporter.say("error: could not list {s} (check the name, your network, or HF_TOKEN for gated repos)", .{resolved.repo});
        }
        return error.PullFailed;
    };
    defer allocator.free(tree_json);
    const files = parseTreeJson(allocator, tree_json) catch {
        reporter.say("error: unexpected listing for {s}", .{resolved.repo});
        return error.PullFailed;
    };
    defer freeRepoFiles(allocator, files);

    // The repo SHAPE is a property of the whole listing, not of any one path
    // (`layoutFor`): whether this is a diffusers multifolder repo, and which
    // components publish an fp16 weight variant.
    const paths = allocator.alloc([]const u8, files.len) catch {
        reporter.say("error: out of memory listing {s}", .{resolved.repo});
        return error.PullFailed;
    };
    defer allocator.free(paths);
    for (files, 0..) |f, i| paths[i] = f.path;
    const layout = layoutFor(resolved, paths);

    var wanted: usize = 0;
    var total_bytes: u64 = 0;
    for (files) |f| {
        if (!wantedFile(resolved, layout, f.path)) continue;
        wanted += 1;
        total_bytes += f.size;
    }
    if (wanted == 0) {
        reporter.say("error: {s} has no downloadable model files", .{resolved.repo});
        return error.PullFailed;
    }
    reporter.say("{d} files, {d} MB total", .{ wanted, total_bytes / (1024 * 1024) });

    var idx: usize = 0;
    for (files) |f| {
        if (!wantedFile(resolved, layout, f.path)) continue;
        idx += 1;
        // A subdir alias strips its prefix so the variant lands as a flat model
        // dir (`q4/unet/…` -> `unet/…`); the download URL still uses the full
        // repo path.
        const rel = subdirRelPath(resolved, f.path);
        const dest_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dest_dir, rel });
        defer allocator.free(dest_path);
        if (f.size > 0) {
            if (fileSizeAt(io, dest_dir, rel)) |have| {
                if (have == f.size) {
                    reporter.say("[{d}/{d}] {s} — already complete", .{ idx, wanted, rel });
                    continue;
                }
            }
        }
        reporter.say("[{d}/{d}] pulling {s} ({d} MB)", .{ idx, wanted, rel, f.size / (1024 * 1024) });
        const url = try std.fmt.allocPrint(allocator, "https://huggingface.co/{s}/resolve/main/{s}", .{ resolved.repo, f.path });
        defer allocator.free(url);
        curlDownload(allocator, io, url, dest_path, show_progress) catch {
            reporter.say("error: download failed for {s} (partial kept — rerun to resume)", .{f.path});
            return error.PullFailed;
        };
    }
    reporter.say("success: {s} ready", .{resolved.repo});
}

/// The diffusers component dirs our backends read: SDXL / SD 1.x bind the
/// `unet` set, SD 3.5 the `transformer` set with a THIRD tower (T5-XXL) beside
/// the two CLIPs. Everything else a stability repo ships under a subfolder —
/// `vae_1_0/`, `vae_decoder/`, `vae_encoder/`, `safety_checker/`,
/// `feature_extractor/`, `onnx/`, `openvino/`, and the ComfyUI-style flat
/// `text_encoders/` drop — is an export for another runtime or a component we
/// never load. Capped at 16 by `RepoLayout`'s bitsets.
const DIFFUSERS_COMPONENTS = [_][]const u8{
    "scheduler",     "tokenizer",      "tokenizer_2", "tokenizer_3",
    "text_encoder",  "text_encoder_2", "text_encoder_3",
    "unet",          "transformer",    "vae",
};

fn diffusersComponentIndex(dir: []const u8) ?usize {
    for (DIFFUSERS_COMPONENTS, 0..) |c, i| {
        if (std.mem.eql(u8, c, dir)) return i;
    }
    return null;
}

/// What the repo's LISTING says about its shape — the part of "is this file
/// wanted?" that a single path cannot answer.
///
/// `model.loadWeights` sweeps EVERY `*.safetensors` in a component dir into one
/// map, so exactly one weight variant may land per dir: two is both a doubled
/// download and a key-for-key collision. Which one is right depends on what the
/// rest of the dir holds, hence the per-component bits.
pub const RepoLayout = struct {
    /// Root `model_index.json` — a diffusers multifolder repo.
    diffusers: bool = false,
    /// Bit i: component i publishes an fp16 weight variant.
    fp16: u16 = 0,
    /// Bit i: component i is SHARDED — it ships a `model.safetensors.index.json`.
    sharded: u16 = 0,

    /// Whether component `i`'s fp16 variant is the one to fetch.
    ///
    /// Only for UNSHARDED components. `model.loadWeights` resolves a sharded dir
    /// through `model_discovery.indexShardSet`, which reads exactly
    /// `model.safetensors.index.json` — and that index names the PLAIN shards.
    /// diffusers puts the fp16 shard map in a differently-named file
    /// (`model.safetensors.index.fp16.json`) our loader does not read, so an
    /// fp16 shard set would be skipped shard-for-shard and the dir would load as
    /// "no usable weights". SD 3.5's `text_encoder_3` is the first component
    /// here that is sharded at all.
    fn wantsFp16(self: RepoLayout, ci: usize) bool {
        const bit = @as(u16, 1) << @intCast(ci);
        return self.fp16 & bit != 0 and self.sharded & bit == 0;
    }
};

/// True for a weight file's fp16 VARIANT spelling.
///
/// diffusers separates the variant from what follows with the ORIGINAL
/// separator, so an unsharded file reads `model.fp16.safetensors` while a shard
/// reads `model.fp16-00001-of-00002.safetensors`. Matching only `".fp16."`
/// misses every shard, which reads as "this component has no fp16 variant" and
/// lands BOTH shard sets in one dir — a doubled 9.5 GB download and a
/// key-for-key collision.
fn isFp16Variant(name: []const u8) bool {
    const at = std.mem.indexOf(u8, name, ".fp16") orelse return false;
    const after = at + ".fp16".len;
    return after < name.len and (name[after] == '.' or name[after] == '-');
}

/// Read the shape off the repo listing. `paths` is every file in the tree.
pub fn layoutFor(resolved: Resolved, paths: []const []const u8) RepoLayout {
    // A single-artifact pull never consults the layout, and a `subdir` variant
    // has its own (already flat, already one-variant) rule.
    if (resolved.gguf_file.len > 0 or resolved.single_file.len > 0 or resolved.subdir.len > 0) return .{};
    var out: RepoLayout = .{};
    for (paths) |path| {
        if (std.mem.eql(u8, path, "model_index.json")) {
            out.diffusers = true;
            continue;
        }
        const slash = std.mem.indexOfScalar(u8, path, '/') orelse continue;
        const ci = diffusersComponentIndex(path[0..slash]) orelse continue;
        const rel = path[slash + 1 ..];
        const bit = @as(u16, 1) << @intCast(ci);
        if (std.mem.eql(u8, rel, "model.safetensors.index.json")) {
            out.sharded |= bit;
            continue;
        }
        if (!std.mem.endsWith(u8, rel, ".safetensors")) continue;
        if (std.mem.indexOf(u8, rel, ".non_ema.") != null) continue;
        if (isFp16Variant(rel)) out.fp16 |= bit;
    }
    return out;
}

/// Keep a file from a diffusers multifolder repo (`sdxl`, `sdxl:turbo`,
/// `sd1`, `sd1:turbo`).
///
/// `shouldDownload`'s flat-repo rule rejects every nested path except `mtp/`,
/// which for these repos means the ENTIRE model: `unet/`, `vae/`,
/// `text_encoder*/`, `tokenizer*/` and `scheduler/` all live one level down.
/// What survived was `model_index.json` plus the root single-file checkpoints —
/// so `mlx-serve pull sdxl` fetched 13.9 GB of merged checkpoint, and
/// `Engine.loadAuto`, seeing the index, took the folder path and failed on the
/// missing `text_encoder/`.
///
/// Mirrors the app's `MediaBundle.sdxlDiffusers` / `.sd1Diffusers` selection:
/// component dirs only, `.safetensors`/`.json`/`.txt` only (which drops the
/// `.bin`, `.msgpack`, `.onnx`/`.onnx_data` and openvino `.xml` exports these
/// repos carry a full copy of the model in), and the fp16 weight variant. fp16
/// is not a compromise here — `sdxl_unet` and both CLIP towers are SERVED at
/// fp16, so those bytes are exactly what reaches the GPU either way, and the
/// VAE's f32 compute (`force_upcast`) reads upcast fp16 weights, which is what
/// diffusers' own `variant="fp16"` distribution does. It halves SDXL base from
/// ~13.9 GB to ~7 GB. A repo publishing no fp16 variant (NoobAI's folder) keeps
/// the plain name.
fn wantedInDiffusers(layout: RepoLayout, path: []const u8) bool {
    if (path.len == 0 or path[0] == '.') return false;
    const slash = std.mem.indexOfScalar(u8, path, '/') orelse {
        // Root: the index and nothing else. The merged single-file checkpoints
        // beside it are a SECOND copy of the whole model that the folder path
        // will never open.
        return std.mem.eql(u8, path, "model_index.json");
    };
    const ci = diffusersComponentIndex(path[0..slash]) orelse return false;
    const rel = path[slash + 1 ..];
    // Components are flat; anything deeper is an export tree.
    if (rel.len == 0 or rel[0] == '.') return false;
    if (std.mem.indexOfScalar(u8, rel, '/') != null) return false;
    // The fp16 shard MAP names shards we deliberately did not fetch; keeping it
    // would leave two index files in one dir, only one of them honest.
    if (std.mem.eql(u8, rel, "model.safetensors.index.fp16.json")) return false;
    if (std.mem.endsWith(u8, rel, ".json") or std.mem.endsWith(u8, rel, ".txt")) return true;
    if (!std.mem.endsWith(u8, rel, ".safetensors")) return false;
    // The non-EMA training checkpoint: diffusers loads it only on request.
    if (std.mem.indexOf(u8, rel, ".non_ema.") != null) return false;
    return isFp16Variant(rel) == layout.wantsFp16(ci);
}

fn wantedFile(resolved: Resolved, layout: RepoLayout, path: []const u8) bool {
    if (resolved.gguf_file.len > 0) {
        // Single-artifact GGUF repos: just that file (plus nothing else).
        return std.mem.eql(u8, path, resolved.gguf_file);
    }
    if (resolved.single_file.len > 0) {
        // Single-file LDM checkpoint: that file and nothing else — in
        // particular NOT a `model_index.json` sitting beside it.
        return std.mem.eql(u8, path, resolved.single_file);
    }
    if (resolved.subdir.len > 0) {
        // A diffusers variant is inherently nested (unet/, vae/, …), so the
        // flat-repo `mtp/`-only rule in `shouldDownload` can't apply. Filter the
        // STRIPPED path: keep nested model files, drop images/readme/dotfiles.
        if (!pathInSubdir(resolved.subdir, path)) return false;
        return wantedInVariant(subdirRelPath(resolved, path));
    }
    if (layout.diffusers) return wantedInDiffusers(layout, path);
    return shouldDownload(path);
}

/// Keep a file inside a `subdir` variant: any nested model file, minus
/// dotfiles, sample images and docs (the assets `shouldDownload` also skips).
fn wantedInVariant(rel: []const u8) bool {
    if (rel.len == 0 or rel[0] == '.') return false;
    if (std.mem.indexOf(u8, rel, "/.") != null) return false; // dot-dir anywhere
    const skip_ext = [_][]const u8{ ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".md" };
    for (skip_ext) |ext| {
        if (rel.len > ext.len and std.ascii.eqlIgnoreCase(rel[rel.len - ext.len ..], ext)) return false;
    }
    const base = std.fs.path.basename(rel);
    for ([_][]const u8{ "README.md", "LICENSE", "LICENSE.txt", "USE_POLICY.md" }) |sk| {
        if (std.ascii.eqlIgnoreCase(base, sk)) return false;
    }
    return true;
}

/// True when `path` lives under `subdir/` (a `subdir`-prefixed entry).
fn pathInSubdir(subdir: []const u8, path: []const u8) bool {
    return path.len > subdir.len and
        std.mem.startsWith(u8, path, subdir) and
        path[subdir.len] == '/';
}

/// `path` with the `subdir/` prefix removed when a subdir alias is in play,
/// so the variant is written flat under the dest dir. Unchanged otherwise.
fn subdirRelPath(resolved: Resolved, path: []const u8) []const u8 {
    if (resolved.subdir.len > 0 and pathInSubdir(resolved.subdir, path)) {
        return path[resolved.subdir.len + 1 ..];
    }
    return path;
}

// ── Commands ────────────────────────────────────────────────────────────

fn stderrReport(impl: *anyopaque, line: []const u8) void {
    _ = impl;
    log.info("{s}\n", .{line});
}

var stderr_reporter_dummy: u8 = 0;
const stderr_reporter = Reporter{ .impl = &stderr_reporter_dummy, .reportFn = &stderrReport };

/// Resolve + download-if-missing; returns the local model dir (owned).
/// Exits the process with a friendly message on unknown names.
pub fn ensureModelAvailable(allocator: std.mem.Allocator, io: std.Io, name: []const u8) ![]u8 {
    // A path that exists locally is used as-is.
    if (std.fs.path.isAbsolute(name)) return allocator.dupe(u8, name);
    const resolved = resolveShortName(name) orelse {
        log.err("unknown model '{s}'\n", .{name});
        printKnownAliases(io);
        std.process.exit(1);
    };
    const dest = try modelDestPath(allocator, homeDir(), resolved.destRepo());
    errdefer allocator.free(dest);
    if (modelPresent(io, dest)) return dest;
    try pullRepo(allocator, io, resolved, dest, stderr_reporter, true);
    return dest;
}

pub fn cmdPull(allocator: std.mem.Allocator, io: std.Io, name: []const u8) !void {
    const dir = try ensureModelAvailable(allocator, io, name);
    defer allocator.free(dir);
    log.info("model at {s}\n", .{dir});
    log.info("run it: mlx-serve run {s}\n", .{name});
}

fn printKnownAliases(io: std.Io) void {
    _ = io;
    log.err("known short names (or use any HuggingFace 'org/repo'):\n", .{});
    for (aliases) |a| {
        if (a.is_default) {
            log.err("  {s} (= {s}:{s}) -> {s}\n", .{ a.name, a.name, a.tag, a.repo });
        } else {
            log.err("  {s}:{s} -> {s}\n", .{ a.name, a.tag, a.repo });
        }
    }
}

/// `mlx-serve list` — models on disk under ~/.mlx-serve/models.
pub fn cmdList(allocator: std.mem.Allocator, io: std.Io) !void {
    const root = try modelsRootPath(allocator, homeDir());
    defer allocator.free(root);

    var out_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writer(io, &out_buf);
    const w = &stdout_w.interface;
    defer w.flush() catch {};

    var dir = std.Io.Dir.openDirAbsolute(io, root, .{ .iterate = true }) catch {
        try w.print("no models yet (looked in {s})\ntry: mlx-serve pull gemma4\n", .{root});
        return;
    };
    defer dir.close(io);

    try w.print("{s: <56} {s: <12} {s: >10}\n", .{ "NAME", "TYPE", "SIZE" });
    var count: usize = 0;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (!treeEntryDescends(entry.kind)) continue;
        if (entry.name.len == 0 or entry.name[0] == '.') continue;
        var sub = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
        defer sub.close(io);
        if (isModelDir(io, allocator, &sub)) {
            try printModelRow(io, allocator, w, &sub, entry.name, root);
            count += 1;
            continue;
        }
        // org/ level: one more hop down.
        var sub_it = sub.iterate();
        while (sub_it.next(io) catch null) |sub_entry| {
            if (!treeEntryDescends(sub_entry.kind)) continue;
            var leaf = sub.openDir(io, sub_entry.name, .{ .iterate = true }) catch continue;
            defer leaf.close(io);
            if (!isModelDir(io, allocator, &leaf)) continue;
            var name_buf: [512]u8 = undefined;
            const full = std.fmt.bufPrint(&name_buf, "{s}/{s}", .{ entry.name, sub_entry.name }) catch continue;
            try printModelRow(io, allocator, w, &leaf, full, root);
            count += 1;
        }
    }
    if (count == 0) {
        try w.print("(none) — try: mlx-serve pull gemma4\n", .{});
    }
}

/// A tree-walk entry worth descending into: a real directory OR a symlink
/// (a checkpoint moved to an external drive and linked back — the H3 mirrors
/// live that way; openDir resolves the link, and model_discovery's own walk
/// already accepts both kinds).
fn treeEntryDescends(kind: std.Io.File.Kind) bool {
    return kind == .directory or kind == .sym_link;
}

fn isModelDir(io: std.Io, allocator: std.mem.Allocator, dir: *std.Io.Dir) bool {
    if (dir.statFile(io, "config.json", .{})) |st| {
        if (st.kind == .file) return true;
    } else |_| {}
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".gguf")) continue;
        const st = dir.statFile(io, entry.name, .{}) catch continue;
        if (st.kind == .file) return true;
    }
    // A MageFlow repo has neither: every config lives in a component subdir and
    // `model_index.json` is the only signal. An mflux FLUX.2 conversion may
    // carry neither file NOR an index — it is recognized by the DiT's own
    // weight names. Both shared with discovery so `list` and `/v1/models`
    // cannot disagree about what counts as a model.
    if (model_discovery.peekMageFlowIndex(io, allocator, dir.*)) return true;
    return model_discovery.peekMfluxFlux2(io, allocator, dir.*);
}

fn printModelRow(io: std.Io, allocator: std.mem.Allocator, w: *std.Io.Writer, dir: *std.Io.Dir, name: []const u8, root: []const u8) !void {
    const bytes = dirBytesOneLevel(io, dir);
    // TYPE from the same classification serving uses (gguf → chat via the
    // embedded engines, media modalities, embed, drafter, unsupported) so
    // the list is honest about which rows `run` can actually chat with.
    const abs = std.fmt.allocPrint(allocator, "{s}/{s}", .{ root, name }) catch null;
    defer if (abs) |a| allocator.free(a);
    const kind_label: []const u8 = blk: {
        const a = abs orelse break :blk "?";
        const kind = model_discovery.classifyModelPath(io, allocator, a) orelse break :blk "?";
        break :blk kind.label();
    };
    var size_buf: [32]u8 = undefined;
    try w.print("{s: <56} {s: <12} {s: >10}\n", .{ name, kind_label, formatSize(&size_buf, bytes) });
}

/// Sum file bytes in a model dir INCLUDING one level of subdirectories —
/// media bundles keep their weights in transformer/ vae/ text_encoder/ etc.
/// (the same one-level layout assumption `modelPresent` makes). Top-level-
/// only summing showed a 7 GB FLUX bundle as "6 KB".
fn dirBytesOneLevel(io: std.Io, dir: *std.Io.Dir) u64 {
    var bytes: u64 = 0;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        switch (entry.kind) {
            .file, .sym_link => {
                const st = dir.statFile(io, entry.name, .{}) catch continue;
                if (st.kind == .file) bytes += @intCast(st.size);
            },
            .directory => {
                var sub = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
                defer sub.close(io);
                var sit = sub.iterate();
                while (sit.next(io) catch null) |se| {
                    if (se.kind != .file and se.kind != .sym_link) continue;
                    const st = sub.statFile(io, se.name, .{}) catch continue;
                    if (st.kind != .file) continue;
                    bytes += @intCast(st.size);
                }
            },
            else => {},
        }
    }
    return bytes;
}

pub fn formatSize(buf: []u8, bytes: u64) []const u8 {
    const gb = 1024 * 1024 * 1024;
    const mb = 1024 * 1024;
    if (bytes >= gb) {
        const whole = bytes / gb;
        const tenth = (bytes % gb) * 10 / gb;
        return std.fmt.bufPrint(buf, "{d}.{d} GB", .{ whole, tenth }) catch "?";
    }
    if (bytes >= mb) return std.fmt.bufPrint(buf, "{d} MB", .{bytes / mb}) catch "?";
    return std.fmt.bufPrint(buf, "{d} KB", .{bytes / 1024}) catch "?";
}

// ── REPL (mlx-serve run) ────────────────────────────────────────────────
//
// The REPL is deliberately a real HTTP client against the server's own
// /api/chat endpoint (via curl, streaming NDJSON) — it dogfoods the Ollama
// surface on every keystroke instead of poking internal functions.

pub const Turn = struct {
    role: []const u8,
    content: []const u8,
};

/// /api/chat request body for the REPL conversation so far.
pub fn buildReplChatBody(allocator: std.mem.Allocator, history: []const Turn) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    const w = &out.writer;
    try w.writeAll("{\"model\":\"mlx-serve\",\"stream\":true,\"messages\":[");
    for (history, 0..) |turn, i| {
        if (i > 0) try w.writeAll(",");
        try w.writeAll("{\"role\":");
        try ollama.writeJsonString(w, turn.role);
        try w.writeAll(",\"content\":");
        try ollama.writeJsonString(w, turn.content);
        try w.writeAll("}");
    }
    try w.writeAll("]}");
    return allocator.dupe(u8, out.written());
}

pub const ReplDelta = struct {
    /// Owned by caller.
    content: []u8,
    done: bool,
    eval_count: u64 = 0,
    eval_duration_ns: u64 = 0,
    err: ?[]u8 = null,
};

/// One NDJSON line from /api/chat → the piece the REPL prints.
pub fn parseReplLine(allocator: std.mem.Allocator, line: []const u8) ?ReplDelta {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const root = parsed.value.object;
    if (root.get("error")) |e| {
        if (e == .string) {
            return .{
                .content = allocator.dupe(u8, "") catch return null,
                .done = true,
                .err = allocator.dupe(u8, e.string) catch return null,
            };
        }
    }
    var content: []const u8 = "";
    if (root.get("message")) |m| {
        if (m == .object) {
            if (m.object.get("content")) |c| {
                if (c == .string) content = c.string;
            }
        }
    }
    const done = if (root.get("done")) |d| (d == .bool and d.bool) else false;
    var eval_count: u64 = 0;
    var eval_ns: u64 = 0;
    if (done) {
        if (root.get("eval_count")) |v| {
            if (v == .integer and v.integer > 0) eval_count = @intCast(v.integer);
        }
        if (root.get("eval_duration")) |v| {
            if (v == .integer and v.integer > 0) eval_ns = @intCast(v.integer);
        }
    }
    return .{
        .content = allocator.dupe(u8, content) catch return null,
        .done = done,
        .eval_count = eval_count,
        .eval_duration_ns = eval_ns,
    };
}

/// Interactive loop on the calling thread. Waits for the server to answer
/// /health, then reads prompts from stdin and streams /api/chat responses.
/// Returns when the user exits (/bye or EOF); caller shuts the server down.
pub fn runRepl(allocator: std.mem.Allocator, io: std.Io, port: u16) !void {
    const health_url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}/health", .{port});
    defer allocator.free(health_url);
    // Big checkpoints take a while to fault in; poll patiently.
    var waited_ms: u64 = 0;
    while (waited_ms < 15 * 60 * 1000) {
        if (curlFetch(allocator, io, health_url)) |body| {
            allocator.free(body);
            break;
        } else |_| {}
        std.Io.sleep(io, .fromMilliseconds(500), .real) catch {};
        waited_ms += 500;
    }

    const chat_url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}/api/chat", .{port});
    defer allocator.free(chat_url);

    var out_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writer(io, &out_buf);
    const w = &stdout_w.interface;
    try w.writeAll("\n>>> chat is live — /bye to exit\n");
    try w.flush();

    var history = std.ArrayList(Turn).empty;
    defer {
        for (history.items) |t| allocator.free(t.content);
        history.deinit(allocator);
    }

    var stdin_buf: [16 * 1024]u8 = undefined;
    var stdin_r = std.Io.File.stdin().reader(io, &stdin_buf);
    const r = &stdin_r.interface;

    while (true) {
        try w.writeAll(">>> ");
        try w.flush();
        const line = r.takeDelimiter('\n') catch break orelse break;
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;
        if (std.mem.eql(u8, trimmed, "/bye") or std.mem.eql(u8, trimmed, "/exit") or std.mem.eql(u8, trimmed, "/quit")) break;

        try history.append(allocator, .{ .role = "user", .content = try allocator.dupe(u8, trimmed) });
        const body = try buildReplChatBody(allocator, history.items);
        defer allocator.free(body);

        const reply = streamOneTurn(allocator, io, chat_url, body, w) catch |err| {
            try w.print("\n[error: {s}]\n", .{@errorName(err)});
            try w.flush();
            continue;
        };
        try history.append(allocator, .{ .role = "assistant", .content = reply });
        try w.writeAll("\n");
        try w.flush();
    }
}

/// POST the body, stream NDJSON, print content deltas as they arrive.
/// Returns the full assistant reply (owned).
fn streamOneTurn(allocator: std.mem.Allocator, io: std.Io, url: []const u8, body: []const u8, w: *std.Io.Writer) ![]u8 {
    var child = std.process.spawn(io, .{
        .argv = &.{ "curl", "-sN", "-X", "POST", "-H", "Content-Type: application/json", "--data-binary", "@-", url },
        .stdin = .pipe,
        .stdout = .pipe,
        .stderr = .ignore,
    }) catch return error.CurlSpawnFailed;
    defer child.kill(io);

    {
        var in_buf: [4096]u8 = undefined;
        var stdin_w = child.stdin.?.writer(io, &in_buf);
        try stdin_w.interface.writeAll(body);
        try stdin_w.interface.flush();
        child.stdin.?.close(io);
        child.stdin = null;
    }

    var full = std.ArrayList(u8).empty;
    errdefer full.deinit(allocator);

    var out_buf: [64 * 1024]u8 = undefined;
    var stdout_r = child.stdout.?.reader(io, &out_buf);
    const r = &stdout_r.interface;
    while (true) {
        const line = r.takeDelimiter('\n') catch break orelse break;
        if (line.len == 0) continue;
        const delta = parseReplLine(allocator, line) orelse continue;
        defer allocator.free(delta.content);
        if (delta.err) |e| {
            defer allocator.free(e);
            try w.print("[server error: {s}]", .{e});
            try w.flush();
            break;
        }
        if (delta.content.len > 0) {
            try w.writeAll(delta.content);
            try w.flush();
            try full.appendSlice(allocator, delta.content);
        }
        if (delta.done) {
            if (delta.eval_count > 0 and delta.eval_duration_ns > 0) {
                const tok_s = @as(f64, @floatFromInt(delta.eval_count)) * 1e9 / @as(f64, @floatFromInt(delta.eval_duration_ns));
                try w.print("\n[{d} tokens, {d:.1} tok/s]", .{ delta.eval_count, tok_s });
            }
            break;
        }
    }
    _ = child.wait(io) catch {};
    return full.toOwnedSlice(allocator);
}

// ── Tests ───────────────────────────────────────────────────────────────

const testing = std.testing;

test "cli: resolveShortName aliases, tags, org/repo, hf.co, unknown" {
    // Bare alias picks the family default.
    try testing.expectEqualStrings("mlx-community/gemma-4-e4b-it-4bit", resolveShortName("gemma4").?.repo);
    try testing.expectEqualStrings("ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve", resolveShortName("qwen3.6").?.repo);
    // Tagged alias.
    try testing.expectEqualStrings("mlx-community/gemma-4-12b-it-4bit", resolveShortName("gemma4:12b").?.repo);
    try testing.expectEqualStrings("mlx-community/gemma-4-26b-a4b-it-8bit", resolveShortName("GEMMA4:26B-8BIT").?.repo);
    // :latest behaves like bare.
    try testing.expectEqualStrings("mlx-community/gemma-4-e4b-it-4bit", resolveShortName("gemma4:latest").?.repo);
    // Direct org/repo passthrough, tag stripped, hf.co prefixes stripped.
    try testing.expectEqualStrings("org/repo", resolveShortName("org/repo").?.repo);
    try testing.expectEqualStrings("org/repo", resolveShortName("org/repo:latest").?.repo);
    try testing.expectEqualStrings("org/repo", resolveShortName("hf.co/org/repo").?.repo);
    try testing.expectEqualStrings("org/repo", resolveShortName("https://huggingface.co/org/repo").?.repo);
    // GGUF single-artifact alias carries its file restriction.
    const ds = resolveShortName("deepseek-v4").?;
    try testing.expect(ds.gguf_file.len > 0);
    // Unknown alias-shaped name → null.
    try testing.expect(resolveShortName("doesnotexist") == null);
    try testing.expect(resolveShortName("gemma4:nosuchtag") == null);
}

test "cli: modelDestPath layout" {
    const allocator = testing.allocator;
    const p = try modelDestPath(allocator, "/Users/x", "org/repo");
    defer allocator.free(p);
    try testing.expectEqualStrings("/Users/x/.mlx-serve/models/org/repo", p);
}

test "cli: SDXL bundles — pony single-file + illustrious subdir variants" {
    // Pony: a flat single-file repo, no subdir.
    const pony = resolveShortName("pony").?;
    try testing.expectEqualStrings("LyliaEngine/Pony_Diffusion_V6_XL", pony.repo);
    try testing.expectEqualStrings("", pony.subdir);
    try testing.expectEqualStrings("LyliaEngine/Pony_Diffusion_V6_XL", pony.destRepo());

    // Bare illustrious = q4 (default), stored under its own flat dest.
    const ill = resolveShortName("illustrious").?;
    try testing.expectEqualStrings("SceneWorks/illustrious-xl-v2-mlx", ill.repo);
    try testing.expectEqualStrings("q4", ill.subdir);
    try testing.expectEqualStrings("SceneWorks/illustrious-xl-v2-q4", ill.destRepo());
    try testing.expectEqualStrings("q8", resolveShortName("illustrious:q8").?.subdir);
    try testing.expectEqualStrings("bf16", resolveShortName("illustrious:bf16").?.subdir);

    // Subdir filtering keeps the variant's nested files, strips the prefix, and
    // rejects the OTHER variants + assets. A subdir alias never reads a layout —
    // its own `q4/model_index.json` must not be mistaken for a root one.
    const ill_layout = layoutFor(ill, &.{ "q4/model_index.json", "q4/unet/diffusion_pytorch_model.safetensors" });
    try testing.expect(!ill_layout.diffusers);
    try testing.expect(wantedFile(ill, ill_layout, "q4/unet/diffusion_pytorch_model.safetensors"));
    try testing.expect(wantedFile(ill, ill_layout, "q4/model_index.json"));
    try testing.expect(!wantedFile(ill, ill_layout, "q8/unet/diffusion_pytorch_model.safetensors"));
    try testing.expect(!wantedFile(ill, ill_layout, "bf16/vae/config.json"));
    try testing.expect(!wantedFile(ill, ill_layout, "q4/images/sample.jpeg"));
    try testing.expect(!wantedFile(ill, ill_layout, "README.md"));
    try testing.expectEqualStrings("unet/diffusion_pytorch_model.safetensors", subdirRelPath(ill, "q4/unet/diffusion_pytorch_model.safetensors"));

    // pathInSubdir is boundary-exact (q4 must not match q40).
    try testing.expect(pathInSubdir("q4", "q4/x"));
    try testing.expect(!pathInSubdir("q4", "q40/x"));
    try testing.expect(!pathInSubdir("q4", "q4"));
}

test "cli: a diffusers multifolder repo pulls its COMPONENTS, not the merged checkpoint" {
    // Regression (PR #301 review): `sdxl`, `sdxl:turbo`, `sd1` and `sd1:turbo`
    // have no `subdir`, so `wantedFile` fell through to `shouldDownload`, whose
    // flat-repo rule rejects every nested path except `mtp/`. Nothing under
    // `unet/`, `vae/`, `text_encoder*/`, `tokenizer*/` or `scheduler/` was ever
    // fetched — `mlx-serve pull sdxl` downloaded `model_index.json` plus 13.9 GB
    // of root single-file checkpoint, and `Engine.loadAuto` then took the folder
    // path (the index is present) and failed on the missing `text_encoder/`.
    //
    // The listing is `stabilityai/stable-diffusion-xl-base-1.0`, trimmed to one
    // entry per shape it ships.
    const sdxl = resolveShortName("sdxl").?;
    const listing = [_][]const u8{
        ".gitattributes",
        "01.png",
        "LICENSE.md",
        "README.md",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "sd_xl_base_1.0.safetensors",
        "sd_xl_base_1.0_0.9vae.safetensors",
        "sd_xl_offset_example-lora_1.0.safetensors",
        "text_encoder/config.json",
        "text_encoder/flax_model.msgpack",
        "text_encoder/model.fp16.safetensors",
        "text_encoder/model.onnx",
        "text_encoder/model.safetensors",
        "text_encoder/openvino_model.bin",
        "text_encoder/openvino_model.xml",
        "text_encoder_2/config.json",
        "text_encoder_2/model.fp16.safetensors",
        "text_encoder_2/model.onnx_data",
        "text_encoder_2/model.safetensors",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "tokenizer_2/merges.txt",
        "tokenizer_2/vocab.json",
        "unet/config.json",
        "unet/diffusion_flax_model.msgpack",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "unet/diffusion_pytorch_model.safetensors",
        "unet/model.onnx",
        "unet/openvino_model.xml",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "vae_1_0/config.json",
        "vae_1_0/diffusion_pytorch_model.safetensors",
        "vae_decoder/model.onnx",
        "vae_encoder/model.onnx",
    };
    const layout = layoutFor(sdxl, &listing);
    try testing.expect(layout.diffusers);

    const want = [_][]const u8{
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/model.fp16.safetensors",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "tokenizer_2/merges.txt",
        "tokenizer_2/vocab.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    };
    var kept: usize = 0;
    for (listing) |path| {
        const keep = wantedFile(sdxl, layout, path);
        var expected = false;
        for (want) |w| {
            if (std.mem.eql(u8, w, path)) expected = true;
        }
        if (keep != expected) {
            std.debug.print("wantedFile mismatch: {s} kept={} want={}\n", .{ path, keep, expected });
            return error.TestUnexpectedResult;
        }
        if (keep) kept += 1;
    }
    // Every component `sdxl_pipeline.Engine.load` binds is covered, so the load
    // cannot fail on a missing dir the way it did before. Named explicitly
    // rather than swept from `DIFFUSERS_COMPONENTS`, which is the union across
    // families and carries SD 3.5's `transformer`/`text_encoder_3` too.
    try testing.expectEqual(want.len, kept);
    const sdxl_binds = [_][]const u8{
        "scheduler", "tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2", "unet", "vae",
    };
    for (sdxl_binds) |c| {
        var seen = false;
        for (want) |w| {
            if (std.mem.startsWith(u8, w, c) and w[c.len] == '/') seen = true;
        }
        if (!seen) {
            std.debug.print("no file kept for component {s}\n", .{c});
            return error.TestUnexpectedResult;
        }
        // And every one of them is a dir the filter recognises at all.
        try testing.expect(diffusersComponentIndex(c) != null);
    }
}

test "cli: exactly ONE weight variant lands per component dir" {
    // `model.loadWeights` sweeps EVERY `*.safetensors` in a component dir into
    // one map, so a second variant is a doubled download AND a key-for-key
    // collision. fp16 wins where the repo publishes it (the UNet and both CLIP
    // towers are SERVED at fp16); the plain name wins where it does not.
    const sd1 = resolveShortName("sd1").?;
    // `stable-diffusion-v1-5`: fp16 siblings everywhere, plus a `non_ema`
    // training checkpoint and torch `.bin` copies of the whole model.
    const with_fp16 = [_][]const u8{
        "model_index.json",
        "unet/diffusion_pytorch_model.bin",
        "unet/diffusion_pytorch_model.fp16.bin",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "unet/diffusion_pytorch_model.non_ema.safetensors",
        "unet/diffusion_pytorch_model.safetensors",
        "safety_checker/model.safetensors",
        "v1-5-pruned-emaonly.safetensors",
        "v1-inference.yaml",
    };
    var l = layoutFor(sd1, &with_fp16);
    try testing.expect(wantedFile(sd1, l, "unet/diffusion_pytorch_model.fp16.safetensors"));
    try testing.expect(!wantedFile(sd1, l, "unet/diffusion_pytorch_model.safetensors"));
    try testing.expect(!wantedFile(sd1, l, "unet/diffusion_pytorch_model.non_ema.safetensors"));
    try testing.expect(!wantedFile(sd1, l, "unet/diffusion_pytorch_model.bin"));
    // A component we never load, and the root LDM checkpoint the folder path
    // will never open.
    try testing.expect(!wantedFile(sd1, l, "safety_checker/model.safetensors"));
    try testing.expect(!wantedFile(sd1, l, "v1-5-pruned-emaonly.safetensors"));
    try testing.expect(!wantedFile(sd1, l, "v1-inference.yaml"));

    // No fp16 sibling: the plain name is the only weight file, so it is kept.
    const no_fp16 = [_][]const u8{
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
    };
    l = layoutFor(sd1, &no_fp16);
    try testing.expect(wantedFile(sd1, l, "unet/diffusion_pytorch_model.safetensors"));
}

test "cli: a SHARDED component takes the variant its index names" {
    // SD 3.5's `text_encoder_3` (T5-XXL) is the first sharded component to reach
    // this filter, and it breaks the fp16 preference two ways at once:
    //
    //   * A shard's variant marker is `.fp16-00001-of-00002`, not `.fp16.`, so a
    //     `".fp16."` substring test sees NO fp16 variant and keeps both shard
    //     sets — 9.5 GB of doubled download into one dir.
    //   * `model.loadWeights` resolves a sharded dir through
    //     `model_discovery.indexShardSet`, which reads exactly
    //     `model.safetensors.index.json`. That index names the PLAIN shards, so
    //     an fp16 shard set is skipped shard-for-shard and the dir loads as "no
    //     usable weights".
    //
    // Listing is `stable-diffusion-3.5-large`'s, trimmed.
    const sd3 = Resolved{ .repo = "stabilityai/stable-diffusion-3.5-large" };
    const listing = [_][]const u8{
        "model_index.json",
        "sd3.5_large.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder/model.safetensors",
        "text_encoder_3/config.json",
        "text_encoder_3/model-00001-of-00002.safetensors",
        "text_encoder_3/model-00002-of-00002.safetensors",
        "text_encoder_3/model.fp16-00001-of-00002.safetensors",
        "text_encoder_3/model.fp16-00002-of-00002.safetensors",
        "text_encoder_3/model.safetensors.index.fp16.json",
        "text_encoder_3/model.safetensors.index.json",
        "vae/diffusion_pytorch_model.safetensors",
    };
    const layout = layoutFor(sd3, &listing);
    try testing.expect(layout.diffusers);

    const want = [_][]const u8{
        "model_index.json",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_3/config.json",
        "text_encoder_3/model-00001-of-00002.safetensors",
        "text_encoder_3/model-00002-of-00002.safetensors",
        "text_encoder_3/model.safetensors.index.json",
        "vae/diffusion_pytorch_model.safetensors",
    };
    var kept: usize = 0;
    for (listing) |path| {
        const keep = wantedFile(sd3, layout, path);
        var expected = false;
        for (want) |w| {
            if (std.mem.eql(u8, w, path)) expected = true;
        }
        if (keep != expected) {
            std.debug.print("wantedFile mismatch: {s} kept={} want={}\n", .{ path, keep, expected });
            return error.TestUnexpectedResult;
        }
        if (keep) kept += 1;
    }
    try testing.expectEqual(want.len, kept);

    // The variant marker itself, both spellings and the near-misses.
    try testing.expect(isFp16Variant("model.fp16.safetensors"));
    try testing.expect(isFp16Variant("model.fp16-00001-of-00002.safetensors"));
    try testing.expect(!isFp16Variant("model.safetensors"));
    try testing.expect(!isFp16Variant("model-00001-of-00002.safetensors"));
    // A basename that merely ENDS in ".fp16" names no variant.
    try testing.expect(!isFp16Variant("weights.fp16"));
}

test "cli: a single-file alias leaves model_index.json on the server" {
    // NoobAI ships a COMPLETE diffusers folder beside its LDM checkpoint, and
    // the V-Pred repo's own `scheduler_config.json` declares epsilon — wrong for
    // it. Only the checkpoint carries the `v_pred`/`ztsnr` markers, so pulling
    // the index would silently sample V-Pred on an epsilon ladder: a plausible,
    // systematically washed-out image with nothing to error on. Twins
    // `SdxlFinetuneCatalogTests`.
    const noobai = resolveShortName("noobai:vpred").?;
    const listing = [_][]const u8{
        "NoobAI-XL-Vpred-v1.0.safetensors",
        "README.md",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "unet/diffusion_pytorch_model.safetensors",
    };
    const l = layoutFor(noobai, &listing);
    try testing.expect(!l.diffusers);
    var kept: usize = 0;
    for (listing) |path| {
        if (wantedFile(noobai, l, path)) {
            try testing.expectEqualStrings("NoobAI-XL-Vpred-v1.0.safetensors", path);
            kept += 1;
        }
    }
    try testing.expectEqual(@as(usize, 1), kept);

    // Pony's repo carries a standalone `sdxl_vae` checkpoint beside the model.
    const pony = resolveShortName("pony").?;
    const pl = layoutFor(pony, &.{});
    try testing.expect(wantedFile(pony, pl, "ponyDiffusionV6XL_v6StartWithThisOne.safetensors"));
    try testing.expect(!wantedFile(pony, pl, "sdxl_vae.safetensors"));
}

test "cli: every image alias resolves to a shape `pull` can actually fetch" {
    // Class guard for the bug above: the three image repo shapes (single-file,
    // subdir variant, diffusers multifolder) each need their own arm, and an
    // alias that matches none falls through to the flat-repo rule, which drops
    // every nested path. A new image alias with no declared shape trips here
    // rather than at the first `pull`.
    for (aliases) |a| {
        const image = std.mem.eql(u8, a.name, "sdxl") or std.mem.eql(u8, a.name, "sd1") or
            std.mem.eql(u8, a.name, "pony") or std.mem.eql(u8, a.name, "noobai") or
            std.mem.eql(u8, a.name, "illustrious");
        if (!image) continue;
        const r = a.resolve();
        if (r.single_file.len > 0 or r.subdir.len > 0) continue;
        // The remainder must be diffusers repos — proven by the fact that a
        // listing WITHOUT a root index keeps nothing usable under a component.
        const l = layoutFor(r, &.{"model_index.json"});
        try testing.expect(l.diffusers);
        try testing.expect(wantedFile(r, l, "unet/config.json"));
    }
}

test "cli: SD 3.5 resolves to three gated diffusers repos, bare name = medium" {
    // The T5 tower alone is 9.5 GB and is shared by all three, so the smallest
    // complete SD 3.5 download is a different CHECKPOINT rather than a narrower
    // quantization of one — which is why bare `sd3.5` is medium and not the
    // 4-bit-default convention the chat aliases use.
    const med = resolveShortName("sd3.5").?;
    try testing.expectEqualStrings("stabilityai/stable-diffusion-3.5-medium", med.repo);
    try testing.expect(med.gated);
    try testing.expectEqualStrings("stabilityai/stable-diffusion-3.5-large", resolveShortName("sd3.5:large").?.repo);
    try testing.expectEqualStrings("stabilityai/stable-diffusion-3.5-large-turbo", resolveShortName("sd3.5:large-turbo").?.repo);

    // All three are diffusers multifolder repos, and none is a single-file or
    // subdir shape — so `layoutFor` must classify them off the index alone.
    for ([_][]const u8{ "sd3.5", "sd3.5:large", "sd3.5:large-turbo" }) |name| {
        const r = resolveShortName(name).?;
        try testing.expect(r.gated);
        try testing.expectEqualStrings("", r.single_file);
        try testing.expectEqualStrings("", r.subdir);
        const l = layoutFor(r, &.{"model_index.json"});
        try testing.expect(l.diffusers);
        // The SD 3.5-only components the SDXL set never had.
        try testing.expect(wantedFile(r, l, "transformer/config.json"));
        try testing.expect(wantedFile(r, l, "text_encoder_3/config.json"));
        try testing.expect(wantedFile(r, l, "tokenizer_3/tokenizer.json"));
        // ...and the 16 GB merged root checkpoint that the folder path never opens.
        try testing.expect(!wantedFile(r, l, "sd3.5_large.safetensors"));
        // The ComfyUI-style flat drop is a THIRD copy of the towers.
        try testing.expect(!wantedFile(r, l, "text_encoders/t5xxl_fp16.safetensors"));
        // A stray "vae copy/" (the medium mirror really ships one).
        try testing.expect(!wantedFile(r, l, "vae copy/diffusion_pytorch_model.safetensors"));
    }
}

test "cli: shouldDownload chat-default selection" {
    try testing.expect(shouldDownload("config.json"));
    try testing.expect(shouldDownload("model.safetensors"));
    try testing.expect(shouldDownload("model-00001-of-00002.safetensors"));
    try testing.expect(shouldDownload("tokenizer.json"));
    try testing.expect(shouldDownload("chat_template.jinja"));
    try testing.expect(shouldDownload("mtp/weights.safetensors"));
    try testing.expect(!shouldDownload(".gitattributes"));
    try testing.expect(!shouldDownload("README.md"));
    try testing.expect(!shouldDownload("assets/demo.png"));
    try testing.expect(!shouldDownload("banner.png"));
    try testing.expect(!shouldDownload("vae/weights.safetensors")); // media subdirs are app-bundle territory
    // A `.bin` the engine READS (qwen4_exp ngram_table) is needed; torch-format
    // shadow weights are a second copy of the same model. Same rule as the app's
    // `DownloadManager.selectNeededFiles` — keep them in sync.
    try testing.expect(shouldDownload("ngram_table.bin"));
    try testing.expect(!shouldDownload("pytorch_model-00001-of-00002.bin"));
    try testing.expect(!shouldDownload("consolidated.pth"));
    try testing.expect(!shouldDownload("flax_model.msgpack"));
}

test "cli: modelPresentInDir requires a COMPLETE checkpoint" {
    // Regression: an interrupted `pull` (Ctrl-C mid-weights) leaves
    // config.json + model.safetensors.partial. modelPresent used to return
    // true on config.json alone, so the rerun skipped the resume and fed a
    // weightless dir to the loader (SIGSEGV). Present now means: no .partial
    // leftovers anywhere (top level or one subdir deep), and config.json +
    // >=1 .safetensors (MLX) or any .gguf.
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    // config.json alone (weights never started): not present.
    try tmp.dir.createDirPath(io, "a");
    try tmp.dir.writeFile(io, .{ .sub_path = "a/config.json", .data = "{}" });
    {
        var d = try tmp.dir.openDir(io, "a", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(!modelPresentInDir(io, d));
    }

    // config.json + interrupted weights: not present (the user's live repro).
    try tmp.dir.writeFile(io, .{ .sub_path = "a/model.safetensors.partial", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "a", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(!modelPresentInDir(io, d));
    }

    // Complete single-file checkpoint: present.
    try tmp.dir.createDirPath(io, "b");
    try tmp.dir.writeFile(io, .{ .sub_path = "b/config.json", .data = "{}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "b/model.safetensors", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "b", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(modelPresentInDir(io, d));
    }

    // Complete weights but another file still partial (e.g. tokenizer.json):
    // not present — resume must finish the pull.
    try tmp.dir.writeFile(io, .{ .sub_path = "b/tokenizer.json.partial", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "b", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(!modelPresentInDir(io, d));
    }

    // Interrupted sidecar one subdir deep (mtp/weights.safetensors.partial):
    // not present.
    try tmp.dir.createDirPath(io, "c/mtp");
    try tmp.dir.writeFile(io, .{ .sub_path = "c/config.json", .data = "{}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "c/model.safetensors", .data = "x" });
    try tmp.dir.writeFile(io, .{ .sub_path = "c/mtp/weights.safetensors.partial", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "c", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(!modelPresentInDir(io, d));
    }

    // GGUF: the file itself is the checkpoint (no config.json needed)…
    try tmp.dir.createDirPath(io, "g");
    try tmp.dir.writeFile(io, .{ .sub_path = "g/model-Q4_K_M.gguf", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "g", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(modelPresentInDir(io, d));
    }

    // …but a partial GGUF is not.
    try tmp.dir.createDirPath(io, "h");
    try tmp.dir.writeFile(io, .{ .sub_path = "h/model-Q4_K_M.gguf.partial", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "h", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(!modelPresentInDir(io, d));
    }
}

test "cli: parseTreeJson uses lfs size and skips directories" {
    const allocator = testing.allocator;
    const files = try parseTreeJson(allocator,
        \\[{"type":"directory","path":"mtp","size":0},
        \\ {"type":"file","path":"config.json","size":1234},
        \\ {"type":"file","path":"model.safetensors","size":135,"lfs":{"size":5300000000,"sha256":"x"}}]
    );
    defer freeRepoFiles(allocator, files);
    try testing.expectEqual(@as(usize, 2), files.len);
    try testing.expectEqualStrings("config.json", files[0].path);
    try testing.expectEqual(@as(u64, 1234), files[0].size);
    try testing.expectEqual(@as(u64, 5_300_000_000), files[1].size);
}

test "cli: buildReplChatBody and parseReplLine round-trip" {
    const allocator = testing.allocator;
    const history = [_]Turn{
        .{ .role = "user", .content = "hi \"there\"\n" },
        .{ .role = "assistant", .content = "hello" },
    };
    const body = try buildReplChatBody(allocator, &history);
    defer allocator.free(body);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const msgs = parsed.value.object.get("messages").?.array.items;
    try testing.expectEqual(@as(usize, 2), msgs.len);
    try testing.expectEqualStrings("hi \"there\"\n", msgs[0].object.get("content").?.string);
    try testing.expect(parsed.value.object.get("stream").?.bool);

    const d1 = parseReplLine(allocator, "{\"model\":\"m\",\"message\":{\"role\":\"assistant\",\"content\":\"Hey\"},\"done\":false}").?;
    defer allocator.free(d1.content);
    try testing.expectEqualStrings("Hey", d1.content);
    try testing.expect(!d1.done);

    const d2 = parseReplLine(allocator, "{\"model\":\"m\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\",\"eval_count\":50,\"eval_duration\":2000000000}").?;
    defer allocator.free(d2.content);
    try testing.expect(d2.done);
    try testing.expectEqual(@as(u64, 50), d2.eval_count);

    const d3 = parseReplLine(allocator, "{\"error\":\"boom\"}").?;
    defer allocator.free(d3.content);
    defer if (d3.err) |e| allocator.free(e);
    try testing.expect(d3.done);
    try testing.expectEqualStrings("boom", d3.err.?);
}

test "cli: formatSize" {
    var buf: [32]u8 = undefined;
    try testing.expectEqualStrings("5.2 GB", formatSize(&buf, 5_600_000_000));
    try testing.expectEqualStrings("35 MB", formatSize(&buf, 36_700_160));
    try testing.expectEqualStrings("2 KB", formatSize(&buf, 2048));
}

test "cli: dirBytesOneLevel counts weight subdirs (FLUX bundle showed 6 KB)" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    // Media-bundle shape: small top-level config + the actual weights one
    // level down in transformer/ and vae/ (the modelPresent layout).
    try tmp.dir.createDirPath(io, "m/transformer");
    try tmp.dir.createDirPath(io, "m/vae");
    try tmp.dir.writeFile(io, .{ .sub_path = "m/config.json", .data = "{}" }); // 2 bytes
    try tmp.dir.writeFile(io, .{ .sub_path = "m/transformer/w.safetensors", .data = "0123456789" }); // 10
    try tmp.dir.writeFile(io, .{ .sub_path = "m/vae/w.safetensors", .data = "0123" }); // 4

    var m = try tmp.dir.openDir(io, "m", .{ .iterate = true });
    defer m.close(io);
    try testing.expectEqual(@as(u64, 16), dirBytesOneLevel(io, &m));
}

test "cli: a serving boot with no model named falls back to the shared models root" {
    // `mlx-serve serve` and `mlx-serve run <m>` already default the discovery
    // root; the `--serve` FLAG form did not, so a bare `mlx-serve --serve`
    // booted a server that had discovered nothing and could only answer 503 —
    // never what anyone meant by "serve".
    try testing.expect(shouldDefaultModelsRoot(.{ .subcommand = true, .serve_mode = true, .has_explicit_model = false }));
    try testing.expect(shouldDefaultModelsRoot(.{ .subcommand = false, .serve_mode = true, .has_explicit_model = false }));

    // `--model <path> --serve` asked for ONE model. Quietly registering the
    // other 28 on disk is a different server than the one requested.
    try testing.expect(!shouldDefaultModelsRoot(.{ .subcommand = false, .serve_mode = true, .has_explicit_model = true }));
    // `mlx-serve run <model>` names a model AND wants the picker populated —
    // the subcommand's existing behavior, which this must not change.
    try testing.expect(shouldDefaultModelsRoot(.{ .subcommand = true, .serve_mode = true, .has_explicit_model = true }));

    // Not serving at all (one-shot `--prompt`) never scans a root.
    try testing.expect(!shouldDefaultModelsRoot(.{ .subcommand = false, .serve_mode = false, .has_explicit_model = true }));
    try testing.expect(!shouldDefaultModelsRoot(.{ .subcommand = false, .serve_mode = false, .has_explicit_model = false }));
}

test "cli: isModelDir accepts a MageFlow repo (model_index.json, no config.json)" {
    const io = std.testing.io;
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    // `list` must agree with `/v1/models` about what a model is. A MageFlow
    // repo carries only model_index.json, so a config-or-gguf test hides it
    // from `list` while discovery serves it — the exact divergence the
    // "one path must never classify two ways" rule exists to prevent.
    try tmp.dir.createDirPath(io, "mage/transformer");
    try tmp.dir.writeFile(io, .{
        .sub_path = "mage/model_index.json",
        .data =
        \\{"_class_name":"MageFlowPipeline"}
        ,
    });
    var mage = try tmp.dir.openDir(io, "mage", .{ .iterate = true });
    defer mage.close(io);
    try testing.expect(isModelDir(io, a, &mage));

    // An unrelated diffusers pipeline is NOT ours — it must stay hidden.
    try tmp.dir.createDirPath(io, "other");
    try tmp.dir.writeFile(io, .{
        .sub_path = "other/model_index.json",
        .data =
        \\{"_class_name":"StableDiffusionPipeline"}
        ,
    });
    var other = try tmp.dir.openDir(io, "other", .{ .iterate = true });
    defer other.close(io);
    try testing.expect(!isModelDir(io, a, &other));

    // An org dir (no config, no index, no gguf) stays a directory to descend.
    try tmp.dir.createDirPath(io, "org/repo");
    var org = try tmp.dir.openDir(io, "org", .{ .iterate = true });
    defer org.close(io);
    try testing.expect(!isModelDir(io, a, &org));
}

test "cli: isModelDir accepts an mflux FLUX.2 repo (no config.json at all)" {
    const io = std.testing.io;
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    // The klein 9B MLX build carries neither config.json nor model_index.json,
    // so this is the third place that has to know the shape. Same rule as
    // MageFlow above: `list` and `/v1/models` answer from one predicate.
    try tmp.dir.createDirPath(io, "klein9/transformer");
    try tmp.dir.writeFile(io, .{
        .sub_path = "klein9/transformer/model.safetensors.index.json",
        .data = "{\"weight_map\":{\"double_stream_modulation_img.linear.weight\":\"0.safetensors\"}}",
    });
    var klein = try tmp.dir.openDir(io, "klein9", .{ .iterate = true });
    defer klein.close(io);
    try testing.expect(isModelDir(io, a, &klein));

    // A different DiT with the same directory shape is not ours.
    try tmp.dir.createDirPath(io, "notflux/transformer");
    try tmp.dir.writeFile(io, .{
        .sub_path = "notflux/transformer/model.safetensors.index.json",
        .data = "{\"weight_map\":{\"blocks.0.attn.qkv.weight\":\"0.safetensors\"}}",
    });
    var notflux = try tmp.dir.openDir(io, "notflux", .{ .iterate = true });
    defer notflux.close(io);
    try testing.expect(!isModelDir(io, a, &notflux));
}

test "cli: an unparsed argument is classified, never silently ignored" {
    // `--flag=value`: main.zig's flag loop matches names EXACTLY and takes the
    // value as the NEXT argument, so the '='-joined form is not a near-miss,
    // it is a shape we have never accepted. It used to fall out of the loop in
    // silence — `--model=<path>` booted a healthy-looking headless server that
    // then auto-picked a different model and crashed on it.
    try testing.expectEqual(ArgReject.equals_form, classifyUnparsedArg("--model=/tmp/m", false));
    try testing.expectEqual(ArgReject.equals_form, classifyUnparsedArg("--port=1234", false));
    // Even in last position the '=' hint is the useful one.
    try testing.expectEqual(ArgReject.equals_form, classifyUnparsedArg("--model=/tmp/m", true));

    // A flag in the LAST position fell out because its value is missing —
    // the loop's `i + 1 < args.len` guard is the only way to reach here.
    try testing.expectEqual(ArgReject.missing_value, classifyUnparsedArg("--model", true));
    try testing.expectEqual(ArgReject.missing_value, classifyUnparsedArg("-h", true));

    // Anything else is simply not a flag we know.
    try testing.expectEqual(ArgReject.unknown, classifyUnparsedArg("--frobnicate", false));
    try testing.expectEqual(ArgReject.unknown, classifyUnparsedArg("stray", false));
    // An '=' outside a flag is not the equals form.
    try testing.expectEqual(ArgReject.unknown, classifyUnparsedArg("a=b", false));
    try testing.expectEqual(ArgReject.unknown, classifyUnparsedArg("a=b", true));

    // Every reason carries actionable advice, and the equals-form one names
    // the shape that actually works.
    for ([_]ArgReject{ .equals_form, .missing_value, .unknown }) |r| {
        try testing.expect(r.hint().len > 0);
    }
    try testing.expect(std.mem.indexOf(u8, ArgReject.equals_form.hint(), "separate argument") != null);
}

test "cli: list tree walk descends into symlinked model dirs" {
    // Moving a big checkpoint to an external drive and symlinking it back is
    // a supported layout (the H3 mirrors live that way): model_discovery's
    // walk accepts .sym_link entries, but `list` had its own private walk
    // that silently skipped them — both MiniMax mirrors vanished from `list`
    // while the server kept serving them. Both loops route through ONE
    // predicate now.
    try testing.expect(treeEntryDescends(.directory));
    try testing.expect(treeEntryDescends(.sym_link));
    try testing.expect(!treeEntryDescends(.file));

    // Source scan: the org-level and leaf-level loops in listModels must both
    // consult the predicate — a reintroduced raw `!= .directory` check is the
    // regression this pins. Needle split so the scan cannot match itself.
    const src = @embedFile("cli.zig");
    const needle = "treeEntry" ++ "Descends(";
    var found: usize = 0;
    var idx: usize = 0;
    while (std.mem.indexOfPos(u8, src, idx, needle)) |p| {
        found += 1;
        idx = p + needle.len;
    }
    // 1 definition + 3 in this test + at least 2 call sites in the walk.
    try testing.expect(found >= 6);
}
