//! A loaded FLUX.2 klein pipeline: text encoder (resident, or per request in low-memory mode), DiT, VAE and the
//! optional VAE encoder (img2img and edits), plus the Qwen3 tokenizer and its fixed-length prompt template.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const flux = @import("flux.zig");
const tok_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const log = @import("../../sdk/util/log.zig");
const sse = @import("../../sdk/media/progress.zig");
const image_request = @import("../../sdk/media/image_request.zig");

/// Qwen2/3's pad token and mflux's tokenizer max_length: the DiT's text positions are built for this length.
const PAD_TOKEN: i32 = 151643;
pub const SEQ_LEN: usize = 512;

/// References one edit takes, the primary included.
pub const MAX_EDIT_IMAGES = 4;

/// One request's options beyond prompt, size, seed and steps.
pub const Opts = struct {
    /// img2img source [1,3,H,W] f32 in [0,1] at the output size; renoised at `strength`.
    init_image: ?mlx.mlx_array = null,
    strength: f32 = 0.6,
    /// Edit references [1,3,h,w] f32 in [0,1], each at its own grid, primary first (clean in-context tokens).
    edit_images: []const mlx.mlx_array = &.{},
    /// Conditioning rebalance: global gain and one weight per tapped encoder layer (3).
    cond_gain: f32 = 1.0,
    cond_weights: ?[]const f32 = null,
    /// Real CFG (the undistilled base klein); 1.0 runs one forward per step.
    guidance_scale: f32 = 1.0,
    negative_prompt: []const u8 = "",
};

pub const Engine = struct {
    s: mlx.mlx_stream,
    /// Null in low-memory mode between requests: loaded per request and freed after the prompt encode.
    te: ?flux.TextEncoder,
    dit: flux.Dit,
    vae: flux.Vae,
    vae_enc: ?flux.VaeEncoder,
    tok: tok_mod.Tokenizer,
    io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []u8,
    low_mem: bool,

    /// Every array of the engine lives and runs on `s`. `low_mem` (the host's policy, `Machine.low_mem`) loads the
    /// encoder per request and frees it after the prompt (~0.1-0.3 s per image, ~1.8 GB lower peak).
    pub fn loadOn(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, s: mlx.mlx_stream, low_mem: bool) !Engine {
        var self: Engine = undefined;
        self.io = io;
        self.allocator = allocator;
        self.low_mem = low_mem;
        self.model_dir = try allocator.dupe(u8, model_dir);
        errdefer allocator.free(self.model_dir);
        self.s = s;
        if (self.low_mem) {
            self.te = null;
            log.info("[image] FLUX low-mem mode: text encoder loads per request\n", .{});
        } else {
            self.te = try flux.loadTextEncoder(io, allocator, self.s, model_dir);
        }
        errdefer if (self.te) |*t| t.deinit();
        self.dit = try flux.loadDit(io, allocator, self.s, model_dir);
        errdefer self.dit.deinit();
        self.vae = try flux.loadVae(io, allocator, self.s, model_dir);
        errdefer self.vae.deinit();
        self.vae_enc = flux.loadVaeEncoder(io, allocator, self.s, model_dir) catch |e| blk: {
            log.warn("[image] FLUX VAE encoder load failed ({}) — image-to-image disabled\n", .{e});
            break :blk null;
        };
        errdefer if (self.vae_enc) |*e| e.deinit();
        const tok_dir = try std.fmt.allocPrint(allocator, "{s}/tokenizer", .{model_dir});
        defer allocator.free(tok_dir);
        self.tok = try tok_mod.loadTokenizerAny(io, allocator, tok_dir);
        log.info("[image] FLUX models + tokenizer ready\n", .{});
        return self;
    }

    pub fn deinit(self: *Engine) void {
        if (self.te) |*t| t.deinit();
        self.dit.deinit();
        self.vae.deinit();
        if (self.vae_enc) |*e| e.deinit();
        self.tok.deinit();
        self.allocator.free(self.model_dir);
    }

    /// mflux's Qwen3 chat template (thinking off adds an empty think block), padded or truncated to `SEQ_LEN`.
    /// The prompt and the negative prompt land on the same shape.
    fn tokenizeFixed(self: *Engine, allocator: std.mem.Allocator, text: []const u8) !struct { ids: []i32, mask: []i32 } {
        const templated = try std.fmt.allocPrint(allocator, "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", .{text});
        defer allocator.free(templated);
        const enc = try self.tok.encode(allocator, templated);
        defer allocator.free(enc);
        const ids = try allocator.alloc(i32, SEQ_LEN);
        errdefer allocator.free(ids);
        const mask = try allocator.alloc(i32, SEQ_LEN);
        errdefer allocator.free(mask);
        const real = @min(enc.len, SEQ_LEN);
        for (0..SEQ_LEN) |i| {
            if (i < real) {
                ids[i] = @intCast(enc[i]);
                mask[i] = 1;
            } else {
                ids[i] = PAD_TOKEN;
                mask[i] = 0;
            }
        }
        return .{ .ids = ids, .mask = mask };
    }

    /// The pipeline → image [1,3,H,W] f32 in [0,1] (owned; caller frees).
    pub fn generateImage(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, opts: Opts, progress: ?sse.Progress) !mlx.mlx_array {
        const pos = try self.tokenizeFixed(allocator, prompt);
        defer allocator.free(pos.ids);
        defer allocator.free(pos.mask);
        var fopts = flux.GenOpts{ .cond_gain = opts.cond_gain, .cond_weights = opts.cond_weights, .guidance_scale = opts.guidance_scale };
        var init_lat: ?mlx.mlx_array = null;
        defer if (init_lat) |l| {
            _ = mlx.mlx_array_free(l);
        };
        var g = ops.Graph.init(allocator, self.s);
        defer g.deinit();
        var ref_lats: [MAX_EDIT_IMAGES]mlx.mlx_array = undefined;
        var ref_lat_n: usize = 0;
        defer for (ref_lats[0..ref_lat_n]) |l| {
            _ = mlx.mlx_array_free(l);
        };
        if (opts.edit_images.len > 0) {
            const ve = if (self.vae_enc) |*e| e else return error.NoVaeEncoder;
            if (opts.edit_images.len > MAX_EDIT_IMAGES) return error.TooManyEditImages;
            for (opts.edit_images) |pix| {
                ref_lats[ref_lat_n] = g.escape(try ve.encode(&g, pix));
                ref_lat_n += 1;
            }
            fopts.ref_latents = ref_lats[0..ref_lat_n];
        } else if (opts.init_image) |pix| {
            const ve = if (self.vae_enc) |*e| e else return error.NoVaeEncoder;
            init_lat = g.escape(try ve.encode(&g, pix));
            fopts.init_latents = init_lat;
            fopts.start_step = image_request.img2imgStartStep(steps, opts.strength);
        }
        // The prompt encode is materialized inside `encodePrompt`, so freeing the encoder after it (low-memory
        // mode) changes no output.
        if (self.te == null) self.te = try flux.loadTextEncoder(self.io, self.allocator, self.s, self.model_dir);
        if (opts.guidance_scale != 1.0) {
            const neg = try self.tokenizeFixed(allocator, opts.negative_prompt);
            defer allocator.free(neg.ids);
            defer allocator.free(neg.mask);
            fopts.neg_enc = try flux.encodePrompt(&self.te.?, neg.ids, neg.mask, fopts);
        }
        const cond = try flux.encodePrompt(&self.te.?, pos.ids, pos.mask, fopts);
        if (self.low_mem) {
            self.te.?.deinit();
            self.te = null;
            _ = mlx.mlx_clear_cache();
            log.info("[image] low-mem: text encoder freed after encode\n", .{});
        }
        return flux.generateFromCondWithOpts(&self.dit, &self.vae, cond, pos.ids.len, seed, steps, height, width, fopts, progress);
    }
};
