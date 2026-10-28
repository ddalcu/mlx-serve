//! The model plugin table (`ModelTable`) and every struct that crosses it. Extern layouts, C calling convention, no
//! globals. Every struct opens with `size`: one passed by pointer grows at its end; one held by value or as an array
//! element is frozen per `version` (its `size` is a check). Per-model values ride `Parsed`; hook tables hold functions.

const Host = @import("host.zig").Host;
const quant = @import("quant.zig");
const Source = quant.Source;
const std = @import("std");
const common = @import("common.zig");

pub const Span = common.Span;
pub const Str = common.Str;
pub const Verdict = common.Verdict;
pub const Array = common.Array;
pub const Stream = common.Stream;
pub const Claim = common.Claim;
pub const PackDir = common.PackDir;

pub const version: u32 = 1;

const nan = std.math.nan(f32);

/// An object the engine owns and the plugin only passes back (a slot's KV cache): opaque until P3 types it.
pub const Handle = ?*anyopaque;

/// The layers this process serves when a model is split across machines; null = all of them.
pub const LayerSpan = extern struct {
    size: u32 = @sizeOf(LayerSpan),
    first: u32 = 0,
    count: u32 = 0,
};

// ── Parsed: what `parse` fills ──

pub const DType = enum(u8) { unset, f16, bf16, f32 };
pub const Tri = enum(u8) { unset, off, on };
pub const QuantMode = enum(u8) { none, affine, nvfp4, mxfp4, mxfp8, gguf };

/// The config's `quantization` block.
pub const Quant = extern struct {
    size: u32 = @sizeOf(Quant),
    bits: u8 = 0,
    mode: QuantMode = .none,
    group_size: u32 = 0,
};

/// Sampling defaults from the pack (`generation_config.json`); NaN or -1 = unset.
pub const GenDefaults = extern struct {
    size: u32 = @sizeOf(GenDefaults),
    temperature: f32 = nan,
    top_p: f32 = nan,
    min_p: f32 = nan,
    top_k: i32 = -1,
    enable_thinking: Tri = .unset,
};

pub const max_eos = 16;

/// The generic shell: the only config the engine reads. Per-layer facts are `CacheHooks.layout`, not here.
pub const Shell = extern struct {
    size: u32 = @sizeOf(Shell),
    vocab_size: u32 = 0,
    hidden_size: u32 = 0,
    /// Leaves once every plugin bills `Bill.prefill_streams`.
    intermediate_size: u32 = 0,
    num_hidden_layers: u32 = 0,
    num_attention_heads: u32 = 0,
    num_key_value_heads: u32 = 0,
    head_dim: u32 = 0,
    /// Temporary (decisions.md): both leave once every plugin's prefill bill lands.
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    max_position_embeddings: u64 = 0,
    /// The derived context the model serves.
    context_cap: u64 = 0,
    quant: Quant = .{},
    gen: GenDefaults = .{},
    bos: i64 = -1, // -1 = none
    eos: [max_eos]u32 = @splat(0),
    eos_count: u32 = 0,
    act_dtype: DType = .bf16,
    state_dtype: DType = .unset,
    /// For display only: the host never compares it (the plugin is the identity).
    model_type: Str = .{},
};

/// Model facts the host keeps once at load. Defaults are the generic model's.
pub const Caps = packed struct(u64) {
    batches_decode: bool = false,
    batch_per_slot_attn: bool = true,
    batch_pads_kv: bool = true,
    /// Per-request decode state on the module, not the KV cache: single-flight, prefix cache through restore only.
    owns_decode_state: bool = false,
    shared_readonly_state: bool = false,
    long_ctx: bool = false,
    moe: bool = false,
    prefix_cache: bool = true,
    /// The exactness contracts (P2 step 7): drafted == plain, restored == cold, batched == solo.
    exact_drafted: bool = false,
    exact_resumed: bool = false,
    exact_concurrent: bool = false,
    /// The forward fills `Forward.capture_layers`.
    dflash_capture: bool = false,
    dflash_media: bool = false,
    spec_rollback: bool = false,
    spec_tree: bool = false,
    whole_prompt_prefill: bool = false,
    /// The forward honours `Forward.offload` and `WeightHooks.layer_weight` lends the offloaded projections.
    ane_prefill: bool = false,
    streamed_weights: bool = false,
    embed: bool = false,
    images: bool = false,
    video: bool = false,
    audio: bool = false,
    /// A pack another pack loads beside it (the ASR forced aligner).
    companion: bool = false,
    drafter: bool = false,
    /// The forward reads `Forward.mask_add`.
    prefill_mask: bool = false,
    /// The model reads a Thunderbolt cluster's plan itself (`distributed.planFor`); a planned pack whose plugin does
    /// not is loaded by the legacy path.
    cluster_plan: bool = false,
    /// DFlash rounds follow the round policy (`dflash_policy.zig`): a sparse target whose verify rows cost.
    dflash_policy: bool = false,
    /// A rejected draft rolls back by truncating the KV cache (an attention trunk): no snapshot, no re-forward.
    verify_truncates: bool = false,
    /// A batch's recurrent state is read and written only through `CacheOps.state_rows` / `set_state_rows`, so the
    /// engine may keep it merged across batched ticks instead of re-stacking every layer each tick.
    merged_state: bool = false,
    /// Module-owned state (`owns_decode_state`) kept per slot (`Forward.slots`, `release_slot`, `activate_slot`):
    /// requests interleave and batch instead of running single-flight.
    slot_state: bool = false,
    _: u34 = 0,
};

pub const ThinkingDefault = enum(u8) { unset, off, on, on_with_tools };
pub const ReasoningDialect = enum(u8) { unset, think_tags, gemma_channel, harmony, muse, inkling, content_thinking };
/// The effort words a template understands; the tables stay in the host.
pub const EffortVocab = enum(u8) { none, generic, qwen38, kolibri, dsv4, dsv41, glm5, k2, inkling };
pub const RoundTrip = enum(u8) { keep, drop_prior_turns, required_field };
/// The dialect `tool_choice` enforcement commits an opener in; parsing stays dialect-agnostic in the host.
pub const ToolDialect = enum(u8) { unset, qwen_xml };

pub const NoThink = extern struct {
    size: u32 = @sizeOf(NoThink),
    template_refuses: bool = false,
    tail_suffix: Str = .{},
};

/// Family sampling defaults when the pack has no generation_config; NaN or -1 = none.
pub const SamplingDefaults = extern struct {
    size: u32 = @sizeOf(SamplingDefaults),
    top_k: i32 = -1,
    top_p: f32 = nan,
    wild_temp_override: f32 = nan,
};

/// Chat facts. A fact left `unset`/empty means the host falls back to its template probe.
pub const ChatFacts = extern struct {
    size: u32 = @sizeOf(ChatFacts),
    thinking_default: ThinkingDefault = .unset,
    reasoning_dialect: ReasoningDialect = .unset,
    effort_vocab: EffortVocab = .none,
    reasoning_round_trip: RoundTrip = .keep,
    tool_dialect: ToolDialect = .unset,
    nothink: NoThink = .{},
    template_family: Str = .{},
    sampling_defaults: SamplingDefaults = .{},
    video_placeholder: Str = .{},
};

pub const MarkerAlias = extern struct {
    size: u32 = @sizeOf(MarkerAlias),
    from: Str = .{},
    to: Str = .{},
};

pub const TokenizerFacts = extern struct {
    size: u32 = @sizeOf(TokenizerFacts),
    extra_eos: [max_eos]u32 = @splat(0),
    extra_eos_count: u32 = 0,
    /// The vocabulary rows that are real tokens; 0 = `vocab_size`.
    defined_vocab: u32 = 0,
    marker_aliases: Span(MarkerAlias) = .{},
};

/// How the engine keeps a recurrent layer's state across a pipelined step.
pub const SnapshotPolicy = enum(u8) { lagged, in_place };

pub const SsmState = extern struct {
    size: u32 = @sizeOf(SsmState),
    /// Per-request recurrent entries; 0 = none.
    entries: u32 = 0,
    dtype: DType = .unset,
    snapshot_policy: SnapshotPolicy = .lagged,
    checkpoint_bytes: u64 = 0,
};

/// `CacheHooks`' per-model values.
pub const CacheFacts = extern struct {
    size: u32 = @sizeOf(CacheFacts),
    /// A fresh SSD-cache root when stored K/V change meaning; empty = the default root.
    layout_namespace: Str = .{},
    /// Rows a sliding layer keeps (window + slack); 0 = the full buffer.
    sliding_ring: u32 = 0,
    ssm_state: SsmState = .{},
};

pub const MtpLayout = enum(u8) { none, sidecar, in_checkpoint, nemotron };

pub const SpecPolicy = extern struct {
    size: u32 = @sizeOf(SpecPolicy),
    drafter_ok: bool = true,
    mtp_group_solo: bool = false,
    seeded_sampling: bool = false,
    /// 0 = the machine's block.
    dflash_block: u32 = 0,
    /// The DFlash yield gate's floor in accepted tokens per round x10; 0 = the calibrated gate.
    dflash_floor_x10: u32 = 0,
    draft_temperature: f32 = nan,
};

/// `SpecHooks`' per-model values.
pub const SpecFacts = extern struct {
    size: u32 = @sizeOf(SpecFacts),
    mtp_default_depth: u8 = 0,
    mtp_layout: MtpLayout = .none,
    /// MTP heads, one per draft depth (MiMo: 3), each with its own history and each reading the trunk's hidden rows
    /// (`hidden_in` stays the trunk's along the round); 0 or 1 = one head chained on its own output. A round's draft k
    /// runs head k (`Forward.head`); a draft past the last head re-runs it and is never accepted.
    mtp_heads: u32 = 0,
    head_module_owned: bool = false,
    head_reads_final_norm: bool = false,
    draft_vocab: u32 = 0,
    /// An id into the engine's measured round-cost tables; 0 = generic.
    cost_profile: u32 = 0,
    native_stages: u32 = 0,
    native_block: u32 = 0,
    policy: SpecPolicy = .{},
    /// The trunk multiplies its embedding rows by this before the first layer (Gemma: sqrt(hidden)); 0 = unscaled.
    /// A drafter that embeds through the trunk's table (`trunk_weight`) applies it itself.
    embed_scale: f32 = 0,
};

/// Placeholder token ids a media item's rows replace; 0 = unset, never token 0.
pub const PlaceholderIds = extern struct {
    size: u32 = @sizeOf(PlaceholderIds),
    image: u32 = 0,
    video: u32 = 0,
    audio: u32 = 0,
    boi: u32 = 0,
    eoi: u32 = 0,
    boa: u32 = 0,
    eoa: u32 = 0,
    vision_start: u32 = 0,
    vision_end: u32 = 0,
};

/// The model's interleaved M-RoPE spectrum, from which the engine builds a prompt chunk's angles off the prompt's
/// position table (`Forward.mrope`).
pub const MropeFacts = extern struct {
    size: u32 = @sizeOf(MropeFacts),
    /// The t/h/w frequency split; all zero = the model takes no M-RoPE table.
    sections: [3]u32 = .{ 0, 0, 0 },
    /// Rotated dims per head.
    rope_dims: u32 = 0,
    /// `rope_dims / 2` inverse frequencies (YaRN-scaled when the model scales), plugin-owned until `free_parsed`.
    inv_freq: Span(f64) = .{},
    /// YaRN's attention factor on the angles; 1 = none.
    mscale: f64 = 1,
};

pub const Pooling = enum(u8) { unset, mean, cls, last_token };

pub const EmbedFacts = extern struct {
    size: u32 = @sizeOf(EmbedFacts),
    pooling: Pooling = .unset,
    has_projection: bool = false,
};

pub const WeightFacts = extern struct {
    size: u32 = @sizeOf(WeightFacts),
    /// The weight-name prefix (probed for packs that differ); empty = none.
    prefix: Str = .{},
    /// The quant plugin that reads this pack's tensors (exl3, modelopt, hadamard, ...); empty = MLX's own.
    quant_layout: Str = .{},
};

/// Every `Str` here points at static text or at memory the plugin owns until `ModelTable.free_parsed`.
pub const Parsed = extern struct {
    size: u32 = @sizeOf(Parsed),
    caps: Caps = .{},
    shell: Shell = .{},
    chat: ChatFacts = .{},
    tokenizer: TokenizerFacts = .{},
    cache: CacheFacts = .{},
    spec: SpecFacts = .{},
    placeholder_ids: PlaceholderIds = .{},
    embed: EmbedFacts = .{},
    weights: WeightFacts = .{},
    mrope: MropeFacts = .{},
    /// The plugin's own parsed config, read by its `bill` and `load`.
    private: ?*anyopaque = null,
};

// ── Bill ──

/// The load-time shape `bill` prices.
pub const Shape = extern struct {
    size: u32 = @sizeOf(Shape),
    context: u64 = 0,
    slots: u32 = 1,
    /// KV precision: 0 = the activation dtype, else 8 or 4.
    kv_bits: u8 = 0,
};

/// The plugin's memory terms. A zero or null term is the engine's default arithmetic over the shell and layout.
pub const Bill = extern struct {
    size: u32 = @sizeOf(Bill),
    kv_per_token: u64 = 0,
    fixed_slot_bytes: u64 = 0,
    state_per_token: u64 = 0,
    prefill_streams: u64 = 0,
    load_bytes: u64 = 0,
    ssm_checkpoint: u64 = 0,
    prefill_score_width: u32 = 0,
    prefill_attn_keys: ?*const fn (p: *const Parsed, seq: u64) callconv(.c) u64 = null,
    prefill_bytes: ?*const fn (p: *const Parsed, seq: u64, chunk: u32, kv_bits: u8, warm: u64) callconv(.c) u64 = null,
    batched_kv_len: ?*const fn (p: *const Parsed, kv: u32, gather_on: bool, min: u32) callconv(.c) u32 = null,
};

// ── Load and forward ──

/// Valid during `load` only: the model keeps none of its pointers.
pub const LoadCtx = extern struct {
    size: u32 = @sizeOf(LoadCtx),
    model_dir: Str,
    dir: *const PackDir,
    parsed: *const Parsed,
    settings_json: Str = .{},
    span: ?*const LayerSpan = null,
    stream: Stream = .{},
    /// The host table (P2 step 8); opaque until it lands.
    host: ?*const Host = null,
    /// The pack's tensors as the engine loaded them (`sdk/weights/resident.zig`); null = the plugin reads its own.
    weights: ?*const Source = null,
    /// GPU bytes a streamed expert bank may fill (with what is resident): the GPU ceiling less one prompt's bill at
    /// 16K + 1K and the wired margin; 0 = none.
    stream_room: u64 = 0,
    /// A two-Mac pair's layers on the other Mac (DeepSeek-V4.1's worker run); `count` 0 = no pair.
    pair_remote: LayerSpan = .{},
    /// The host's wired-limit margin (`--wired-margin-gib`): the GPU ceiling's stop a streamed bank leaves unplanned.
    wired_margin_bytes: u64 = 8 << 30,
};

pub const Mode = enum(u8) { prefill, decode, verify, batched_rows };
pub const Logits = enum(u8) { last_row, all_rows, none };

/// One forward. `tokens` holds `rows` tokens per slot, slot after slot; `slots`, `positions` and `caches` hold one
/// entry per slot (its id, its first position, its KV cache).
pub const Forward = extern struct {
    size: u32 = @sizeOf(Forward),
    mode: Mode,
    logits: Logits = .last_row,
    want_hidden: bool = false,
    rows: u32,
    tokens: Span(u32),
    slots: Span(u32) = .{},
    positions: Span(u32) = .{},
    caches: Span(Handle) = .{},
    /// Layers whose hidden rows the forward publishes (DFlash); filled into `ForwardOut.captured`.
    capture_layers: Span(u32) = .{},
    /// Media rows `[n, hidden]` to splice at the placeholder ids (`Parsed.placeholder_ids`) in token order, from row
    /// `media_offset` on (a chunked prefill resumes there).
    media: Array = .{},
    media_offset: u32 = 0,
    /// A prompt chunk's M-RoPE angles, cos and sin stacked `[2, 1, 1, rows, rope_dims]` (built from `Parsed.mrope`);
    /// null = 1-D positions, which then carry the slot's M-RoPE delta.
    mrope: Array = .{},
    /// How the forward reads and writes `caches`; null = a forward that keeps no state (an encoder).
    cache_ops: ?*const CacheOps = null,
    /// The token ids as a device array `[slots, rows]` (a still-lazy sampled token); when set, `tokens` is empty.
    ids: Array = .{},
    /// A wide forward evaluates its residual after every this many layers and after the last (0 = never): the
    /// engine's bound on a prompt chunk's transient memory.
    eval_every: u32 = 0,
    /// An MTP head's input: the trunk's hidden rows `[1, rows, hidden]` beside its token rows.
    hidden_in: Array = .{},
    /// The engine reads nothing off the result before the next `forward_settle`: a host-side input the forward
    /// needs (an n-gram lookup of a still-lazy token) may be deferred to it rather than synced inside the forward.
    defer_host: bool = false,
    /// A padded encoder batch's additive key mask `[slots, rows]` (0 = a real token, a large negative = padding);
    /// empty = none. An encoder batch (`cache_ops` null) carries one `positions` entry per sequence.
    key_mask: Array = .{},
    /// An MTP head forward's head (`SpecFacts.mtp_heads`): its history is layer `head` of `caches[0]`. A history
    /// append (`.none` without `want_hidden`) appends to every head.
    head: u32 = 0,
    /// A DFlash drafter is bound: projections and attention read the row-exact kernels wherever the model's weights
    /// fit them, so a verify row reproduces serial decode bit for bit (`Caps.exact_drafted`).
    row_exact: bool = false,
    /// An engine unit computing a prompt chunk's projections beside the GPU (`Caps.ane_prefill`); null = none.
    offload: ?*const Offload = null,
    /// A verify whose rows form a draft tree (`Caps.spec_tree`); null `table` = a chain.
    tree: Tree = .{},
    /// A verify hands the residual built so far to the GPU every this many layers (0 = never), so the GPU runs while
    /// the rest of the graph is built.
    pipeline_every: u32 = 0,
    /// An additive attention mask `[1, 1, rows, rows]` over a prompt forwarded whole from position 0 (causal itself,
    /// finite: a left-padded prompt's pads masked), summed into every layer's scores (`Caps.prefill_mask`); empty =
    /// none.
    mask_add: Array = .{},
    /// A native drafter's round (`NativeDrafter.step`): the most drafts it may keep, the tokens still allowed less one.
    accept_cap: u32 = 0,
    /// The sampler's suppress mask `[vocab]` (true = never drawn), which a round's greedy accept applies; empty = none.
    suppress_mask: Array = .{},
};

/// The engine's calls a forward makes at a layer's projections: each returns true with new references in its outputs
/// when it computed them from input `x` (`[1, rows, hidden]`), false to leave them to the model.
pub const Offload = extern struct {
    size: u32 = @sizeOf(Offload),
    ctx: *anyopaque,
    /// The dense MLP's output.
    mlp: *const fn (ctx: *anyopaque, layer: u32, x: Array, out: *Array) callconv(.c) bool,
    /// A GatedDeltaNet layer's joined q|k|v and z projections.
    gdn_proj: *const fn (ctx: *anyopaque, layer: u32, x: Array, qkv: *Array, z: *Array) callconv(.c) bool,
};

/// Draft-tree verify rows: row w sits at its slot's position + `depth[w]`, its keys and recurrence follow its
/// ancestors. A recurrent layer leaves the round's input state and per-row prework (`StateSlot.state_in`..) for the
/// engine to replay the accepted path from.
pub const Tree = extern struct {
    /// `[6 W]` int32: each row's parent (-1 for row 0), then its conv-input sources (`gdn_decode.treeTable`).
    table: Array = .{},
    /// `[W]` int32 depths, and `[W, max_depth + 1]` int32: each row's ancestor at every depth.
    depth: Array = .{},
    path: Array = .{},
    max_depth: i32 = 0,
};

/// What a `defer_host` forward left owed (`ForwardOut.owed`), settled by `forward_settle`.
pub const Owed = enum(u8) {
    none,
    /// A host-filled input of this forward's graph: settled before anything of the forward is evaluated.
    input,
    /// Host bookkeeping off the forward's own tokens only: the engine may settle it later (with the tokens it read).
    history,
};

pub const Settle = enum(u8) { flush, discard };

// ── Cache access ──

/// One layer's keys and values as attention reads them: dense `[B, H, T, D]`, owned by the view only when `owned`
/// (`CacheOps.release` frees them), plus the stored quantized triples (borrowed) when the cache keeps them.
pub const KvView = extern struct {
    size: u32 = @sizeOf(KvView),
    k: Array = .{},
    v: Array = .{},
    owned: bool = false,
    /// The request reads the triples directly where a packed attention kernel serves the shape.
    packed_reads: bool = false,
    /// 0 = dense storage, else the quantized width of the triples below.
    bits: u8 = 0,
    group_size: u32 = 0,
    /// Axis-2 capacity of the triples' buffers (the packed kernels stride by it); 0 = unknown.
    capacity: u32 = 0,
    k_q: Array = .{},
    k_scales: Array = .{},
    k_biases: Array = .{},
    v_q: Array = .{},
    v_scales: Array = .{},
    v_biases: Array = .{},
};

/// A recurrent layer's states: the convolution window and the scan state, and what a verify forward leaves for the
/// partial-accept rollback: the conv input behind the window (`conv_input` [B, K-1+S, C]) and the state after every
/// row (`state_seq` [S, B, Hv, Dv, Dk]); a tree verify leaves the round's input state (`state_in`) and its rows'
/// prework (`prework_k`, `prework_v`, `prework_gb`) in place of `state_seq`. Writing a null array clears a slot.
pub const StateSlot = enum(u8) { conv, ssm, conv_input, state_seq, state_in, prework_k, prework_v, prework_gb };

/// The engine's calls on a slot's cache (`Handle`), handed in `Forward.cache_ops`: blocks append and read KV and
/// recurrent state through it and never see the engine's types. Every `Array` passed in stays the caller's.
pub const CacheOps = extern struct {
    size: u32 = @sizeOf(CacheOps),
    /// Rows `layer` holds, counted from position 0 (a ring counts the rows it dropped); 0 before its first append.
    len: *const fn (cache: Handle, layer: u32) callconv(.c) u64,
    /// Appends `k`/`v` `[1, H, T, D]` at `layer`, then writes the view attention reads: the newest `keep` rows, 0 = all.
    append: *const fn (cache: Handle, layer: u32, k: Array, v: Array, keep: u32, s: Stream, out: *KvView) callconv(.c) Verdict,
    /// `append` on a sliding layer whose buffer holds only its newest `ring` rows (`ring >= window`).
    append_ring: *const fn (cache: Handle, layer: u32, k: Array, v: Array, keep: u32, window: u32, ring: u32, s: Stream, out: *KvView) callconv(.c) Verdict,
    /// The rows `layer` already holds (a layer that reads another layer's KV).
    view: *const fn (cache: Handle, layer: u32, s: Stream, out: *KvView) callconv(.c) Verdict,
    /// Frees what `out` of `append`/`view` owns.
    release: *const fn (view: *KvView) callconv(.c) void,
    /// Appends row `i` of the stacked `k`/`v` `[N, H, T, D]` to `caches[i]` (batched rows).
    append_rows: *const fn (caches: [*]const Handle, n: u32, layer: u32, k: Array, v: Array, keep: u32, s: Stream) callconv(.c) Verdict,
    /// The state, borrowed; a null array before its first write.
    state: *const fn (cache: Handle, layer: u32, slot: StateSlot) callconv(.c) Array,
    /// Replaces the state with a new reference on `value`.
    set_state: *const fn (cache: Handle, layer: u32, slot: StateSlot, value: Array) callconv(.c) Verdict,
    /// Layer `layer`'s `slot` over `caches[0..n]`, stacked on the batch axis (axis 1 for `.state_seq`): a new
    /// reference the caller frees; a null array when any cache holds none yet. An engine that keeps a batch's
    /// states merged across ticks hands its group's array, with no concat.
    state_rows: *const fn (caches: [*]const Handle, n: u32, layer: u32, slot: StateSlot, s: Stream) callconv(.c) Array,
    /// Replaces layer `layer`'s `slot` of `caches[0..n]` with the rows of `value` (the engine takes its own
    /// reference). A merged group keeps `value` whole and splits a row out only when that cache is read alone or
    /// leaves the group.
    set_state_rows: *const fn (caches: [*]const Handle, n: u32, layer: u32, slot: StateSlot, value: Array, s: Stream) callconv(.c) Verdict,
    /// Each handle is an SDK `cache_ops.Slot` (`sdk/kv/cache_ops.zig`), which an in-tree model may read directly. A
    /// flag, not the table's address: every compile unit holds its own copy of the table.
    sdk_slots: bool = false,

    pub fn sdkSlots(self: *const CacheOps) bool {
        return self.size >= @sizeOf(CacheOps) and self.sdk_slots;
    }
};

/// The forward's results: every array is a new reference the host owns. `logits` is [R, vocab] with R the number
/// of slots (`.last_row`) or of tokens (`.all_rows`).
pub const max_round_tokens = 32;

/// What a native drafter's round committed.
pub const Round = extern struct {
    /// Tokens committed, `t1` first: `tokens[0..n]`.
    n: u32 = 0,
    /// Drafts kept of those verified (`n - 1`).
    accepted: u32 = 0,
    /// The trunk's token after them, the next round's `t1`.
    next_token: u32 = 0,
    /// Verify rows run, `t1` included, for the engine's stats.
    rows: u32 = 0,
    /// The model's position after the round.
    position: u64 = 0,
    tokens: [max_round_tokens]u32 = @splat(0),
};

pub const ForwardOut = extern struct {
    size: u32 = @sizeOf(ForwardOut),
    logits: Array = .{},
    hidden: Array = .{},
    /// Host-allocated, `capture_layers.len` entries.
    captured: ?[*]Array = null,
    /// What each slot of a `defer_host` forward left owed (one value for every slot of the forward).
    owed: Owed = .none,
    /// `NativeDrafter.step`'s result.
    round: Round = .{},
};

// ── Hooks ──

pub const LayerKind = enum(u8) { none, full, sliding, linear, conv, ssm };

pub const LayerCache = extern struct {
    size: u32 = @sizeOf(LayerCache),
    kind: LayerKind = .full,
    /// No rotary position on this layer.
    nope: bool = false,
    /// The layer whose KV this one reads; -1 = its own.
    kv_source: i16 = -1,
    window: u32 = 0,
    kv_heads: u32 = 0,
    k_dim: u32 = 0,
    v_dim: u32 = 0,
};

/// Snapshot metadata; `ids` is borrowed until the snapshot's `free`, and `n == ids.len`.
pub const AuxSnapshotInfo = extern struct {
    size: u32 = @sizeOf(AuxSnapshotInfo),
    n: u64 = 0,
    ids: Span(u32) = .{},
    prefix_key: u64 = 0,
};

/// A plugin's opaque per-slot state beside the KV cache, under the RAM prefix cache, the SSD tier and restores.
/// Every call runs on the inference thread, except `free` at process teardown, which runs on the thread tearing the
/// model down after the inference thread has exited. A snapshot's caller owns it until `free`.
pub const AuxHooks = extern struct {
    size: u32 = @sizeOf(AuxHooks),
    reserve: *const fn (m: *anyopaque, slot: u32, tokens: u64) callconv(.c) Verdict,
    commit: *const fn (m: *anyopaque, slot: u32, len: u64) callconv(.c) Verdict,
    /// Transfers the slot's completed capture to the caller; null means no checkpoint was reached.
    snapshot: *const fn (m: *anyopaque, slot: u32, out: *Handle) callconv(.c) Verdict,
    /// Accepts only `pos == snap.n`; builds the replacement before installing it. The caller keeps the snapshot.
    restore_at: *const fn (m: *anyopaque, slot: u32, snap: Handle, pos: u64) callconv(.c) Verdict,
    /// Clones references when `len == snap.n`; any other length returns named `.unsupported`.
    slice: *const fn (m: *anyopaque, snap: Handle, len: u64, out: *Handle) callconv(.c) Verdict,
    handoff: *const fn (m: *anyopaque, from_slot: u32, to_slot: u32) callconv(.c) Verdict,
    /// Releases the complete snapshot and its host allocations exactly once.
    free: *const fn (m: *anyopaque, snap: Handle) callconv(.c) void,
    /// Resident bytes, including retained buffer capacities; encoded size is a separate quantity.
    bytes: *const fn (m: *anyopaque, snap: Handle) callconv(.c) u64,
    /// Writes the encoding into `buf` up to `cap`, setting `len`; refuses insufficient capacity by name.
    serialize: *const fn (m: *anyopaque, snap: Handle, buf: [*]u8, cap: usize, len: *usize) callconv(.c) Verdict,
    deserialize: *const fn (m: *anyopaque, bytes: Str, out: *Handle) callconv(.c) Verdict,
    /// Whether this loaded instance supports auxiliary caching, including its runtime configuration.
    enabled: *const fn (m: *anyopaque) callconv(.c) bool,
    /// Describes the snapshot's own boundary and identity; `ids` remains borrowed until its `free`.
    describe: *const fn (m: *anyopaque, snap: Handle, out: *AuxSnapshotInfo) callconv(.c) Verdict,
    /// Computes identity for `ids` and only the media rows belonging to that prefix.
    prefix_key: *const fn (m: *anyopaque, ids: Span(u32), media_rows: Array, out: *u64) callconv(.c) Verdict,
    /// Activates `slot`; zero `restored_at` resets it, otherwise verifies its position. Arms that slot's capture
    /// at `prompt.len - 1` only when `capture` is true; media rows are borrowed for this call.
    begin_prompt: *const fn (m: *anyopaque, slot: u32, prompt: Span(u32), media_rows: Array, restored_at: u64, capture: bool) callconv(.c) Verdict,
};

pub const CacheHooks = extern struct {
    size: u32 = @sizeOf(CacheHooks),
    /// Writes one entry per served layer into `out` (`cap` entries) and returns the count; null = every layer full
    /// attention at the shell's geometry.
    layout: ?*const fn (p: *const Parsed, span: ?*const LayerSpan, out: [*]LayerCache, cap: u32) callconv(.c) u32 = null,
    aux: ?*const AuxHooks = null,
    /// Module-owned conversation state (`Caps.owns_decode_state`): called before slot `slot`'s prefill with the whole
    /// prompt and its media rows (empty = none); returns the prompt tokens the slot's state already holds, from which
    /// the engine prefills and which it reports as cached. 0 = none held: the plugin has reset that state for a cold
    /// prefill.
    resume_prompt: ?*const fn (m: *anyopaque, slot: u32, prompt: Span(u32), media_rows: Array) callconv(.c) u64 = null,
    /// Bytes that module-owned state holds now (the cache budget's resident figures).
    state_bytes: ?*const fn (m: *anyopaque) callconv(.c) u64 = null,
};

/// An MTP head's calls. `forward` takes the head's token rows (`ids` or `tokens`), the trunk's hidden rows in
/// `hidden_in`, the head's own cache in `caches[0]` (through `cache_ops`) at `positions[0]`, and writes the head's
/// normed output for the LAST row (`[1, 1, hidden]`, the vector the trunk's head reads) to `out.hidden`; the engine
/// projects it. `logits = .last_row`: the engine reads that row (leading rows may append K/V only); `.none` with
/// `want_hidden`: every row runs, the last one's output is read; `.none` without it: history only (K/V, no output).
pub const HeadHooks = extern struct {
    size: u32 = @sizeOf(HeadHooks),
    load: *const fn (m: *anyopaque, ctx: *const LoadCtx) callconv(.c) Verdict,
    activate: *const fn (m: *anyopaque, slot: u32) callconv(.c) Verdict,
    release: *const fn (m: *anyopaque, slot: u32) callconv(.c) void,
    forward: *const fn (m: *anyopaque, f: *const Forward, out: *ForwardOut) callconv(.c) Verdict,
    truncate: *const fn (m: *anyopaque, slot: u32, len: u64) callconv(.c) void,
    adopt: *const fn (m: *anyopaque, slot: u32, snap: Handle) callconv(.c) Verdict,
    mark: *const fn (m: *anyopaque, slot: u32) callconv(.c) void,
    draft_select: *const fn (m: *anyopaque, logits: Array, out: *Array) callconv(.c) Verdict,
    shortlist: ?*const fn (m: *anyopaque, hidden: Array, k: u32, out: *Array) callconv(.c) Verdict = null,
    rerank: ?*const fn (m: *anyopaque, hidden: Array, shortlist: Array, out: *Array) callconv(.c) Verdict = null,
    bill: *const fn (p: *const Parsed) callconv(.c) u64,
    /// One draft step for several requests as the rows of one head forward: row k of `ids` `[n, 1]` and of
    /// `hidden_in` `[1, n, hidden]` runs at `positions[k]` over `caches[k]`; `out.hidden` = the normed rows
    /// `[1, n, hidden]`. Null = the engine drafts each request alone.
    forward_lanes: ?*const fn (m: *anyopaque, f: *const Forward, out: *ForwardOut) callconv(.c) Verdict = null,
    /// The head's half of the engine's measured cost fingerprint: `bits << 16 | group` when the loaded head is the
    /// measured dense layout (bf16 fc `[2H, H]` and norms, every projection affine at that one width); 0 = none.
    cost_fingerprint: ?*const fn (m: *anyopaque) callconv(.c) u32 = null,
};

/// A drafter the model carries in its own forward (DSpark); stages and block are `SpecFacts`.
pub const NativeDrafter = extern struct {
    size: u32 = @sizeOf(NativeDrafter),
    /// One greedy round on the model's own state: `f.tokens` is the committed token `t1` (not yet in the state, which
    /// stands at `f.positions[0]`), `f.accept_cap` the most drafts to keep. The model drafts, verifies, keeps the
    /// drafts its own argmax agrees with (under `f.suppress_mask`), rewinds its state to them and writes `out.round`.
    step: *const fn (m: *anyopaque, f: *const Forward, out: *ForwardOut) callconv(.c) Verdict,
    /// The first half of a round the engine accepts itself (a sampled request): the model drafts from `t1` and verifies
    /// the block, leaving the verify open. A `verify` while one is open discards the open one. `out.round.tokens[0..n]` are the verified rows' ids (`t1` first) and
    /// `out.logits` their logits `[n, vocab]` (f32). Needs `finish`.
    verify: ?*const fn (m: *anyopaque, f: *const Forward, out: *ForwardOut) callconv(.c) Verdict = null,
    /// The second half: keeps `t1` and the first `accepted` drafts of the open verify, rewinds the state to them and
    /// writes the model's position into `out.round.position`.
    finish: ?*const fn (m: *anyopaque, accepted: u32, out: *ForwardOut) callconv(.c) Verdict = null,
};

/// A trunk tensor a drafter reads beside the trunk's forward.
pub const TrunkWeight = enum(u8) { embed, head };

pub const SpecHooks = extern struct {
    size: u32 = @sizeOf(SpecHooks),
    head: ?*const HeadHooks = null,
    /// The trunk's embedding table or LM head as loaded (borrowed until `unload`; a tied head is the table), for
    /// drafters that read them (DFlash, the Gemma assistant, MTP heads). Null = the model serves no drafter.
    trunk_weight: ?*const fn (m: *anyopaque, which: TrunkWeight, out: *quant.Weight) callconv(.c) Verdict = null,
    native_drafter: ?*const NativeDrafter = null,
    /// Writes the verify widths to warm into `out` (`cap` entries); returns the count.
    warmup_shapes: ?*const fn (p: *const Parsed, out: [*]u32, cap: u32) callconv(.c) u32 = null,
    /// Once a drafter binds, re-orders the trunk's projections in place for row-exact lane `lane` (the bases'
    /// `row_exact.Lane`), the GPU synchronized; the model releases them at `unload`.
    tile_lanes: ?*const fn (m: *anyopaque, lane: u32) callconv(.c) Verdict = null,
};

pub const MediaKind = enum(u8) { image, video, audio };

/// One prepared media item: its pixels (or samples) as a new reference the host owns, and its row plan.
pub const MediaItem = extern struct {
    size: u32 = @sizeOf(MediaItem),
    kind: MediaKind = .image,
    /// Rows the item adds to the prompt.
    rows: u32 = 0,
    /// Grid in patches: time, height, width.
    grid: [3]u32 = .{ 0, 0, 0 },
    /// Encoder calls the item takes (crops plus a thumbnail); 1 = a single pass.
    tiles: u32 = 1,
    pixels: Array = .{},
    /// The pixels as host memory instead of `pixels`: float32 little-endian, `grid[1] * grid[2]` rows of equal width.
    /// Plugin-owned until the host has copied it and called `free`. `prepare` runs on a connection thread, which never
    /// calls MLX, so an outside plugin returns its pixels this way.
    bytes: Span(u8) = .{},
    free: ?*const fn (item: *MediaItem) callconv(.c) void = null,
};

pub const MediaHooks = extern struct {
    size: u32 = @sizeOf(MediaHooks),
    /// The tower forward: the item's rows.
    tower: *const fn (m: *anyopaque, item: *const MediaItem, out: *Array) callconv(.c) Verdict,
    /// Decodes and resizes one item's `bytes` within the pixel `budget`.
    prepare: *const fn (p: *const Parsed, kind: MediaKind, bytes: Str, budget: u64, out: *MediaItem) callconv(.c) Verdict,
    /// Writes the item's placeholder token run into `out` (`cap` entries); returns its length.
    placeholder_rows: ?*const fn (p: *const Parsed, item: *const MediaItem, out: [*]u32, cap: u32) callconv(.c) u32 = null,
    /// The M-RoPE table over the expanded prompt `ids`.
    positions: ?*const fn (p: *const Parsed, ids: Span(u32), items: Span(MediaItem), out: *Array) callconv(.c) Verdict = null,
    video_prune: ?*const fn (m: *anyopaque, item: *const MediaItem, rows: Array, out: *Array) callconv(.c) Verdict = null,
};

pub const EmbedHooks = extern struct {
    size: u32 = @sizeOf(EmbedHooks),
    /// The projection head over pooled rows; pooling itself is `EmbedFacts.pooling`.
    project: ?*const fn (m: *anyopaque, pooled: Array, out: *Array) callconv(.c) Verdict = null,
};

pub const DistHooks = extern struct {
    size: u32 = @sizeOf(DistHooks),
    /// Splits the model across `hosts` machines: one span per host into `out`; returns the count.
    plan: *const fn (p: *const Parsed, hosts: u32, out: [*]LayerSpan, cap: u32) callconv(.c) u32,
    /// This machine's part of a forward.
    serve: *const fn (m: *anyopaque, f: *const Forward, out: *ForwardOut) callconv(.c) Verdict,
};

/// A layer projection the engine reads beside the forward (`WeightHooks.layer_weight`).
pub const LayerWeight = enum(u8) { mlp_gate, mlp_up, mlp_down, gdn_qkv, gdn_z };

pub const WeightHooks = extern struct {
    size: u32 = @sizeOf(WeightHooks),
    /// After the weights load: folds, narrows, residents.
    transform: ?*const fn (m: *anyopaque) callconv(.c) Verdict = null,
    /// Whether the loader keeps tensor `key`; null = all.
    keep_key: ?*const fn (p: *const Parsed, key: Str) callconv(.c) bool = null,
    /// Layer `layer`'s projection `which` as loaded (borrowed until `unload`); refused where the layer has none.
    layer_weight: ?*const fn (m: *anyopaque, layer: u32, which: LayerWeight, out: *quant.Weight) callconv(.c) Verdict = null,
    /// The bytes a load of the pack at `model_dir` holds resident, from `shards` (the safetensors bytes its index
    /// names) and the load's `settings_json` (`LoadCtx.settings_json`); null = `shards`.
    resident_bytes: ?*const fn (p: *const Parsed, model_dir: Str, settings_json: Str, shards: u64, out: *u64) callconv(.c) Verdict = null,
};

// ── The table ──

/// A model plugin's table, exported as `exportName(name, version)`. `claims` and `parse` never reach MLX.
pub const ModelTable = extern struct {
    size: u32 = @sizeOf(ModelTable),
    version: u32 = version,
    name: Str,
    claims: *const fn (model_type: Str, config_json: Str, dir: *const PackDir) callconv(.c) Claim,
    /// `stub`: discovery's parse (no sidecar beyond what the shell needs). A refusal leaves nothing to free.
    parse: *const fn (dir: *const PackDir, config_json: Str, settings_json: Str, span: ?*const LayerSpan, stub: bool, out: *Parsed) callconv(.c) Verdict,
    /// Frees what `parse` allocated (`Parsed.private` and the text it points at).
    free_parsed: *const fn (p: *Parsed) callconv(.c) void,
    bill: *const fn (p: *const Parsed, shape: *const Shape, out: *Bill) callconv(.c) void,
    load: *const fn (ctx: *const LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque,
    unload: *const fn (m: *anyopaque) callconv(.c) void,
    /// A mode the model does not declare is refused `unsupported`.
    forward: *const fn (m: *anyopaque, f: *const Forward, out: *ForwardOut) callconv(.c) Verdict,
    /// Settles what `slots` were left owed: `.flush` reads the tokens off the device once for every slot, or takes
    /// `tokens` (one slot's rows, already read); `.discard` drops it (the forward is abandoned).
    /// `tokens` by pointer: a trailing by-value span past the argument registers miscompiles on x86_64 (LLVM byval).
    forward_settle: ?*const fn (m: *anyopaque, slots: Span(u32), how: Settle, tokens: *const Span(u32)) callconv(.c) Verdict = null,
    /// A slot id the engine handed in `Forward.slots` is gone: per-slot state the plugin keeps for it is freed.
    release_slot: ?*const fn (m: *anyopaque, slot: u32) callconv(.c) void = null,
    /// Engine code is about to drive the model's module directly for `slot` (a native draft round,
    /// `Caps.slot_state`): its state becomes the module's current one.
    activate_slot: ?*const fn (m: *anyopaque, slot: u32) callconv(.c) Verdict = null,
    cache: *const CacheHooks,
    spec: ?*const SpecHooks = null,
    media: ?*const MediaHooks = null,
    embed: ?*const EmbedHooks = null,
    dist: ?*const DistHooks = null,
    weights: *const WeightHooks,
    /// `slot`'s arm counters for its request's summary lines (each a full log line), in memory the plugin owns until
    /// the next call on `m`; the counters reset.
    meters: ?*const fn (m: *anyopaque, slot: u32, out: *Str) callconv(.c) void = null,
};

/// The modes a model serves: prefill, decode and verify always; the rest as its caps declare. An encoder (`embed`)
/// serves prefill only, its result the hidden rows.
pub fn declaresMode(caps: Caps, mode: Mode) bool {
    if (caps.embed) return mode == .prefill;
    return switch (mode) {
        .prefill, .decode, .verify => true,
        .batched_rows => caps.batches_decode,
    };
}

// ── tests ──

const testing = std.testing;

test "abi plugin: caps fit one word with the generic defaults" {
    try testing.expectEqual(@as(usize, 8), @sizeOf(Caps));
    const c: Caps = .{};
    try testing.expect(c.prefix_cache and c.batch_per_slot_attn and c.batch_pads_kv and !c.batches_decode);
}

test "abi plugin: every struct opens with its size, set by default" {
    inline for (.{ LayerSpan, Quant, GenDefaults, Shell, NoThink, SamplingDefaults, ChatFacts, MarkerAlias, TokenizerFacts, SsmState, CacheFacts, SpecPolicy, SpecFacts, PlaceholderIds, EmbedFacts, WeightFacts, MropeFacts, Parsed, Shape, Bill, ForwardOut, KvView, LayerCache, CacheHooks, SpecHooks, EmbedHooks, WeightHooks, MediaItem }) |T| {
        try testing.expectEqual(@as(usize, 0), @offsetOf(T, "size"));
        const v: T = .{};
        try testing.expectEqual(@as(u32, @sizeOf(T)), v.size);
    }
    inline for (.{ LoadCtx, Forward, Offload, CacheOps, AuxHooks, HeadHooks, NativeDrafter, MediaHooks, DistHooks, ModelTable }) |T|
        try testing.expectEqual(@as(usize, 0), @offsetOf(T, "size"));
}

test "abi plugin: auxiliary callbacks are required function pointers" {
    const info = @typeInfo(AuxHooks).@"struct";
    inline for (info.field_names, info.field_types, info.field_attrs) |name, T, attrs| {
        if (comptime std.mem.eql(u8, name, "size")) continue;
        try testing.expect(@typeInfo(T) == .pointer);
        try testing.expect(attrs.defaultValue(T) == null);
    }
}

test "abi plugin: unset generation defaults read as unset" {
    const g: GenDefaults = .{};
    try testing.expect(std.math.isNan(g.temperature) and std.math.isNan(g.top_p) and g.top_k == -1);
}

test "abi plugin: prefill, decode and verify are always declared; batched rows follow the caps" {
    try testing.expect(declaresMode(.{}, .verify) and !declaresMode(.{}, .batched_rows));
    try testing.expect(declaresMode(.{ .batches_decode = true }, .batched_rows));
    try testing.expect(declaresMode(.{ .embed = true }, .prefill) and !declaresMode(.{ .embed = true }, .decode));
}
