//! GLM-5.3-Flash's streamed-expert routes at their defaults (the stack of record): the read pool's staging, the decode
//! lane's route and predictor, the prompt residents, the EXL3 decode forms, the demand reads and the calibrated split of
//! each routed layer's slot rows.

const std = @import("std");
const sdk = @import("sdk");
const expert_usage = @import("../sdk_ext.zig").expert.usage;
const exl3_forms = @import("exl3_forms.zig");

/// The most model layers a split names.
const max_layers = 64;

pub const default_tally_ratio: f64 = 0;
/// The reader's staging per worker: Flash's gate/up span (9,437,192 B) plus its page alignment and a page (bank.zig).
pub const reader_staging_bytes: u64 = 10 << 20;

pub const DecodeMoe = enum { stock, fused, hitpass };
pub const DecodePredict = enum { f32, bf16 };
pub const PromptResidents = expert_usage.PromptResidents;

pub const Exl3GateUpForm = enum { seam, guone };
pub const Exl3DownForm = enum { seam, pair };
pub const Exl3Forms = struct {
    gateup: Exl3GateUpForm = .guone,
    down: Exl3DownForm = .pair,

    /// The seam chain, by name (`"exl3_gateup_form":"seam","exl3_down_form":"seam"`): the control.
    pub const seam_chain: Exl3Forms = .{ .gateup = .seam, .down = .seam };

    /// The forms exl3_forms.zig installs for these settings (every name maps).
    pub fn forms(f: Exl3Forms) exl3_forms.Forms {
        return .{
            .gu = switch (f.gateup) {
                .seam => .seam,
                .guone => .guone,
            },
            .dn = switch (f.down) {
                .seam => .seam,
                .pair => .pair,
            },
        };
    }

    /// The seam chain itself (`seam_chain`): Math binds lane B's kernels, no forms install.
    pub fn isSeam(f: Exl3Forms) bool {
        return std.meta.eql(f, seam_chain);
    }
};

/// The demand reads: staged through the read pool, or `direct` into the slot rows (the EXL3 bank's default).
pub const MissReads = enum { staged, direct };

/// `expert_rows_by_layer` (MiMo's mimo/layer-rows, mimo_v2/settings.zig `LayerRows`): the plan's decode rows split
/// between the routed layers at the SAME total, once at construction. `uniform` (the default) is every layer at the plan's
/// rows, today's construction exactly. `offsets` is one integer per routed layer (routed layer 0 first, model layer
/// `first_k_dense`), summing to 0, added to the plan's DECODE rows; each layer's prompt rows are min(the plan's prompt rows,
/// its decode rows), so prompt <= decode on every layer (the stream's grow adds rows, never moves them). The bill is
/// unchanged: its one row term is linear in the rows (bill.zig `slotWiring`), the decode total is the plan's, and the
/// prompt total can only fall. No table ships: a table comes from a calibration trace (the per-layer miss rates of
/// non-benchmark prompts at uniform rows; the replay in reports/.../mimo/glm5next-per-layer-rows-20261012.md weights
/// rows by sqrt(miss rate)), never from the benchmark prompt's own trace. Setting: `"uniform"` or an array of integers
/// (each within i16) summing to 0, of length the model's routed layers; anything else refuses by name in `validate`;
/// a layer over its experts is held at them, the excess moved to the narrowest layers; plan rows over the experts refuse
/// at construction (`layerRows`).
pub const LayerRows = union(enum) {
    uniform,
    offsets: Offsets,
    /// Not `"uniform"` nor an array of integers within i16 (`Glm5NextExpertRowsByLayer`), an array longer than
    /// `max_layers` or of another length than the model's routed layers (`Glm5NextExpertRowsByLayerLength`), or one
    /// whose entries do not sum to 0 (`Glm5NextExpertRowsByLayerSum`).
    invalid: enum { setting, length, sum },

    pub const Offsets = struct {
        len: u8,
        d: [max_layers]i16 = @splat(0),

        pub fn slice(o: *const Offsets) []const i16 {
            return o.d[0..o.len];
        }
    };
};

/// glm5_next's default decode split (`expert_rows_by_layer`): 42 integer offsets summing to 0, the calibration of the
/// three NON-benchmark prompts (reports/.../mimo/cells/glm5next-calib-20261013/expert-rows-by-layer.json). The
/// g21d-vs-g21d-lr pair (lane on, 115/131): decode +3.40 % (noisy), stacked.
pub const default_layer_offsets = [_]i16{ 63, 61, 62, 58, 34, 40, 35, 42, 22, 6, 2, 2, 0, 0, -14, -30, -32, -26, -26, -29, -11, -12, -24, -21, -22, -13, -18, -18, -10, -5, -22, -18, -13, -11, -12, -9, -10, -7, -7, -9, -6, 8 };

pub fn defaultLayerRows() LayerRows {
    var o: LayerRows.Offsets = .{ .len = default_layer_offsets.len };
    @memcpy(o.d[0..default_layer_offsets.len], &default_layer_offsets);
    return .{ .offsets = o };
}

comptime {
    var sum: i32 = 0;
    for (default_layer_offsets) |x| sum += x;
    if (sum != 0 or default_layer_offsets.len != 42) @compileError("glm5_next: the default per-layer offsets must be 42 summing to 0");
}

/// The per-layer rows of both phases (routed-layer order) into `prompt` / `decode` (each of the model's routed layers'
/// length), from the plan's uniform `rows`, once at construction. `uniform`: every layer at `rows` (today's construction).
/// `offsets` (lr2): the prompt rows stay uniform at `rows.prompt` in every layer (equal prompt bytes; the split applies at
/// the phase change); decode = rows.decode + offset, held at rows.prompt at least (the stream grows a layer at the phase
/// change, never shrinks it), and the rows that floor adds are taken back from the widest decode layers (the first widest
/// first, one row at a time, never under rows.prompt) so the decode rows sum to the plan's: both phases bill exactly as
/// the uniform rows. A layer is also held at `n_experts` at most (all its experts resident), and the rows that ceiling
/// removes go to the narrowest decode layers (the first narrowest first, one row at a time, never over `n_experts`), so
/// the plan's decode rows can go up to `n_experts` under any offsets (before: n_experts - the largest offset, 225 for the
/// default table). Refused by name when the plan's rows are over `n_experts` or the floor cannot be paid back.
pub fn layerRows(split: LayerRows, rows: sdk.Rows, n_experts: u32, prompt: []u32, decode: []u32) !void {
    if (prompt.len != decode.len) return error.Glm5NextExpertRowsByLayerLength;
    switch (split) {
        .uniform => {
            @memset(prompt, rows.prompt);
            @memset(decode, rows.decode);
        },
        .offsets => |*o| {
            if (o.len != decode.len) return error.Glm5NextExpertRowsByLayerLength;
            if (rows.decode > n_experts or rows.prompt > rows.decode) return error.Glm5NextExpertRowsByLayerRange;
            var surplus: i64 = 0;
            for (prompt, decode, o.slice()) |*p, *q, off| {
                const dd = @as(i64, rows.decode) + off;
                const held = std.math.clamp(dd, @as(i64, rows.prompt), @as(i64, n_experts));
                surplus += held - dd;
                q.* = @intCast(held);
                p.* = rows.prompt;
            }
            while (surplus < 0) : (surplus += 1) {
                var ni: usize = 0;
                for (decode, 0..) |d, i| {
                    if (d < decode[ni]) ni = i;
                }
                if (decode[ni] >= n_experts) return error.Glm5NextExpertRowsByLayerRange;
                decode[ni] += 1;
            }
            while (surplus > 0) : (surplus -= 1) {
                var wi: usize = 0;
                for (decode, 0..) |d, i| {
                    if (d > decode[wi]) wi = i;
                }
                if (decode[wi] <= rows.prompt) return error.Glm5NextExpertRowsByLayerRange;
                decode[wi] -= 1;
            }
        },
        .invalid => return error.Glm5NextExpertRowsByLayer,
    }
}
