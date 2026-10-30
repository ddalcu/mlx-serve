//! The streaming packs' glue as the engine sees it: mlx-stream compiled in (`stream_glue.zig`), in the module archs'
//! compile unit under `-Dunits` (`stream_table.zig`), or absent (`stream_stub.zig`, Linux and iOS).

const build_options = @import("build_options");
const impl = if (!build_options.mlx_stream)
    @import("stream_stub.zig")
else if (@import("split.zig").active)
    @import("stream_table.zig")
else
    @import("stream_glue.zig");

pub const built = impl.built;
pub const Model = impl.Model;
pub const SamplingParams = impl.SamplingParams;
pub const setForeignBytes = impl.setForeignBytes;
pub const loadBytes = impl.loadBytes;
pub const contextLength = impl.contextLength;
pub const noteMaxOutput = impl.noteMaxOutput;
pub const open = impl.open;
pub const close = impl.close;
pub const begin = impl.begin;
pub const end = impl.end;
pub const forward = impl.forward;
pub const position = impl.position;
pub const blockSize = impl.blockSize;
pub const laneName = impl.laneName;
pub const arm = impl.arm;
pub const round = impl.round;
pub const laneReceipt = impl.laneReceipt;
