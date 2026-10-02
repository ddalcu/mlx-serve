//! gen surface for builds without MLX.

const std = @import("std");

pub const common = @import("gen_common.zig");

pub const Modality = common.Modality;
pub const modalityFromType = common.modalityFromType;
pub const peekModelType = common.peekModelType;
pub const detectModality = common.detectModality;
pub const incompleteMediaDir = common.incompleteMediaDir;
pub const requiredMarkerFor = common.requiredMarkerFor;
pub const media_model_types = common.media_model_types;
pub const GenRoute = common.GenRoute;
pub const AudioBackendKind = common.AudioBackendKind;
pub const audioBackendKindForType = common.audioBackendKindForType;
pub const StubCpuState = common.StubCpuState;
pub const buildStubCpuState = common.buildStubCpuState;
pub const freeStubCpuState = common.freeStubCpuState;
pub const estimateResidentBytes = common.estimateResidentBytes;
pub const stagedPeakBytes = common.stagedPeakBytes;
pub const ltxPeakBytes = common.ltxPeakBytes;
pub const h3DitResidentBytes = common.h3DitResidentBytes;
pub const h3PeakBytes = common.h3PeakBytes;
pub const estimatePeakResidentBytesIn = common.estimatePeakResidentBytesIn;
pub const estimatePeakResidentBytes = common.estimatePeakResidentBytes;
pub const H3_DIT_RESIDENT_PCT = common.H3_DIT_RESIDENT_PCT;
pub const H3_ACTIVATION_BYTES = common.H3_ACTIVATION_BYTES;
pub const MUSIC3_GEN_BUFFER_BYTES = common.MUSIC3_GEN_BUFFER_BYTES;

pub const Error = error{MlxUnavailable};

fn MediaEngine(comptime what: []const u8) type {
    return struct {
        const Self = @This();

        backend: common.AudioBackendKind = .tts,

        pub fn load(_: std.Io, _: std.mem.Allocator, _: []const u8) anyerror!*Self {
            @branchHint(.cold);
            _ = what; // named in the error path's message at the call site
            return Error.MlxUnavailable;
        }

        pub fn deinit(_: *Self) void {}
    };
}

pub const ImageEngine = MediaEngine("image");
pub const AudioEngine = MediaEngine("audio");
pub const VideoEngine = MediaEngine("video");
pub const MeshEngine = MediaEngine("mesh");

pub const estimatePeakResident = common.estimatePeakResidentBytes;

pub fn computeEmbeddingsBatch(_: std.mem.Allocator, _: anytype, _: anytype) anyerror![][]f32 {
    return Error.MlxUnavailable;
}

pub fn handleImage(_: std.mem.Allocator, _: anytype, _: []const u8, _: *ImageEngine) anyerror!void {
    return Error.MlxUnavailable;
}
pub fn handleAudio(_: std.mem.Allocator, _: anytype, _: []const u8, _: *AudioEngine) anyerror!void {
    return Error.MlxUnavailable;
}
pub fn handleMusic(_: std.mem.Allocator, _: anytype, _: []const u8, _: *AudioEngine) anyerror!void {
    return Error.MlxUnavailable;
}
pub fn handleVideo(_: std.Io, _: std.mem.Allocator, _: anytype, _: []const u8, _: *VideoEngine) anyerror!void {
    return Error.MlxUnavailable;
}
pub fn handleMesh(_: std.mem.Allocator, _: anytype, _: []const u8, _: *MeshEngine) anyerror!void {
    return Error.MlxUnavailable;
}

pub const EditFormError = common.EditFormError;
pub const openaiEditFormToJson = common.openaiEditFormToJson;
pub const editFormErrorMessage = common.editFormErrorMessage;

pub const DecisionEngine = struct {
    batch_window_us: u32 = 0,

    pub fn load(_: std.Io, _: std.mem.Allocator, _: []const u8) Error!*DecisionEngine {
        return error.MlxUnavailable;
    }
    pub fn deinit(_: *DecisionEngine) void {}
};

pub const DecisionRequest = struct {
    pub fn deinit(_: *DecisionRequest, _: std.mem.Allocator) void {}
    pub fn count(_: *const DecisionRequest) usize {
        return 0;
    }
};
pub const DecisionJob = struct {
    allocator: std.mem.Allocator,
    conn: *@import("server.zig").Conn,
    req: *const DecisionRequest,
};
pub fn prepareDecisions(_: std.mem.Allocator, _: anytype, _: []const u8, _: *DecisionEngine) Error!?DecisionRequest {
    return error.MlxUnavailable;
}
pub fn handleDecisions(_: *DecisionEngine, _: []const u8, jobs: []const DecisionJob) void {
    const body = "{\"error\":{\"message\":\"decision models require MLX\"}}";
    const head = std.fmt.comptimePrint("HTTP/1.1 501 Not Implemented\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n", .{body.len});
    for (jobs) |job| job.conn.writeAll(head ++ body) catch {};
}
