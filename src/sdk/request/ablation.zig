//! A request's directional ablation (`abi.Ablation`), resolved by the host from a profile the server loaded at launch.

pub const Request = struct {
    /// The profile's unit direction, `[hidden]` f32; owned by the server's profile set, which outlives every request.
    direction: []const f32,
    /// In (0, 1]: a strength of 0 never reaches the engine (the request runs as a standard one).
    strength: f32,
    /// The profile's content hash with the strength folded in (`abi.Ablation.key`).
    key: u64,
};
