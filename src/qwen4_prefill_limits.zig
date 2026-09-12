//! An 8192-token chunk can absorb a final remainder of up to 511 tokens.
//! Keep specializations active for that real forward width. A generate.zig
//! regression ties this bound to nextChunkEnd and TAIL_MERGE_MAX.
pub const tail_slack: c_int = 511;
pub const max_seq: c_int = 8192 + tail_slack;
