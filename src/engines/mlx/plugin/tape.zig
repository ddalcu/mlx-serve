//! The bit-exact tapes live in the SDK (`sdk/testing/tape.zig`).

const tape = @import("../../../sdk/testing/tape.zig");
pub const Tape = tape.Tape;
pub const deviceName = tape.deviceName;
