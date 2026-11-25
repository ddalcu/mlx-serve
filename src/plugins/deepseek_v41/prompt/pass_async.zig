//! MLX_SERVE_DSV41_PROMPT_ASYNC_EVAL (on by default, =0 off; exact): a prompt pass's last wide layer is committed, not waited
//! for, as every wide layer before it already is (`runLayers`). With the wait the GPU sits idle from the pass's end
//! while the host builds the next graph (the CED replay and the head, or the lead's compressor) up to its first eval;
//! without it that build runs under the pass's last command buffers. MLX keeps the host within 10 committed command
//! buffers of the GPU (`MAX_ACTIVE_TASKS`), so the overlap is that build, not the next graph's run. The same arrays in
//! the same order, so the same words. A streamed bank (its slot rows rewritten from one call to the next), a pair and a
//! tensor split keep the wait.

const env = @import("../../../env.zig");
const lev = @import("../levers.zig");
const log = @import("../../../sdk/util/log.zig");

/// The passes that left their last wide layer unwaited, one count across `-Dunits` units (pass_async_test.zig reads it).
pub const shared_state = @import("../../../sdk/util/shared.zig").Shared(usize, "dsv41_pass_async", 0);

/// Whether a wide pass over `rows` rows leaves its last layer unwaited; each such pass is counted and logged.
pub fn on(rows: usize) bool {
    if (!lev.onUnlessZero(.PROMPT_ASYNC_EVAL)) return false;
    shared_state.get().* += 1;
    log.info(lev.line(.PROMPT_ASYNC_EVAL) ++ ": a {d}-row pass's last wide layer goes to the GPU without the host's wait\n", .{rows});
    return true;
}
