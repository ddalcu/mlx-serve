/* FFI layout cross-check for src/ds4_ffi.zig, compiled against the REAL
 * lib/ds4/ds4.h. Upstream inserts fields mid-struct on upgrades; a stale Zig
 * mirror silently corrupts ds4_engine_options at open time (generation hangs
 * with no error). The "EngineOptions layout" test in ds4_ffi.zig compares
 * these against @sizeOf/@offsetOf so drift fails `zig build test` instead. */
#include <stddef.h>
#include "../lib/ds4/ds4.h"

size_t mlxserve_ds4_sizeof_engine_options(void) {
    return sizeof(ds4_engine_options);
}
size_t mlxserve_ds4_offsetof_mtp_draft_tokens(void) {
    return offsetof(ds4_engine_options, mtp_draft_tokens);
}
size_t mlxserve_ds4_offsetof_ssd_streaming(void) {
    return offsetof(ds4_engine_options, ssd_streaming);
}
size_t mlxserve_ds4_offsetof_placement_session_count_hint(void) {
    return offsetof(ds4_engine_options, placement_session_count_hint);
}
size_t mlxserve_ds4_offsetof_distributed(void) {
    return offsetof(ds4_engine_options, distributed);
}
size_t mlxserve_ds4_sizeof_distributed_options(void) {
    return sizeof(ds4_distributed_options);
}
size_t mlxserve_ds4_sizeof_tokens(void) {
    return sizeof(ds4_tokens);
}
size_t mlxserve_ds4_sizeof_context_memory(void) {
    return sizeof(ds4_context_memory);
}
size_t mlxserve_ds4_sizeof_session_snapshot(void) {
    return sizeof(ds4_session_snapshot);
}
