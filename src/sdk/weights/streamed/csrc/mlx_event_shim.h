/* Event-gated pass-through primitives over MLX (mlx_event_shim.cpp: the DSV4.1 program's q3ev4 extension as a C
 * ABI for mlx-c callers). A wait's outputs ALIAS its inputs (no copy, no kernel); every consumer of an output
 * depends on it, so MLX encodes it after its inputs and before every consumer. On a GPU stream it ends the open
 * compute pass and encodes MTLCommandBuffer.encodeWaitForEvent:value: into the stream's current command buffer, so
 * the GPU waits for the pool to hand the event the value (lookahead pool, q3ld_ev_config). On a CPU stream a HOST
 * event (an int64 word the pool release-stores) makes the evaluating thread wait before the consumers are
 * dispatched. Built against MLX 0.32.2's metal backend headers: re-verify on every MLX bump. */
#ifndef MLX_SERVE_MLX_EVENT_SHIM_H
#define MLX_SERVE_MLX_EVENT_SHIM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mlx/c/array.h"
#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DSV41EV_ABI 2026092801

int32_t dsv41ev_abi(void);
/* An MTLSharedEvent on MLX's GPU device, signaled value = start, kept for the process life. Returns its id
 * (>= 1) and writes the id<MTLSharedEvent> (the pool's EVK_MTL object) to *object; 0 on failure. */
int32_t dsv41ev_create_metal(uint64_t start, uint64_t *object);
/* A host event over an 8-aligned int64 word; a CPU-stream wait gives up (counted) after timeout_ns. 0 on failure. */
int32_t dsv41ev_create_host(int64_t *word, int64_t timeout_ns);
/* outs[i] (mlx_array_new() handles) alias xs[i], n >= 1, value >= 1; deps are ordering-only inputs; track_inputs
 * orders the post-wait GPU pass after the producers of xs (only for xs produced by in-flight GPU work). A host
 * event on a GPU stream is refused. 0, or -1 (dsv41ev_last_error). */
int dsv41ev_wait(const mlx_array *xs, size_t n, int32_t event, uint64_t value, const mlx_array *deps, size_t n_deps,
                 bool track_inputs, mlx_stream s, mlx_array *outs);
/* outs alias xs; the GPU hands `value` to a metal event after every pass encoded before it (probes). Metal events
 * only. 0 or -1. */
int dsv41ev_signal(const mlx_array *xs, size_t n, int32_t event, uint64_t value, mlx_stream s, mlx_array *outs);
/* The event's current value (0 for an unknown id). */
uint64_t dsv41ev_value(int32_t event);
/* gpu waits encoded, gpu signals encoded, host waits already satisfied, host waits that blocked, host wait
 * timeouts, host wait ns, cpu pass-throughs, events created. */
void dsv41ev_stats(int64_t out[8]);
/* The calling thread's last error message. */
const char *dsv41ev_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
