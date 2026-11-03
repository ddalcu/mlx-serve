/* PROFILE builds only (-Ddsv41-decode-timers=true): every command buffer MLX's GPU queue commits while a tag is set,
 * with its host commit time, Metal's GPUStartTime / GPUEndTime and its completion handler's host time
 * (dsv41_cb_timeline.mm; MLX's current buffer from dsv41_tl_mlx.cpp). Times are mach_absolute_time in ns
 * (clock_gettime_nsec_np(CLOCK_UPTIME_RAW)), the time base Metal's GPU timestamps use. Built against MLX 0.32.2's
 * CommandEncoder (one queue per stream): re-verify on every MLX bump. */
#ifndef MLX_SERVE_DSV41_CB_TIMELINE_H
#define MLX_SERVE_DSV41_CB_TIMELINE_H

#include <stddef.h>
#include <stdint.h>

#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint64_t host_commit;
  uint64_t gpu_start;
  uint64_t gpu_end;
  uint64_t host_done;
  uint64_t tag;
  uint64_t status; /* MTLCommandBufferStatus at completion (4 = completed, 5 = error) */
  uint64_t queue;  /* 1: the queue of the buffer the install saw (MLX's stream), 0: another queue */
} Dsv41tlRow;

/* MLX's current command buffer on stream s (an id<MTLCommandBuffer>), NULL when the stream has no Metal encoder. */
void *dsv41tl_mlx_buffer(mlx_stream s);
/* Hooks -commit on every class of cmdbuf's chain that implements it, and records every tagged commit into
 * rows[0..cap) from index 0 (the row says whether it was cmdbuf's queue). 0, or -1 (no buffer / no rows), -2 (no
 * -commit along the chain), -3 (class table full). */
int dsv41tl_install(void *cmdbuf, Dsv41tlRow *rows, uint32_t cap);
/* The same for a later buffer's chain (its classes not yet hooked): the classes added, or -1, -2, -3. */
int dsv41tl_rehook(void *cmdbuf);
/* Every override entry, the tagged ones, the tagged ones of another queue, the hooked classes. */
void dsv41tl_hook_stats(uint64_t out[4]);
/* The hooked classes' names, comma-separated (NUL-terminated); returns the length. */
size_t dsv41tl_hooked_names(char *out, size_t cap);
/* The host self-test over a fake buffer hierarchy (no Metal): 0 or the first failed check's number. */
int dsv41tl_selftest(Dsv41tlRow *rows, uint32_t cap);
/* Rows record while tag != 0 (the tag is stored in each row). */
void dsv41tl_tag(uint64_t tag);
/* committed (incl. dropped), completed (recorded rows whose handler ran), dropped (past cap). */
void dsv41tl_counts(uint32_t out[3]);
uint64_t dsv41tl_now(void);

#ifdef __cplusplus
}
#endif

#endif
