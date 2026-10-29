/* Lookahead read pool (q3_lookahead4_exl3.c + q3_event_shim.mm, copied unchanged). The demand path is the
 * native-issue pool's: pthread workers read each record's gate/up span, then its down span (page-aligned
 * F_NOCACHE preadv into a staging buffer, scatter into slot rows), publish every range's status words, then its
 * ticket in an ordered log, then the sequence (RELEASE). Three construction-time classes ride on it, all keyed
 * by file offset or ticket (never by layer, expert or slot):
 *   speculative  whole records read ahead into their own staging slots, only while demand is idle; a demand
 *                submit claims the record holding its range and copies it out instead of reading;
 *   pre-read     a layer call's certain misses queued as ranges before its plan; the submit binds them;
 *   event gate   per-wave gates over tickets; the satisfied prefix is handed to an MTLSharedEvent (or a host
 *                word) by the publishing worker, and a watchdog forces a gate whose bytes never land.
 * A queued submit (its range jobs) can be withdrawn unread by its first ticket (q3ld_withdraw). The pool never calls
 * MLX. State is static: one pool per process. macOS only (mach time). */
#ifndef MLX_SERVE_Q3_LOOKAHEAD4_H
#define MLX_SERVE_Q3_LOOKAHEAD4_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* res[t * RES_W + k]: 0 status, 1 preadv calls, 2 bytes returned, 3 payload landed, 4 errno,
 * 5 t_start (monotonic ns), 6 t_end, 7 worker index. */
#define Q3LD_RES_W 8
#define Q3LD_MAX_WORKERS 8
#define Q3LD_MAX_COMP 6
#define Q3LD_MAX_ITEMS 8
#define Q3LD_MAX_SPEC 16
#define Q3LD_MAX_SPEC_THREADS 3
#define Q3LD_MAX_PRE 32
#define Q3LD_MAX_GATES 256
#define Q3LD_MAX_GATE_TICKETS 256
#define Q3LD_ST_PENDING (-1)
#define Q3LD_ST_OK 0
#define Q3LD_ST_SHORT 1
#define Q3LD_ST_OSERR 2
#define Q3LD_ST_DEADLINE 3
#define Q3LD_ST_SKIPPED 4
#define Q3LD_ST_WITHDRAWN 5
#define Q3LD_EVK_OFF 0
#define Q3LD_EVK_MTL 1
#define Q3LD_EVK_HOST 2

int q3ld_init(void);
int64_t q3ld_monotonic_ns(void);
/* Before q3ld_start (always, nthreads 0 = no speculative class): nslots page-aligned staging slots of slot_bytes,
 * rec_len = the largest record's payload, chunk = bytes per speculative preadv (page multiple), counters =
 * int64[q3ld_counters_n()] (caller-owned, zeroed, live until q3ld_stop). 0 or -1. */
int q3ld_spec_config(int32_t nthreads, const uint64_t *bufs, int32_t nslots, int64_t slot_bytes, int64_t rec_len,
                     int64_t chunk, int64_t *counters);
int q3ld_spec_streams(int32_t idle_busy);   /* stopped pool, after spec_config: 0 or 1; 0 or -1 */
int q3ld_start(int32_t nw, const uint64_t *staging_ptrs, int64_t sbytes, int64_t psize,
               int64_t *res_arr, int64_t n_tickets, int64_t *log_arr, int64_t n_log, int64_t *gauge_arr);
/* One job of n <= MAX_ITEMS records on tickets first .. first + 2n - 1: GU(0..n-1) then DOWN(0..n-1); claims the
 * speculative records and binds the pre-ranges that hold its ranges. 0, -1, -2 ticket pending, -3 queue full. */
int q3ld_submit(int32_t fd, int64_t file_size, int64_t deadline, int32_t n, int32_t ngu, int32_t ndown,
                const int64_t *offsets, const uint64_t *const *rows, const int64_t *lens, int64_t first);
/* q3ld_submit's jobs with each record's part c read from its own file: fds / sizes / offsets[i * (ngu + ndown) + c].
 * No speculative claim, no pre-range bind. */
int q3ld_submit_scatter(int32_t n, int32_t ngu, int32_t ndown, const int32_t *fds, const int64_t *sizes,
                        const int64_t *offsets, const uint64_t *const *rows, const int64_t *lens, int64_t first);
/* The submit whose first ticket is `first`, while no worker has started it, leaves the queue unread: every range job of
 * it, all or none; its tickets are published with ST_WITHDRAWN (no row written). 1 withdrawn, 0 not queued (a worker
 * started it and it runs as submitted / it is done / no submit starts at `first`), -1 not running / bad ticket, -2
 * refused (a ticket bound to a pre-range or waited on by a live event gate). */
int q3ld_withdraw(int64_t first);
/* A layer call's step: settle every unclaimed record with tag <= cur, then queue n records (bases) with tag
 * cur + 1 (len = their payload bytes). Queued count, -1 off / not running. */
int32_t q3ld_spec_step(int32_t fd, int64_t file_size, int64_t cur, int32_t n, const int64_t *bases);
int32_t q3ld_spec_step_len(int32_t fd, int64_t file_size, int64_t cur, int32_t n, const int64_t *bases, int64_t len);
int32_t q3ld_spec_stepn(int32_t fd, int64_t file_size, int64_t cur, int32_t nh, const int32_t *nval,
                        const int64_t *val, const int32_t *nissue, const int64_t *iss);
int32_t q3ld_spec_state(int64_t *out);      /* 11 words per slot x MAX_SPEC; returns the slot count */
int64_t q3ld_seq(void);                     /* ACQUIRE load of the completion sequence */
int64_t q3ld_wait(int64_t seen, int64_t timeout_ns);
void q3ld_gauge(int64_t *out);              /* 6 words */
int q3ld_quiesce(int64_t timeout_ns);       /* 0 quiescent, 1 timeout */
int q3ld_stop(void);                        /* drain, join, release every live gate; 0, -1 not running */
/* Running pool with the speculative class configured: arm (lens = ngu GU then ndown DOWN lengths, none negative) or
 * disarm (0, 0). All or nothing: -1 leaves the class as it was. */
int q3ld_pre_config(int32_t ngu, int32_t ndown, const int64_t *lens);
/* A layer call's certain misses (record bases, route order) before its plan; tag = this call's settle value. */
int32_t q3ld_pre_read(int32_t fd, int64_t file_size, int64_t tag, int32_t n, const int64_t *bases);
int32_t q3ld_pre_read_lens(int32_t fd, int64_t file_size, int64_t tag, int32_t n, const int64_t *bases,
                           const int64_t *lens);
int32_t q3ld_pre_state(int64_t *out);       /* 5 words per entry x MAX_PRE; returns pre_on */
/* Running pool with counters, no live gate: kind EVK_MTL (obj = id<MTLSharedEvent>) or EVK_HOST (obj = an int64
 * word); timeout 1 ms .. 600 s per gate; start = the event's current value. 0, -1 refused, -2 thread. */
int q3ld_ev_config(int32_t kind, uint64_t obj, int64_t timeout_ns, uint64_t start);
/* n gates in GPU encode order: values strictly increasing above every earlier value; gate i waits for counts[i]
 * tickets. n; -1 off; -2 bad value / ticket (nothing registered); -3 ring full. */
int32_t q3ld_ev_gates(int32_t n, const uint64_t *values, const int32_t *counts, const int64_t *tickets);
int32_t q3ld_ev_release(uint64_t value);    /* force every live gate <= value; forced count, -1 off */
int32_t q3ld_ev_state(int64_t *out);        /* 10 words; returns the live gate count */
int32_t q3ld_abi(void);                     /* 2026102401 */
int32_t q3ld_max_gates(void);
int32_t q3ld_max_gate_tickets(void);
int32_t q3ld_max_pre(void);
int32_t q3ld_spec_idle_busy(void);
int32_t q3ld_counters_n(void);
int32_t q3ld_max_spec(void);
int32_t q3ld_max_h(void);

/* q3_event_shim.mm: [MTLSharedEvent setSignaledValue:] from any thread. */
void q3ld_mtl_signal(void *event, uint64_t value);

#ifdef Q3LD_INJECT
/* Test builds: scripted preadv rules keyed by the syscall's aligned offset, each used once (code 1 EINTR,
 * 2 EIO, 3 zero return, 4 truncate to arg bytes, 5 sleep arg ns), a random per-range delay, a speculative
 * per-chunk delay, and the event / signal logs. */
void q3ld_test_rules(int32_t n, const int64_t *off, const int64_t *code, const int64_t *arg);
void q3ld_test_delay(uint64_t seed, int64_t max_ns);
void q3ld_test_spec_delay(int64_t max_ns);
/* The speculative hold: `starts` >= 0 turns it on with that many records still allowed to start (-1 turns it off);
 * a started unclaimed record's chunk waits at its start while it is on (a claim or the hold's end lets it go).
 * q3ld_test_spec_held waits until n records wait there: 0, or 1 at the timeout. */
void q3ld_test_spec_hold(int32_t starts);
int32_t q3ld_test_spec_held(int32_t n, int64_t timeout_ns);
int64_t q3ld_test_events(int64_t *buf, int64_t cap);
int64_t q3ld_test_ev_log(int64_t *buf, int64_t cap);
#endif

#ifdef __cplusplus
}
#endif

#endif
