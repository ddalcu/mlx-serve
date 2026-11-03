/* DSV41_LOOKAHEAD4 pool (2026-09-25, q3-lookahead4-20260925.md): lookahead3/q3_lookahead.c (sha256 abd6ec68...,
 * symbols renamed q3lc_ -> q3ld_) + ONE addition, the EVENT-GATE class (the GPU waits on a shared event for the bytes):
 *   the host builds a parked layer-call's miss waves over event-gated bank aliases (the MLX extension ext/q3ev4.cpp
 *   encodes MTLCommandBuffer.encodeWaitForEvent:value: in front of them) and commits them with the next router barrier
 *   BEFORE the bytes land; q3ld_ev_gates registers, in the GPU's encode order, one value per gate with the tickets the
 *   gate waits for (the GU tickets of every fill of the call, then per part the DOWN tickets of the part's fills).
 *   publish(t) (a ticket became terminal, under the pool mutex) decrements its gate; the SATISFIED PREFIX (every gate
 *   <= v has all its tickets terminal) is handed to the event (q3ld_mtl_signal = [MTLSharedEvent setSignaledValue:],
 *   q3_event_shim.mm; or a host int64 word for the CPU backend) by the publishing worker right after it drops the
 *   mutex, under sig_mu, never decreasing.  A failed ticket is terminal too (the GPU runs the wave, the host drain
 *   raises the stock error and discards it).  A WATCHDOG thread forces the oldest unsatisfied gate after its deadline
 *   (SC_EV_WD_FORCED: the host raises, the output may be invalid): a lost signal can never hang the GPU; stop releases
 *   every live gate.  Keyed by TICKET only (plans / slots / policy never see a gate).  No new staging.
 *
 * The lookahead3 pool's own header follows.
 *
 * DSV41_LOOKAHEAD3 pool (2026-09-25, q3-lookahead3-20260925.md): lookahead2/q3_lookahead.c (sha256 b6b127b5...,
 * symbols renamed q3lb_ -> q3ld_) + ONE addition, the PRE-READ class (lever A: host resume -> first demand read):
 *   q3ld_pre_read(fd, size, tag, n, bases), called by the generation thread right after the router barrier returns
 *   (before the flush / observe / planner / transaction glue that the stock issue waits for), takes the layer-call's
 *   CERTAIN misses (the route's unique experts not in the bank's resident map; only the planner edits that map, so it
 *   is the plan's miss set) as record bases.  Per record: a live speculative record containing it is CLAIMED now (the
 *   claim the demand submit would make ~0.3 ms later; a QUEUED one is cancelled and the bytes pre-read instead, as the
 *   submit would read them); otherwise its GU range and DOWN range are queued as PRE-RANGES (every GU first, then every
 *   DOWN: the stock batch-fill order).  A demand worker with no job queued takes the oldest pre-range and runs the
 *   STOCK range loop (run_range: aligned F_NOCACHE preadv into ITS OWN 9 MiB staging buffer, short-read
 *   continuation, the same syscalls and bytes) up to the first scatter copy, where it waits for the range's rows.
 *   q3ld_submit BINDS each job range (fd, offset, component lengths all equal; job without a deadline) to its pre-range:
 *   in flight / waiting -> the rows are attached and that worker scatters and finishes the loop; still QUEUED -> the
 *   pre-range is cancelled and the job reads the range itself (stock).  The range's own job, reaching a bound range,
 *   waits until the pre-range is DONE and publishes ITS result words.  Unbound pre-ranges
 *   expire at the layer-call's settle (spec_step / spec_stepn), at the next pre_read and at quiesce.  A pre-running
 *   worker counts as demand busy (the speculative queue rule sees it) and in the read gauge.  Keyed by FILE OFFSET only;
 *   plans / slots / policy never see a pre-range; exactness: a range's bytes and result words are those of the same
 *   stock read, started earlier.  Counters SC_PRE_*; inject event kinds 9 (pre start) / 10 (pre served).
 *
 * The lookahead2 pool's own header follows.
 *
DSV41_LOOKAHEAD2 pool (2026-09-25, q3-lookahead2-20260925.md): lookahead/q3_lookahead.c (sha256 bb894ccc...,
 * symbols renamed q3la_ -> q3ld_) with ONE policy change, the DEMAND-IDLE SECOND STREAM:
 *   speculative thread 0 keeps the lane's queue rule (an unclaimed chunk starts only while qlen == 0 and demand
 *   busy <= 1); speculative threads >= 1 start / continue an UNCLAIMED chunk only while qlen == 0 and demand busy
 *   <= spec_idle_busy (q3ld_spec_streams, set at construction; 0 = no demand job executes; 1 = the lane's rule for
 *   every thread, i.e. the lookahead pool unchanged).  Claimed records ignore the rule (unchanged).  The replay
 *   (replay_route_lookahead.py tune5idle) prices the lane's second stream as mostly contention with the demand read
 *   it overlaps; keeping it off the drive while a demand job reads keeps its demand-idle work.  New counters:
 *   SC_IDLE_CHUNKS (unclaimed chunks started by threads >= 1), SC_MAX_BUSY_AT_START_IDLE (<= spec_idle_busy).
 *   Event kind 8 = an unclaimed chunk start by a thread >= 1 (kind 3 = by thread 0).
 *
 * The lookahead pool's own header follows.
 *
 * DSV41_LOOKAHEAD pool (2026-09-24, q3-lookahead-20260924.md).  The DSV41_NATIVE_ISSUE pool
 * (nativeissue/q3_nativeissue.c, sha256 6866b04b..., symbols renamed q3ni_ -> q3ld_, the demand path
 * transcribed unchanged) plus ONE addition: a low-priority, CHUNKED, SPECULATIVE op class, split into
 * per-HORIZON priority classes (prio 0 = the next layer, prio h-1 = h layers ahead).
 *
 * DEMAND: workers, one per stock 9 MiB staging buffer, a FIFO job queue of ONE RANGE per job (a submit
 * queues each of its ranges, gate/up ranges first, so idle workers read them side by side; each range's
 * status is its own), per range the stock aligned F_NOCACHE preadv loop + the scatter copy into the slot
 * rows, the result words + an ordered completion log published with a RELEASE store.  Two additions on the demand
 * path, both keyed by FILE OFFSET only (never by layer / expert / slot):
 *   - q3ld_submit CLAIMS every speculative record that contains one of the job's ranges: an in-flight,
 *     parked or landed record is promoted (it now reads its remaining chunks back to back, ignoring the
 *     queue rule); a record still QUEUED (never started) is cancelled (the demand job reads those bytes);
 *   - before a range reads, a CLAIMED / LANDED record that contains the range serves it: the worker waits
 *     (never past the job's deadline) until the record's contiguous landed prefix covers the range, then
 *     memcpy-scatters it from the speculative staging (0 preadv).  Otherwise the stock read runs unchanged.
 *
 * SPECULATIVE op class (records live in their OWN staging slots and are read by their OWN threads):
 *   - a record is read as ceil(span / chunk) page-aligned CHUNKS (span = the page-aligned whole record);
 *   - QUEUE RULE (demand first, then nearer horizons): an UNCLAIMED record starts, and starts each further
 *     chunk, only while the demand queue is EMPTY (qlen == 0) and at most SPEC_MAX_DEMAND_BUSY (= 1) demand
 *     job executes, and (prio > 0) no record of a NEARER horizon is QUEUED.  At a boundary where that fails
 *     a prio-0 record pauses in place (the horizon-1 behaviour), a prio > 0 record PARKS (its thread is freed,
 *     its landed chunks kept; any thread resumes it later).  Pick order: claimed, then prio, then age;
 *   - q3ld_spec_step(cur, bases): horizon 1 (the h1 lane, unchanged): settle tag <= cur, issue tag cur + 1;
 *   - q3ld_spec_stepn(cur, nh, val, issue): horizon N: settle tag <= cur; RE-VALIDATE every unclaimed record
 *     with cur < tag <= cur + nh whose class is farther than its remaining distance (tag - cur) against the
 *     fresh candidate list for that target (val[tag - cur]): kept -> promoted to class tag - cur - 1, else
 *     dropped (queued / parked: now; running: at its next chunk boundary); then issue per horizon h the
 *     records issue[h] with tag cur + h, class h - 1;
 *   - reclaim: FREE first, else the oldest LANDED / FAILED slot with no reader and tag < cur;
 *   - a demand range holds `readers` on the slot while copying: a slot is never reclaimed under a copy.
 *
 * Counters (Python-owned int64[SC_N], written under the pool mutex, read at teardown): see SC_*.
 * Profile builds (-DQ3LD_EVSIG, -Ddsv41-decode-timers=true) keep only the event's signal log (q3ld_test_ev_log).
 * Test build (-DQ3LD_INJECT): the q3ni injection rules/delays, a speculative per-chunk delay and an event
 * log of demand submit / demand start / speculative chunk start with (qlen, busy, nearer-queued) at that
 * instant.
 */
/* EXL3_LANE_PORT (exl3/runtime/ports/gen_exl3_ports.py; q3-exl3-lane-ports-20260927.md): lookahead4/q3_lookahead.c with ONE change,
 * a per-record payload length in the speculative and pre-read classes, for the EXL3 variant bank whose layers carry
 * different record lengths (K = 2 layers: 8,891,904 B logical records inside 13,315,584 B slots):
 *   - spec_t.len = the record's payload bytes; spec_containing, the chunk span and the landed cap use it instead of the
 *     configured rec_len, which becomes the LARGEST record (the staging slot is sized from it, as before);
 *   - q3ld_spec_step_len(fd, size, cur, n, bases, len): q3ld_spec_step whose records carry the target layer's length
 *     (0 < len <= rec_len): a speculative read covers [base, base + len) and never the next record;
 *   - q3ld_pre_read_lens(fd, size, tag, n, bases, lens): q3ld_pre_read with the call's layer component lengths (ngu then
 *     ndown, the counts armed by q3ld_pre_config); the demand submit of that layer passes the same lengths, so the bind
 *     (same fd, offset, component lengths) holds, and a pre-range covers exactly the layer's span;
 *   - the stock entry points (q3ld_spec_step / _spec_stepn / _pre_read) behave as before (len = rec_len, the armed
 *     lengths): a uniform-record bank reads exactly as with the original pool.  ABI 2026092704 (original 2026092504).
 * ABI 2026100201.
 *
 * WITHDRAW (mimo/ahead-trim, 2026-10-10; ABI 2026101001; over range jobs ABI 2026101201): q3ld_withdraw(first) takes
 *   a submit that no worker has started back out of the queue, by its first ticket: every range job of it (a submit
 *   queues one job per range, sub_len[first] of them, adjacent in the FIFO and in ticket order), all or none; the jobs
 *   behind them keep their order.  Its tickets get ST_WITHDRAWN words (nothing read: no preadv, payload 0, no row
 *   written) and are published in ticket order under the pool mutex, so they are terminal for waits and free for the
 *   ring's next submit.  A submit a worker has started (its first range taken: the FIFO takes its ranges in ticket
 *   order) runs as submitted, every range of it (0 = not queued).  Refused (-2) for a submit whose bytes are promised
 *   elsewhere: a ticket bound to a pre-range or waited on by a live event gate.  Its owner calls it when the job's records are no longer
 *   needed before the drive reached them (P7 early_whole_trim: the read-ahead jobs still queued when the prompt call's
 *   plans exist).  It never touches the speculative class (no claim is released; the prompt pass issues none).
 *   Inject event kind 11 (withdrawn).
 */
#include <errno.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#define RES_W 8
#define MAX_WORKERS 8
#define MAX_COMP 6
#define MAX_ITEMS 8
#define ST_PENDING (-1)
#define ST_OK 0
#define ST_SHORT 1
#define ST_OSERR 2
#define ST_DEADLINE 3
#define ST_SKIPPED 4
#define ST_WITHDRAWN 5       /* q3ld_withdraw: the job left the queue unstarted; nothing read, no row written */

#define MAX_SPEC 16
#define MAX_SPEC_THREADS 3
#define MAX_H 4
#define SPEC_MAX_DEMAND_BUSY 1
#define SP_FREE 0
#define SP_QUEUED 1
#define SP_INFLIGHT 2
#define SP_LANDED 3
#define SP_FAILED 4

enum {
    SC_SUBMITTED = 0,     /* records queued (every horizon) */
    SC_REFRESHED,         /* issue hit a live record of the same base (tag refreshed) */
    SC_NOSLOT,            /* no reclaimable slot: skipped */
    SC_STARTED,           /* records started */
    SC_LANDED,            /* records fully landed */
    SC_FAILED,
    SC_EXPIRED,           /* QUEUED records dropped at settle / re-validation / quiesce */
    SC_CANCELLED_BY_DEMAND, /* QUEUED records whose bytes a demand job claimed (it reads them itself) */
    SC_CLAIMED,           /* in-flight / landed records claimed by a demand submit */
    SC_CLAIMED_INFLIGHT,  /* ... of which still in flight (running or parked) at the claim (partial lead) */
    SC_ABANDONED,         /* started records dropped at a chunk boundary / while parked */
    SC_ABANDONED_BYTES,   /* physical bytes those had read */
    SC_DISCARDED,         /* LANDED records reclaimed / stopped without ever serving a range */
    SC_ADOPT_RANGES,      /* demand ranges served from speculative staging */
    SC_ADOPT_BYTES,
    SC_ADOPT_WAITS,       /* demand ranges that waited for chunks still in flight */
    SC_ADOPT_WAIT_NS,
    SC_SPEC_BYTES,        /* physical bytes read by speculative chunks */
    SC_SPEC_CHUNKS,       /* speculative preadv calls */
    SC_PAUSES,            /* chunk boundaries where an unclaimed prio-0 record waited for the queue rule */
    SC_MAX_BUSY_AT_START, /* max demand busy at an UNCLAIMED chunk start (<= SPEC_MAX_DEMAND_BUSY) */
    SC_MAX_QLEN_AT_START, /* max demand qlen at an UNCLAIMED chunk start (== 0) */
    /* horizon classes (index = the horizon the record was ISSUED at, 1..MAX_H -> 0..MAX_H-1) */
    SC_PARKS,             /* prio > 0 records parked at a chunk boundary */
    SC_MAX_NEARER_AT_START, /* max count of QUEUED nearer-class records at an unclaimed prio > 0 chunk start (== 0) */
    SC_H_SUBMITTED,
    SC_H_KEPT = SC_H_SUBMITTED + MAX_H,      /* re-validations that kept (promoted) the record */
    SC_H_DROPPED = SC_H_KEPT + MAX_H,        /* re-validations that dropped it */
    SC_H_CLAIMED = SC_H_DROPPED + MAX_H,     /* claimed by a demand submit */
    SC_H_ADOPT_RANGES = SC_H_CLAIMED + MAX_H,
    SC_IDLE_CHUNKS = SC_H_ADOPT_RANGES + MAX_H,   /* unclaimed chunks started by speculative threads >= 1 */
    SC_MAX_BUSY_AT_START_IDLE,                    /* max demand busy at such a start (<= spec_idle_busy) */
    /* the pre-read class (lookahead3) */
    SC_PRE_CALLS,         /* q3ld_pre_read calls with >= 1 record */
    SC_PRE_ISSUED,        /* pre-ranges queued */
    SC_PRE_NOSLOT,        /* no free pre-range entry / range larger than a staging buffer: not pre-read */
    SC_PRE_CLAIMS,        /* live speculative records claimed at pre-read time */
    SC_PRE_STARTED,       /* pre-ranges a worker started */
    SC_PRE_BOUND,         /* pre-ranges bound to a demand job range (in flight / waiting / done) */
    SC_PRE_CANCELLED,     /* QUEUED pre-ranges cancelled by a demand submit (the job reads the range) */
    SC_PRE_EXPIRED,       /* unbound pre-ranges dropped at settle / next pre_read / quiesce */
    SC_PRE_SERVED,        /* job ranges whose result came from a pre-range */
    SC_PRE_WAITS,         /* ... of which the job worker waited for the pre-range */
    SC_PRE_WAIT_NS,
    SC_PRE_BIND_WAIT_NS,  /* pre-range workers waiting for their rows */
    SC_PRE_MAX_INFLIGHT,  /* max pre-ranges running at a start */
    /* the event-gate class (lookahead4) */
    SC_EV_CALLS,          /* q3ld_ev_gates calls */
    SC_EV_GATES,          /* gates registered */
    SC_EV_TICKETS,        /* tickets still pending at registration (mapped to their gate) */
    SC_EV_IMMEDIATE,      /* gates already satisfied at registration */
    SC_EV_SIGNALS,        /* values handed to the event (strictly increasing) */
    SC_EV_SIGNAL_NS,      /* time inside the event backend call */
    SC_EV_LAG_NS,         /* satisfied prefix advanced -> backend call start */
    SC_EV_MAX_LIVE,       /* max live gates */
    SC_EV_WD_FORCED,      /* gates the watchdog forced (bytes may not have landed: the host raises) */
    SC_EV_WD_LAST_VALUE,  /* the value of the last forced gate */
    SC_EV_HOST_RELEASED,  /* gates the host released (error paths) */
    SC_EV_STOP_RELEASED,  /* live gates released at stop */
    SC_DIRECT_RANGES,     /* demand ranges read straight into their rows (q3ld_direct_config: no staging, no copy) */
    SC_N
};

static uint32_t tb_numer = 0, tb_denom = 0;

int q3ld_init(void) {
    mach_timebase_info_data_t tb;
    if (mach_timebase_info(&tb) != KERN_SUCCESS || tb.denom == 0) return -1;
    tb_numer = tb.numer;
    tb_denom = tb.denom;
    return 0;
}

int64_t q3ld_monotonic_ns(void) {
    uint64_t ticks = mach_absolute_time();
    int64_t t = (int64_t)ticks;
    int64_t intpart = t / tb_denom;
    int64_t rem = t % tb_denom;
    return intpart * (int64_t)tb_numer + (rem * (int64_t)tb_numer) / (int64_t)tb_denom;
}

typedef struct {
    int64_t offset;
    int32_t ndst;
    uint64_t dst[MAX_COMP];
    int64_t len[MAX_COMP];
    int32_t scatter;              /* SCATTER: part d is read from fds[d] at offs[d], never from `offset` */
    int32_t fds[MAX_COMP];
    int64_t sizes[MAX_COMP];
    int64_t offs[MAX_COMP];
} range_t;

typedef struct {
    int fd;
    int64_t file_size;
    int64_t deadline;
    int64_t ticket;     /* the job's one range */
} job_t;

typedef struct {
    int state;
    int claimed;        /* a demand submit needs its bytes: reads on, never abandoned */
    int abandon;        /* settle / re-validation: drop at the next chunk boundary (unless claimed) */
    int running;        /* a speculative thread owns it right now (INFLIGHT && !running = parked) */
    int prio;           /* horizon class: 0 = next layer, h - 1 = h layers ahead */
    int origin;         /* the horizon it was issued at (1..MAX_H) */
    int fd;
    int64_t file_size;
    int64_t tag;
    uint64_t order;
    int64_t base;       /* file offset of the record */
    int64_t len;        /* EXL3_LANE_PORT: the record's payload bytes (<= spec_rec_len) */
    int64_t aligned;    /* page-floor of base: buf[0] holds file byte `aligned` */
    int64_t span;       /* physical bytes to read from `aligned` (page-aligned end, capped at EOF / slot) */
    int64_t got;        /* physical bytes read so far (contiguous from `aligned`) */
    int64_t landed;     /* payload bytes from `base` present in buf */
    int32_t readers;    /* demand ranges copying out of buf */
    int32_t adopted;    /* ranges served since it was queued */
    char *buf;
} spec_t;

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t work_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t done_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t spec_cv = PTHREAD_COND_INITIALIZER;   /* speculative threads: something changed */
static pthread_cond_t land_cv = PTHREAD_COND_INITIALIZER;   /* a speculative chunk landed / a read ended */
static pthread_t threads[MAX_WORKERS];
static pthread_t spec_threads[MAX_SPEC_THREADS];
static char *staging[MAX_WORKERS];
static int nworkers = 0, running = 0, stopping = 0, busy = 0;
static int64_t staging_bytes = 0, page_size = 0;
static int64_t *res = 0, nt = 0, *logbuf = 0, logn = 0, *gauge = 0;
static range_t *ranges = 0;
static job_t *queue = 0;
static int64_t qcap = 0, qhead = 0, qlen = 0;
static uint64_t seq = 0;

static spec_t spec[MAX_SPEC];
static int nspec = 0, nspec_threads = 0, nspec_started = 0, spec_inflight = 0, spec_configured = 0;
static int64_t spec_slot_bytes = 0, spec_rec_len = 0, spec_chunk = 0;
static int spec_idle_busy = SPEC_MAX_DEMAND_BUSY;   /* the queue rule's demand-busy limit for threads >= 1 */
static uint64_t spec_order = 0;
static int64_t *sc = 0;

/* ---- the pre-read class (lookahead3) */
#define MAX_PRE 32
#define PR_FREE 0
#define PR_QUEUED 1
#define PR_INFLIGHT 2     /* a worker owns it: reading, or waiting at the first scatter for its rows */
#define PR_DONE 3         /* words[] final; a bound DONE range is freed by its job worker, an unbound one by expiry */
#define PRE_DROPPED (-100)
typedef struct {
    int state;
    int bound;            /* a demand submit attached its rows (rg.dst) */
    int cancel;           /* expired while unbound (a cancelled range is never bound): no scatter */
    int fd;
    int64_t file_size;
    int64_t tag;
    uint64_t order;
    range_t rg;           /* offset + component lengths at issue; dst at bind */
    int64_t words[RES_W]; /* the range's result words once DONE (words[0] = status) */
} pre_t;
static pre_t pre[MAX_PRE];
static pthread_cond_t pre_cv = PTHREAD_COND_INITIALIZER;
static int pre_on = 0;
static int32_t pre_ngu = 0, pre_ndown = 0;
static int64_t pre_lens[2 * MAX_COMP];
static int64_t pre_gu_total = 0;
static uint64_t pre_order = 0;
static int32_t *pre_of = 0;   /* per ticket: 1 + the pre-range bound to that range at submit, 0 = none */
static int32_t *sub_len = 0;  /* WITHDRAW: per ticket, the range jobs of the submit it starts (0: it starts none) */

/* ---- the event-gate class (lookahead4) */
#define MAX_GATES 512         /* the live gates of every route a stream holds: 6 routes x (48 parts + 1) <= 512 */
#define MAX_GATE_TICKETS 768  /* tickets of one registration: a gate/up and a down ticket per record, 2 x 48 ids x 8 minis */
#define EVK_OFF 0
#define EVK_MTL 1         /* ev_obj = an MTLSharedEvent (q3ld_mtl_signal) */
#define EVK_HOST 2        /* ev_obj = an int64 word (release store; the CPU backend and the checks) */
#define GF_WATCHDOG 1
#define GF_HOST 2
#define GF_STOP 3
typedef struct {
    int64_t serial;       /* the ring index this entry holds; gate_of[t] == serial + 1 while t is mapped to it */
    uint64_t value;       /* the event value the GPU waits for in front of the gated waves */
    int32_t remaining;    /* mapped tickets not yet terminal */
    int32_t forced;       /* 0, GF_WATCHDOG, GF_HOST or GF_STOP: released without its tickets */
    int64_t deadline;     /* monotonic ns: the watchdog forces the gate after this */
    int64_t t_reg, t_sat; /* registration; the last ticket terminal (or forced) */
} gate_t;
static gate_t gates[MAX_GATES];
static int64_t g_head = 0, g_tail = 0;   /* live gates are the ring indices [g_head, g_tail), in value order */
static int64_t *gate_of = 0;             /* per ticket: 1 + the serial of the gate waiting for it, 0 = none */
static int ev_kind = EVK_OFF;            /* written under mu + sig_mu at config (atomic loads elsewhere) */
static void *ev_obj = 0;                 /* read under sig_mu */
static uint64_t ev_hi = 0;               /* highest value registered / released (under mu) */
static uint64_t ev_target = 0;           /* the satisfied prefix (written under mu, atomic) */
static int64_t ev_target_t = 0;          /* when ev_target last advanced (written under mu, atomic) */
static uint64_t ev_sent = 0;             /* the last value handed to the event (written under sig_mu, atomic) */
static int64_t ev_timeout_ns = 0;
static pthread_mutex_t sig_mu = PTHREAD_MUTEX_INITIALIZER;   /* orders the backend calls: values never decrease */
static pthread_cond_t wd_cv = PTHREAD_COND_INITIALIZER;
static pthread_t wd_thread;
static int wd_running = 0, wd_stop = 0;

void q3ld_mtl_signal(void *event, uint64_t value);          /* q3_event_shim.mm */

#ifdef Q3LD_INJECT
#define MAX_RULES 16
static int64_t rule_off[MAX_RULES], rule_code[MAX_RULES], rule_arg[MAX_RULES], rule_used[MAX_RULES];
static int nrules = 0;
static uint64_t rng = 0;
static int64_t delay_max = 0, spec_delay_max = 0;
/* event log: kind (1 demand submit, 2 demand start, 3 spec start by thread 0, 8 spec start by a thread >= 1, 7 claimed
 * chunk start, 4 spec end, 5 adopt, 9 pre-range start, 10 pre-range served, 11 demand job withdrawn), qlen, busy, t,
 * arg */
static int64_t *evlog = 0, evcap = 0, evn = 0;

void q3ld_test_rules(int32_t n, const int64_t *off, const int64_t *code, const int64_t *arg) {
    pthread_mutex_lock(&mu);
    nrules = n > MAX_RULES ? MAX_RULES : n;
    for (int i = 0; i < nrules; i++) { rule_off[i] = off[i]; rule_code[i] = code[i]; rule_arg[i] = arg[i]; rule_used[i] = 0; }
    pthread_mutex_unlock(&mu);
}

void q3ld_test_delay(uint64_t seed, int64_t max_ns) {
    pthread_mutex_lock(&mu);
    rng = seed ? seed : 88172645463325252ULL;
    delay_max = max_ns;
    pthread_mutex_unlock(&mu);
}

void q3ld_test_spec_delay(int64_t max_ns) {
    pthread_mutex_lock(&mu);
    spec_delay_max = max_ns;
    pthread_mutex_unlock(&mu);
}

/* HOLD (spec_hold >= 0): only spec_hold more records may start (SPEC_MAY_START, SPEC_STARTED), and a started
 * unclaimed record's chunk waits at its start (spec_hold_wait); spec_held counts the records waiting there, hold_cv
 * tells q3ld_test_spec_held of a change.  A claim (spec_cv) or the hold's end (-1) lets a record go. */
static int spec_hold = -1, spec_held = 0;
static pthread_cond_t hold_cv = PTHREAD_COND_INITIALIZER;

void q3ld_test_spec_hold(int32_t starts) {
    pthread_mutex_lock(&mu);
    spec_hold = starts < 0 ? -1 : starts;
    pthread_cond_broadcast(&spec_cv);
    pthread_mutex_unlock(&mu);
}

/* Python-owned int64[5 * cap]; n = 0 resets.  Returns entries written so far. */
int64_t q3ld_test_events(int64_t *buf, int64_t cap) {
    pthread_mutex_lock(&mu);
    if (buf) { evlog = buf; evcap = cap; evn = 0; }
    int64_t n = evn;
    pthread_mutex_unlock(&mu);
    return n;
}

static void ev(int64_t kind, int64_t arg) {   /* caller holds mu */
    if (evlog && evn < evcap) {
        int64_t *e = evlog + 5 * evn++;
        e[0] = kind; e[1] = qlen; e[2] = busy; e[3] = q3ld_monotonic_ns(); e[4] = arg;
    }
}

static void sleep_ns(int64_t ns) {
    struct timespec ts = { (time_t)(ns / 1000000000), (long)(ns % 1000000000) };
    nanosleep(&ts, 0);
}

static int64_t rand_below(int64_t max) {      /* caller holds mu */
    if (max <= 0) return 0;
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (int64_t)(rng % (uint64_t)max);
}

static void range_delay(void) {
    pthread_mutex_lock(&mu);
    int64_t d = rand_below(delay_max);
    pthread_mutex_unlock(&mu);
    if (d > 0) sleep_ns(d);
}

static void spec_delay(void) {   /* per chunk: uniform in [max / 2, max) */
    pthread_mutex_lock(&mu);
    if (!rng) rng = 88172645463325252ULL;
    int64_t d = spec_delay_max / 2 + rand_below(spec_delay_max - spec_delay_max / 2);
    pthread_mutex_unlock(&mu);
    if (d > 0) sleep_ns(d);
}

static ssize_t do_preadv(int fd, const struct iovec *iov, int n, off_t off) {
    int64_t code = 0, arg = 0;
    pthread_mutex_lock(&mu);
    for (int i = 0; i < nrules; i++) {
        if (!rule_used[i] && rule_off[i] == (int64_t)off) { rule_used[i] = 1; code = rule_code[i]; arg = rule_arg[i]; break; }
    }
    pthread_mutex_unlock(&mu);
    if (code == 2) { errno = EIO; return -1; }
    if (code == 3) return 0;
    if (code == 4) {
        struct iovec one = iov[0];
        if ((int64_t)one.iov_len > arg) one.iov_len = (size_t)arg;
        return preadv(fd, &one, 1, off);
    }
    if (code == 5) sleep_ns(arg);
    ssize_t r;
    do { r = preadv(fd, iov, n, off); } while (r < 0 && errno == EINTR);   /* code 1: EINTR retried inside */
    return r;
}
#define RANGE_DELAY() range_delay()
#define SPEC_DELAY() spec_delay()
#define EV(kind, arg) ev((kind), (arg))
#define SPEC_HOLD(s) spec_hold_wait(s)
#define SPEC_MAY_START(s) (spec_hold != 0 || (s)->state != SP_QUEUED)
#define SPEC_STARTED() do { if (spec_hold > 0) spec_hold -= 1; } while (0)
#else
static ssize_t do_preadv(int fd, const struct iovec *iov, int n, off_t off) {
    ssize_t r;
    do { r = preadv(fd, iov, n, off); } while (r < 0 && errno == EINTR);
    return r;
}
#define RANGE_DELAY() ((void)0)
#define SPEC_DELAY() ((void)0)
#define EV(kind, arg) ((void)0)
#define SPEC_HOLD(s) ((void)0)
#define SPEC_MAY_START(s) 1
#define SPEC_STARTED() ((void)0)
#endif

#define LATE(deadline) ((deadline) >= 0 && q3ld_monotonic_ns() >= (deadline))

static void gauge_enter(void) {
    pthread_mutex_lock(&mu);
    if (gauge[0] == 0) gauge[5] = q3ld_monotonic_ns();
    gauge[0] += 1;
    if (gauge[0] > gauge[1]) gauge[1] = gauge[0];
    gauge[2] += gauge[0];
    gauge[3] += 1;
    pthread_mutex_unlock(&mu);
}

static void gauge_exit(void) {
    pthread_mutex_lock(&mu);
    if (gauge[0] > 0) {
        gauge[0] -= 1;
        if (gauge[0] == 0 && gauge[5]) { gauge[4] += q3ld_monotonic_ns() - gauge[5]; gauge[5] = 0; }
    }
    pthread_mutex_unlock(&mu);
}

/* DIRECT (miss_reads, 2026-10-14; ABI 2026101401): the demand ranges' read ROUTE, chosen once on a stopped pool by
 * q3ld_direct_config.  On: every demand range is read straight into its rows by preadv (one iovec per part), no staging
 * buffer and no scatter copy; off (the default): the staging loop + the copy.  No per-range eligibility test: the owner
 * proves once, at construction, that every range it will submit is aligned to the file's block size (the record
 * offsets, the part lengths, the slot rows; sdk_ext/expert/stream.zig checkDirectReads), so the F_NOCACHE preadv takes
 * the uncached path.  Either route lands the file's bytes; the route never decides correctness, only the copy. */
static int direct_reads = 0;
static int run_range_staged(const job_t *job, const range_t *rg, char *stage, int64_t *out);
static int run_range_direct_route(const job_t *job, const range_t *rg, char *stage, int64_t *out);
static int run_range_scatter(const job_t *job, const range_t *rg, char *stage, int64_t *out);
static int (*run_range)(const job_t *, const range_t *, char *, int64_t *) = run_range_staged;

int q3ld_direct_config(int32_t on) {
    pthread_mutex_lock(&mu);
    if (running) { pthread_mutex_unlock(&mu); return -1; }
    direct_reads = on != 0;
    run_range = direct_reads ? run_range_direct_route : run_range_staged;
    pthread_mutex_unlock(&mu);
    return 0;
}

/* the DIRECT read: preadv into the rows until every part is full; the status words as run_range's */
static int run_range_direct(const job_t *job, const range_t *rg, int64_t *out) {
    struct iovec iov[MAX_COMP];
    int n = 0;
    int64_t want = 0;
    for (int32_t d = 0; d < rg->ndst; d++) {
        if (!rg->len[d]) continue;
        iov[n].iov_base = (void *)(uintptr_t)rg->dst[d];
        iov[n].iov_len = (size_t)rg->len[d];
        want += rg->len[d];
        n++;
    }
    int64_t calls = 0, returned = 0, total = 0, t_start = 0;
    int status = ST_OK, first = 0;
    out[4] = 0;
    while (total < want) {
        if (LATE(job->deadline)) { status = ST_DEADLINE; break; }
        calls++;
        if (!t_start) t_start = q3ld_monotonic_ns();
        ssize_t r = do_preadv(job->fd, iov + first, n - first, (off_t)(rg->offset + total));
        if (r < 0) { out[4] = errno; status = ST_OSERR; break; }
        if (r == 0) { status = ST_SHORT; break; }
        returned += r;
        total += r;
        int64_t left = r;
        while (first < n && left >= (int64_t)iov[first].iov_len) { left -= (int64_t)iov[first].iov_len; first++; }
        if (first < n && left > 0) { iov[first].iov_base = (char *)iov[first].iov_base + left; iov[first].iov_len -= (size_t)left; }
    }
    pthread_mutex_lock(&mu);
    sc[SC_DIRECT_RANGES] += 1;
    pthread_mutex_unlock(&mu);
    out[1] = calls;
    out[2] = returned;
    out[3] = total;
    out[5] = t_start ? t_start : q3ld_monotonic_ns();
    out[6] = q3ld_monotonic_ns();
    return status;
}

static int run_range_direct_route(const job_t *job, const range_t *rg, char *stage, int64_t *out) {
    (void)stage;
    return run_range_direct(job, rg, out);
}

/* one range: stock _readv_range_into_impl loop + _read_some steps; returns the status (q3_nativeissue.c) */
static int run_range_staged(const job_t *job, const range_t *rg, char *stage, int64_t *out) {
    int64_t calls = 0, returned = 0, read_total = 0, t_start = 0;
    int status = ST_OK;
    int32_t d = 0;
    int64_t d_off = 0;
    out[4] = 0;
    while (d < rg->ndst && rg->len[d] == 0) d++;
    while (d < rg->ndst) {
        if (LATE(job->deadline)) { status = ST_DEADLINE; break; }         /* impl loop top */
        if (LATE(job->deadline)) { status = ST_DEADLINE; break; }         /* _read_some entry */
        int64_t off = rg->offset + read_total;
        if (off >= job->file_size) { status = ST_SHORT; break; }
        int64_t requested = rg->len[d] - d_off;
        for (int32_t i = d + 1; i < rg->ndst; i++) requested += rg->len[i];
        int64_t aligned_offset = off / page_size * page_size;
        int64_t skip = off - aligned_offset;
        int64_t aligned_end = (off + requested + page_size - 1) / page_size * page_size;
        int64_t physical = staging_bytes;
        if (aligned_end - aligned_offset < physical) physical = aligned_end - aligned_offset;
        if (job->file_size - aligned_offset < physical) physical = job->file_size - aligned_offset;
        if (physical <= 0) { status = ST_SHORT; break; }
        struct iovec iov = { stage, (size_t)physical };
        ssize_t r;
        for (;;) {
            if (LATE(job->deadline)) { status = ST_DEADLINE; break; }
            calls++;
            if (!t_start) t_start = q3ld_monotonic_ns();
            r = do_preadv(job->fd, &iov, 1, (off_t)aligned_offset);
            if (r < 0) { out[4] = errno; status = ST_OSERR; break; }
            returned += r;
            if (LATE(job->deadline)) { status = ST_DEADLINE; break; }
            break;   /* a read with no byte past the skip ends the range short below: the file ends inside it */
        }
        if (status) break;
        int64_t payload = r - skip;
        if (payload < 0) payload = 0;
        if (payload > requested) payload = requested;
        if (payload <= 0) { status = ST_SHORT; break; }
        const char *src = stage + skip;
        int64_t left = payload;
        while (left > 0) {
            int64_t room = rg->len[d] - d_off;
            int64_t n = left < room ? left : room;
            memcpy((char *)(uintptr_t)rg->dst[d] + d_off, src, (size_t)n);
            src += n; left -= n; d_off += n;
            if (d_off == rg->len[d]) { d++; d_off = 0; while (d < rg->ndst && rg->len[d] == 0) d++; }
        }
        read_total += payload;
    }
    out[1] = calls;
    out[2] = returned;
    out[3] = read_total;
    out[5] = t_start ? t_start : q3ld_monotonic_ns();
    out[6] = q3ld_monotonic_ns();
    return status;
}

/* A PRE-RANGE: run_range verbatim (the job has no deadline), except that before the FIRST scatter copy the worker
 * waits (holding its staged bytes) until a demand submit binds the rows, or the range is cancelled / the pool stops
 * unbound: then PRE_DROPPED, no copy.  Every later loop step is the stock one. */
static int run_range_pre(const job_t *job, pre_t *pr, char *stage, int64_t *out) {
    const range_t *rg = &pr->rg;
    int64_t calls = 0, returned = 0, read_total = 0, t_start = 0;
    int status = ST_OK;
    int32_t d = 0;
    int64_t d_off = 0;
    int bound_seen = 0;
    out[4] = 0;
    while (d < rg->ndst && rg->len[d] == 0) d++;
    while (d < rg->ndst) {
        if (LATE(job->deadline)) { status = ST_DEADLINE; break; }         /* impl loop top */
        if (LATE(job->deadline)) { status = ST_DEADLINE; break; }         /* _read_some entry */
        int64_t off = rg->offset + read_total;
        if (off >= job->file_size) { status = ST_SHORT; break; }
        int64_t requested = rg->len[d] - d_off;
        for (int32_t i = d + 1; i < rg->ndst; i++) requested += rg->len[i];
        int64_t aligned_offset = off / page_size * page_size;
        int64_t skip = off - aligned_offset;
        int64_t aligned_end = (off + requested + page_size - 1) / page_size * page_size;
        int64_t physical = staging_bytes;
        if (aligned_end - aligned_offset < physical) physical = aligned_end - aligned_offset;
        if (job->file_size - aligned_offset < physical) physical = job->file_size - aligned_offset;
        if (physical <= 0) { status = ST_SHORT; break; }
        struct iovec iov = { stage, (size_t)physical };
        ssize_t r;
        for (;;) {
            if (LATE(job->deadline)) { status = ST_DEADLINE; break; }
            calls++;
            if (!t_start) t_start = q3ld_monotonic_ns();
            r = do_preadv(job->fd, &iov, 1, (off_t)aligned_offset);
            if (r < 0) { out[4] = errno; status = ST_OSERR; break; }
            returned += r;
            if (LATE(job->deadline)) { status = ST_DEADLINE; break; }
            break;   /* a read with no byte past the skip ends the range short below: the file ends inside it */
        }
        if (status) break;
        int64_t payload = r - skip;
        if (payload < 0) payload = 0;
        if (payload > requested) payload = requested;
        if (payload <= 0) { status = ST_SHORT; break; }
        if (!bound_seen) {                                  /* PRE: the rows, before the first copy */
            pthread_mutex_lock(&mu);
            int64_t tw = q3ld_monotonic_ns();
            while (!pr->bound && !pr->cancel && !stopping) pthread_cond_wait(&pre_cv, &mu);
            int drop = !pr->bound || pr->cancel;
            sc[SC_PRE_BIND_WAIT_NS] += q3ld_monotonic_ns() - tw;
            pthread_mutex_unlock(&mu);
            if (drop) return PRE_DROPPED;
            bound_seen = 1;
        }
        const char *src = stage + skip;
        int64_t left = payload;
        while (left > 0) {
            int64_t room = rg->len[d] - d_off;
            int64_t n = left < room ? left : room;
            memcpy((char *)(uintptr_t)rg->dst[d] + d_off, src, (size_t)n);
            src += n; left -= n; d_off += n;
            if (d_off == rg->len[d]) { d++; d_off = 0; while (d < rg->ndst && rg->len[d] == 0) d++; }
        }
        read_total += payload;
    }
    out[1] = calls;
    out[2] = returned;
    out[3] = read_total;
    out[5] = t_start ? t_start : q3ld_monotonic_ns();
    out[6] = q3ld_monotonic_ns();
    return status;
}

/* ---------------------------------------------------------------- pre-read: shared helpers (caller holds mu) */
static int pre_pick(void) {   /* the oldest QUEUED pre-range, -1 if none */
    int best = -1;
    for (int i = 0; i < MAX_PRE; i++) {
        if (pre[i].state == PR_QUEUED && (best < 0 || pre[i].order < pre[best].order)) best = i;
    }
    return best;
}

/* the live unbound pre-range serving exactly this job range (same fd, offset, components), or -1 */
static int pre_find(int fd, const range_t *rg) {
    for (int i = 0; i < MAX_PRE; i++) {
        pre_t *p = &pre[i];
        if (p->state == PR_FREE || p->bound || p->cancel || p->fd != fd || p->rg.offset != rg->offset
            || p->rg.ndst != rg->ndst) continue;
        int same = 1;
        for (int32_t c = 0; c < rg->ndst; c++) if (p->rg.len[c] != rg->len[c]) { same = 0; break; }
        if (same) return i;
    }
    return -1;
}

/* drop every UNBOUND pre-range with tag <= cur: queued / done -> free, in flight -> cancelled at its bind point */
static int pre_expire(int64_t cur) {
    int wake = 0;
    for (int i = 0; i < MAX_PRE; i++) {
        pre_t *p = &pre[i];
        if (p->state == PR_FREE || p->bound || p->tag > cur) continue;
        if (p->state == PR_INFLIGHT) {
            if (!p->cancel) { p->cancel = 1; sc[SC_PRE_EXPIRED] += 1; wake = 1; }
        } else {
            if (p->state == PR_QUEUED || !p->cancel) sc[SC_PRE_EXPIRED] += 1;
            p->state = PR_FREE;
            p->cancel = 0;
        }
    }
    if (wake) pthread_cond_broadcast(&pre_cv);
    return wake;
}

/* a worker runs pre-range i (enters and leaves holding mu) */
static void run_pre(int i, int w, char *stage) {
    pre_t *p = &pre[i];
    p->state = PR_INFLIGHT;
    busy += 1;
    sc[SC_PRE_STARTED] += 1;
    int n = 0;
    for (int k = 0; k < MAX_PRE; k++) n += pre[k].state == PR_INFLIGHT;
    if (n > sc[SC_PRE_MAX_INFLIGHT]) sc[SC_PRE_MAX_INFLIGHT] = n;
    EV(9, p->rg.offset);
    job_t job = { p->fd, p->file_size, -1, 0 };
    pthread_mutex_unlock(&mu);
    RANGE_DELAY();
    int64_t out[RES_W] = { 0 };
    gauge_enter();
    int status = run_range_pre(&job, p, stage, out);
    gauge_exit();
    pthread_mutex_lock(&mu);
    busy -= 1;
    if (status == PRE_DROPPED) {
        p->state = PR_FREE;                 /* expired unbound: nobody waits for it */
        p->cancel = 0;
    } else {
        for (int k = 1; k < RES_W; k++) p->words[k] = out[k];
        p->words[7] = w;
        p->words[0] = status;
        p->state = PR_DONE;
    }
    pthread_cond_broadcast(&pre_cv);
    pthread_cond_broadcast(&done_cv);
    pthread_cond_broadcast(&spec_cv);             /* the demand side got shallower */
}

/* ---------------------------------------------------------------- speculative: shared helpers */
static void abs_realtime(struct timespec *dl, int64_t ns_from_now) {
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    int64_t ns = (int64_t)now.tv_nsec + ns_from_now;
    dl->tv_sec = now.tv_sec + (time_t)(ns / 1000000000);
    dl->tv_nsec = (long)(ns % 1000000000);
}

static int64_t range_total(const range_t *rg) {
    int64_t total = 0;
    for (int32_t i = 0; i < rg->ndst; i++) total += rg->len[i];
    return total;
}

static int spec_live(const spec_t *s) {
    return s->state == SP_QUEUED || s->state == SP_INFLIGHT || s->state == SP_LANDED;
}

static int hidx(const spec_t *s) {   /* the counter index of the issuing horizon */
    int h = s->origin < 1 ? 1 : (s->origin > MAX_H ? MAX_H : s->origin);
    return h - 1;
}

/* the live record of fd whose [base, base + rec_len) contains [off, off + total), or -1 (caller holds mu) */
static int spec_containing(int fd, int64_t off, int64_t total) {
    for (int i = 0; i < nspec; i++) {
        spec_t *s = &spec[i];
        if (spec_live(s) && s->fd == fd && off >= s->base && off + total <= s->base + s->len) return i;
    }
    return -1;
}

/* ---------------------------------------------------------------- speculative: demand-side adoption */
/* Serve one demand range from a speculative record.  Returns -1 when no record serves it (the caller reads
 * it the stock way), else the range status (ST_OK, or ST_DEADLINE when the deadline passed while waiting). */
static int try_adopt(const job_t *job, const range_t *rg, int64_t *out) {
    if (!nspec) return -1;
    int64_t total = range_total(rg);
    int64_t t0 = q3ld_monotonic_ns(), waited = 0;
    pthread_mutex_lock(&mu);
    int i;
    for (;;) {
        i = spec_containing(job->fd, rg->offset, total);
        if (i < 0) { pthread_mutex_unlock(&mu); return -1; }
        spec_t *s = &spec[i];
        int64_t need = rg->offset + total - s->base;
        if (s->landed >= need && (s->state == SP_LANDED || s->state == SP_INFLIGHT)) break;
        /* adoptable only if it will land: a landed-short record (EOF) or an unclaimed / queued one is not */
        if (s->state != SP_INFLIGHT || !s->claimed) { pthread_mutex_unlock(&mu); return -1; }
        if (!waited) { waited = 1; sc[SC_ADOPT_WAITS] += 1; }
        if (job->deadline >= 0) {
            int64_t left = job->deadline - q3ld_monotonic_ns();
            if (left <= 0) {
                pthread_mutex_unlock(&mu);
                out[1] = out[2] = out[3] = out[4] = 0;
                out[5] = t0; out[6] = q3ld_monotonic_ns();
                return ST_DEADLINE;
            }
            struct timespec dl;
            abs_realtime(&dl, left);
            pthread_cond_timedwait(&land_cv, &mu, &dl);
        } else {
            pthread_cond_wait(&land_cv, &mu);
        }
    }
    spec_t *s = &spec[i];
    s->readers += 1;
    /* An adopting copy reads s->buf until it drops `readers`. A landed record a demand range adopts without a claim
     * (its bytes landed before any submit claimed it) could otherwise be abandoned (settle, re-validation, quiesce) and
     * its slot reissued while the copy runs: spec_end frees it without looking at `readers`, issue() takes the free
     * slot, and its preadv overwrites the buffer mid-copy (the range then publishes OK with torn bytes). Claiming it
     * here, under `mu`, keeps it out of every abandon path until the copy is done. */
    s->claimed = 1;
    s->abandon = 0;
    if (waited) sc[SC_ADOPT_WAIT_NS] += q3ld_monotonic_ns() - t0;
    EV(5, rg->offset);
    const char *src = s->buf + (s->base - s->aligned) + (rg->offset - s->base);
    pthread_mutex_unlock(&mu);
    int64_t t_start = q3ld_monotonic_ns();
    for (int32_t d = 0; d < rg->ndst; d++) {
        if (rg->len[d]) memcpy((char *)(uintptr_t)rg->dst[d], src, (size_t)rg->len[d]);
        src += rg->len[d];
    }
    pthread_mutex_lock(&mu);
    s->readers -= 1;
    s->adopted += 1;
    sc[SC_ADOPT_RANGES] += 1;
    sc[SC_ADOPT_BYTES] += total;
    sc[SC_H_ADOPT_RANGES + hidx(s)] += 1;
    pthread_mutex_unlock(&mu);
    out[1] = 0;          /* preadv calls */
    out[2] = 0;          /* bytes returned by preadv */
    out[3] = total;      /* payload landed */
    out[4] = 0;
    out[5] = t_start;
    out[6] = q3ld_monotonic_ns();
    return ST_OK;
}

/* a demand job's ranges claim the records that contain them (caller holds mu) */
static void claim_ranges(int fd, int64_t first, int32_t count) {
    int woke = 0;
    for (int32_t i = 0; i < count; i++) {
        if (pre_of && pre_of[first + i]) continue;          /* PRE: served by its pre-range */
        const range_t *rg = &ranges[first + i];
        int k = spec_containing(fd, rg->offset, range_total(rg));
        if (k < 0) continue;
        spec_t *s = &spec[k];
        if (s->state == SP_QUEUED) {
            s->state = SP_FREE;
            sc[SC_CANCELLED_BY_DEMAND] += 1;
        } else if (!s->claimed) {
            s->claimed = 1;
            s->abandon = 0;
            sc[SC_CLAIMED] += 1;
            sc[SC_H_CLAIMED + hidx(s)] += 1;
            if (s->state == SP_INFLIGHT) { sc[SC_CLAIMED_INFLIGHT] += 1; woke = 1; }
        }
    }
    if (woke) pthread_cond_broadcast(&spec_cv);
}

/* ---------------------------------------------------------------- the event-gate class: the prefix and the signal */
static int gate_live(int64_t serial) {   /* caller holds mu */
    return serial >= g_head && serial < g_tail && gates[serial % MAX_GATES].serial == serial;
}

/* retire every satisfied / forced gate at the head, in value order (caller holds mu); 1 when the prefix moved */
static int ev_advance(void) {
    int moved = 0;
    uint64_t target = ev_target;
    while (g_head < g_tail) {
        gate_t *g = &gates[g_head % MAX_GATES];
        if (g->remaining > 0 && !g->forced) break;
        target = g->value;
        g_head += 1;
        moved = 1;
    }
    if (moved) {
        __atomic_store_n(&ev_target_t, q3ld_monotonic_ns(), __ATOMIC_RELEASE);
        __atomic_store_n(&ev_target, target, __ATOMIC_RELEASE);
        pthread_cond_signal(&wd_cv);                /* the watchdog re-reads the oldest live deadline */
    }
    return moved;
}

/* ticket t just became terminal (caller holds mu, from publish) */
static void gate_ticket_done(int64_t t) {
    int64_t s = gate_of[t] - 1;
    gate_of[t] = 0;
    if (!gate_live(s)) return;                      /* its gate was forced and retired meanwhile */
    gate_t *g = &gates[s % MAX_GATES];
    if (g->remaining > 0 && --g->remaining == 0 && !g->forced) {
        g->t_sat = q3ld_monotonic_ns();
        ev_advance();
    }
}

#if defined(Q3LD_INJECT) || defined(Q3LD_EVSIG)
static int64_t *evsig = 0, evsig_cap = 0, evsig_n = 0;   /* signal log (under sig_mu): value, t_call, t_target, ns */
#endif

/* Hand the satisfied prefix to the event (caller holds NO pool lock).  Called by the thread that moved the prefix
 * right after it dropped mu; sig_mu orders concurrent callers so the event's value never decreases. */
static void ev_flush(void) {
    if (__atomic_load_n(&ev_kind, __ATOMIC_ACQUIRE) == EVK_OFF) return;
    uint64_t target = __atomic_load_n(&ev_target, __ATOMIC_ACQUIRE);
    if (target <= __atomic_load_n(&ev_sent, __ATOMIC_ACQUIRE)) return;
    pthread_mutex_lock(&sig_mu);
    target = __atomic_load_n(&ev_target, __ATOMIC_ACQUIRE);
    if (target > ev_sent && ev_obj) {
        int64_t tt = __atomic_load_n(&ev_target_t, __ATOMIC_ACQUIRE);
        int64_t t0 = q3ld_monotonic_ns();
        if (ev_kind == EVK_MTL) q3ld_mtl_signal(ev_obj, target);
        else __atomic_store_n((int64_t *)ev_obj, (int64_t)target, __ATOMIC_RELEASE);
        int64_t t1 = q3ld_monotonic_ns();
        __atomic_store_n(&ev_sent, target, __ATOMIC_RELEASE);
        if (sc) {
            __atomic_fetch_add(&sc[SC_EV_SIGNALS], 1, __ATOMIC_RELAXED);
            __atomic_fetch_add(&sc[SC_EV_SIGNAL_NS], t1 - t0, __ATOMIC_RELAXED);
            __atomic_fetch_add(&sc[SC_EV_LAG_NS], t0 > tt ? t0 - tt : 0, __ATOMIC_RELAXED);
        }
#if defined(Q3LD_INJECT) || defined(Q3LD_EVSIG)
        if (evsig && evsig_n < evsig_cap) {
            int64_t *e = evsig + 4 * evsig_n++;
            e[0] = (int64_t)target; e[1] = t0; e[2] = tt; e[3] = t1 - t0;
        }
#endif
    }
    pthread_mutex_unlock(&sig_mu);
}

static void publish(int64_t t) {
    /* caller holds mu; res[t] already written */
    logbuf[seq % (uint64_t)logn] = t;
    __atomic_store_n(&seq, seq + 1, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&done_cv);
    if (gate_of && gate_of[t]) gate_ticket_done(t);          /* EVENT: its gate (the flush follows the unlock) */
}

static void *worker(void *arg) {
    int w = (int)(intptr_t)arg;
    char *stage = staging[w];
    for (;;) {
        pthread_mutex_lock(&mu);
        int pi = -1;
        while (!stopping && qlen == 0 && (pi = pre_pick()) < 0) pthread_cond_wait(&work_cv, &mu);
        if (qlen == 0 && stopping) { pthread_mutex_unlock(&mu); return 0; }
        if (qlen == 0 && pi >= 0) {                   /* PRE: no job queued -> the oldest pre-range */
            run_pre(pi, w, stage);
            pthread_mutex_unlock(&mu);
            continue;
        }
        job_t job = queue[qhead];
        qhead = (qhead + 1) % qcap;
        __atomic_store_n(&qlen, qlen - 1, __ATOMIC_RELAXED);
        EV(2, job.ticket);
        busy += 1;
        pthread_mutex_unlock(&mu);
        const int64_t t = job.ticket;
        int64_t *out = res + t * RES_W;
        int pb = pre_of ? pre_of[t] : 0;
        int status;
        int ww = w;
        if (pb) {                                      /* PRE: the bound pre-range's result */
            pre_t *p = &pre[pb - 1];
            pthread_mutex_lock(&mu);
            if (p->state != PR_DONE) {
                int64_t tw = q3ld_monotonic_ns();
                sc[SC_PRE_WAITS] += 1;
                while (p->state != PR_DONE) pthread_cond_wait(&pre_cv, &mu);
                sc[SC_PRE_WAIT_NS] += q3ld_monotonic_ns() - tw;
            }
            for (int k = 1; k < RES_W; k++) out[k] = p->words[k];
            status = (int)p->words[0];
            ww = (int)p->words[7];
            p->state = PR_FREE; p->bound = 0; p->cancel = 0;
            pre_of[t] = 0;
            sc[SC_PRE_SERVED] += 1;
            EV(10, ranges[t].offset);
            pthread_mutex_unlock(&mu);
        } else {
            RANGE_DELAY();
            status = ranges[t].scatter ? -1 : try_adopt(&job, &ranges[t], out);
            if (status < 0) {
                gauge_enter();
                status = ranges[t].scatter ? run_range_scatter(&job, &ranges[t], stage, out) : run_range(&job, &ranges[t], stage, out);
                gauge_exit();
            }
        }
        out[7] = ww;
        pthread_mutex_lock(&mu);
        out[0] = status;
        publish(t);
        pthread_mutex_unlock(&mu);
        ev_flush();                                    /* EVENT: the gate this range completed, if any */
        pthread_mutex_lock(&mu);
        busy -= 1;
        pthread_cond_broadcast(&done_cv);
        pthread_cond_broadcast(&spec_cv);          /* the demand side got shallower */
        pthread_mutex_unlock(&mu);
    }
}

/* ---------------------------------------------------------------- speculative: reader threads */
static int rule_ok(int ti) {   /* THE QUEUE RULE, demand side, for speculative thread ti (caller holds mu) */
    return qlen == 0 && busy <= (ti == 0 ? SPEC_MAX_DEMAND_BUSY : spec_idle_busy);
}

static int nearer_queued(int prio) {   /* QUEUED records of a nearer horizon class (caller holds mu) */
    int n = 0;
    for (int i = 0; i < nspec; i++) {
        if (spec[i].state == SP_QUEUED && spec[i].prio < prio) n++;
    }
    return n;
}

static int pickable(const spec_t *s) {   /* queued, or parked (in flight, unowned) */
    return s->state == SP_QUEUED || (s->state == SP_INFLIGHT && !s->running);
}

/* caller holds mu: claimed (parked) first, then the nearest class, then the oldest; -1 if none */
static int spec_pick(void) {
    int best = -1;
    for (int i = 0; i < nspec; i++) {
        spec_t *s = &spec[i];
        if (!pickable(s)) continue;
        if (best < 0) { best = i; continue; }
        spec_t *b = &spec[best];
        if (s->claimed != b->claimed) { if (s->claimed) best = i; continue; }
        if (s->prio != b->prio) { if (s->prio < b->prio) best = i; continue; }
        if (s->order < b->order) best = i;
    }
    return best;
}

static int startable(const spec_t *s, int ti) {   /* caller holds mu */
    if (s->claimed) return 1;
    return rule_ok(ti) && (s->prio == 0 || nearer_queued(s->prio) == 0);
}

static void spec_end(spec_t *s, int state) {   /* caller holds mu; s is running */
    s->state = state;
    s->running = 0;
    spec_inflight -= 1;
    EV(4, s->base);
    pthread_cond_broadcast(&land_cv);
    pthread_cond_broadcast(&done_cv);
}

static void spec_drop_parked(spec_t *s) {   /* caller holds mu; s is INFLIGHT && !running && !claimed */
    sc[SC_ABANDONED] += 1;
    sc[SC_ABANDONED_BYTES] += s->got;
    s->state = SP_FREE;
    pthread_cond_broadcast(&land_cv);
}

#ifdef Q3LD_INJECT
static void spec_hold_wait(const spec_t *s) {   /* caller holds mu */
    if (spec_hold < 0 || s->claimed) return;
    spec_held += 1;
    pthread_cond_broadcast(&hold_cv);
    while (spec_hold >= 0 && !s->claimed && !stopping) pthread_cond_wait(&spec_cv, &mu);
    spec_held -= 1;
    pthread_cond_broadcast(&hold_cv);
}

int32_t q3ld_test_spec_held(int32_t n, int64_t timeout_ns) {
    struct timespec dl;
    abs_realtime(&dl, timeout_ns);
    pthread_mutex_lock(&mu);
    int32_t rc = 0;
    while (spec_held < n) {
        if (pthread_cond_timedwait(&hold_cv, &mu, &dl) == ETIMEDOUT) { rc = 1; break; }
    }
    pthread_mutex_unlock(&mu);
    return rc;
}
#endif

static void *spec_worker(void *arg) {
    int ti = (int)(intptr_t)arg;
    pthread_mutex_lock(&mu);
    for (;;) {
        int i;
        while (!stopping && !((i = spec_pick()) >= 0 && startable(&spec[i], ti) && SPEC_MAY_START(&spec[i]))) pthread_cond_wait(&spec_cv, &mu);
        if (stopping) break;
        spec_t *s = &spec[i];
        if (s->state == SP_QUEUED) {
            SPEC_STARTED();
            s->state = SP_INFLIGHT;
            sc[SC_STARTED] += 1;
            int64_t want_end = s->base + s->len;                  /* EXL3_LANE_PORT: this record's length */
            if (want_end > s->file_size) want_end = s->file_size;
            int64_t aligned_end = (want_end + page_size - 1) / page_size * page_size;
            s->span = aligned_end - s->aligned;
            if (s->span > spec_slot_bytes) s->span = spec_slot_bytes;
            if (s->file_size - s->aligned < s->span) s->span = s->file_size - s->aligned;
            s->got = 0;
            s->landed = 0;
        }
        s->running = 1;
        spec_inflight += 1;
        for (;;) {
            /* chunk boundary (also before the first chunk) */
            if ((s->abandon || stopping) && !s->claimed) {
                sc[SC_ABANDONED] += 1;
                sc[SC_ABANDONED_BYTES] += s->got;
                spec_end(s, SP_FREE);
                break;
            }
            if (s->got >= s->span) {
                if (s->landed > 0) { sc[SC_LANDED] += 1; spec_end(s, SP_LANDED); }
                else { sc[SC_FAILED] += 1; spec_end(s, SP_FAILED); }
                break;
            }
            if (!s->claimed && !stopping && s->prio > 0 && (!rule_ok(ti) || nearer_queued(s->prio))) {
                /* PARK: free this thread for demand-nearer work; any thread resumes the record later */
                sc[SC_PARKS] += 1;
                s->running = 0;
                spec_inflight -= 1;
                pthread_cond_broadcast(&done_cv);
                pthread_cond_broadcast(&spec_cv);
                break;
            }
            if (!s->claimed && !stopping && !rule_ok(ti)) {
                sc[SC_PAUSES] += 1;
                pthread_cond_wait(&spec_cv, &mu);
                continue;
            }
            int nq = 0;
            if (!s->claimed) {
                nq = s->prio ? nearer_queued(s->prio) : 0;
                if (busy > sc[SC_MAX_BUSY_AT_START]) sc[SC_MAX_BUSY_AT_START] = busy;
                if (qlen > sc[SC_MAX_QLEN_AT_START]) sc[SC_MAX_QLEN_AT_START] = qlen;
                if (nq > sc[SC_MAX_NEARER_AT_START]) sc[SC_MAX_NEARER_AT_START] = nq;
                if (ti > 0) {
                    sc[SC_IDLE_CHUNKS] += 1;
                    if (busy > sc[SC_MAX_BUSY_AT_START_IDLE]) sc[SC_MAX_BUSY_AT_START_IDLE] = busy;
                }
            }
            EV(s->claimed ? 7 : (ti > 0 ? 8 : 3), s->claimed ? s->base : nq);
            SPEC_HOLD(s);
            int64_t n = s->span - s->got;
            if (n > spec_chunk) n = spec_chunk;
            char *dst = s->buf + s->got;
            int fd = s->fd;
            off_t off = (off_t)(s->aligned + s->got);
            pthread_mutex_unlock(&mu);
            SPEC_DELAY();
            int64_t done = 0;
            int err = 0;
            while (done < n) {
                struct iovec iov = { dst + done, (size_t)(n - done) };
                ssize_t r = do_preadv(fd, &iov, 1, off + done);
                if (r < 0) { err = 1; break; }
                if (r == 0) break;
                done += r;
            }
            pthread_mutex_lock(&mu);
            sc[SC_SPEC_CHUNKS] += 1;
            sc[SC_SPEC_BYTES] += done;
            if (err) { s->landed = 0; sc[SC_FAILED] += 1; spec_end(s, SP_FAILED); break; }
            s->got += done;
            int64_t payload = s->aligned + s->got - s->base;
            if (payload > s->len) payload = s->len;               /* EXL3_LANE_PORT */
            s->landed = payload > 0 ? payload : 0;
            if (done < n) {                                   /* EOF: what landed is all there is */
                if (s->landed > 0) { sc[SC_LANDED] += 1; spec_end(s, SP_LANDED); }
                else { sc[SC_FAILED] += 1; spec_end(s, SP_FAILED); }
                break;
            }
            pthread_cond_broadcast(&land_cv);
        }
    }
    pthread_mutex_unlock(&mu);
    return 0;
}

/* Configure the speculative class BEFORE q3ld_start.  bufs: nslots page-aligned buffers of slot_bytes;
 * rec_len: the record length; chunk: bytes per speculative preadv (page multiple); counters: int64[SC_N]
 * (Python-owned).  nthreads 0 = no speculative class.  Returns 0 or -1. */
int q3ld_spec_config(int32_t nthreads, const uint64_t *bufs, int32_t nslots, int64_t slot_bytes, int64_t rec_len,
                     int64_t chunk, int64_t *counters) {
    pthread_mutex_lock(&mu);
    if (running || nthreads < 0 || nthreads > MAX_SPEC_THREADS || nslots < 0 || nslots > MAX_SPEC
        || (nthreads > 0 && (nslots < 1 || rec_len <= 0 || slot_bytes < rec_len || chunk <= 0 || !counters))) {
        pthread_mutex_unlock(&mu);
        return -1;
    }
    nspec_threads = nthreads;
    nspec = nthreads ? nslots : 0;
    spec_slot_bytes = slot_bytes;
    spec_rec_len = rec_len;
    spec_chunk = chunk;
    sc = counters;
    for (int i = 0; i < MAX_SPEC; i++) {
        memset(&spec[i], 0, sizeof(spec_t));
        spec[i].fd = -1;
        spec[i].state = SP_FREE;
        if (i < nspec) spec[i].buf = (char *)(uintptr_t)bufs[i];
    }
    spec_order = 0;
    spec_inflight = 0;
    spec_configured = 1;
    pthread_mutex_unlock(&mu);
    return 0;
}

/* The demand-busy limit of the queue rule for speculative threads >= 1 (thread 0 keeps SPEC_MAX_DEMAND_BUSY), set
 * on a STOPPED pool after q3ld_spec_config.  0 = start / continue an unclaimed chunk only while no demand job executes.
 * Returns 0 or -1. */
int q3ld_spec_streams(int32_t idle_busy) {
    pthread_mutex_lock(&mu);
    if (running || idle_busy < 0 || idle_busy > SPEC_MAX_DEMAND_BUSY) { pthread_mutex_unlock(&mu); return -1; }
    spec_idle_busy = idle_busy;
    pthread_mutex_unlock(&mu);
    return 0;
}

/* Start the pool.  staging_ptrs: nw page-aligned buffers of sbytes.  Returns 0 or -errno / -1 / -2. */
int q3ld_start(int32_t nw, const uint64_t *staging_ptrs, int64_t sbytes, int64_t psize,
               int64_t *res_arr, int64_t n_tickets, int64_t *log_arr, int64_t n_log, int64_t *gauge_arr) {
    if (tb_denom == 0 && q3ld_init() != 0) return -1;
    pthread_mutex_lock(&mu);
    if (running || !spec_configured || nw < 1 || nw > MAX_WORKERS || psize <= 0 || sbytes % psize
        || (nspec && (spec_slot_bytes % psize || spec_chunk % psize))) { pthread_mutex_unlock(&mu); return -1; }
    for (int i = 0; i < nspec; i++) {
        if (((uintptr_t)spec[i].buf) % (uintptr_t)psize) { pthread_mutex_unlock(&mu); return -1; }
    }
    ranges = (range_t *)calloc((size_t)n_tickets, sizeof(range_t));
    queue = (job_t *)calloc((size_t)n_tickets, sizeof(job_t));
    pre_of = (int32_t *)calloc((size_t)n_tickets, sizeof(int32_t));
    sub_len = (int32_t *)calloc((size_t)n_tickets, sizeof(int32_t));
    gate_of = (int64_t *)calloc((size_t)n_tickets, sizeof(int64_t));
    if (!ranges || !queue || !pre_of || !sub_len || !gate_of) {
        free(ranges); free(queue); free(pre_of); free(sub_len); free(gate_of);
        ranges = 0; queue = 0; pre_of = 0; sub_len = 0; gate_of = 0;
        pthread_mutex_unlock(&mu);
        return -ENOMEM;
    }
    for (int i = 0; i < MAX_PRE; i++) { memset(&pre[i], 0, sizeof(pre_t)); pre[i].fd = -1; }
    pre_on = 0; pre_order = 0;
    memset(gates, 0, sizeof(gates));                  /* EVENT: armed later by q3ld_ev_config */
    g_head = g_tail = 0; ev_hi = ev_target = ev_sent = 0; ev_target_t = 0; ev_kind = EVK_OFF; ev_obj = 0;
    wd_stop = 0; wd_running = 0;
    qcap = n_tickets; qhead = 0; __atomic_store_n(&qlen, 0, __ATOMIC_RELAXED);
    res = res_arr; nt = n_tickets; logbuf = log_arr; logn = n_log; gauge = gauge_arr;
    staging_bytes = sbytes; page_size = psize;
    __atomic_store_n(&stopping, 0, __ATOMIC_RELAXED); busy = 0; seq = 0;
    for (int i = 0; i < nw; i++) staging[i] = (char *)(uintptr_t)staging_ptrs[i];
    nworkers = 0;
    for (int i = 0; i < nw; i++) {
        if (pthread_create(&threads[i], 0, worker, (void *)(intptr_t)i) != 0) break;
        nworkers++;
    }
    nspec_started = 0;
    for (int i = 0; i < nspec_threads; i++) {
        if (pthread_create(&spec_threads[i], 0, spec_worker, (void *)(intptr_t)i) != 0) break;
        nspec_started++;
    }
    running = 1;
    pthread_mutex_unlock(&mu);
    if (nworkers != nw || nspec_started != nspec_threads) return -2;
    return 0;
}

/* SCATTER (mlx-serve, ABI 2026102401): a range whose parts live apart (an MLX pack keeps an expert's weight, scales and
 * biases in separate tensors, possibly in separate shards).  Each part runs the stock range loop on its own file and
 * offset; the range's status is the first failing part's, its counters the parts' sums. */
static int run_range_scatter(const job_t *job, const range_t *rg, char *stage, int64_t *out) {
    int64_t calls = 0, returned = 0, total = 0, t_start = 0;
    int status = ST_OK;
    out[4] = 0;
    for (int32_t d = 0; d < rg->ndst && status == ST_OK; d++) {
        job_t part = { rg->fds[d], rg->sizes[d], job->deadline, job->ticket };
        range_t one = { 0 };
        one.offset = rg->offs[d];
        one.ndst = 1;
        one.dst[0] = rg->dst[d];
        one.len[0] = rg->len[d];
        int64_t o[RES_W] = { 0 };
        status = run_range(&part, &one, stage, o);
        calls += o[1]; returned += o[2]; total += o[3];
        if (o[4]) out[4] = o[4];
        if (!t_start) t_start = o[5];
    }
    out[1] = calls;
    out[2] = returned;
    out[3] = total;
    out[5] = t_start ? t_start : q3ld_monotonic_ns();
    out[6] = q3ld_monotonic_ns();
    return status;
}

/* One job (a stock fill unit) -- q3_nativeissue.c's q3ni_submit, plus the claim of the speculative records
 * containing its ranges.  Returns 0; -1 bad args / not running; -2 a ticket is still pending; -3 queue full. */
int q3ld_submit(int32_t fd, int64_t file_size, int64_t deadline, int32_t n, int32_t ngu, int32_t ndown,
                const int64_t *offsets, const uint64_t *const *rows, const int64_t *lens, int64_t first) {
    if (n < 1 || n > MAX_ITEMS || ngu < 1 || ndown < 1 || ngu > MAX_COMP || ndown > MAX_COMP) return -1;
    int32_t count = 2 * n;
    if (first < 0 || first + count > nt) return -1;
    pthread_mutex_lock(&mu);
    if (!running || stopping) { pthread_mutex_unlock(&mu); return -1; }
    for (int32_t i = 0; i < count; i++) {
        if (res[(first + i) * RES_W] == ST_PENDING) { pthread_mutex_unlock(&mu); return -2; }
    }
    if (qlen + count > qcap) { pthread_mutex_unlock(&mu); return -3; }
    for (int32_t i = 0; i < count; i++) {
        int down = i >= n;
        int32_t item = down ? i - n : i;
        range_t *rg = &ranges[first + i];
        rg->offset = offsets[i];
        rg->scatter = 0;
        rg->ndst = down ? ndown : ngu;
        for (int32_t c = 0; c < rg->ndst; c++) {
            int32_t k = down ? ngu + c : c;
            rg->dst[c] = rows[item][k];
            rg->len[c] = lens[k];
        }
        int64_t *out = res + (first + i) * RES_W;
        for (int k = 1; k < RES_W; k++) out[k] = 0;
        out[0] = ST_PENDING;
        pre_of[first + i] = 0;
        sub_len[first + i] = i == 0 ? count : 0;       /* WITHDRAW: the submit's range jobs, on its first ticket */
        gate_of[first + i] = 0;                        /* EVENT: a gate maps a ticket only after this submit */
    }
    if (pre_on && deadline < 0) {                  /* PRE: bind (in flight / waiting / done) or cancel (queued) */
        int woke = 0;
        for (int32_t i = 0; i < count; i++) {
            range_t *rg = &ranges[first + i];
            int p = pre_find(fd, rg);
            if (p < 0) continue;
            pre_t *pr = &pre[p];
            if (pr->state == PR_QUEUED) {
                pr->state = PR_FREE;
                sc[SC_PRE_CANCELLED] += 1;
                continue;
            }
            for (int32_t c = 0; c < rg->ndst; c++) pr->rg.dst[c] = rg->dst[c];
            pr->bound = 1;
            pre_of[first + i] = p + 1;
            sc[SC_PRE_BOUND] += 1;
            woke = 1;
        }
        if (woke) pthread_cond_broadcast(&pre_cv);
    }
    if (nspec) claim_ranges(fd, first, count);
    for (int32_t i = 0; i < count; i++) {          /* a job per range, in ticket order: every gate/up, then every down */
        job_t *job = &queue[(qhead + qlen) % qcap];
        job->fd = fd; job->file_size = file_size; job->deadline = deadline; job->ticket = first + i;
        __atomic_store_n(&qlen, qlen + 1, __ATOMIC_RELAXED);
    }
    EV(1, first);
    pthread_cond_broadcast(&work_cv);
    pthread_mutex_unlock(&mu);
    return 0;
}

/* SCATTER: q3ld_submit's 2n range jobs (every gate/up range, then every down range) whose part c of record i is read
 * from fds[i * (ngu + ndown) + c] (file size sizes[...]) at offsets[...].  Never claims a speculative record or binds a
 * pre-range: those are keyed by one file's offsets.  Returns q3ld_submit's codes. */
int q3ld_submit_scatter(int32_t n, int32_t ngu, int32_t ndown, const int32_t *fds, const int64_t *sizes,
                        const int64_t *offsets, const uint64_t *const *rows, const int64_t *lens, int64_t first) {
    if (n < 1 || n > MAX_ITEMS || ngu < 1 || ndown < 1 || ngu > MAX_COMP || ndown > MAX_COMP) return -1;
    int32_t count = 2 * n, nc = ngu + ndown;
    if (first < 0 || first + count > nt) return -1;
    pthread_mutex_lock(&mu);
    if (!running || stopping) { pthread_mutex_unlock(&mu); return -1; }
    for (int32_t i = 0; i < count; i++) {
        if (res[(first + i) * RES_W] == ST_PENDING) { pthread_mutex_unlock(&mu); return -2; }
    }
    if (qlen + count > qcap) { pthread_mutex_unlock(&mu); return -3; }
    for (int32_t i = 0; i < count; i++) {
        int down = i >= n;
        int32_t item = down ? i - n : i;
        range_t *rg = &ranges[first + i];
        rg->offset = 0;
        rg->scatter = 1;
        rg->ndst = down ? ndown : ngu;
        for (int32_t c = 0; c < rg->ndst; c++) {
            int32_t k = down ? ngu + c : c;
            rg->dst[c] = rows[item][k];
            rg->len[c] = lens[k];
            rg->fds[c] = fds[item * nc + k];
            rg->sizes[c] = sizes[item * nc + k];
            rg->offs[c] = offsets[item * nc + k];
        }
        int64_t *out = res + (first + i) * RES_W;
        for (int k = 1; k < RES_W; k++) out[k] = 0;
        out[0] = ST_PENDING;
        pre_of[first + i] = 0;
        sub_len[first + i] = i == 0 ? count : 0;
        gate_of[first + i] = 0;
    }
    for (int32_t i = 0; i < count; i++) {
        job_t *job = &queue[(qhead + qlen) % qcap];
        job->fd = -1; job->file_size = 0; job->deadline = -1; job->ticket = first + i;
        __atomic_store_n(&qlen, qlen + 1, __ATOMIC_RELAXED);
    }
    EV(1, first);
    pthread_cond_broadcast(&work_cv);
    pthread_mutex_unlock(&mu);
    return 0;
}

/* WITHDRAW: take the submit whose first ticket is `first` back out of the FIFO while no worker has started it: its
 * sub_len[first] range jobs, adjacent in the queue in ticket order (one submit queues them under the pool mutex), leave
 * it together, the jobs behind them keeping their order; its tickets get ST_WITHDRAWN words (no preadv, payload 0, no
 * row written) and are published in ticket order.  The FIFO takes a submit's ranges in ticket order, so its first
 * range still queued means none of it was taken.  Returns 1 withdrawn; 0 not queued (a worker started it and every
 * range of it runs as submitted, it is done, or no submit starts at `first`); -1 not running / bad ticket; -2 refused
 * (a ticket bound to a pre-range or waited on by a live event gate: its bytes are promised elsewhere). */
int q3ld_withdraw(int64_t first) {
    if (first < 0) return -1;
    pthread_mutex_lock(&mu);
    if (!running || first >= nt) { pthread_mutex_unlock(&mu); return -1; }
    const int32_t count = sub_len[first];
    if (count < 1) { pthread_mutex_unlock(&mu); return 0; }
    int64_t k = 0;
    while (k < qlen && queue[(qhead + k) % qcap].ticket != first) k++;
    if (k == qlen) { pthread_mutex_unlock(&mu); return 0; }
    for (int32_t i = 0; i < count; i++) {           /* the submit's ranges, adjacent and in ticket order */
        if (k + i >= qlen || queue[(qhead + k + i) % qcap].ticket != first + i) { pthread_mutex_unlock(&mu); return 0; }
        int64_t t = first + i;
        if (pre_of[t] || (gate_of[t] && gate_live(gate_of[t] - 1))) { pthread_mutex_unlock(&mu); return -2; }
    }
    for (int64_t m = k; m + count < qlen; m++) queue[(qhead + m) % qcap] = queue[(qhead + m + count) % qcap];
    __atomic_store_n(&qlen, qlen - count, __ATOMIC_RELAXED);
    sub_len[first] = 0;
    int64_t now = q3ld_monotonic_ns();
    for (int32_t i = 0; i < count; i++) {
        int64_t *out = res + (first + i) * RES_W;
        out[1] = out[2] = out[3] = out[4] = 0;
        out[5] = out[6] = now;
        out[7] = -1;
        out[0] = ST_WITHDRAWN;
        publish(first + i);
    }
    EV(11, first);
    pthread_cond_broadcast(&spec_cv);          /* the demand queue got shallower: the speculative queue rule may pass */
    pthread_mutex_unlock(&mu);
    return 1;
}

/* settle every unclaimed record with tag <= cur: QUEUED dropped, parked dropped, running abandoned at the next
 * chunk boundary (caller holds mu).  Returns 1 when a running record was flagged. */
static int settle(int64_t cur) {
    int wake = pre_on ? pre_expire(cur) : 0;
    for (int i = 0; i < nspec; i++) {
        spec_t *s = &spec[i];
        if (s->tag > cur || s->claimed) continue;
        if (s->state == SP_QUEUED) { s->state = SP_FREE; sc[SC_EXPIRED] += 1; }
        else if (s->state == SP_INFLIGHT && !s->running) spec_drop_parked(s);
        else if (s->state == SP_INFLIGHT && !s->abandon) { s->abandon = 1; wake = 1; }
    }
    return wake;
}

/* issue n records (bases) with tag, class prio, issuing horizon origin (caller holds mu).  Returns queued. */
static int32_t issue(int32_t fd, int64_t file_size, int64_t cur, int64_t tag, int prio, int origin, int32_t n,
                     const int64_t *bases, int64_t len) {
    int32_t queued = 0;
    for (int32_t k = 0; k < n; k++) {
        int64_t b = bases[k];
        int hit = -1;
        for (int i = 0; i < nspec; i++) {
            if (spec_live(&spec[i]) && spec[i].fd == fd && spec[i].base == b) { hit = i; break; }
        }
        if (hit >= 0) {
            spec_t *s = &spec[hit];
            if (s->tag < tag) s->tag = tag;
            if (prio < s->prio) s->prio = prio;
            s->abandon = 0;
            sc[SC_REFRESHED] += 1;
            continue;
        }
        int slot = -1;
        for (int i = 0; i < nspec; i++) {
            if (spec[i].state == SP_FREE) { slot = i; break; }
        }
        if (slot < 0) {
            for (int i = 0; i < nspec; i++) {
                spec_t *s = &spec[i];
                if ((s->state == SP_LANDED || s->state == SP_FAILED) && s->readers == 0 && s->tag < cur
                    && (slot < 0 || s->order < spec[slot].order)) slot = i;
            }
        }
        if (slot < 0) { sc[SC_NOSLOT] += 1; continue; }
        spec_t *s = &spec[slot];
        if (s->state == SP_LANDED && s->adopted == 0) sc[SC_DISCARDED] += 1;
        s->state = SP_QUEUED;
        s->claimed = 0;
        s->abandon = 0;
        s->running = 0;
        s->prio = prio;
        s->origin = origin;
        s->fd = fd;
        s->file_size = file_size;
        s->tag = tag;
        s->order = spec_order++;
        s->base = b;
        s->len = len;                                   /* EXL3_LANE_PORT */
        s->aligned = b / page_size * page_size;
        s->span = s->got = s->landed = 0;
        s->adopted = 0;
        sc[SC_SUBMITTED] += 1;
        sc[SC_H_SUBMITTED + origin - 1] += 1;
        queued += 1;
    }
    return queued;
}

/* Horizon 1 (the h1 lane): one layer-call's step (PyDLL: GIL held, never blocks beyond the pool mutex), called
 * right AFTER that call's demand submits.  settle tag <= cur, then issue n records with tag cur + 1.
 * Returns the number newly queued, -1 when the class is off / the pool is not running. */
int32_t q3ld_spec_step(int32_t fd, int64_t file_size, int64_t cur, int32_t n, const int64_t *bases) {
    pthread_mutex_lock(&mu);
    if (!running || stopping || !nspec) { pthread_mutex_unlock(&mu); return -1; }
    int wake = settle(cur);
    int32_t queued = issue(fd, file_size, cur, cur + 1, 0, 1, n, bases, spec_rec_len);
    if (queued || wake) pthread_cond_broadcast(&spec_cv);
    pthread_mutex_unlock(&mu);
    return queued;
}

/* EXL3_LANE_PORT: q3ld_spec_step whose n records all carry len payload bytes (the target layer's record length,
 * 0 < len <= the configured rec_len).  Returns the number newly queued, -1 when off / not running / bad len. */
int32_t q3ld_spec_step_len(int32_t fd, int64_t file_size, int64_t cur, int32_t n, const int64_t *bases, int64_t len) {
    pthread_mutex_lock(&mu);
    if (!running || stopping || !nspec || len <= 0 || len > spec_rec_len) { pthread_mutex_unlock(&mu); return -1; }
    int wake = settle(cur);
    int32_t queued = issue(fd, file_size, cur, cur + 1, 0, 1, n, bases, len);
    if (queued || wake) pthread_cond_broadcast(&spec_cv);
    pthread_mutex_unlock(&mu);
    return queued;
}

/* q3ld_spec_step whose record k carries lens[k] payload bytes (0 < lens[k] <= the configured rec_len): one step over the
 * records of several layers of different record lengths (GLM-5.3's EXL3 bank: an expert of the next routed layer's
 * K3 or K4 bank layer).  Returns the number newly queued, -1 when off / not running / a bad length. */
int32_t q3ld_spec_step_lens(int32_t fd, int64_t file_size, int64_t cur, int32_t n, const int64_t *bases, const int64_t *lens) {
    pthread_mutex_lock(&mu);
    if (!running || stopping || !nspec) { pthread_mutex_unlock(&mu); return -1; }
    for (int32_t k = 0; k < n; k++) {
        if (lens[k] <= 0 || lens[k] > spec_rec_len) { pthread_mutex_unlock(&mu); return -1; }
    }
    int wake = settle(cur);
    int32_t queued = 0;
    for (int32_t k = 0; k < n; k++) queued += issue(fd, file_size, cur, cur + 1, 0, 1, 1, &bases[k], lens[k]);
    if (queued || wake) pthread_cond_broadcast(&spec_cv);
    pthread_mutex_unlock(&mu);
    return queued;
}

/* Horizon N: settle tag <= cur; re-validate every unclaimed record with cur < tag <= cur + nh whose class is
 * farther than tag - cur - 1 against val[tag - cur] (kept -> class tag - cur - 1, else dropped); then issue per
 * horizon h = 1..nh the records issue[h] with tag cur + h, class h - 1.  nval[h-1] / nissue[h-1] count the
 * concatenated val / issue lists.  Returns the number newly queued, -1 when off / not running. */
int32_t q3ld_spec_stepn(int32_t fd, int64_t file_size, int64_t cur, int32_t nh, const int32_t *nval,
                        const int64_t *val, const int32_t *nissue, const int64_t *iss) {
    if (nh < 0 || nh > MAX_H) return -1;
    pthread_mutex_lock(&mu);
    if (!running || stopping || !nspec) { pthread_mutex_unlock(&mu); return -1; }
    int wake = settle(cur);
    const int64_t *vh[MAX_H];
    const int64_t *ih[MAX_H];
    int64_t vo = 0, io = 0;
    for (int h = 0; h < nh; h++) {
        vh[h] = val + vo; vo += nval[h];
        ih[h] = iss + io; io += nissue[h];
    }
    for (int i = 0; i < nspec; i++) {
        spec_t *s = &spec[i];
        if (s->claimed || !(s->state == SP_QUEUED || s->state == SP_INFLIGHT)) continue;
        int64_t d = s->tag - cur;                 /* remaining distance: 1..nh */
        if (d < 1 || d > nh || s->prio <= d - 1) continue;
        int keep = 0;
        for (int32_t k = 0; k < nval[d - 1]; k++) {
            if (vh[d - 1][k] == s->base) { keep = 1; break; }
        }
        if (keep) {
            s->prio = (int)(d - 1);
            sc[SC_H_KEPT + hidx(s)] += 1;
            wake = 1;
            continue;
        }
        sc[SC_H_DROPPED + hidx(s)] += 1;
        if (s->state == SP_QUEUED) { s->state = SP_FREE; sc[SC_EXPIRED] += 1; }
        else if (!s->running) spec_drop_parked(s);
        else { s->abandon = 1; wake = 1; }
    }
    int32_t queued = 0;
    for (int h = 1; h <= nh; h++)
        queued += issue(fd, file_size, cur, cur + h, h - 1, h, nissue[h - 1], ih[h - 1], spec_rec_len);
    if (queued || wake) pthread_cond_broadcast(&spec_cv);
    pthread_mutex_unlock(&mu);
    return queued;
}

/* Snapshot (PyDLL): per slot state, tag, base, landed, readers, adopted, claimed, abandon, prio, running, origin
 * -> out[11 * MAX_SPEC]; returns nspec. */
int32_t q3ld_spec_state(int64_t *out) {
    pthread_mutex_lock(&mu);
    for (int i = 0; i < MAX_SPEC; i++) {
        spec_t *s = &spec[i];
        int64_t *o = out + 11 * i;
        o[0] = s->state; o[1] = s->tag; o[2] = s->base; o[3] = s->landed; o[4] = s->readers; o[5] = s->adopted;
        o[6] = s->claimed; o[7] = s->abandon; o[8] = s->prio; o[9] = s->running; o[10] = s->origin;
    }
    int32_t n = nspec;
    pthread_mutex_unlock(&mu);
    return n;
}

/* ACQUIRE load of the completion sequence (PyDLL: GIL held, no blocking). */
int64_t q3ld_seq(void) {
    return (int64_t)__atomic_load_n(&seq, __ATOMIC_ACQUIRE);
}

/* Block (CDLL: GIL released) until seq != seen or timeout_ns elapses; returns seq (ACQUIRE). */
int64_t q3ld_wait(int64_t seen, int64_t timeout_ns) {
    pthread_mutex_lock(&mu);
    if ((int64_t)seq == seen && timeout_ns > 0) {
        struct timespec dl;
        abs_realtime(&dl, timeout_ns);
        while ((int64_t)seq == seen) {
            if (pthread_cond_timedwait(&done_cv, &mu, &dl) == ETIMEDOUT) break;
        }
    }
    pthread_mutex_unlock(&mu);
    return (int64_t)__atomic_load_n(&seq, __ATOMIC_ACQUIRE);
}

/* Copy the gauge under the mutex (PyDLL). */
void q3ld_gauge(int64_t *out) {
    pthread_mutex_lock(&mu);
    for (int i = 0; i < 6; i++) out[i] = gauge[i];
    pthread_mutex_unlock(&mu);
}

/* Wait (CDLL) until the demand queue is empty, no demand worker is busy and no speculative read is running.
 * QUEUED / parked unclaimed records are dropped and running unclaimed ones abandoned first.  0 = quiescent,
 * 1 = timeout. */
int q3ld_quiesce(int64_t timeout_ns) {
    struct timespec dl;
    abs_realtime(&dl, timeout_ns);
    pthread_mutex_lock(&mu);
    if (pre_on) pre_expire(INT64_MAX);
    for (int i = 0; i < nspec; i++) {
        spec_t *s = &spec[i];
        if (s->state == SP_QUEUED) { s->state = SP_FREE; sc[SC_EXPIRED] += 1; }
        else if (s->state == SP_INFLIGHT && !s->claimed && !s->running) spec_drop_parked(s);
        else if (s->state == SP_INFLIGHT && !s->claimed) s->abandon = 1;
    }
    pthread_cond_broadcast(&spec_cv);
    int rc = 0;
    for (;;) {
        int parked_claimed = 0;
        for (int i = 0; i < nspec; i++) {
            if (spec[i].state == SP_INFLIGHT && !spec[i].running) parked_claimed = 1;
        }
        int pre_live = 0;
        for (int i = 0; i < MAX_PRE; i++) pre_live |= pre[i].state == PR_INFLIGHT || pre[i].state == PR_QUEUED;
        if (!(qlen || busy || spec_inflight || parked_claimed || pre_live)) break;
        if (pthread_cond_timedwait(&done_cv, &mu, &dl) == ETIMEDOUT) { rc = 1; break; }
    }
    pthread_mutex_unlock(&mu);
    return rc;
}

/* Stop and join every thread (CDLL), after draining the demand queue.  0 ok, -1 not running. */
int q3ld_stop(void) {
    pthread_mutex_lock(&mu);
    if (!running) { pthread_mutex_unlock(&mu); return -1; }
    __atomic_store_n(&stopping, 1, __ATOMIC_RELAXED);
    pthread_cond_broadcast(&work_cv);
    pthread_cond_broadcast(&spec_cv);
    pthread_cond_broadcast(&pre_cv);             /* PRE: a worker held at an unbound range's bind point drops it */
    pthread_mutex_unlock(&mu);
    for (int i = 0; i < nworkers; i++) pthread_join(threads[i], 0);
    for (int i = 0; i < nspec_started; i++) pthread_join(spec_threads[i], 0);
    /* EVENT: no ticket publishes any more.  Stop the watchdog, then release every live gate (no GPU wait outlives
     * the pool) and hand the final prefix to the event before the class is disarmed. */
    pthread_mutex_lock(&mu);
    wd_stop = 1;
    pthread_cond_broadcast(&wd_cv);
    pthread_mutex_unlock(&mu);
    if (wd_running) { pthread_join(wd_thread, 0); wd_running = 0; }
    pthread_mutex_lock(&mu);
    for (int64_t i = g_head; i < g_tail; i++) {
        gate_t *g = &gates[i % MAX_GATES];
        if (g->remaining > 0 && !g->forced) {
            g->forced = GF_STOP;
            g->t_sat = q3ld_monotonic_ns();
            if (sc) sc[SC_EV_STOP_RELEASED] += 1;
        }
    }
    ev_advance();
    pthread_mutex_unlock(&mu);
    ev_flush();
    pthread_mutex_lock(&sig_mu);
    __atomic_store_n(&ev_kind, EVK_OFF, __ATOMIC_RELEASE);
    ev_obj = 0;
    pthread_mutex_unlock(&sig_mu);
    pthread_mutex_lock(&mu);
    g_head = g_tail = 0;
    free(gate_of); gate_of = 0;
    for (int i = 0; i < nspec; i++) {
        if (spec[i].state == SP_LANDED && spec[i].adopted == 0) sc[SC_DISCARDED] += 1;
        if (spec[i].state == SP_INFLIGHT) { sc[SC_ABANDONED] += 1; sc[SC_ABANDONED_BYTES] += spec[i].got; }
        memset(&spec[i], 0, sizeof(spec_t));
        spec[i].fd = -1;
    }
    running = 0; nworkers = 0; nspec_started = 0; __atomic_store_n(&stopping, 0, __ATOMIC_RELAXED); nspec = 0; nspec_threads = 0; spec_configured = 0;
    spec_idle_busy = SPEC_MAX_DEMAND_BUSY;
    direct_reads = 0;
    run_range = run_range_staged;
    free(ranges); free(queue); free(pre_of); free(sub_len); ranges = 0; queue = 0; pre_of = 0; sub_len = 0;
    for (int i = 0; i < MAX_PRE; i++) { memset(&pre[i], 0, sizeof(pre_t)); pre[i].fd = -1; }
    pre_on = 0; pre_ngu = pre_ndown = 0; pre_gu_total = 0;
    res = 0; logbuf = 0; gauge = 0; sc = 0;
    pthread_mutex_unlock(&mu);
    return 0;
}

/* Arm (ngu, ndown >= 1) or disarm (0, 0) the pre-read class on a RUNNING pool with the speculative class configured
 * and no pre-range live.  lens: the ngu GU then ndown DOWN component lengths (the lane geometry).  0 or -1. */
int q3ld_pre_config(int32_t ngu, int32_t ndown, const int64_t *lens) {
    pthread_mutex_lock(&mu);
    int live = 0;
    for (int i = 0; i < MAX_PRE; i++) live |= pre[i].state != PR_FREE;
    int off = ngu == 0 && ndown == 0;
    if (!running || stopping || !sc || !nspec || live
        || (!off && (ngu < 1 || ndown < 1 || ngu > MAX_COMP || ndown > MAX_COMP || !lens))) {
        pthread_mutex_unlock(&mu);
        return -1;
    }
    if (!off) for (int32_t c = 0; c < ngu + ndown; c++) {   /* all or nothing: a refusal leaves the class as it was */
        if (lens[c] < 0) { pthread_mutex_unlock(&mu); return -1; }
    }
    pre_ngu = off ? 0 : ngu;
    pre_ndown = off ? 0 : ndown;
    pre_gu_total = 0;
    for (int32_t c = 0; c < pre_ngu + pre_ndown; c++) {
        pre_lens[c] = lens[c];
        if (c < pre_ngu) pre_gu_total += lens[c];
    }
    pre_on = !off;
    pthread_mutex_unlock(&mu);
    return 0;
}

/* queue one pre-range (caller holds mu); 1 queued, 0 not */
static int pre_issue(int32_t fd, int64_t file_size, int64_t tag, int64_t offset, int down, const int64_t *lens_all) {
    int32_t nd = down ? pre_ndown : pre_ngu;
    const int64_t *lens = lens_all + (down ? pre_ngu : 0);          /* EXL3_LANE_PORT: the call's lengths */
    int64_t total = 0;
    for (int32_t c = 0; c < nd; c++) total += lens[c];
    int64_t a0 = offset / page_size * page_size;
    int64_t a1 = (offset + total + page_size - 1) / page_size * page_size;
    if (a1 - a0 > staging_bytes) { sc[SC_PRE_NOSLOT] += 1; return 0; }   /* the stock loop would read it in steps */
    int slot = -1;
    for (int i = 0; i < MAX_PRE; i++) {
        pre_t *p = &pre[i];
        if (p->state != PR_FREE && !p->cancel && !p->bound && p->fd == fd && p->rg.offset == offset) {
            if (p->tag < tag) p->tag = tag;
            return 0;                                /* already queued / reading */
        }
        if (slot < 0 && p->state == PR_FREE) slot = i;
    }
    if (slot < 0) { sc[SC_PRE_NOSLOT] += 1; return 0; }
    pre_t *p = &pre[slot];
    memset(p, 0, sizeof(pre_t));
    p->state = PR_QUEUED;
    p->fd = fd;
    p->file_size = file_size;
    p->tag = tag;
    p->order = pre_order++;
    p->rg.offset = offset;
    p->rg.ndst = nd;
    for (int32_t c = 0; c < nd; c++) p->rg.len[c] = lens[c];
    sc[SC_PRE_ISSUED] += 1;
    return 1;
}

/* One layer-call's pre-read (PyDLL: GIL held, never blocks beyond the pool mutex), right after the router barrier
 * returns: bases = the call's certain-miss record offsets (route order), tag = the clock value of this call's settle.
 * Expires unbound pre-ranges of older calls; claims the live speculative records containing a base (a QUEUED one is
 * cancelled and pre-read instead); queues every remaining record's GU range, then every DOWN range.  Returns the
 * number of pre-ranges queued, -1 when the class is off / the pool is not running. */
static int32_t pre_read_with(int32_t fd, int64_t file_size, int64_t tag, int32_t n, const int64_t *bases,
                             const int64_t *call_lens) {
    pthread_mutex_lock(&mu);
    if (!running || stopping || !pre_on || n < 0) { pthread_mutex_unlock(&mu); return -1; }
    /* EXL3_LANE_PORT: the call's layer component lengths (NULL = the armed lane geometry: the stock entry point) */
    const int64_t *lens_all = call_lens ? call_lens : pre_lens;
    int64_t gu_total = pre_gu_total;
    if (call_lens) {
        gu_total = 0;
        for (int32_t c = 0; c < pre_ngu + pre_ndown; c++) {
            if (call_lens[c] < 0) { pthread_mutex_unlock(&mu); return -1; }
            if (c < pre_ngu) gu_total += call_lens[c];
        }
    }
    pre_expire(tag - 1);
    int64_t todo[2 * MAX_PRE];
    int32_t nt = 0, queued = 0;
    int woke_spec = 0;
    for (int32_t k = 0; k < n; k++) {
        int64_t b = bases[k];
        int s = nspec ? spec_containing(fd, b, gu_total) : -1;
        if (s >= 0) {
            spec_t *r = &spec[s];
            if (r->state != SP_QUEUED) {             /* claim_ranges' claim, made now */
                if (!r->claimed) {
                    r->claimed = 1;
                    r->abandon = 0;
                    sc[SC_CLAIMED] += 1;
                    sc[SC_H_CLAIMED + hidx(r)] += 1;
                    if (r->state == SP_INFLIGHT) { sc[SC_CLAIMED_INFLIGHT] += 1; woke_spec = 1; }
                }
                sc[SC_PRE_CLAIMS] += 1;
                continue;
            }
            r->state = SP_FREE;                      /* claim_ranges' cancel of a QUEUED record, made now */
            sc[SC_CANCELLED_BY_DEMAND] += 1;
        }
        if (nt < 2 * MAX_PRE) todo[nt++] = b;
    }
    for (int32_t k = 0; k < nt; k++) queued += pre_issue(fd, file_size, tag, todo[k], 0, lens_all);
    for (int32_t k = 0; k < nt; k++) queued += pre_issue(fd, file_size, tag, todo[k] + gu_total, 1, lens_all);
    if (n) sc[SC_PRE_CALLS] += 1;
    if (queued) pthread_cond_broadcast(&work_cv);
    if (woke_spec) pthread_cond_broadcast(&spec_cv);
    pthread_mutex_unlock(&mu);
    return queued;
}

/* One layer-call's pre-read with the armed lane geometry (the stock entry point; see pre_read_with). */
int32_t q3ld_pre_read(int32_t fd, int64_t file_size, int64_t tag, int32_t n, const int64_t *bases) {
    return pre_read_with(fd, file_size, tag, n, bases, 0);
}

/* EXL3_LANE_PORT: one layer-call's pre-read with that layer's component lengths (ngu then ndown). */
int32_t q3ld_pre_read_lens(int32_t fd, int64_t file_size, int64_t tag, int32_t n, const int64_t *bases,
                           const int64_t *lens) {
    if (!lens) return -1;
    return pre_read_with(fd, file_size, tag, n, bases, lens);
}

/* Snapshot (PyDLL): per pre-range entry state, bound, cancel, offset, tag -> out[5 * MAX_PRE]; returns pre_on. */
int32_t q3ld_pre_state(int64_t *out) {
    pthread_mutex_lock(&mu);
    for (int i = 0; i < MAX_PRE; i++) {
        int64_t *o = out + 5 * i;
        o[0] = pre[i].state; o[1] = pre[i].bound; o[2] = pre[i].cancel; o[3] = pre[i].rg.offset; o[4] = pre[i].tag;
    }
    int32_t on = pre_on;
    pthread_mutex_unlock(&mu);
    return on;
}

/* ---------------------------------------------------------------- the event-gate class: API (lookahead4) */
/* The watchdog: forces the oldest live gate once its deadline passed (a lost signal can never hang the GPU). */
static void *wd_main(void *arg) {
    (void)arg;
    pthread_mutex_lock(&mu);
    while (!wd_stop) {
        if (g_head >= g_tail) { pthread_cond_wait(&wd_cv, &mu); continue; }
        gate_t *g = &gates[g_head % MAX_GATES];
        int64_t now = q3ld_monotonic_ns();
        if (now < g->deadline) {
            struct timespec dl;
            abs_realtime(&dl, g->deadline - now);
            pthread_cond_timedwait(&wd_cv, &mu, &dl);
            continue;
        }
        g->forced = GF_WATCHDOG;
        g->t_sat = now;
        sc[SC_EV_WD_FORCED] += 1;
        sc[SC_EV_WD_LAST_VALUE] = (int64_t)g->value;
        ev_advance();
        pthread_mutex_unlock(&mu);
        ev_flush();
        pthread_mutex_lock(&mu);
    }
    pthread_mutex_unlock(&mu);
    return 0;
}

/* Arm the event-gate class on a RUNNING pool with counters (q3ld_spec_config) and no live gate: kind EVK_MTL (obj =
 * an MTLSharedEvent*, whose signaled value is `start`) or EVK_HOST (obj = an int64 word holding `start`); timeout_ns
 * = the watchdog deadline of every gate (1 ms .. 600 s).  Starts the watchdog thread.  kind EVK_OFF disarms (no live
 * gate).  0, -1 refused, -2 thread. */
int q3ld_ev_config(int32_t kind, uint64_t obj, int64_t timeout_ns, uint64_t start) {
    pthread_mutex_lock(&mu);
    if (!running || stopping || !sc || !gate_of || g_head != g_tail) { pthread_mutex_unlock(&mu); return -1; }
    if (kind == EVK_OFF) {
        if (ev_kind == EVK_OFF) { pthread_mutex_unlock(&mu); return 0; }
        wd_stop = 1;
        pthread_cond_broadcast(&wd_cv);
        pthread_mutex_unlock(&mu);
        if (wd_running) { pthread_join(wd_thread, 0); wd_running = 0; }
        pthread_mutex_lock(&sig_mu);
        __atomic_store_n(&ev_kind, EVK_OFF, __ATOMIC_RELEASE);
        ev_obj = 0;
        pthread_mutex_unlock(&sig_mu);
        return 0;
    }
    if ((kind != EVK_MTL && kind != EVK_HOST) || !obj || timeout_ns < 1000000 || timeout_ns > 600000000000LL
        || ev_kind != EVK_OFF || wd_running) {
        pthread_mutex_unlock(&mu);
        return -1;
    }
    pthread_mutex_lock(&sig_mu);
    ev_obj = (void *)(uintptr_t)obj;
    __atomic_store_n(&ev_sent, start, __ATOMIC_RELEASE);
    ev_hi = start;
    __atomic_store_n(&ev_target, start, __ATOMIC_RELEASE);
    __atomic_store_n(&ev_target_t, q3ld_monotonic_ns(), __ATOMIC_RELEASE);
    ev_timeout_ns = timeout_ns;
    g_head = g_tail = 0;
    memset(gates, 0, sizeof(gates));
    memset(gate_of, 0, (size_t)nt * sizeof(int64_t));
    __atomic_store_n(&ev_kind, kind, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&sig_mu);
    wd_stop = 0;
    int rc = pthread_create(&wd_thread, 0, wd_main, 0);
    if (rc == 0) wd_running = 1;
    else {
        pthread_mutex_lock(&sig_mu);
        __atomic_store_n(&ev_kind, EVK_OFF, __ATOMIC_RELEASE);
        ev_obj = 0;
        pthread_mutex_unlock(&sig_mu);
    }
    pthread_mutex_unlock(&mu);
    return rc == 0 ? 0 : -2;
}

/* Register n gates (PyDLL: GIL held, never blocks beyond the pool mutex), in the order the GPU meets their waits:
 * values strictly increasing and above every value registered / released so far; gate i waits for the counts[i]
 * tickets that follow in `tickets` (terminal ones are not waited for).  Satisfied gates at the head are handed to
 * the event before returning.  Returns n; -1 off / not running / bad n; -2 bad value / ticket (range, repeated,
 * already waited for by a live gate) -- nothing registered; -3 the gate ring is full. */
int32_t q3ld_ev_gates(int32_t n, const uint64_t *values, const int32_t *counts, const int64_t *tickets) {
    if (n < 1 || n > MAX_GATES) return -1;
    pthread_mutex_lock(&mu);
    if (!running || stopping || ev_kind == EVK_OFF || !gate_of) { pthread_mutex_unlock(&mu); return -1; }
    if (g_tail - g_head + n > MAX_GATES) { pthread_mutex_unlock(&mu); return -3; }
    uint64_t prev = ev_hi;
    int64_t total = 0;
    for (int32_t i = 0; i < n; i++) total += counts[i] < 0 ? MAX_GATE_TICKETS + 1 : counts[i];
    if (total > MAX_GATE_TICKETS) { pthread_mutex_unlock(&mu); return -2; }
    int64_t k = 0;
    for (int32_t i = 0; i < n; i++) {                   /* validate everything before any change */
        if (values[i] <= prev) { pthread_mutex_unlock(&mu); return -2; }
        prev = values[i];
        for (int32_t j = 0; j < counts[i]; j++, k++) {
            int64_t t = tickets[k];
            if (t < 0 || t >= nt || (gate_of[t] && gate_live(gate_of[t] - 1))) { pthread_mutex_unlock(&mu); return -2; }
            for (int64_t m = 0; m < k; m++) {
                if (tickets[m] == t) { pthread_mutex_unlock(&mu); return -2; }
            }
        }
    }
    int64_t now = q3ld_monotonic_ns();
    k = 0;
    for (int32_t i = 0; i < n; i++) {
        int64_t idx = g_tail++;
        gate_t *g = &gates[idx % MAX_GATES];
        memset(g, 0, sizeof(gate_t));
        g->serial = idx;
        g->value = values[i];
        g->t_reg = now;
        g->deadline = now + ev_timeout_ns;
        for (int32_t j = 0; j < counts[i]; j++, k++) {
            int64_t t = tickets[k];
            if (res[t * RES_W] == ST_PENDING) { gate_of[t] = idx + 1; g->remaining += 1; }
        }
        sc[SC_EV_TICKETS] += g->remaining;
        if (!g->remaining) { g->t_sat = now; sc[SC_EV_IMMEDIATE] += 1; }
    }
    ev_hi = values[n - 1];
    sc[SC_EV_CALLS] += 1;
    sc[SC_EV_GATES] += n;
    if (g_tail - g_head > sc[SC_EV_MAX_LIVE]) sc[SC_EV_MAX_LIVE] = g_tail - g_head;
    ev_advance();
    pthread_cond_signal(&wd_cv);
    pthread_mutex_unlock(&mu);
    ev_flush();
    return n;
}

/* Host release (error paths; PyDLL): force every live gate with value <= `value`; with no live gate left, a value
 * that was allocated but never registered is handed to the event as well.  Returns the gates forced, -1 when off. */
int32_t q3ld_ev_release(uint64_t value) {
    pthread_mutex_lock(&mu);
    if (!running || ev_kind == EVK_OFF) { pthread_mutex_unlock(&mu); return -1; }
    int32_t n = 0;
    int64_t now = q3ld_monotonic_ns();
    for (int64_t i = g_head; i < g_tail; i++) {
        gate_t *g = &gates[i % MAX_GATES];
        if (g->value > value) break;
        if (g->remaining > 0 && !g->forced) { g->forced = GF_HOST; g->t_sat = now; n += 1; }
    }
    sc[SC_EV_HOST_RELEASED] += n;
    ev_advance();
    if (g_head == g_tail && value > ev_target) {
        __atomic_store_n(&ev_target_t, now, __ATOMIC_RELEASE);
        __atomic_store_n(&ev_target, value, __ATOMIC_RELEASE);
    }
    if (value > ev_hi) ev_hi = value;
    pthread_mutex_unlock(&mu);
    ev_flush();
    return n;
}

/* Snapshot (PyDLL) -> out[10]: kind, live gates, head value (0 = none), head remaining, head forced, highest value
 * registered, satisfied prefix, value handed to the event, watchdog running, timeout ns.  Returns the live count. */
int32_t q3ld_ev_state(int64_t *out) {
    pthread_mutex_lock(&mu);
    int32_t live = (int32_t)(g_tail - g_head);
    gate_t *g = live ? &gates[g_head % MAX_GATES] : 0;
    out[0] = ev_kind; out[1] = live; out[2] = g ? (int64_t)g->value : 0; out[3] = g ? g->remaining : 0;
    out[4] = g ? g->forced : 0; out[5] = (int64_t)ev_hi; out[6] = (int64_t)__atomic_load_n(&ev_target, __ATOMIC_ACQUIRE);
    out[7] = (int64_t)__atomic_load_n(&ev_sent, __ATOMIC_ACQUIRE); out[8] = wd_running; out[9] = ev_timeout_ns;
    pthread_mutex_unlock(&mu);
    return live;
}

#if defined(Q3LD_INJECT) || defined(Q3LD_EVSIG)
/* Python-owned int64[4 * cap] signal log (value, backend call start, prefix-advance time, backend ns); buf = 0 keeps
 * the current log, cap < 0 detaches it (the owner frees the buffer).  Returns the entries written so far. */
int64_t q3ld_test_ev_log(int64_t *buf, int64_t cap) {
    pthread_mutex_lock(&sig_mu);
    if (buf) { evsig = buf; evsig_cap = cap; evsig_n = 0; }
    int64_t n = evsig_n;
    if (!buf && cap < 0) { evsig = 0; evsig_cap = 0; evsig_n = 0; }
    pthread_mutex_unlock(&sig_mu);
    return n;
}
#endif

int32_t q3ld_abi(void) { return 2026102401; }              /* SCATTER ranges; 2026101401 DIRECT as a construction route; 2026101201 WITHDRAW over range jobs; 2026101101 range jobs, 2026101001 WITHDRAW, EXL3_LANE_PORT 2026092704, original 2026092504 */
int32_t q3ld_max_gates(void) { return MAX_GATES; }
int32_t q3ld_max_gate_tickets(void) { return MAX_GATE_TICKETS; }
int32_t q3ld_max_pre(void) { return MAX_PRE; }
int32_t q3ld_spec_idle_busy(void) { return spec_idle_busy; }
int32_t q3ld_counters_n(void) { return SC_N; }
int32_t q3ld_max_spec(void) { return MAX_SPEC; }
int32_t q3ld_max_h(void) { return MAX_H; }
