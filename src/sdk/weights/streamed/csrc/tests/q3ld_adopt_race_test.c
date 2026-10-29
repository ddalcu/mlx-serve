/* q3ld_adopt_race_test.c: the read pool's adopt race (CPU only, no Metal; build and run standalone):
 *   xcrun clang -O1 -DQ3LD_INJECT -Icsrc -o /tmp/q3ld_adopt_race csrc/tests/q3ld_adopt_race_test.c csrc/q3_lookahead4_exl3.c -lpthread
 *   nice -n 19 /tmp/q3ld_adopt_race 30 /tmp/q3ld_adopt_race.bin      (creates a 160 MB pattern file; exit 0 = no torn rows)
 * Mode A: a demand miss served by CLAIMING an in-flight speculative record (the adopt memcpy stalled 30 ms mid-copy by an
 * mprotect fault) under settle / reissue pressure: its event gate must never pass before the rows hold the file bytes.
 * Mode B: a demand range adopting an UNCLAIMED landed record that is then abandoned and its staging slot reissued while
 * the copy runs. Before the fix (try_adopt claims the record under `mu`) the reissued preadv overwrote the buffer
 * mid-copy and the range published OK with torn rows (30 iterations: 6 bad); with it, 0. Exit 1 on any bad row. */
/* rowrace: does an event gate (EVK_HOST word) ever reach a ticket's value before that ticket's rows hold the file
 * bytes?  A: demand miss served by CLAIMING an in-flight speculative record (adopt memcpy).  B: the unclaimed-adopt
 * path with the record abandoned + its slot reissued during the copy (pool-level hazard probe). */
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
static char *slow_lo = 0, *slow_hi = 0; static volatile int stalls = 0;
static void on_segv(int sig, siginfo_t *si, void *u) { (void)sig; (void)u; char *a = (char *)si->si_addr;
  if (a >= slow_lo && a < slow_hi) { struct timespec t = { 0, 30000000 }; nanosleep(&t, 0); stalls++; mprotect(slow_lo, slow_hi - slow_lo, PROT_READ | PROT_WRITE); return; }
  signal(SIGSEGV, SIG_DFL); }
int q3ld_spec_config(int32_t, const uint64_t *, int32_t, int64_t, int64_t, int64_t, int64_t *);
int q3ld_start(int32_t, const uint64_t *, int64_t, int64_t, int64_t *, int64_t, int64_t *, int64_t, int64_t *);
int q3ld_submit(int32_t, int64_t, int64_t, int32_t, int32_t, int32_t, const int64_t *, const uint64_t *const *, const int64_t *, int64_t);
int32_t q3ld_spec_step(int32_t, int64_t, int64_t, int32_t, const int64_t *);
int q3ld_ev_config(int32_t, uint64_t, int64_t, uint64_t);
int32_t q3ld_ev_gates(int32_t, const uint64_t *, const int32_t *, const int64_t *);
int q3ld_stop(void);
int q3ld_quiesce(int64_t);
void q3ld_test_spec_delay(int64_t);
void q3ld_test_delay(uint64_t, int64_t);
int32_t q3ld_counters_n(void);
int32_t q3ld_spec_state(int64_t *);
void q3ld_mtl_signal(void *e, uint64_t v) { (void)e; (void)v; }

#define PG 16384
#define MB (1024 * 1024)
static uint8_t pat(int64_t o) { return (uint8_t)(((uint64_t)(o / PG) * 2654435761u) >> 13) ^ (uint8_t)(o * 7); }
static void *pa(size_t n) { void *p = 0; posix_memalign(&p, PG, n); return p; }
static void nsleep(int64_t ns) { struct timespec t = { ns / 1000000000, ns % 1000000000 }; nanosleep(&t, 0); }

static volatile int64_t word = 0;
typedef struct { uint64_t v; uint8_t *row; int64_t off, len; int bad; int seen; } chk_t;
static chk_t chk[2];
static volatile int stop_chk = 0;
static void *checker(void *a) {
    (void)a;
    while (!stop_chk) {
        int64_t w = __atomic_load_n(&word, __ATOMIC_ACQUIRE);
        for (int i = 0; i < 2; i++) {
            chk_t *c = &chk[i];
            if (c->v && !c->seen && (uint64_t)w >= c->v) {
                c->seen = 1;
                for (int64_t k = 0; k < c->len; k++) if (c->row[k] != pat(c->off + k)) { c->bad = 1; break; }
            }
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 200;
    const char *path = argc > 2 ? argv[2] : "/tmp/q3ld_adopt_race.bin";
    const int64_t fsz = 160 * (int64_t)MB;
    int fd = open(path, O_RDWR | O_CREAT, 0644);
    struct stat; off_t cur = lseek(fd, 0, SEEK_END);
    if (cur != fsz) {
        uint8_t *b = malloc(MB);
        for (int64_t o = 0; o < fsz; o += MB) { for (int k = 0; k < MB; k++) b[k] = pat(o + k); pwrite(fd, b, MB, o); }
        free(b);
    }
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = on_segv; sa.sa_flags = SA_SIGINFO; sigaction(SIGSEGV, &sa, 0); sigaction(SIGBUS, &sa, 0);
    int64_t *sc = calloc(q3ld_counters_n(), 8);
    const int64_t rec = 64 * MB, chunk = 4 * MB;
    uint64_t sb[1] = { (uint64_t)pa(rec) };            /* ONE speculative slot: any reissue reuses this buffer */
    uint64_t st[2] = { (uint64_t)pa(8 * MB), (uint64_t)pa(8 * MB) };
    const int64_t NT = 64;
    int64_t *res = calloc(NT * 8, 8), *log = calloc(NT, 8), *gauge = calloc(16, 8);
    int64_t gu_len = 24 * MB, dn_len = 24 * MB;          /* GU [b, b+24M) DOWN [b+24M, b+48M): inside one record */
    int64_t gu_len_b = PG, dn_len_b = 40 * MB;
    uint8_t *rg = pa(gu_len), *rd = pa(dn_len_b);
    uint64_t v = 0;
    int reissuedA = 0, badA = 0, badB = 0, adoptedA = 0, adoptedB = 0, reissued = 0, live_at_reissue = 0;
    for (int mode = 0; mode < 2; mode++) for (int it = 0; it < iters; it++) {
        uint64_t seed = 1469598103934665603ULL ^ (uint64_t)(it * 1000003 + mode);
        if (q3ld_spec_config(1, sb, 1, rec, rec, chunk, sc)) { puts("spec_config"); return 1; }
        if (q3ld_start(mode ? 1 : 2, st, 8 * MB, PG, res, NT, log, NT, gauge)) { puts("start"); return 1; }
        word = (int64_t)v;
        if (q3ld_ev_config(2, (uint64_t)&word, 10000000000LL, v)) { puts("ev_config"); return 1; }
        memset(rg, 0xA5, gu_len); memset(rd, 0x5A, dn_len_b);
        int64_t base = 0, other = 96 * (int64_t)MB;
        int64_t pre_adopt = sc[13];
        q3ld_test_spec_delay(mode ? 1500000 : 200000 + (int64_t)(seed % 1500000));
        memset(chk, 0, sizeof chk);
        int64_t gl0 = mode ? gu_len_b : gu_len, dl0 = mode ? dn_len_b : dn_len;
        chk[0] = (chk_t){ v + 1, rg, base, gl0, 0, 0 };
        chk[1] = (chk_t){ v + 2, rd, base + gl0, dl0, 0, 0 };
        stop_chk = 0;
        pthread_t th; pthread_create(&th, 0, checker, 0);
        int64_t gl = mode ? gu_len_b : gu_len, dl = mode ? dn_len_b : dn_len;
        int64_t offs[2] = { base, base + gl };
        int64_t lens[2] = { gl, dl };
        uint64_t row0[2] = { (uint64_t)rg, (uint64_t)rd };
        const uint64_t *rows[1] = { row0 };
        int64_t first = (it % 4) * 2 + 8 * mode;
        if (mode == 0) {
            /* A: record in flight, then the demand submit CLAIMS it; the worker adopts (memcpy from spec staging) */
            q3ld_spec_step(fd, fsz, 0, 1, &base);
            nsleep((int64_t)(seed % 3000000));
            slow_lo = (char *)rd + 12 * MB; slow_hi = (char *)rd + 24 * MB;   /* the adopt copy stalls 30 ms */
            mprotect(slow_lo, slow_hi - slow_lo, PROT_NONE);
            if (q3ld_submit(fd, fsz, -1, 1, 1, 1, offs, rows, lens, first)) { puts("submit"); return 1; }
            int64_t sub0 = sc[0];
            for (int k = 0; k < 400; k++) { q3ld_spec_step(fd, fsz, 1 + k, 1, &other); nsleep(100000); }  /* settle + reclaim pressure on the ONE slot */
            if (sc[0] != sub0) reissuedA++;
        } else {
            /* B: submit FIRST (no record -> no claim), both workers held by the range delay; the record issued after
             * the submit is unclaimed; once a worker holds it as a reader, settle abandons it and a reissue reuses the
             * slot (issue() takes a FREE slot without a readers check) while the copy runs */
            q3ld_test_delay(seed, 30000000);
            slow_lo = (char *)rd + 20 * MB; slow_hi = (char *)rd + 40 * MB;   /* the copy stalls 30 ms at the half */
            mprotect(slow_lo, slow_hi - slow_lo, PROT_NONE);
            if (q3ld_submit(fd, fsz, -1, 1, 1, 1, offs, rows, lens, first)) { puts("submit"); return 1; }
            q3ld_spec_step(fd, fsz, 0, 1, &base);              /* tag 1, unclaimed */
            int64_t so[16 * 11];
            int saw = 0;
            for (int k = 0; k < 40000 && !saw; k++) {
                q3ld_spec_state(so);
                if (so[4] > 0 && so[0] == 2) saw = 1;            /* INFLIGHT with a demand reader copying */
                else if (so[0] != 2 && so[0] != 1) break;
                else nsleep(5000);
            }
            if (saw) {
                q3ld_test_spec_delay(0);
                int64_t sub0 = sc[0];
                for (int k = 0; k < 4000 && sc[0] == sub0; k++) { q3ld_spec_step(fd, fsz, 1 + k, 1, &other); if (sc[0] == sub0) nsleep(2000); }
                q3ld_spec_state(so);
                if (sc[0] != sub0) { reissued++; if (so[2] == other) { /* slot now holds `other`: was the copy live? */
                    int64_t a0 = sc[13]; (void)a0; if (sc[13] == pre_adopt) live_at_reissue++; } }
            }
        }
        uint64_t vals[2] = { v + 1, v + 2 };
        int32_t cnt[2] = { 1, 1 };
        int64_t tk[2] = { first, first + 1 };
        int32_t rc = q3ld_ev_gates(2, vals, cnt, tk);
        if (rc != 2) { printf("ev_gates %d\n", rc); return 1; }
        while (__atomic_load_n(&word, __ATOMIC_ACQUIRE) < (int64_t)(v + 2)) nsleep(100000);
        nsleep(2000000);
        stop_chk = 1; pthread_join(th, 0);
        q3ld_test_delay(0, 0);
        if (slow_lo) { mprotect(slow_lo, slow_hi - slow_lo, PROT_READ | PROT_WRITE); slow_lo = slow_hi = 0; }
        q3ld_quiesce(5000000000LL);
        /* the rows after the run, independent of the gate timing */
        int final_bad = 0;
        for (int64_t k = 0; k < gl; k++) if (rg[k] != pat(base + k)) { final_bad = 1; break; }
        for (int64_t k = 0; k < dl; k++) if (rd[k] != pat(base + gl + k)) { final_bad = 1; break; }
        int adopted = sc[13] > pre_adopt;
        int bad = chk[0].bad || chk[1].bad || final_bad || res[first * 8] || res[(first + 1) * 8];
        if (mode == 0) { badA += bad; adoptedA += adopted; } else { badB += bad; adoptedB += adopted; }
        if (bad) printf("mode %c it %d: gate-early gu=%d down=%d final_bad=%d status=%lld/%lld adopted=%d\n", 'A' + mode, it,
                        chk[0].bad, chk[1].bad, final_bad, (long long)res[first * 8], (long long)res[(first + 1) * 8], adopted);
        q3ld_stop();
        v += 2;
    }
    printf("A: reissues into the slot during/after the claimed adopt %d\n", reissuedA);
    printf("A (claimed in-flight adopt): %d iters, %d adopted, %d bad\n", iters, adoptedA, badA);
    printf("B: copy stalls %d\n", stalls);
    printf("B: reissued into the slot %d, of which with the adopt copy still running %d\n", reissued, live_at_reissue);
    printf("B (unclaimed adopt + abandon/reissue): %d iters, %d adopted, %d bad\n", iters, adoptedB, badB);
    return (badA || badB) ? 1 : 0;
}
