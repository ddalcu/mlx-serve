#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static _Atomic uint64_t witness_stats;
static int output_fd = -1;

static int probe_fstatat(int fd, const char *path, struct stat *out, int flags) {
    int result = fstatat(fd, path, out, flags);
    if (strncmp(path, "cache-io-witness-", 17) == 0)
        atomic_fetch_add_explicit(&witness_stats, 1, memory_order_relaxed);
    return result;
}

static void report(int sig) {
    (void)sig;
    uint64_t value = atomic_load_explicit(&witness_stats, memory_order_relaxed);
    if (output_fd >= 0) (void)write(output_fd, &value, sizeof(value));
}

__attribute__((constructor)) static void initialize(void) {
    const char *fd = getenv("CACHE_IO_PROBE_FD");
    if (fd) {
        output_fd = atoi(fd);
        signal(SIGUSR2, report);
    }
}

__attribute__((used)) static struct {
    const void *replacement;
    const void *original;
} interpose __attribute__((section("__DATA,__interpose"))) = {
    (const void *)probe_fstatat, (const void *)fstatat
};
