// Linux stubs for the ANE prefill-MLP offload C ABI (lib/ane/ane_mlp.h).
//
// The real implementations are ARC Objective-C against the private Apple
// Neural Engine framework — macOS-only. Linux compiles this file instead
// (see build.zig addLinuxServe); every entry point reports "unavailable"
// so ane.zig's `available()` gate is false and no other path is reached.
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct MsvAneMlp MsvAneMlp;
typedef struct MsvAneBank MsvAneBank;
typedef struct MsvAnePlane MsvAnePlane;

int msv_ane_available(void) { return 0; }
uint64_t msv_ane_internal_free_disk(void) { return 0; }
void msv_ane_cache_lineage(const char *group, const char *variant) {
    (void)group;
    (void)variant;
}
void msv_ane_cache_variant(const char *group, char *out, int out_len) {
    (void)group;
    (void)out;
    (void)out_len;
}

MsvAnePlane *msv_ane_plane_create(size_t bytes) {
    (void)bytes;
    return NULL;
}
void msv_ane_plane_free(MsvAnePlane *p) { (void)p; }
uint16_t *msv_ane_plane_base(MsvAnePlane *p) {
    (void)p;
    return NULL;
}

MsvAneBank *msv_ane_bank_create(void) { return NULL; }
void msv_ane_bank_free(MsvAneBank *b) { (void)b; }
uint32_t msv_ane_bank_count(const MsvAneBank *b) {
    (void)b;
    return 0;
}
uint64_t msv_ane_bank_bytes(const MsvAneBank *b) {
    (void)b;
    return 0;
}
int msv_ane_bank_add_mlp(MsvAneBank *b, uint32_t hidden, uint32_t ffn,
                         uint32_t rows, const int8_t *gate_q,
                         const float *gate_s, const int8_t *up_q,
                         const float *up_s, const int8_t *down_q,
                         const float *down_s, char *err, size_t err_size) {
    (void)b; (void)hidden; (void)ffn; (void)rows; (void)gate_q; (void)gate_s;
    (void)up_q; (void)up_s; (void)down_q; (void)down_s;
    if (err != NULL && err_size > 0)
        snprintf(err, err_size, "ANE unavailable on Linux");
    return -1;
}
int msv_ane_bank_add_gdn(MsvAneBank *b, uint32_t hidden, uint32_t qkv_out,
                         uint32_t z_out, uint32_t rows, const int8_t *qkv_q,
                         const float *qkv_s, const int8_t *z_q,
                         const float *z_s, char *err, size_t err_size) {
    (void)b; (void)hidden; (void)qkv_out; (void)z_out; (void)rows;
    (void)qkv_q; (void)qkv_s; (void)z_q; (void)z_s;
    if (err != NULL && err_size > 0)
        snprintf(err, err_size, "ANE unavailable on Linux");
    return -1;
}
MsvAneMlp *msv_ane_bank_finish(MsvAneBank *b, const char *name,
                               int ane_instance, MsvAnePlane *input_plane,
                               MsvAnePlane *output_plane, char *err,
                               size_t err_size) {
    (void)b; (void)name; (void)ane_instance; (void)input_plane;
    (void)output_plane;
    if (err != NULL && err_size > 0)
        snprintf(err, err_size, "ANE unavailable on Linux");
    return NULL;
}

void msv_ane_mlp_free(MsvAneMlp *m) { (void)m; }
uint16_t *msv_ane_mlp_input(MsvAneMlp *m) {
    (void)m;
    return NULL;
}
uint16_t *msv_ane_mlp_output(MsvAneMlp *m) {
    (void)m;
    return NULL;
}
int msv_ane_mlp_eval(MsvAneMlp *m, uint32_t procedure, char *err,
                     size_t err_size) {
    (void)m; (void)procedure;
    if (err != NULL && err_size > 0)
        snprintf(err, err_size, "ANE unavailable on Linux");
    return -1;
}
double msv_ane_mlp_compile_seconds(const MsvAneMlp *m) {
    (void)m;
    return 0.0;
}
int msv_ane_mlp_cache_hit(const MsvAneMlp *m) {
    (void)m;
    return 0;
}

// kv_disk_cache.zig's "what would a write actually get" probe (ANE bridge on
// macOS: purgeable space released on demand). 0 = "no better estimate", which
// makes volumeSpace() fall back to statfs' f_bavail — the correct Linux answer.
uint64_t msv_volume_free_for_use(const char *path) {
    (void)path;
    return 0;
}

// Avahi's libdns_sd compat layer (Linux Bonjour) does not implement
// DNSServiceGetAddrInfo, so lan.zig's peer address resolution gets an
// explicit NotSupported (-65537) instead of a link error. Peer BROWSE and
// RESOLVE still work through the real Avahi symbols.
int DNSServiceGetAddrInfo(void **ref, unsigned int flags, unsigned int interface,
                          unsigned int protocol, const char *hostname,
                          void *cb, void *ctx) {
    (void)flags; (void)interface; (void)protocol; (void)hostname; (void)cb; (void)ctx;
    if (ref != NULL)
        *ref = NULL;
    return -65537; /* kDNSServiceErr_NotSupported */
}
