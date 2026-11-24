/* Dynamic ibverbs shim for macOS Thunderbolt 5 RDMA (librdma.dylib).
 * Uses dlopen("librdma.dylib") so mlx-serve has zero dyld load-time dependency
 * and degrades cleanly to cache_transfer.rdma=false on hosts without TB5 RDMA.
 *
 * Transport design (verified on Apple Silicon Thunderbolt 5 rdma_en* hardware):
 * - macOS librdma.dylib supports IBV_QPT_UC (Unreliable Connected) two-sided
 *   SEND/RECV only (IBV_QPT_RC returns EOPNOTSUPP and ibv_reg_mr rkey is 0).
 * - Each HCA enforces max_mr_size (~15.6 MiB), max_mr (100), and < 4 MiB of
 *   simultaneously posted IBV_WR_RECV bytes per QP.
 * - Each QP owns one pre-registered ~4 MiB staging ring MR (8 x 512 KiB data
 *   slots + 16 KiB control region) created once at QP setup. Per-transfer
 *   ibv_reg_mr latency is 0 ms, arbitrary KV cache sizes (1 B to 100+ GiB) are
 *   supported, and credit-paced 512 KiB x 4 (2 MiB window) streaming saturates
 *   the Thunderbolt 5 link (~8.71 GiB/s) while remaining well below the 4 MiB
 *   posted-recv ceiling. End-to-end integrity is verified by mlx-serve's 64-bit
 *   peer_wire Wyhash checksum on import. */

#include "rdma_shim.h"

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#if defined(__APPLE__) && TARGET_OS_OSX && __has_include(<infiniband/verbs.h>)

#include <dlfcn.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MSV_CHUNK_BYTES   (512ULL * 1024ULL) /* 512 KiB */
#define MSV_DATA_SLOTS    8U                 /* 4 MiB ring (8 x 512 KiB) */
#define MSV_WINDOW        4U                 /* 2 MiB inflight (< 4 MiB TB5 RX cap) */
#define MSV_CREDIT_BATCH  2U                 /* 1 MiB credit refill cadence */
#define MSV_CTRL_RECV_N   8U
#define MSV_CTRL_SEND_N   8U
#define MSV_CTRL_STRIDE   64U

#define MSV_CTRL_MAGIC    0x4D535652U /* 'MSVR' */
#define MSV_CTRL_REQ      1U
#define MSV_CTRL_CRED     2U

#define WR_TAG_MASK       0xF000000000000000ULL
#define WR_IDX_MASK       0x0FFFFFFFFFFFFFFFULL
#define WR_TAG_CTRL_RECV  0x1000000000000000ULL
#define WR_TAG_CTRL_SEND  0x2000000000000000ULL
#define WR_TAG_DATA_RECV  0x3000000000000000ULL
#define WR_TAG_DATA_SEND  0x4000000000000000ULL

typedef struct {
    uint32_t magic;
    uint32_t kind;
    uint8_t tid[16];
    uint64_t total_bytes;
    uint32_t chunk_bytes;
    uint32_t credits;
    uint8_t pad[24];
} MsvCtrlMsg;

_Static_assert(sizeof(MsvCtrlMsg) == MSV_CTRL_STRIDE, "MsvCtrlMsg must be 64 bytes");

typedef struct {
    void *handle;
    int loaded;

    struct ibv_device **(*get_device_list)(int *num_devices);
    const char *(*get_device_name)(struct ibv_device *device);
    struct ibv_context *(*open_device)(struct ibv_device *device);
    void (*free_device_list)(struct ibv_device **list);
    int (*close_device)(struct ibv_context *context);

    struct ibv_pd *(*alloc_pd)(struct ibv_context *context);
    int (*dealloc_pd)(struct ibv_pd *pd);
    struct ibv_cq *(*create_cq)(struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vector);
    int (*destroy_cq)(struct ibv_cq *cq);
    struct ibv_qp *(*create_qp)(struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr);
    int (*destroy_qp)(struct ibv_qp *qp);

    int (*query_port)(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr);
    int (*query_gid)(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid);
    int (*modify_qp)(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
    struct ibv_mr *(*reg_mr)(struct ibv_pd *pd, void *addr, size_t length, int access);
    int (*dereg_mr)(struct ibv_mr *mr);
} IbvApi;

static IbvApi g_ibv;
static pthread_once_t g_ibv_once = PTHREAD_ONCE_INIT;

static void set_err(char *buf, size_t cap, const char *fmt, ...) {
    if (!buf || cap == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, cap, fmt, ap);
    va_end(ap);
}

static uint64_t mono_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)(ts.tv_nsec / 1000000L);
}

static inline uint64_t compute_deadline_ms(uint64_t timeout_ms) {
    if (timeout_ms == 0) return 0;
    uint64_t now = mono_ms();
    if (now == 0) return 0;
    return (UINT64_MAX - now < timeout_ms) ? UINT64_MAX : (now + timeout_ms);
}

static inline int deadline_exceeded(uint64_t deadline_ms) {
    if (deadline_ms == 0) return 0;
    uint64_t now = mono_ms();
    return now > 0 && now >= deadline_ms;
}

static inline int stop_requested(volatile int *stop_flag) {
    return stop_flag && __atomic_load_n(stop_flag, __ATOMIC_ACQUIRE) != 0;
}

static void load_ibv_once(void) {
    memset(&g_ibv, 0, sizeof(g_ibv));
    void *h = dlopen("librdma.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) return;

#define LOAD_SYM(sym, field)                                            \
    do {                                                                \
        void *p = dlsym(h, #sym);                                       \
        if (!p) {                                                       \
            dlclose(h);                                                 \
            memset(&g_ibv, 0, sizeof(g_ibv));                           \
            return;                                                     \
        }                                                               \
        memcpy(&g_ibv.field, &p, sizeof(p));                            \
    } while (0)

    LOAD_SYM(ibv_get_device_list, get_device_list);
    LOAD_SYM(ibv_get_device_name, get_device_name);
    LOAD_SYM(ibv_open_device, open_device);
    LOAD_SYM(ibv_free_device_list, free_device_list);
    LOAD_SYM(ibv_close_device, close_device);
    LOAD_SYM(ibv_alloc_pd, alloc_pd);
    LOAD_SYM(ibv_dealloc_pd, dealloc_pd);
    LOAD_SYM(ibv_create_cq, create_cq);
    LOAD_SYM(ibv_destroy_cq, destroy_cq);
    LOAD_SYM(ibv_create_qp, create_qp);
    LOAD_SYM(ibv_destroy_qp, destroy_qp);
    LOAD_SYM(ibv_query_port, query_port);
    LOAD_SYM(ibv_query_gid, query_gid);
    LOAD_SYM(ibv_modify_qp, modify_qp);
    LOAD_SYM(ibv_reg_mr, reg_mr);
    LOAD_SYM(ibv_dereg_mr, dereg_mr);
#undef LOAD_SYM

    g_ibv.handle = h;
    g_ibv.loaded = 1;
}

static int ensure_ibv(void) {
    pthread_once(&g_ibv_once, load_ibv_once);
    return g_ibv.loaded;
}

struct MsvRdmaDev {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    union ibv_gid gid;
    uint16_t lid;
    uint8_t gid_index;
    uint8_t active_mtu;
};

struct MsvRdmaQp {
    MsvRdmaDev *dev;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    uint8_t *ring_buf;
    size_t ring_bytes;
    struct ibv_mr *ring_mr;
    uint32_t local_psn;
    int is_sender;
    int ctrl_recvs_armed;
};

static inline uint8_t *qp_data_slot(MsvRdmaQp *qp, uint32_t slot) {
    return qp->ring_buf + (size_t)(slot % MSV_DATA_SLOTS) * (size_t)MSV_CHUNK_BYTES;
}

static inline MsvCtrlMsg *qp_ctrl_recv_slot(MsvRdmaQp *qp, uint32_t slot) {
    size_t base = (size_t)MSV_DATA_SLOTS * (size_t)MSV_CHUNK_BYTES;
    return (MsvCtrlMsg *)(qp->ring_buf + base + (size_t)(slot % MSV_CTRL_RECV_N) * MSV_CTRL_STRIDE);
}

static inline MsvCtrlMsg *qp_ctrl_send_slot(MsvRdmaQp *qp, uint32_t slot) {
    size_t base = (size_t)MSV_DATA_SLOTS * (size_t)MSV_CHUNK_BYTES
                + (size_t)MSV_CTRL_RECV_N * MSV_CTRL_STRIDE;
    return (MsvCtrlMsg *)(qp->ring_buf + base + (size_t)(slot % MSV_CTRL_SEND_N) * MSV_CTRL_STRIDE);
}

static int post_ctrl_recv(MsvRdmaQp *qp, uint32_t slot, char *err_buf, size_t err_cap) {
    struct ibv_sge sge;
    sge.addr = (uint64_t)(uintptr_t)qp_ctrl_recv_slot(qp, slot);
    sge.length = MSV_CTRL_STRIDE;
    sge.lkey = qp->ring_mr->lkey;

    struct ibv_recv_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = WR_TAG_CTRL_RECV | (uint64_t)slot;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    struct ibv_recv_wr *bad = NULL;
    int rc = ibv_post_recv(qp->qp, &wr, &bad);
    if (rc != 0) {
        set_err(err_buf, err_cap, "ibv_post_recv(ctrl slot=%u) failed (rc=%d)", slot, rc);
        return -1;
    }
    return 0;
}

static int post_data_recv(MsvRdmaQp *qp, uint32_t slot, uint32_t exact_len, char *err_buf, size_t err_cap) {
    struct ibv_sge sge;
    sge.addr = (uint64_t)(uintptr_t)qp_data_slot(qp, slot);
    sge.length = exact_len;
    sge.lkey = qp->ring_mr->lkey;

    struct ibv_recv_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = WR_TAG_DATA_RECV | (uint64_t)slot;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    struct ibv_recv_wr *bad = NULL;
    int rc = ibv_post_recv(qp->qp, &wr, &bad);
    if (rc != 0) {
        set_err(err_buf, err_cap, "ibv_post_recv(data slot=%u len=%u) failed (rc=%d)", slot, exact_len, rc);
        return -1;
    }
    return 0;
}

static int post_ctrl_send(MsvRdmaQp *qp, uint32_t slot, char *err_buf, size_t err_cap) {
    struct ibv_sge sge;
    sge.addr = (uint64_t)(uintptr_t)qp_ctrl_send_slot(qp, slot);
    sge.length = MSV_CTRL_STRIDE;
    sge.lkey = qp->ring_mr->lkey;

    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = WR_TAG_CTRL_SEND | (uint64_t)slot;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;

    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send(qp->qp, &wr, &bad);
    if (rc != 0) {
        set_err(err_buf, err_cap, "ibv_post_send(ctrl slot=%u) failed (rc=%d)", slot, rc);
        return -1;
    }
    return 0;
}

static int post_data_send(MsvRdmaQp *qp, uint32_t slot, uint32_t len, char *err_buf, size_t err_cap) {
    struct ibv_sge sge;
    sge.addr = (uint64_t)(uintptr_t)qp_data_slot(qp, slot);
    sge.length = len;
    sge.lkey = qp->ring_mr->lkey;

    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = WR_TAG_DATA_SEND | (uint64_t)slot;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;

    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send(qp->qp, &wr, &bad);
    if (rc != 0) {
        set_err(err_buf, err_cap, "ibv_post_send(data slot=%u len=%u) failed (rc=%d)", slot, len, rc);
        return -1;
    }
    return 0;
}

static int is_ipv4_mapped_gid(const union ibv_gid *gid) {
    for (int i = 0; i < 10; i++) {
        if (gid->raw[i] != 0) return 0;
    }
    return gid->raw[10] == 0xff && gid->raw[11] == 0xff;
}

int msv_rdma_available(void) {
    if (!ensure_ibv()) return 0;
    int count = 0;
    struct ibv_device **list = g_ibv.get_device_list(&count);
    if (!list) return 0;
    int found = 0;
    for (int i = 0; i < count; i++) {
        const char *name = g_ibv.get_device_name(list[i]);
        if (name && name[0] != '\0') {
            found = 1;
            break;
        }
    }
    g_ibv.free_device_list(list);
    return found;
}

int msv_rdma_list_devices(char out[][64], size_t cap) {
    if (!out || cap == 0) return 0;
    if (!ensure_ibv()) return 0;
    int count = 0;
    struct ibv_device **list = g_ibv.get_device_list(&count);
    if (!list) return 0;
    size_t written = 0;
    for (int i = 0; i < count && written < cap; i++) {
        const char *name = g_ibv.get_device_name(list[i]);
        if (name && name[0] != '\0' && strlen(name) < 64) {
            snprintf(out[written], 64, "%s", name);
            written++;
        }
    }
    g_ibv.free_device_list(list);
    return (int)written;
}

MsvRdmaDev *msv_rdma_dev_open(const char *dev_name, char *err_buf, size_t err_cap) {
    if (!dev_name || dev_name[0] == '\0') {
        set_err(err_buf, err_cap, "empty rdma device name");
        return NULL;
    }
    if (!ensure_ibv()) {
        set_err(err_buf, err_cap, "librdma.dylib unavailable");
        return NULL;
    }

    int count = 0;
    struct ibv_device **list = g_ibv.get_device_list(&count);
    if (!list || count <= 0) {
        if (list) g_ibv.free_device_list(list);
        set_err(err_buf, err_cap, "no rdma devices found");
        return NULL;
    }

    struct ibv_context *ctx = NULL;
    for (int i = 0; i < count; i++) {
        const char *name = g_ibv.get_device_name(list[i]);
        if (name && strcmp(name, dev_name) == 0) {
            ctx = g_ibv.open_device(list[i]);
            break;
        }
    }
    g_ibv.free_device_list(list);

    if (!ctx) {
        set_err(err_buf, err_cap, "could not open rdma device %s", dev_name);
        return NULL;
    }

    struct ibv_pd *pd = g_ibv.alloc_pd(ctx);
    if (!pd) {
        g_ibv.close_device(ctx);
        set_err(err_buf, err_cap, "ibv_alloc_pd failed for %s (errno=%d)", dev_name, errno);
        return NULL;
    }

    struct ibv_port_attr port_attr;
    memset(&port_attr, 0, sizeof(port_attr));
    if (g_ibv.query_port(ctx, 1, &port_attr) != 0) {
        g_ibv.dealloc_pd(pd);
        g_ibv.close_device(ctx);
        set_err(err_buf, err_cap, "ibv_query_port failed for %s (errno=%d)", dev_name, errno);
        return NULL;
    }

    union ibv_gid chosen_gid;
    memset(&chosen_gid, 0, sizeof(chosen_gid));
    int chosen_idx = -1;
    int max_gid = port_attr.gid_tbl_len > 0 ? port_attr.gid_tbl_len : 16;
    if (max_gid > 256) max_gid = 256;
    for (int i = 0; i < max_gid; i++) {
        union ibv_gid tmp;
        memset(&tmp, 0, sizeof(tmp));
        if (g_ibv.query_gid(ctx, 1, i, &tmp) == 0 && is_ipv4_mapped_gid(&tmp)) {
            chosen_gid = tmp;
            chosen_idx = i;
            break;
        }
    }
    if (chosen_idx < 0) {
        g_ibv.dealloc_pd(pd);
        g_ibv.close_device(ctx);
        set_err(err_buf, err_cap, "no IPv4-mapped GID on %s (assign IPv4 to interface)", dev_name);
        return NULL;
    }

    MsvRdmaDev *dev = (MsvRdmaDev *)calloc(1, sizeof(MsvRdmaDev));
    if (!dev) {
        g_ibv.dealloc_pd(pd);
        g_ibv.close_device(ctx);
        set_err(err_buf, err_cap, "out of memory allocating rdma dev");
        return NULL;
    }
    dev->ctx = ctx;
    dev->pd = pd;
    dev->gid = chosen_gid;
    dev->lid = port_attr.lid ? port_attr.lid : 1;
    dev->gid_index = (uint8_t)chosen_idx;
    dev->active_mtu = (port_attr.active_mtu >= IBV_MTU_256 && port_attr.active_mtu <= IBV_MTU_4096)
        ? (uint8_t)port_attr.active_mtu
        : (uint8_t)IBV_MTU_1024;
    return dev;
}

void msv_rdma_dev_close(MsvRdmaDev *dev) {
    if (!dev) return;
    if (dev->pd) g_ibv.dealloc_pd(dev->pd);
    if (dev->ctx) g_ibv.close_device(dev->ctx);
    free(dev);
}

MsvRdmaQp *msv_rdma_qp_create(
    MsvRdmaDev *dev,
    uint32_t psn,
    int is_sender,
    MsvRdmaEndpoint *local_out,
    char *err_buf,
    size_t err_cap
) {
    if (!dev || !local_out) {
        set_err(err_buf, err_cap, "invalid qp_create arguments");
        return NULL;
    }

    size_t ring_bytes = (size_t)MSV_DATA_SLOTS * (size_t)MSV_CHUNK_BYTES + 16384U;
    void *ring_mem = NULL;
    if (posix_memalign(&ring_mem, 16384, ring_bytes) != 0 || !ring_mem) {
        set_err(err_buf, err_cap, "failed to allocate %zu-byte RDMA staging ring", ring_bytes);
        return NULL;
    }
    memset(ring_mem, 0, ring_bytes);

    struct ibv_mr *ring_mr = g_ibv.reg_mr(
        dev->pd,
        ring_mem,
        ring_bytes,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE
    );
    if (!ring_mr) {
        int e = errno;
        free(ring_mem);
        set_err(err_buf, err_cap, "ibv_reg_mr(staging ring %zu B) failed (errno=%d)", ring_bytes, e);
        return NULL;
    }

    struct ibv_cq *cq = g_ibv.create_cq(dev->ctx, 256, NULL, NULL, 0);
    if (!cq) {
        int e = errno;
        g_ibv.dereg_mr(ring_mr);
        free(ring_mem);
        set_err(err_buf, err_cap, "ibv_create_cq failed (errno=%d)", e);
        return NULL;
    }

    struct ibv_qp_init_attr init_attr;
    memset(&init_attr, 0, sizeof(init_attr));
    init_attr.qp_context = dev->ctx;
    init_attr.send_cq = cq;
    init_attr.recv_cq = cq;
    init_attr.cap.max_send_wr = 64;
    init_attr.cap.max_recv_wr = 64;
    init_attr.cap.max_send_sge = 1;
    init_attr.cap.max_recv_sge = 1;
    init_attr.cap.max_inline_data = 0;
    init_attr.qp_type = IBV_QPT_UC;
    init_attr.sq_sig_all = 0;

    struct ibv_qp *qp = g_ibv.create_qp(dev->pd, &init_attr);
    if (!qp) {
        int e = errno;
        g_ibv.destroy_cq(cq);
        g_ibv.dereg_mr(ring_mr);
        free(ring_mem);
        set_err(err_buf, err_cap, "ibv_create_qp(UC) failed (errno=%d)", e);
        return NULL;
    }

    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE;
    int mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    int rc = g_ibv.modify_qp(qp, &attr, mask);
    if (rc != 0) {
        g_ibv.destroy_qp(qp);
        g_ibv.destroy_cq(cq);
        g_ibv.dereg_mr(ring_mr);
        free(ring_mem);
        set_err(err_buf, err_cap, "ibv_modify_qp(INIT) failed (rc=%d)", rc);
        return NULL;
    }

    MsvRdmaQp *wrap = (MsvRdmaQp *)calloc(1, sizeof(MsvRdmaQp));
    if (!wrap) {
        g_ibv.destroy_qp(qp);
        g_ibv.destroy_cq(cq);
        g_ibv.dereg_mr(ring_mr);
        free(ring_mem);
        set_err(err_buf, err_cap, "out of memory allocating qp wrapper");
        return NULL;
    }
    wrap->dev = dev;
    wrap->cq = cq;
    wrap->qp = qp;
    wrap->ring_buf = (uint8_t *)ring_mem;
    wrap->ring_bytes = ring_bytes;
    wrap->ring_mr = ring_mr;
    wrap->local_psn = psn & 0x00ffffffU;
    wrap->is_sender = is_sender ? 1 : 0;
    wrap->ctrl_recvs_armed = 0;

    memcpy(local_out->gid, dev->gid.raw, 16);
    local_out->qpn = qp->qp_num;
    local_out->psn = wrap->local_psn;
    local_out->lid = dev->lid;
    local_out->gid_index = dev->gid_index;
    local_out->active_mtu = dev->active_mtu;
    return wrap;
}

int msv_rdma_qp_connect(
    MsvRdmaQp *qp,
    const uint8_t peer_gid[16],
    uint32_t peer_qpn,
    uint32_t peer_psn,
    uint16_t peer_lid,
    char *err_buf,
    size_t err_cap
) {
    if (!qp || !qp->dev || !qp->qp || !peer_gid) {
        set_err(err_buf, err_cap, "invalid qp_connect arguments");
        return -1;
    }

    struct ibv_qp_attr rtr;
    memset(&rtr, 0, sizeof(rtr));
    rtr.qp_state = IBV_QPS_RTR;
    rtr.path_mtu = (enum ibv_mtu)qp->dev->active_mtu;
    rtr.dest_qp_num = peer_qpn;
    rtr.rq_psn = peer_psn & 0x00ffffffU;
    rtr.ah_attr.is_global = 1;
    memcpy(rtr.ah_attr.grh.dgid.raw, peer_gid, 16);
    rtr.ah_attr.grh.sgid_index = qp->dev->gid_index;
    rtr.ah_attr.grh.hop_limit = 1;
    rtr.ah_attr.dlid = peer_lid ? peer_lid : qp->dev->lid;
    rtr.ah_attr.sl = 0;
    rtr.ah_attr.src_path_bits = 0;
    rtr.ah_attr.port_num = 1;

    int rtr_mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;
    int rc = g_ibv.modify_qp(qp->qp, &rtr, rtr_mask);
    if (rc != 0) {
        rtr.path_mtu = IBV_MTU_1024;
        rc = g_ibv.modify_qp(qp->qp, &rtr, rtr_mask);
        if (rc != 0) {
            set_err(err_buf, err_cap, "ibv_modify_qp(UC RTR) failed (rc=%d)", rc);
            return -1;
        }
    }

    struct ibv_qp_attr rts;
    memset(&rts, 0, sizeof(rts));
    rts.qp_state = IBV_QPS_RTS;
    rts.sq_psn = qp->local_psn;

    int rts_mask = IBV_QP_STATE | IBV_QP_SQ_PSN;
    rc = g_ibv.modify_qp(qp->qp, &rts, rts_mask);
    if (rc != 0) {
        set_err(err_buf, err_cap, "ibv_modify_qp(UC RTS) failed (rc=%d)", rc);
        return -1;
    }

    if (qp->is_sender && !qp->ctrl_recvs_armed) {
        for (uint32_t s = 0; s < MSV_CTRL_RECV_N; s++) {
            if (post_ctrl_recv(qp, s, err_buf, err_cap) != 0) {
                return -1;
            }
        }
        qp->ctrl_recvs_armed = 1;
    }
    return 0;
}

void msv_rdma_qp_destroy(MsvRdmaQp *qp) {
    if (!qp) return;
    if (qp->qp) g_ibv.destroy_qp(qp->qp);
    if (qp->cq) g_ibv.destroy_cq(qp->cq);
    if (qp->ring_mr) g_ibv.dereg_mr(qp->ring_mr);
    free(qp->ring_buf);
    free(qp);
}

int msv_rdma_serve_offer(
    MsvRdmaQp *qp,
    const void *src_buf,
    uint64_t len,
    const uint8_t tid_16[16],
    volatile int *stop_flag,
    uint64_t timeout_ms,
    char *err_buf,
    size_t err_cap
) {
    if (!qp || !qp->qp || !qp->cq || !qp->ring_mr || !src_buf || !tid_16) {
        set_err(err_buf, err_cap, "invalid serve_offer arguments");
        return -2;
    }
    if (len == 0) return 0;
    if (!qp->is_sender || !qp->ctrl_recvs_armed) {
        set_err(err_buf, err_cap, "sender QP not armed for control doorbells");
        return -2;
    }

    const uint64_t deadline_ms = compute_deadline_ms(timeout_ms);

    /* Phase 1: wait for matching REQ(tid_16) doorbell from dst. */
    uint32_t chunk_bytes = (uint32_t)MSV_CHUNK_BYTES;
    uint64_t credits = 0;
    int got_req = 0;
    uint32_t idle_spins = 0;

    while (!got_req) {
        if (stop_requested(stop_flag)) {
            return 1;
        }
        struct ibv_wc wcs[16];
        int n = ibv_poll_cq(qp->cq, 16, wcs);
        if (n < 0) {
            set_err(err_buf, err_cap, "ibv_poll_cq(wait REQ) failed (n=%d)", n);
            return -2;
        }
        if (n == 0) {
            idle_spins++;
            if ((idle_spins & 0x3fU) == 0) {
                if (deadline_exceeded(deadline_ms)) {
                    return 1;
                }
                usleep(50);
            }
            continue;
        }
        idle_spins = 0;
        for (int i = 0; i < n; i++) {
            if (wcs[i].status != IBV_WC_SUCCESS) {
                set_err(err_buf, err_cap, "sender CQ error waiting REQ status=%d vendor=0x%x",
                        (int)wcs[i].status, wcs[i].vendor_err);
                return -2;
            }
            uint64_t tag = wcs[i].wr_id & WR_TAG_MASK;
            uint32_t slot = (uint32_t)(wcs[i].wr_id & WR_IDX_MASK);
            if (tag == WR_TAG_CTRL_RECV) {
                MsvCtrlMsg msg;
                int full_ctrl = (wcs[i].byte_len == sizeof(MsvCtrlMsg));
                if (full_ctrl) {
                    memcpy(&msg, qp_ctrl_recv_slot(qp, slot), sizeof(msg));
                }
                if (post_ctrl_recv(qp, slot, err_buf, err_cap) != 0) {
                    return -2;
                }
                if (full_ctrl && msg.magic == MSV_CTRL_MAGIC && memcmp(msg.tid, tid_16, 16) == 0) {
                    if (msg.kind == MSV_CTRL_REQ && !got_req) {
                        if (msg.chunk_bytes > 0 && msg.chunk_bytes <= MSV_CHUNK_BYTES) {
                            chunk_bytes = msg.chunk_bytes;
                        }
                        if (msg.total_bytes != len) {
                            set_err(err_buf, err_cap, "REQ byte_len mismatch (%llu vs %llu)",
                                    (unsigned long long)msg.total_bytes, (unsigned long long)len);
                            return -2;
                        }
                        credits += msg.credits;
                        got_req = 1;
                    } else if (msg.kind == MSV_CTRL_CRED) {
                        credits += msg.credits;
                    }
                }
            }
        }
    }

    /* Phase 2: stream all chunks over credit-paced IBV_WR_SEND using the staging ring. */
    const uint8_t *src = (const uint8_t *)src_buf;
    const uint64_t total_chunks = 1ULL + (len - 1ULL) / (uint64_t)chunk_bytes;
    uint64_t sent_chunks = 0;
    uint32_t send_inflight = 0;
    uint8_t slot_busy[MSV_DATA_SLOTS];
    memset(slot_busy, 0, sizeof(slot_busy));
    uint32_t next_slot = 0;
    uint32_t loop_iters = 0;
    idle_spins = 0;

    while (sent_chunks < total_chunks || send_inflight > 0) {
        if (stop_requested(stop_flag)) {
            set_err(err_buf, err_cap, "sender cancelled mid-transfer");
            return -1;
        }
        if ((++loop_iters & 0x3fU) == 0 && deadline_exceeded(deadline_ms)) {
            set_err(err_buf, err_cap, "sender timed out mid-transfer");
            return -1;
        }

        while (sent_chunks < total_chunks && credits > 0 && send_inflight < MSV_DATA_SLOTS) {
            uint32_t slot = next_slot % MSV_DATA_SLOTS;
            if (slot_busy[slot]) break;

            uint64_t off = sent_chunks * (uint64_t)chunk_bytes;
            uint64_t rem = len - off;
            uint32_t seg = rem > chunk_bytes ? chunk_bytes : (uint32_t)rem;

            memcpy(qp_data_slot(qp, slot), src + off, seg);
            if (post_data_send(qp, slot, seg, err_buf, err_cap) != 0) {
                return -2;
            }
            slot_busy[slot] = 1;
            next_slot++;
            sent_chunks++;
            credits--;
            send_inflight++;
        }

        struct ibv_wc wcs[32];
        int n = ibv_poll_cq(qp->cq, 32, wcs);
        if (n < 0) {
            set_err(err_buf, err_cap, "ibv_poll_cq(stream) failed (n=%d)", n);
            return -2;
        }
        if (n == 0) {
            idle_spins++;
            if ((idle_spins & 0x3ffU) == 0) {
                if (stop_requested(stop_flag)) {
                    set_err(err_buf, err_cap, "sender cancelled mid-transfer");
                    return -1;
                }
                if (deadline_exceeded(deadline_ms)) {
                    set_err(err_buf, err_cap, "sender timed out waiting for credits");
                    return -1;
                }
                usleep(10);
            }
            continue;
        }
        idle_spins = 0;
        for (int i = 0; i < n; i++) {
            if (wcs[i].status != IBV_WC_SUCCESS) {
                set_err(err_buf, err_cap, "sender completion error status=%d vendor=0x%x",
                        (int)wcs[i].status, wcs[i].vendor_err);
                return -2;
            }
            uint64_t tag = wcs[i].wr_id & WR_TAG_MASK;
            uint32_t slot = (uint32_t)(wcs[i].wr_id & WR_IDX_MASK);
            if (tag == WR_TAG_DATA_SEND) {
                if (slot < MSV_DATA_SLOTS) slot_busy[slot] = 0;
                if (send_inflight > 0) send_inflight--;
            } else if (tag == WR_TAG_CTRL_RECV) {
                MsvCtrlMsg msg;
                int full_ctrl = (wcs[i].byte_len == sizeof(MsvCtrlMsg));
                if (full_ctrl) {
                    memcpy(&msg, qp_ctrl_recv_slot(qp, slot), sizeof(msg));
                }
                if (post_ctrl_recv(qp, slot, err_buf, err_cap) != 0) {
                    return -2;
                }
                if (full_ctrl && msg.magic == MSV_CTRL_MAGIC && msg.kind == MSV_CTRL_CRED
                    && memcmp(msg.tid, tid_16, 16) == 0) {
                    credits += msg.credits;
                }
            }
        }
    }

    return 0;
}

int msv_rdma_pull_offer(
    MsvRdmaQp *qp,
    void *dst_buf,
    uint64_t len,
    const uint8_t tid_16[16],
    volatile int *stop_flag,
    uint64_t timeout_ms,
    char *err_buf,
    size_t err_cap
) {
    if (!qp || !qp->qp || !qp->cq || !qp->ring_mr || !dst_buf || !tid_16) {
        set_err(err_buf, err_cap, "invalid pull_offer arguments");
        return -2;
    }
    if (len == 0) return 0;
    if (qp->is_sender) {
        set_err(err_buf, err_cap, "pull_offer requires receiver QP");
        return -2;
    }
    if (stop_requested(stop_flag)) {
        set_err(err_buf, err_cap, "rdma pull cancelled");
        return -1;
    }

    const uint64_t deadline_ms = compute_deadline_ms(timeout_ms);
    uint8_t *dst = (uint8_t *)dst_buf;
    const uint32_t chunk_bytes = (uint32_t)MSV_CHUNK_BYTES;
    const uint64_t total_chunks = 1ULL + (len - 1ULL) / (uint64_t)chunk_bytes;
    const uint32_t init_win = total_chunks < MSV_WINDOW ? (uint32_t)total_chunks : MSV_WINDOW;

    /* 1. Post initial credit-bounded data RECVs (<= 2 MiB) BEFORE sending REQ.
     *    Match exact chunk length (including trailing partial chunk) because Apple's
     *    librdma.dylib returns IBV_WC_LOC_LEN_ERR if recv SGE length != send length. */
    uint64_t posted_recvs = 0;
    for (uint32_t i = 0; i < init_win; i++) {
        uint32_t slot = (uint32_t)(posted_recvs % MSV_DATA_SLOTS);
        uint64_t off = posted_recvs * (uint64_t)chunk_bytes;
        uint64_t rem = len - off;
        uint32_t exp_len = rem > chunk_bytes ? chunk_bytes : (uint32_t)rem;
        if (post_data_recv(qp, slot, exp_len, err_buf, err_cap) != 0) {
            return -2;
        }
        posted_recvs++;
    }

    /* 2. Send REQ doorbell to src. */
    uint8_t ctrl_busy[MSV_CTRL_SEND_N];
    memset(ctrl_busy, 0, sizeof(ctrl_busy));
    uint32_t ctrl_next = 0;
    uint32_t ctrl_inflight = 0;

    MsvCtrlMsg *req = qp_ctrl_send_slot(qp, ctrl_next);
    memset(req, 0, sizeof(*req));
    req->magic = MSV_CTRL_MAGIC;
    req->kind = MSV_CTRL_REQ;
    memcpy(req->tid, tid_16, 16);
    req->total_bytes = len;
    req->chunk_bytes = chunk_bytes;
    req->credits = init_win;
    if (post_ctrl_send(qp, ctrl_next, err_buf, err_cap) != 0) {
        return -2;
    }
    ctrl_busy[ctrl_next] = 1;
    ctrl_next = (ctrl_next + 1) % MSV_CTRL_SEND_N;
    ctrl_inflight++;

    uint64_t recvd_chunks = 0;
    uint64_t granted_credits = init_win;
    uint32_t pending_credit = 0;
    uint32_t idle_spins = 0;
    uint32_t loop_iters = 0;
    uint64_t last_req_ms = mono_ms();
    int req_retries = 0;

    /* 3. Receive all chunks, copy into dst_buf, repost RECVs, send CREDIT doorbells,
     *    and drain all ctrl send completions before returning so QP stays clean. */
    while (recvd_chunks < total_chunks || ctrl_inflight > 0) {
        if (stop_requested(stop_flag)) {
            set_err(err_buf, err_cap, "rdma pull cancelled (%llu/%llu chunks)",
                    (unsigned long long)recvd_chunks, (unsigned long long)total_chunks);
            return -1;
        }
        if ((++loop_iters & 0x3fU) == 0 && deadline_exceeded(deadline_ms)) {
            set_err(err_buf, err_cap, "rdma pull timed out (%llu/%llu chunks)",
                    (unsigned long long)recvd_chunks, (unsigned long long)total_chunks);
            return -1;
        }

        struct ibv_wc wcs[32];
        int n = ibv_poll_cq(qp->cq, 32, wcs);
        if (n < 0) {
            set_err(err_buf, err_cap, "ibv_poll_cq(pull) failed (n=%d)", n);
            return -2;
        }
        if (n == 0) {
            idle_spins++;
            if ((idle_spins & 0x3ffU) == 0) {
                if (stop_requested(stop_flag)) {
                    set_err(err_buf, err_cap, "rdma pull cancelled (%llu/%llu chunks)",
                            (unsigned long long)recvd_chunks, (unsigned long long)total_chunks);
                    return -1;
                }
                if (deadline_exceeded(deadline_ms)) {
                    set_err(err_buf, err_cap, "rdma pull timed out (%llu/%llu chunks)",
                            (unsigned long long)recvd_chunks, (unsigned long long)total_chunks);
                    return -1;
                }
                uint64_t now = mono_ms();
                /* If initial REQ was lost before any data chunk arrived, retry REQ up to 2 times. */
                if (recvd_chunks == 0 && req_retries < 2 && now > 0 && now >= last_req_ms
                    && (now - last_req_ms) >= 200 && !ctrl_busy[ctrl_next]) {
                    MsvCtrlMsg *r2 = qp_ctrl_send_slot(qp, ctrl_next);
                    memset(r2, 0, sizeof(*r2));
                    r2->magic = MSV_CTRL_MAGIC;
                    r2->kind = MSV_CTRL_REQ;
                    memcpy(r2->tid, tid_16, 16);
                    r2->total_bytes = len;
                    r2->chunk_bytes = chunk_bytes;
                    r2->credits = init_win;
                    if (post_ctrl_send(qp, ctrl_next, err_buf, err_cap) != 0) {
                        return -2;
                    }
                    ctrl_busy[ctrl_next] = 1;
                    ctrl_next = (ctrl_next + 1) % MSV_CTRL_SEND_N;
                    ctrl_inflight++;
                    last_req_ms = now;
                    req_retries++;
                }
                usleep(10);
            }
            continue;
        }
        idle_spins = 0;

        for (int i = 0; i < n; i++) {
            if (wcs[i].status != IBV_WC_SUCCESS) {
                set_err(err_buf, err_cap, "pull completion error status=%d wr_id=0x%llx vendor=0x%x",
                        (int)wcs[i].status, (unsigned long long)wcs[i].wr_id, wcs[i].vendor_err);
                return -2;
            }
            uint64_t tag = wcs[i].wr_id & WR_TAG_MASK;
            uint32_t slot = (uint32_t)(wcs[i].wr_id & WR_IDX_MASK);

            if (tag == WR_TAG_CTRL_SEND) {
                if (slot < MSV_CTRL_SEND_N) ctrl_busy[slot] = 0;
                if (ctrl_inflight > 0) ctrl_inflight--;
            } else if (tag == WR_TAG_DATA_RECV) {
                if (recvd_chunks >= total_chunks) {
                    set_err(err_buf, err_cap, "unexpected extra data chunk");
                    return -2;
                }
                uint64_t off = recvd_chunks * (uint64_t)chunk_bytes;
                uint64_t rem = len - off;
                uint32_t expected = rem > chunk_bytes ? chunk_bytes : (uint32_t)rem;
                if (wcs[i].byte_len != expected) {
                    set_err(err_buf, err_cap, "chunk %llu length mismatch (%u vs %u)",
                            (unsigned long long)recvd_chunks, wcs[i].byte_len, expected);
                    return -2;
                }
                memcpy(dst + off, qp_data_slot(qp, slot), expected);
                recvd_chunks++;

                /* Repost RECV on a fresh slot BEFORE granting credit so src never outruns RQ. */
                if (posted_recvs < total_chunks) {
                    uint32_t next_rx_slot = (uint32_t)(posted_recvs % MSV_DATA_SLOTS);
                    uint64_t next_off = posted_recvs * (uint64_t)chunk_bytes;
                    uint64_t next_rem = len - next_off;
                    uint32_t next_len = next_rem > chunk_bytes ? chunk_bytes : (uint32_t)next_rem;
                    if (post_data_recv(qp, next_rx_slot, next_len, err_buf, err_cap) != 0) {
                        return -2;
                    }
                    posted_recvs++;
                    pending_credit++;
                }
            }
        }

        /* Grant batched credits once corresponding RECVs are already posted. */
        while (pending_credit > 0 && granted_credits < total_chunks) {
            uint64_t remaining_to_grant = total_chunks - granted_credits;
            if (pending_credit < MSV_CREDIT_BATCH && pending_credit < remaining_to_grant) {
                break;
            }
            if (ctrl_busy[ctrl_next]) break;
            uint32_t grant = pending_credit > remaining_to_grant
                ? (uint32_t)remaining_to_grant
                : pending_credit;
            MsvCtrlMsg *cm = qp_ctrl_send_slot(qp, ctrl_next);
            memset(cm, 0, sizeof(*cm));
            cm->magic = MSV_CTRL_MAGIC;
            cm->kind = MSV_CTRL_CRED;
            memcpy(cm->tid, tid_16, 16);
            cm->credits = grant;
            if (post_ctrl_send(qp, ctrl_next, err_buf, err_cap) != 0) {
                return -2;
            }
            ctrl_busy[ctrl_next] = 1;
            ctrl_next = (ctrl_next + 1) % MSV_CTRL_SEND_N;
            ctrl_inflight++;
            granted_credits += grant;
            pending_credit -= grant;
        }
    }

    return 0;
}

#else

int msv_rdma_available(void) { return 0; }
int msv_rdma_list_devices(char out[][64], size_t cap) {
    (void)out;
    (void)cap;
    return 0;
}
MsvRdmaDev *msv_rdma_dev_open(const char *dev_name, char *err_buf, size_t err_cap) {
    (void)dev_name;
    (void)err_buf;
    (void)err_cap;
    return NULL;
}
void msv_rdma_dev_close(MsvRdmaDev *dev) { (void)dev; }
MsvRdmaQp *msv_rdma_qp_create(MsvRdmaDev *dev, uint32_t psn, int is_sender, MsvRdmaEndpoint *local_out, char *err_buf, size_t err_cap) {
    (void)dev;
    (void)psn;
    (void)is_sender;
    (void)local_out;
    (void)err_buf;
    (void)err_cap;
    return NULL;
}
int msv_rdma_qp_connect(MsvRdmaQp *qp, const uint8_t peer_gid[16], uint32_t peer_qpn, uint32_t peer_psn, uint16_t peer_lid, char *err_buf, size_t err_cap) {
    (void)qp;
    (void)peer_gid;
    (void)peer_qpn;
    (void)peer_psn;
    (void)peer_lid;
    (void)err_buf;
    (void)err_cap;
    return -1;
}
void msv_rdma_qp_destroy(MsvRdmaQp *qp) { (void)qp; }
int msv_rdma_serve_offer(MsvRdmaQp *qp, const void *src_buf, uint64_t len, const uint8_t tid_16[16], volatile int *stop_flag, uint64_t timeout_ms, char *err_buf, size_t err_cap) {
    (void)qp;
    (void)src_buf;
    (void)len;
    (void)tid_16;
    (void)stop_flag;
    (void)timeout_ms;
    (void)err_buf;
    (void)err_cap;
    return -2;
}
int msv_rdma_pull_offer(MsvRdmaQp *qp, void *dst_buf, uint64_t len, const uint8_t tid_16[16], volatile int *stop_flag, uint64_t timeout_ms, char *err_buf, size_t err_cap) {
    (void)qp;
    (void)dst_buf;
    (void)len;
    (void)tid_16;
    (void)stop_flag;
    (void)timeout_ms;
    (void)err_buf;
    (void)err_cap;
    return -2;
}

#endif
