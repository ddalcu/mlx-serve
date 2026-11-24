#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MsvRdmaDev MsvRdmaDev;
typedef struct MsvRdmaQp MsvRdmaQp;

typedef struct {
    uint8_t gid[16];
    uint32_t qpn;
    uint32_t psn;
    uint16_t lid;
    uint8_t gid_index;
    uint8_t active_mtu;
} MsvRdmaEndpoint;

/* Returns 1 if librdma.dylib is loaded and at least one RDMA device exists. */
int msv_rdma_available(void);

/* Copies up to `cap` available RDMA device names (NUL-terminated, < 64 bytes each)
 * into `out` and returns the number of names written. */
int msv_rdma_list_devices(char out[][64], size_t cap);

/* Open RDMA device by name (e.g. "rdma_en2"), allocate PD, locate IPv4-mapped GID. */
MsvRdmaDev *msv_rdma_dev_open(const char *dev_name, char *err_buf, size_t err_cap);
void msv_rdma_dev_close(MsvRdmaDev *dev);

/* Allocate IBV_QPT_UC QP + CQ + pre-registered 4 MiB staging ring MR, and
 * transition RESET -> INIT.
 * is_sender: 0 = recv (dst/puller), 1 = send (src/offerer). */
MsvRdmaQp *msv_rdma_qp_create(
    MsvRdmaDev *dev,
    uint32_t psn,
    int is_sender,
    MsvRdmaEndpoint *local_out,
    char *err_buf,
    size_t err_cap
);

/* Transition IBV_QPT_UC QP from INIT -> RTR -> RTS targeting peer endpoint.
 * For sender QPs, also pre-posts control RECV buffers for incoming REQ/CREDIT doorbells.
 * Returns 0 on success. */
int msv_rdma_qp_connect(
    MsvRdmaQp *qp,
    const uint8_t peer_gid[16],
    uint32_t peer_qpn,
    uint32_t peer_psn,
    uint16_t peer_lid,
    char *err_buf,
    size_t err_cap
);

/* Destroy UC QP, its CQ, and its pre-registered staging ring MR. */
void msv_rdma_qp_destroy(MsvRdmaQp *qp);

/* Sender side (runs on src for a pinned offer):
 * Waits on the sender QP for a matching REQ(tid_16) doorbell from dst, then streams
 * `len` bytes from `src_buf` over credit-paced IBV_QPT_UC IBV_WR_SEND using the QP's
 * pre-registered staging ring (zero per-transfer ibv_reg_mr).
 * Polls `*stop_flag` while waiting and streaming so DELETE/quiesce can cancel cleanly.
 * `timeout_ms` is a relative timeout measured on CLOCK_MONOTONIC inside this call (0 = none).
 * Returns:
 *   0  on complete transfer,
 *   1  if cancelled/expired via *stop_flag or timeout before transfer started,
 *  -1  if cancelled/timed out mid-transfer (caller should reset QP),
 *  -2  on verbs/WC error (caller should reset QP). */
int msv_rdma_serve_offer(
    MsvRdmaQp *qp,
    const void *src_buf,
    uint64_t len,
    const uint8_t tid_16[16],
    volatile int *stop_flag,
    uint64_t timeout_ms,
    char *err_buf,
    size_t err_cap
);

/* Receiver side (runs on dst inside POST /v1/cache/rdma/pull):
 * Posts credit-bounded IBV_WR_RECV chunks (<= 2 MiB inflight, staying below macOS TB5's
 * 4 MiB posted-recv limit), sends REQ(tid_16) over the UC QP, refills RECVs + sends
 * CREDIT doorbells, and copies all `len` bytes into `dst_buf`.
 * Polls `*stop_flag` (optional) so quiesce/shutdown can abort cleanly without UAF.
 * `timeout_ms` is a relative timeout measured on CLOCK_MONOTONIC inside this call (0 = none).
 * Returns:
 *   0  on complete transfer,
 *  -1  on timeout or cancellation (caller should reset QP),
 *  -2  on verbs/WC/protocol error (caller should reset QP). */
int msv_rdma_pull_offer(
    MsvRdmaQp *qp,
    void *dst_buf,
    uint64_t len,
    const uint8_t tid_16[16],
    volatile int *stop_flag,
    uint64_t timeout_ms,
    char *err_buf,
    size_t err_cap
);

#ifdef __cplusplus
}
#endif
