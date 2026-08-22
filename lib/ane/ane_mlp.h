#ifndef MLXSERVE_ANE_MLP_H
#define MLXSERVE_ANE_MLP_H

#include <stddef.h>
#include <stdint.h>

/* Transformer-layer projections compiled as ANE programs at a FIXED row
 * count. Weights are int8 per-output-row with fp16 scales (constexpr-
 * dequantized in-graph); activations run fp16 with a power-of-two
 * (1/16 .. x16) wrap around the down conv for accumulator headroom. The
 * down projection's K axis is chunked in-graph — a single K=17408 conv
 * measured a 2.6x throughput cliff (perf-plan-aug-17 P5).
 *
 * Programs are BANKS: every covered layer's slice is one `procedureNNN`
 * function inside a single program, because the private runtime accepts
 * only ~121 resident model handles (oMLX probe) and a dual-ANE build wants
 * twice our 112. One bank = one compile, one load, one cache entry; each
 * procedure gets its own _ANERequest and is dispatched by index.
 *
 * I/O planes are IOSurface-backed fp16 in CHANNEL-major layout:
 *   input  [hidden][rows]  (row r, channel c at plane[c*rows + r])
 *   output [width][rows]
 * fp16 planes halve seam traffic vs the v1 f32 planes and are numerics-
 * neutral: the graph computed in fp16 either way (bf16→fp16 is exact in
 * fp16's normal range). Every procedure in a bank shares the bank's one
 * input and one output surface, so a bank is homogeneous (all MLP or all
 * GDN). NB: eval returns 1 on SUCCESS. */

typedef struct msv_ane_mlp msv_ane_mlp;   /* one compiled bank */
typedef struct msv_ane_bank msv_ane_bank; /* the builder for one */
typedef struct msv_ane_plane msv_ane_plane; /* opaque IOSurface handle */

int msv_ane_available(void);

/* Free bytes on the INTERNAL volume's /private/tmp — the ANE compiler
 * service (aned) holds per-compile scratch there for the client's lifetime
 * regardless of where OUR staging lives, so this is the number that bounds
 * a compile session's program budget (~free / program-bytes). 0 on probe
 * failure (no information). */
uint64_t msv_ane_internal_free_disk(void);

/* Shared I/O planes (A9): within ONE unit evals are strictly serial (one
 * in-flight kick/wait), so every program of a shape class binds the SAME
 * IOSurface pair — two planes' worth of memory instead of two per program
 * (~11 GB back on the 27B). Concurrent units (dual ANE) each get their OWN
 * planes: overlapping evals cannot share an output surface. A create's
 * `input_plane`/`output_plane` of NULL keeps a per-program allocation; a
 * non-NULL plane is retained by the program and must be at least the
 * program's plane bytes. */
msv_ane_plane *msv_ane_plane_create(size_t bytes);
void msv_ane_plane_free(msv_ane_plane *p);
__fp16 *msv_ane_plane_base(msv_ane_plane *p);

/* ── Bank building ── */

msv_ane_bank *msv_ane_bank_create(void);
void msv_ane_bank_free(msv_ane_bank *b);
uint32_t msv_ane_bank_count(const msv_ane_bank *b);
uint64_t msv_ane_bank_bytes(const msv_ane_bank *b);

/* Append one SwiGLU MLP (gate/up/SiLU/down) procedure. Returns the
 * procedure index, or -1 with `error` filled. */
int msv_ane_bank_add_mlp(msv_ane_bank *b, uint32_t hidden, uint32_t ffn,
                         uint32_t rows,
                         const int8_t *gate_q, const float *gate_s,
                         const int8_t *up_q, const float *up_s,
                         const int8_t *down_q, const float *down_s,
                         char *error, size_t error_size);

/* Append one GDN input-projection procedure: in_proj_qkv + in_proj_z as ONE
 * fused conv over the stacked [qkv_out + z_out, hidden] weight — each
 * output row is an independent dot product, so stacking along the
 * output-channel axis is exactly the two convs concatenated. Output rows
 * are qkv first. Returns the procedure index, or -1. */
int msv_ane_bank_add_gdn(msv_ane_bank *b, uint32_t hidden, uint32_t qkv_out,
                         uint32_t z_out, uint32_t rows,
                         const int8_t *qkv_q, const float *qkv_s,
                         const int8_t *z_q, const float *z_s,
                         char *error, size_t error_size);

/* Compile (or restore from the compile cache) and load the accumulated
 * bank as ONE program pinned to `ane_instance` (0 = no affinity hint, the
 * single-ANE path; 1..4 name a die on a multi-instance machine). The
 * builder is CONSUMED either way. */
msv_ane_mlp *msv_ane_bank_finish(msv_ane_bank *b, const char *name,
                                 int ane_instance,
                                 msv_ane_plane *input_plane,
                                 msv_ane_plane *output_plane,
                                 char *error, size_t error_size);

void msv_ane_mlp_free(msv_ane_mlp *m);

__fp16 *msv_ane_mlp_input(msv_ane_mlp *m);
__fp16 *msv_ane_mlp_output(msv_ane_mlp *m);
/* Dispatch procedure `procedure` of the bank. */
int msv_ane_mlp_eval(msv_ane_mlp *m, uint32_t procedure, char *error,
                     size_t error_size);

double msv_ane_mlp_compile_seconds(const msv_ane_mlp *m);
int msv_ane_mlp_cache_hit(const msv_ane_mlp *m);

#endif
