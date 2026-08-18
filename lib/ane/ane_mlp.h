#ifndef MLXSERVE_ANE_MLP_H
#define MLXSERVE_ANE_MLP_H

#include <stddef.h>
#include <stdint.h>

/* One transformer-layer SwiGLU MLP (gate/up/SiLU/down) compiled as a single
 * ANE program at a FIXED row count. Weights are int8 per-output-row with
 * fp16 scales (constexpr-dequantized in-graph); activations run fp16 with a
 * power-of-two (1/16 .. x16) wrap around the down conv for accumulator
 * headroom. The down projection's K axis is chunked in-graph — a single
 * K=17408 conv measured a 2.6x throughput cliff (perf-plan-aug-17 P5).
 *
 * I/O planes are IOSurface-backed fp16 in CHANNEL-major layout:
 *   input  [hidden][rows]  (row r, channel c at plane[c*rows + r])
 *   output [hidden][rows]
 * fp16 planes halve seam traffic vs the v1 f32 planes and are numerics-
 * neutral: the graph computed in fp16 either way (bf16→fp16 is exact in
 * fp16's normal range). NB: eval returns 1 on SUCCESS. */

typedef struct msv_ane_mlp msv_ane_mlp;
typedef struct msv_ane_plane msv_ane_plane; /* opaque IOSurface handle */

int msv_ane_available(void);

/* Free bytes on the INTERNAL volume's /private/tmp — the ANE compiler
 * service (aned) holds per-compile scratch there for the client's lifetime
 * regardless of where OUR staging lives, so this is the number that bounds
 * a compile session's program budget (~free / program-bytes). 0 on probe
 * failure (no information). */
uint64_t msv_ane_internal_free_disk(void);

/* Shared I/O planes (A9): evals are strictly serial (one in-flight
 * kick/wait), so every program of a shape class can bind the SAME
 * IOSurface pair — two planes' worth of memory instead of two per program
 * (~11 GB back on the 27B). A create's `input_plane`/`output_plane` of NULL
 * keeps the old per-program allocation; a non-NULL plane is retained by the
 * program and must be at least the program's plane bytes. */
msv_ane_plane *msv_ane_plane_create(size_t bytes);
void msv_ane_plane_free(msv_ane_plane *p);
__fp16 *msv_ane_plane_base(msv_ane_plane *p);

msv_ane_mlp *msv_ane_mlp_create(const char *name,
                                uint32_t hidden, uint32_t ffn, uint32_t rows,
                                const int8_t *gate_q, const float *gate_s,
                                const int8_t *up_q, const float *up_s,
                                const int8_t *down_q, const float *down_s,
                                msv_ane_plane *input_plane,
                                msv_ane_plane *output_plane,
                                char *error, size_t error_size);

/* GDN input projections (in_proj_qkv + in_proj_z, per-token-independent
 * linears) as ONE fused conv over the stacked [qkv_out + z_out, hidden]
 * weight — each output row is an independent dot product, so stacking along
 * the output-channel axis is exactly the two convs concatenated. Output
 * plane is [(qkv_out + z_out)][rows] fp16, qkv rows first. Same handle
 * type: input/output/eval/free are shared with the MLP programs. */
msv_ane_mlp *msv_ane_gdn_create(const char *name,
                                uint32_t hidden, uint32_t qkv_out,
                                uint32_t z_out, uint32_t rows,
                                const int8_t *qkv_q, const float *qkv_s,
                                const int8_t *z_q, const float *z_s,
                                msv_ane_plane *input_plane,
                                msv_ane_plane *output_plane,
                                char *error, size_t error_size);
void msv_ane_mlp_free(msv_ane_mlp *m);

__fp16 *msv_ane_mlp_input(msv_ane_mlp *m);
__fp16 *msv_ane_mlp_output(msv_ane_mlp *m);
int msv_ane_mlp_eval(msv_ane_mlp *m, char *error, size_t error_size);

double msv_ane_mlp_compile_seconds(const msv_ane_mlp *m);
int msv_ane_mlp_cache_hit(const msv_ane_mlp *m);

#endif
