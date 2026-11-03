#ifndef MLXSERVE_ANE_BRIDGE_H
#define MLXSERVE_ANE_BRIDGE_H

#include <IOSurface/IOSurface.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* The only interface to the private AppleNeuralEngine framework. Everything
 * above this file speaks MIL text, weight blobs, and IOSurfaces; everything
 * below it is objc_msgSend into _ANEInMemoryModel and friends.
 *
 * Binding rule: request inputs bind to MIL function parameters in
 * ALPHABETICAL name order, not declaration order. Callers must name their
 * parameters so the sort order matches the surface indices (i0_*, i1_*, or
 * x0..x7). Same-shape mismatches swap data silently. */

typedef struct msv_ane_model msv_ane_model;

int msv_ane_bridge_available(void);

/* 16KB-aligned, host-mapped, zero-fill on create is the caller's job. */
IOSurfaceRef msv_ane_bridge_surface(size_t bytes);

/* Compile (or restore from the compile cache) and load one MIL program.
 * Takes ownership of the malloc'd weights blob. The input surfaces bind to
 * indices 0..input_count-1 and `output` to output index 0.
 *
 * `procedure_count` is how many `procedureNNN` functions the MIL declares
 * (1 for a plain `main` program): one _ANERequest is built per procedure
 * and dispatched by index, all sharing the same surfaces.
 *
 * `ane_instance` pins the program to one Neural Engine die: 0 keeps the
 * empty options dict (no affinity — the driver schedules it), 1..4 pass
 * kANEFAneInstanceHint together with the single-ANE procedure-variant hint.
 * BOTH keys are required — the scheduler only honours an instance hint for
 * the single-ANE variant. */
msv_ane_model *msv_ane_model_create(const char *name, const char *mil,
                                  void *weights, size_t weight_bytes,
                                  IOSurfaceRef *inputs, uint32_t input_count,
                                  IOSurfaceRef output,
                                  uint32_t procedure_count, int ane_instance,
                                  char *error, size_t error_size);
void msv_ane_model_free(msv_ane_model *model);

int msv_ane_model_eval(msv_ane_model *model, uint32_t procedure, char *error,
                       size_t error_size);

/* Residency rotation: unload releases the wired compiled net but keeps the
 * handle and request; reload re-wires it from the compile cache. */
int msv_ane_model_unload(msv_ane_model *model, char *error, size_t error_size);
int msv_ane_model_reload(msv_ane_model *model, char *error, size_t error_size);

/* Load cost of the create: a fresh compile or a cached load. */
double msv_ane_model_compile_seconds(const msv_ane_model *model);
bool msv_ane_model_cache_hit(const msv_ane_model *model);

/* Compiled models persist in ~/.mlx-serve/ane-cache/entries/<content-hash>
 * (the hash covers the MIL text AND the weights, so same-shape
 * different-weight models never collide; MLX_SERVE_ANE_CACHE_DIR overrides
 * the root). Only the compiled artifacts are kept; the `data` file embeds
 * the constants. */

/* Names the (seam + shape, share) the next compiles belong to: entries of
 * the same group at a different variant are pruned on the cold-compile
 * path, so a share sweep replaces its predecessors. Empty group = untagged. */
void msv_ane_cache_lineage(const char *group, const char *variant);
/* The variant of the most recently used entry tagged with `group` (empty
 * when none), so a later build can reuse the share that set was built at. */
void msv_ane_cache_variant(const char *group, char *out, int out_len);
/* Apply the cache's caps now (as every cold store does): other variants of
 * the current group first, then the oldest entries outside the current set. */
void msv_ane_cache_prune(void);
/* After a build: drop the current group's entries it did not restore or store. */
void msv_ane_cache_sweep(void);

/* ── programs (the ANE LM engine) ──
 * One MIL program of several functions ("procedures") over one weight blob, loaded as `instances`
 * model objects of the same compiled program: one weight copy, and each instance its own queue (two
 * instances evaluate at once on two ANE clusters; one instance's requests run in turn). Inputs and
 * outputs bind by the names in the compiled model's attributes, in the order `msv_ane_program_io`
 * lists them; their surfaces follow its strides (rows padded to 64 bytes). */
typedef struct msv_ane_program msv_ane_program;
typedef struct msv_ane_request msv_ane_request;

typedef struct {
    char name[128];
    uint32_t dtype; /* 0 fp16, 1 fp32, 2 int8, 3 uint8 */
    uint32_t batches, channels, depth, height, width;
    uint64_t batch_stride, depth_stride, plane_stride, row_stride;
} msv_ane_io;

/* Takes ownership of the malloc'd weights blob (freed once compiled). Loads it `instances` times;
 * `ane_hints` (NULL, or one per instance; 0 = none) pins each load to an ANE instance: two loads
 * evaluated at once without hints can land on one cluster. */
msv_ane_program *msv_ane_program_create(const char *name, const char *mil,
                                        void *weights, size_t weight_bytes,
                                        uint32_t instances, const uint32_t *ane_hints,
                                        char *error, size_t error_size);
void msv_ane_program_free(msv_ane_program *program);
/* The procedure index of the MIL function `function`; -1 when absent. */
int msv_ane_program_procedure(msv_ane_program *program, const char *function);
uint32_t msv_ane_program_io_count(msv_ane_program *program, uint32_t procedure, int output);
int msv_ane_program_io(msv_ane_program *program, uint32_t procedure, int output,
                       uint32_t index, msv_ane_io *out);
msv_ane_request *msv_ane_request_create(msv_ane_program *program, uint32_t instance,
                                        uint32_t procedure, IOSurfaceRef *inputs,
                                        uint32_t input_count, IOSurfaceRef *outputs,
                                        uint32_t output_count, char *error,
                                        size_t error_size);
/* 1 on success. */
int msv_ane_request_eval(msv_ane_request *request, char *error, size_t error_size);
void msv_ane_request_free(msv_ane_request *request);
double msv_ane_program_compile_seconds(const msv_ane_program *program);
bool msv_ane_program_cache_hit(const msv_ane_program *program);

#endif
