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
 * the constants. MLX_SERVE_ANE_CACHE=0 disables reuse, and freeing a model
 * while disabled also evicts its entry. */
bool msv_ane_cache_enabled(void);

#endif
