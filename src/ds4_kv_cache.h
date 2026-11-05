#ifndef MLXSERVE_DS4_KV_CACHE_H
#define MLXSERVE_DS4_KV_CACHE_H

#include "../lib/ds4/ds4.h"

typedef struct mlxserve_ds4_kv_cache mlxserve_ds4_kv_cache;

/* One engine/thread owns this pool and the native pool enforces its budget.
 * budget_bytes is this pool's own fixed allowance; zero is rejected so a
 * disabled cache never opens and never falls back to the native default. */
mlxserve_ds4_kv_cache *mlxserve_ds4_kv_cache_open(
    ds4_engine *engine, const char *dir, const char *namespace_key,
    uint64_t budget_bytes, int min_tokens, char *err, size_t errlen);
void mlxserve_ds4_kv_cache_close(mlxserve_ds4_kv_cache *cache);
/* 1 stores a checkpoint, 0 when the session is ineligible or the native pool
 * declined the write, -1 on a failed write. */
int mlxserve_ds4_kv_cache_store(mlxserve_ds4_kv_cache *cache,
    ds4_session *session, const int *tokens, int count, const char *reason,
    char *err, size_t errlen);
/* A hit leaves exactly target[0..return_value] in the session. Failure after
 * payload mutation invalidates the session. No key bytes are ever tokenized. */
int mlxserve_ds4_kv_cache_restore(mlxserve_ds4_kv_cache *cache,
    ds4_session *session, const int *target, int count, char *err, size_t errlen);

#endif
