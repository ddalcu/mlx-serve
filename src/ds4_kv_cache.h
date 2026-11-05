#ifndef MLXSERVE_DS4_KV_CACHE_H
#define MLXSERVE_DS4_KV_CACHE_H

#include "../lib/ds4/ds4.h"

typedef struct mlxserve_ds4_kv_cache mlxserve_ds4_kv_cache;

/* One engine/thread owns the shared flat pool. budget_bytes is an exact
 * allowance supplied by the combined cache coordinator; zero disables writes. */
mlxserve_ds4_kv_cache *mlxserve_ds4_kv_cache_open(
    ds4_engine *engine, const char *dir, const char *namespace_key,
    uint64_t budget_bytes, int min_tokens, char *err, size_t errlen);
void mlxserve_ds4_kv_cache_close(mlxserve_ds4_kv_cache *cache);
int mlxserve_ds4_kv_cache_store(mlxserve_ds4_kv_cache *cache,
    ds4_session *session, const int *tokens, int count, const char *reason,
    char *err, size_t errlen);
/* A hit leaves exactly target[0..return_value] in the session. Failure after
 * payload mutation invalidates the session. No key bytes are ever tokenized. */
int mlxserve_ds4_kv_cache_restore(mlxserve_ds4_kv_cache *cache,
    ds4_session *session, const int *target, int count, char *err, size_t errlen);
/* Includes invalid/unindexed files. UINT64_MAX means accounting failed. */
uint64_t mlxserve_ds4_kv_cache_physical_bytes(mlxserve_ds4_kv_cache *cache);
int mlxserve_ds4_kv_cache_set_budget(mlxserve_ds4_kv_cache *cache,
    uint64_t bytes, char *err, size_t errlen);
int mlxserve_ds4_kv_cache_prune(mlxserve_ds4_kv_cache *cache,
    char *err, size_t errlen);

#endif
