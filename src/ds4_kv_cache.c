#include "ds4_kv_cache.h"
#include "../lib/ds4/ds4_kvstore.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DS4_KV_CACHE_MIB (1024ull * 1024ull)

struct mlxserve_ds4_kv_cache {
    ds4_engine *engine;
    ds4_kvstore store;
    char *namespace_key;
};

static int fail(char *err, size_t len, const char *message) {
    if (err && len) snprintf(err, len, "%s", message);
    return -1;
}

/* Fixed-width IDs preserve token boundaries under native byte-prefix lookup.
 * The namespace is opaque to ds4: effective_prompt must always be NULL. */
static char *token_key(mlxserve_ds4_kv_cache *c, const int *tokens, int n) {
    if (n < 0 || (n && !tokens)) return NULL;
    size_t prefix = strlen(c->namespace_key);
    if ((size_t)n > (SIZE_MAX - prefix - 2) / 9) return NULL;
    char *key = malloc(prefix + 2 + (size_t)n * 9);
    if (!key) return NULL;
    memcpy(key, c->namespace_key, prefix);
    key[prefix++] = ':';
    for (int i = 0; i < n; i++) {
        if (tokens[i] < 0) { free(key); return NULL; }
        snprintf(key + prefix + (size_t)i * 9, 10, "%08x;", (unsigned)tokens[i]);
    }
    key[prefix + (size_t)n * 9] = '\0';
    return key;
}

static bool exact_prefix(const ds4_tokens *live, const int *target, int n) {
    return live && live->len >= 0 && live->len <= n &&
        (!live->len || (live->v && target &&
         !memcmp(live->v, target, (size_t)live->len * sizeof(int))));
}

mlxserve_ds4_kv_cache *mlxserve_ds4_kv_cache_open(
    ds4_engine *engine, const char *dir, const char *namespace_key,
    uint64_t budget_bytes, int min_tokens, char *err, size_t errlen) {
    if (!engine || !dir || !namespace_key || !namespace_key[0] || min_tokens < 0) {
        fail(err, errlen, "invalid ds4 cache options");
        return NULL;
    }
    /* The native pool budgets in whole MiB and reads zero as "use the
     * default". A zero allowance would silently become that default, so the
     * engine bypasses the cache instead of opening it. */
    if (!budget_bytes) {
        fail(err, errlen, "ds4 cache budget must be greater than zero");
        return NULL;
    }
    uint64_t budget_mb = budget_bytes / DS4_KV_CACHE_MIB;
    if (budget_bytes % DS4_KV_CACHE_MIB) budget_mb++;
    if (budget_mb > UINT64_MAX / DS4_KV_CACHE_MIB) {
        fail(err, errlen, "ds4 cache budget is too large");
        return NULL;
    }
    mlxserve_ds4_kv_cache *c = calloc(1, sizeof(*c));
    if (!c) { fail(err, errlen, "cannot allocate ds4 cache"); return NULL; }
    c->engine = engine;
    c->namespace_key = strdup(namespace_key);
    ds4_kvstore_options opt = ds4_kvstore_default_options();
    if (min_tokens) opt.min_tokens = min_tokens;
    if (!c->namespace_key || !ds4_kvstore_open(&c->store, dir, budget_mb,
                                              true, opt, "mlx-serve ds4", NULL, NULL)) {
        fail(err, errlen, "cannot open ds4 cache pool");
        mlxserve_ds4_kv_cache_close(c);
        return NULL;
    }
    /* The native budget is whole MiB; the exact allowance is applied here and
     * the native eviction enforces it once. */
    c->store.budget_bytes = budget_bytes;
    ds4_kvstore_evict(&c->store, NULL, 0, NULL);
    return c;
}

void mlxserve_ds4_kv_cache_close(mlxserve_ds4_kv_cache *c) {
    if (!c) return;
    ds4_kvstore_close(&c->store);
    free(c->namespace_key);
    free(c);
}

int mlxserve_ds4_kv_cache_store(mlxserve_ds4_kv_cache *c,
    ds4_session *session, const int *tokens, int count, const char *reason,
    char *err, size_t errlen) {
    if (err && errlen) err[0] = '\0';
    if (count < c->store.opt.min_tokens ||
        ds4_session_payload_bytes(session) == 0 || ds4_session_has_vision_state(session)) return 0;
    const ds4_tokens *live = ds4_session_tokens(session);
    if (!exact_prefix(live, tokens, count) || live->len != count) return 0;
    char *key = token_key(c, tokens, count);
    if (!key) return fail(err, errlen, "invalid ds4 cache token key");
    /* The native pool pre-evicts to make room, renames its temporary file and
     * removes a failed write, so the wrapper adds no second budget pass. */
    bool ok = ds4_kvstore_store_live_prefix_text(&c->store, c->engine, session,
        live, count, reason, key, 0, "namespaced-token-ids", NULL, err, errlen);
    free(key);
    if (!ok) return err && errlen && err[0] ? -1 : 0;
    return 1;
}

int mlxserve_ds4_kv_cache_restore(mlxserve_ds4_kv_cache *c,
    ds4_session *session, const int *target, int count, char *err, size_t errlen) {
    if (err && errlen) err[0] = '\0';
    if (count < c->store.opt.min_tokens) return 0;
    char *key = token_key(c, target, count);
    if (!key) return fail(err, errlen, "invalid ds4 cache token key");
    ds4_kvstore_load_result result = {0};
    int loaded = ds4_kvstore_try_load_text(&c->store, c->engine, session,
        key, NULL, &result, NULL, false);
    if (loaded > 0) {
        const ds4_tokens *live = ds4_session_tokens(session);
        /* Verify payload tokens AND the exact key boundary. A matching hash or
         * same-length token history alone cannot authorize graph reuse. */
        bool valid = ds4_session_payload_bytes(session) > 0 &&
            !ds4_session_has_vision_state(session) && exact_prefix(live, target, count) &&
            live->len == loaded &&
            result.text_bytes == strlen(c->namespace_key) + 1 + (size_t)loaded * 9;
        if (!valid) {
            ds4_session_invalidate(session);
            if (result.path) unlink(result.path);
            loaded = 0;
        }
    }
    ds4_kvstore_load_result_free(&result);
    free(key);
    return loaded;
}

#ifdef MLXSERVE_DS4_KV_CACHE_TEST
/* Hermetic integration test: the real native kvstore, with only graph payload
 * I/O replaced. Build this TU + ds4_kvstore.c with the define and pass a temp dir. */
#include <assert.h>
#include <sys/stat.h>
#include <time.h>

struct ds4_engine { int unused; };
struct ds4_session { ds4_tokens tokens; bool valid, vision; };
static const char *test_dir;
static int staged_count;

void ds4_tokens_free(ds4_tokens *v) { free(v->v); memset(v, 0, sizeof(*v)); }
void ds4_tokens_push(ds4_tokens *v, int token) {
    if (v->len == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 16;
        v->v = realloc(v->v, (size_t)v->cap * sizeof(int));
        assert(v->v);
    }
    v->v[v->len++] = token;
}
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src) {
    dst->len = 0;
    for (int i = 0; i < src->len; i++) ds4_tokens_push(dst, src->v[i]);
}
bool ds4_tokens_starts_with(const ds4_tokens *v, const ds4_tokens *p) {
    return exact_prefix(p, v->v, v->len);
}
const ds4_tokens *ds4_session_tokens(ds4_session *s) { return &s->tokens; }
uint64_t ds4_session_payload_bytes(ds4_session *s) { return s && s->valid ? 1 : 0; }
bool ds4_session_has_vision_state(const ds4_session *s) { return s->vision; }
void ds4_session_invalidate(ds4_session *s) { s->valid = false; s->tokens.len = 0; }
int ds4_session_ctx(ds4_session *s) { (void)s; return 8192; }
int ds4_engine_model_id(ds4_engine *e) { (void)e; return 0; }
int ds4_engine_routed_quant_bits(ds4_engine *e) { (void)e; return 4; }
char *ds4_token_text(ds4_engine *e, int token, size_t *len) {
    (void)e; (void)token; (void)len;
    abort(); /* Namespaced token keys must never use rendering/tokenization. */
}
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out) {
    (void)e; (void)text; (void)out;
    abort();
}
int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *p,
                              char *err, size_t errlen) {
    (void)err; (void)errlen;
    staged_count++;
    p->path = ds4_kvstore_path_join(test_dir, "payload.XXXXXX");
    int fd = mkstemp(p->path);
    assert(fd >= 0);
    FILE *fp = fdopen(fd, "wb");
    assert(fp);
    assert(fwrite(&s->tokens.len, sizeof(int), 1, fp) == 1);
    assert(fwrite(s->tokens.v, sizeof(int), (size_t)s->tokens.len, fp) == (size_t)s->tokens.len);
    p->bytes = sizeof(int) * (1ull + (uint64_t)s->tokens.len);
    assert(fclose(fp) == 0);
    return 0;
}
int ds4_session_write_staged_payload(const ds4_session_payload_file *p,
                                     FILE *fp, char *err, size_t errlen) {
    (void)err; (void)errlen;
    FILE *src = fopen(p->path, "rb");
    assert(src);
    char buf[256];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), src))) assert(fwrite(buf, 1, n, fp) == n);
    assert(!ferror(src));
    assert(fclose(src) == 0);
    return 0;
}
void ds4_session_payload_file_free(ds4_session_payload_file *p) {
    assert(unlink(p->path) == 0);
    free(p->path);
    memset(p, 0, sizeof(*p));
}
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t bytes,
                             char *err, size_t errlen) {
    (void)err; (void)errlen;
    int n;
    if (fread(&n, sizeof(n), 1, fp) != 1 || n < 0 ||
        (uint64_t)n + 1 != bytes / sizeof(int)) return 1;
    s->tokens.len = 0;
    for (int i = 0; i < n; i++) {
        int token;
        if (fread(&token, sizeof(token), 1, fp) != 1) return 1;
        ds4_tokens_push(&s->tokens, token);
    }
    s->valid = true;
    return 0;
}

static ds4_session make_session(const int *tokens, int n) {
    ds4_session s = {.valid = true};
    for (int i = 0; i < n; i++) ds4_tokens_push(&s.tokens, tokens[i]);
    return s;
}

static char *test_subdir(const char *name) {
    char *path = ds4_kvstore_path_join(test_dir, name);
    assert(mkdir(path, 0700) == 0);
    return path;
}

static char *test_entry_path(mlxserve_ds4_kv_cache *c, const int *tokens, int n) {
    char *key = token_key(c, tokens, n);
    char sha[41];
    ds4_kvstore_sha1_bytes_hex(key, strlen(key), sha);
    free(key);
    return ds4_kvstore_path_for_sha(&c->store, sha);
}

/* Test-only view of the native index. The native eviction refreshes it. */
static uint64_t indexed_bytes(mlxserve_ds4_kv_cache *c) {
    ds4_kvstore_evict(&c->store, NULL, 0, NULL);
    uint64_t total = 0;
    for (int i = 0; i < c->store.len; i++) total += c->store.entry[i].file_size;
    return total;
}

/* Test-only: a checkpoint whose stored text is shorter than its token list. */
static char *write_mismatched_entry(mlxserve_ds4_kv_cache *c, const int *tokens,
                                     int n, int text_tokens) {
    char *text = token_key(c, tokens, text_tokens);
    assert(text);
    char sha[41];
    ds4_kvstore_sha1_bytes_hex(text, strlen(text), sha);
    char *path = ds4_kvstore_path_for_sha(&c->store, sha);
    FILE *fp = fopen(path, "wb");
    assert(fp);
    uint64_t now = (uint64_t)time(NULL);
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    ds4_kvstore_fill_header(h, 0, 4, DS4_KVSTORE_REASON_COLD, 0,
                            (uint32_t)n, 0, 8192, now, now,
                            sizeof(int) * (1ull + (uint64_t)n));
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, (uint32_t)strlen(text));
    assert(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
    assert(fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb));
    assert(fwrite(text, 1, strlen(text), fp) == strlen(text));
    int count = n;
    assert(fwrite(&count, sizeof(count), 1, fp) == 1);
    assert(fwrite(tokens, sizeof(int), (size_t)n, fp) == (size_t)n);
    assert(fclose(fp) == 0);
    free(text);
    return path;
}

int main(int argc, char **argv) {
    assert(argc == 2);
    test_dir = argv[1];
    ds4_engine engine = {0};
    ds4_session dest = {0};
    int prompt[513];
    for (int i = 0; i < 513; i++) prompt[i] = i;
    ds4_session source = make_session(prompt, 512);
    char err[256] = {0};

    /* A zero allowance would become the native default budget, so the engine
     * bypasses the cache instead of opening it. */
    err[0] = '\0';
    assert(mlxserve_ds4_kv_cache_open(&engine, test_dir, "checkpoint-A", 0, 0,
                                      err, sizeof(err)) == NULL);
    assert(err[0] != '\0');

    const uint64_t budget = 1024 * 1024;
    mlxserve_ds4_kv_cache *c = mlxserve_ds4_kv_cache_open(&engine, test_dir,
        "checkpoint-A", budget, 0, err, sizeof(err));
    assert(c && c->store.opt.min_tokens == 512);
    assert(c->store.budget_bytes == budget);
    /* Below the minimum, without a payload or with vision state: nothing is stored. */
    assert(mlxserve_ds4_kv_cache_store(c, &source, prompt, 511, "cold", err, sizeof(err)) == 0);
    source.valid = false;
    assert(mlxserve_ds4_kv_cache_store(c, &source, prompt, 512, "cold", err, sizeof(err)) == 0);
    source.valid = true; source.vision = true;
    assert(mlxserve_ds4_kv_cache_store(c, &source, prompt, 512, "cold", err, sizeof(err)) == 0);
    source.vision = false;
    assert(staged_count == 0);
    assert(mlxserve_ds4_kv_cache_store(c, &source, prompt, 512, "cold", err, sizeof(err)) == 1);
    assert(staged_count == 1);
    /* Identical prompt checkpoint is idempotent and does not stage again. */
    assert(mlxserve_ds4_kv_cache_store(c, &source, prompt, 512, "shutdown", err, sizeof(err)) == 1);
    assert(staged_count == 1);
    /* Another namespace cannot see this checkpoint. */
    mlxserve_ds4_kv_cache_close(c);
    c = mlxserve_ds4_kv_cache_open(&engine, test_dir, "checkpoint-B", budget, 0, err, sizeof(err));
    assert(c && mlxserve_ds4_kv_cache_restore(c, &dest, prompt, 513, err, sizeof(err)) == 0);
    mlxserve_ds4_kv_cache_close(c);
    c = mlxserve_ds4_kv_cache_open(&engine, test_dir, "checkpoint-A", budget, 0, err, sizeof(err));
    assert(c && mlxserve_ds4_kv_cache_restore(c, &dest, prompt, 513, err, sizeof(err)) == 512);
    assert(dest.valid && exact_prefix(&dest.tokens, prompt, 513));
    /* A different token history is not a hit. */
    prompt[511]++;
    assert(mlxserve_ds4_kv_cache_restore(c, &dest, prompt, 513, err, sizeof(err)) == 0);
    prompt[511]--;
    /* A corrupt payload is refused and the session stays invalid. */
    char *path = test_entry_path(c, prompt, 512);
    FILE *fp = fopen(path, "r+b");
    assert(fp);
    ds4_kvstore_entry header = {0}; uint32_t text_bytes;
    assert(ds4_kvstore_read_header(fp, &header, &text_bytes));
    long payload = DS4_KVSTORE_FIXED_HEADER + 4 + text_bytes;
    assert(fseek(fp, payload, SEEK_SET) == 0);
    int bad_count = -1;
    assert(fwrite(&bad_count, sizeof(int), 1, fp) == 1 && fclose(fp) == 0);
    assert(mlxserve_ds4_kv_cache_restore(c, &dest, prompt, 513, err, sizeof(err)) == 0);
    assert(!dest.valid);
    assert(unlink(path) == 0);
    /* A payload with other tokens is not a hit and its checkpoint is removed. */
    assert(mlxserve_ds4_kv_cache_store(c, &source, prompt, 512, "cold", err, sizeof(err)) == 1);
    fp = fopen(path, "r+b"); assert(fp);
    assert(fseek(fp, payload + sizeof(int), SEEK_SET) == 0);
    int wrong_token = 1234;
    assert(fwrite(&wrong_token, sizeof(int), 1, fp) == 1 && fclose(fp) == 0);
    assert(mlxserve_ds4_kv_cache_restore(c, &dest, prompt, 513, err, sizeof(err)) == 0);
    assert(!dest.valid && access(path, F_OK) != 0);
    free(path);
    /* A checkpoint whose stored text is shorter than its token list is not a
     * hit, and it is removed. */
    char *mismatched = write_mismatched_entry(c, prompt, 512, 511);
    assert(mlxserve_ds4_kv_cache_restore(c, &dest, prompt, 513, err, sizeof(err)) == 0);
    assert(!dest.valid && access(mismatched, F_OK) != 0);
    free(mismatched);
    /* The stored tokens must be the live token history. */
    int wrong[512];
    for (int i = 0; i < 512; i++) wrong[i] = prompt[i];
    wrong[0] = 999;
    assert(mlxserve_ds4_kv_cache_store(c, &source, wrong, 512, "cold", err, sizeof(err)) == 0);
    char *wrong_path = test_entry_path(c, wrong, 512);
    assert(access(wrong_path, F_OK) != 0);
    free(wrong_path);
    mlxserve_ds4_kv_cache_close(c);

    /* An allowance below one MiB is kept exactly, not turned into the native
     * default, and a checkpoint larger than it is declined. */
    char *small_dir = test_subdir("small");
    const uint64_t small_budget = 4096;
    mlxserve_ds4_kv_cache *small = mlxserve_ds4_kv_cache_open(&engine, small_dir,
        "checkpoint-A", small_budget, 0, err, sizeof(err));
    assert(small && small->store.budget_bytes == small_budget);
    assert(small->store.budget_bytes !=
           (uint64_t)DS4_KVSTORE_DEFAULT_MB * 1024ull * 1024ull);
    assert(mlxserve_ds4_kv_cache_store(small, &source, prompt, 512, "cold", err, sizeof(err)) == 0);
    char *small_path = test_entry_path(small, prompt, 512);
    assert(access(small_path, F_OK) != 0);
    assert(indexed_bytes(small) == 0);
    free(small_path);
    mlxserve_ds4_kv_cache_close(small);
    free(small_dir);

    /* A pool that holds more than its allowance drops the excess when it opens. */
    char *reopen_dir = test_subdir("reopen");
    mlxserve_ds4_kv_cache *roomy = mlxserve_ds4_kv_cache_open(&engine, reopen_dir,
        "checkpoint-A", budget, 0, err, sizeof(err));
    assert(roomy);
    assert(mlxserve_ds4_kv_cache_store(roomy, &source, prompt, 512, "cold", err, sizeof(err)) == 1);
    char *roomy_path = test_entry_path(roomy, prompt, 512);
    assert(access(roomy_path, F_OK) == 0);
    mlxserve_ds4_kv_cache_close(roomy);
    mlxserve_ds4_kv_cache *tight = mlxserve_ds4_kv_cache_open(&engine, reopen_dir,
        "checkpoint-A", small_budget, 0, err, sizeof(err));
    assert(tight && tight->store.budget_bytes == small_budget);
    assert(access(roomy_path, F_OK) != 0);
    assert(indexed_bytes(tight) == 0);
    free(roomy_path);
    mlxserve_ds4_kv_cache_close(tight);
    free(reopen_dir);

    /* An allowance that is not a whole MiB is kept exactly. */
    char *exact_dir = test_subdir("exact");
    const uint64_t exact_budget = 1024 * 1024 + 1234;
    mlxserve_ds4_kv_cache *exact = mlxserve_ds4_kv_cache_open(&engine, exact_dir,
        "checkpoint-A", exact_budget, 0, err, sizeof(err));
    assert(exact && exact->store.budget_bytes == exact_budget);
    mlxserve_ds4_kv_cache_close(exact);
    free(exact_dir);

    /* The pool stays inside its allowance: the third checkpoint evicts the
     * least useful one, the restored checkpoint survives. */
    char *pool_dir = test_subdir("pool");
    const uint64_t two_checkpoints = 14000;
    mlxserve_ds4_kv_cache *pool = mlxserve_ds4_kv_cache_open(&engine, pool_dir,
        "checkpoint-A", two_checkpoints, 0, err, sizeof(err));
    assert(pool && pool->store.budget_bytes == two_checkpoints);
    int second_prompt[513], third_prompt[513];
    for (int i = 0; i < 513; i++) { second_prompt[i] = i + 1; third_prompt[i] = i + 2; }
    ds4_session second = make_session(second_prompt, 512);
    ds4_session third = make_session(third_prompt, 512);
    assert(mlxserve_ds4_kv_cache_store(pool, &source, prompt, 512, "cold", err, sizeof(err)) == 1);
    assert(mlxserve_ds4_kv_cache_store(pool, &second, second_prompt, 512, "cold", err, sizeof(err)) == 1);
    assert(indexed_bytes(pool) <= two_checkpoints);
    assert(mlxserve_ds4_kv_cache_restore(pool, &dest, prompt, 513, err, sizeof(err)) == 512);
    assert(mlxserve_ds4_kv_cache_store(pool, &third, third_prompt, 512, "cold", err, sizeof(err)) == 1);
    char *keep = test_entry_path(pool, prompt, 512);
    char *drop = test_entry_path(pool, second_prompt, 512);
    char *added = test_entry_path(pool, third_prompt, 512);
    assert(access(keep, F_OK) == 0);
    assert(access(drop, F_OK) != 0);
    assert(access(added, F_OK) == 0);
    assert(indexed_bytes(pool) <= two_checkpoints);
    free(keep); free(drop); free(added);
    mlxserve_ds4_kv_cache_close(pool);
    free(pool_dir);

    ds4_tokens_free(&source.tokens); ds4_tokens_free(&dest.tokens);
    ds4_tokens_free(&second.tokens); ds4_tokens_free(&third.tokens);
    puts("ds4 native kvstore integration: passed");
    return 0;
}
#endif
