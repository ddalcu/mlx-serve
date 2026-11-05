#include "ds4_kv_cache.h"
#include "../lib/ds4/ds4_kvstore.h"

#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

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

static bool native_temporary(const char *name) {
    if (strlen(name) <= 48 || strncmp(name + 43, ".tmp.", 5)) return false;
    char base[44], sha[41];
    memcpy(base, name, 43);
    base[43] = '\0';
    return ds4_kvstore_sha_hex_name(base, sha) &&
        strspn(name + 48, "0123456789") == strlen(name + 48);
}

/* Count every regular file, including failed native writes and corrupt entries.
 * Only native-format files are eligible for cleanup; never follow symlinks. */
static uint64_t scan(mlxserve_ds4_kv_cache *c, bool clean) {
    DIR *dir = opendir(c->store.dir);
    if (!dir) return UINT64_MAX;
    uint64_t bytes = 0;
    struct dirent *de;
    errno = 0;
    while ((de = readdir(dir))) {
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        char *path = ds4_kvstore_path_join(c->store.dir, de->d_name);
        struct stat st;
        if (lstat(path, &st) != 0) { free(path); bytes = UINT64_MAX; break; }
        if (S_ISREG(st.st_mode)) {
            char sha[41];
            ds4_kvstore_entry entry = {0};
            bool native = ds4_kvstore_sha_hex_name(de->d_name, sha);
            bool valid = !native || ds4_kvstore_read_entry_file(path, sha, &entry);
            ds4_kvstore_entry_free(&entry);
            // All pool operations are serialized, so a native temp here is a crash leftover.
            if (clean && ((native && !valid) || native_temporary(de->d_name)) && unlink(path) == 0) {
                free(path);
                errno = 0;
                continue;
            }
            uint64_t size = st.st_size > 0 ? (uint64_t)st.st_size : 0;
            bytes = UINT64_MAX - bytes < size ? UINT64_MAX : bytes + size;
        }
        free(path);
        errno = 0;
    }
    if (errno) bytes = UINT64_MAX;
    closedir(dir);
    return bytes;
}

uint64_t mlxserve_ds4_kv_cache_physical_bytes(mlxserve_ds4_kv_cache *c) {
    return scan(c, false);
}

int mlxserve_ds4_kv_cache_prune(mlxserve_ds4_kv_cache *c, char *err, size_t errlen) {
    if (scan(c, true) == UINT64_MAX) return fail(err, errlen, "cannot scan ds4 cache pool");
    uint64_t allowance = c->store.budget_bytes;
    /* Native zero means unlimited. One byte evicts every native checkpoint. */
    c->store.budget_bytes = allowance ? allowance : 1;
    ds4_kvstore_evict(&c->store, NULL, 0, NULL);
    c->store.budget_bytes = allowance;
    if (scan(c, false) > allowance)
        return fail(err, errlen, "ds4 cache pool exceeds allowance after pruning");
    return 0;
}

int mlxserve_ds4_kv_cache_set_budget(mlxserve_ds4_kv_cache *c, uint64_t bytes,
                                    char *err, size_t errlen) {
    c->store.budget_bytes = bytes;
    /* Exhausted write headroom must not destroy a useful checkpoint before
     * lookup. Explicit prune still enforces zero by deleting all entries. */
    if (!bytes) return 0;
    return mlxserve_ds4_kv_cache_prune(c, err, errlen);
}

mlxserve_ds4_kv_cache *mlxserve_ds4_kv_cache_open(
    ds4_engine *engine, const char *dir, const char *namespace_key,
    uint64_t budget_bytes, int min_tokens, char *err, size_t errlen) {
    if (!engine || !dir || !namespace_key || !namespace_key[0] || min_tokens < 0) {
        fail(err, errlen, "invalid ds4 cache options");
        return NULL;
    }
    mlxserve_ds4_kv_cache *c = calloc(1, sizeof(*c));
    if (!c) { fail(err, errlen, "cannot allocate ds4 cache"); return NULL; }
    c->engine = engine;
    c->namespace_key = strdup(namespace_key);
    ds4_kvstore_options opt = ds4_kvstore_default_options();
    if (min_tokens) opt.min_tokens = min_tokens;
    /* Native open eagerly evicts. Scan first without eviction, then apply the
     * exact allowance below, preserving read-only restores at zero headroom. */
    uint64_t mb = UINT64_MAX / (1024 * 1024);
    if (!c->namespace_key || !ds4_kvstore_open(&c->store, dir, mb,
                                              true, opt, "mlx-serve ds4", NULL, NULL)) {
        fail(err, errlen, "cannot open ds4 cache pool");
        mlxserve_ds4_kv_cache_close(c);
        return NULL;
    }
    if (mlxserve_ds4_kv_cache_set_budget(c, budget_bytes, err, errlen) != 0) {
        mlxserve_ds4_kv_cache_close(c);
        return NULL;
    }
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
    if (!c->store.budget_bytes || count < c->store.opt.min_tokens ||
        ds4_session_payload_bytes(session) == 0 || ds4_session_has_vision_state(session)) return 0;
    const ds4_tokens *live = ds4_session_tokens(session);
    if (!exact_prefix(live, tokens, count) || live->len != count) return 0;
    char *key = token_key(c, tokens, count);
    if (!key) return fail(err, errlen, "invalid ds4 cache token key");
    bool ok = ds4_kvstore_store_live_prefix_text(&c->store, c->engine, session,
        live, count, reason, key, 0, "namespaced-token-ids", NULL, err, errlen);
    free(key);
    if (!ok) return err && errlen && err[0] ? -1 : 0;
    if (mlxserve_ds4_kv_cache_prune(c, err, errlen) != 0) return -1;
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

static char *test_entry_path(mlxserve_ds4_kv_cache *c, const int *tokens, int n) {
    char *key = token_key(c, tokens, n);
    char sha[41];
    ds4_kvstore_sha1_bytes_hex(key, strlen(key), sha);
    free(key);
    return ds4_kvstore_path_for_sha(&c->store, sha);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    test_dir = argv[1];
    ds4_engine engine = {0};
    ds4_session source = {.valid = true}, dest = {0};
    int prompt[513];
    for (int i = 0; i < 513; i++) { prompt[i] = i; ds4_tokens_push(&source.tokens, i); }
    source.tokens.len = 512;
    char err[256] = {0};
    const uint64_t budget = 1024 * 1024;
    mlxserve_ds4_kv_cache *c = mlxserve_ds4_kv_cache_open(&engine, test_dir,
        "checkpoint-A", budget, 0, err, sizeof(err));
    assert(c && c->store.opt.min_tokens == 512);
    assert(mlxserve_ds4_kv_cache_store(c, &source, prompt, 511, "cold", err, sizeof(err)) == 0);
    source.valid = false;
    assert(mlxserve_ds4_kv_cache_store(c, &source, prompt, 512, "cold", err, sizeof(err)) == 0);
    source.valid = true; source.vision = true;
    assert(mlxserve_ds4_kv_cache_store(c, &source, prompt, 512, "cold", err, sizeof(err)) == 0);
    source.vision = false;
    assert(staged_count == 0);
    assert(mlxserve_ds4_kv_cache_store(c, &source, prompt, 512, "cold", err, sizeof(err)) == 1);
    uint64_t saved = mlxserve_ds4_kv_cache_physical_bytes(c);
    assert(saved > 0 && saved < budget && staged_count == 1);
    /* Identical prompt checkpoint is idempotent and does not stage again. */
    assert(mlxserve_ds4_kv_cache_store(c, &source, prompt, 512, "shutdown", err, sizeof(err)) == 1);
    assert(staged_count == 1);
    mlxserve_ds4_kv_cache_close(c);
    c = mlxserve_ds4_kv_cache_open(&engine, test_dir, "checkpoint-B", budget, 0, err, sizeof(err));
    assert(c && mlxserve_ds4_kv_cache_restore(c, &dest, prompt, 513, err, sizeof(err)) == 0);
    mlxserve_ds4_kv_cache_close(c);
    c = mlxserve_ds4_kv_cache_open(&engine, test_dir, "checkpoint-A", budget, 0, err, sizeof(err));
    assert(c && mlxserve_ds4_kv_cache_restore(c, &dest, prompt, 513, err, sizeof(err)) == 512);
    assert(dest.valid && exact_prefix(&dest.tokens, prompt, 513));
    prompt[511]++;
    assert(mlxserve_ds4_kv_cache_restore(c, &dest, prompt, 513, err, sizeof(err)) == 0);
    prompt[511]--;
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
    assert(mlxserve_ds4_kv_cache_store(c, &source, prompt, 512, "cold", err, sizeof(err)) == 1);
    fp = fopen(path, "r+b"); assert(fp);
    assert(fseek(fp, payload + sizeof(int), SEEK_SET) == 0);
    int wrong_token = 1234;
    assert(fwrite(&wrong_token, sizeof(int), 1, fp) == 1 && fclose(fp) == 0);
    assert(mlxserve_ds4_kv_cache_restore(c, &dest, prompt, 513, err, sizeof(err)) == 0);
    assert(!dest.valid && access(path, F_OK) != 0);
    /* Physical accounting includes invalid native files, which pruning removes. */
    fp = fopen(path, "wb"); assert(fp);
    assert(fwrite("broken", 1, 6, fp) == 6 && fclose(fp) == 0);
    assert(mlxserve_ds4_kv_cache_physical_bytes(c) == 6);
    assert(mlxserve_ds4_kv_cache_prune(c, err, sizeof(err)) == 0);
    assert(mlxserve_ds4_kv_cache_physical_bytes(c) == 0);
    char temporary[4096];
    assert(snprintf(temporary, sizeof(temporary), "%s.tmp.1", path) < (int)sizeof(temporary));
    fp = fopen(temporary, "wb"); assert(fp);
    assert(fwrite("leftover", 1, 8, fp) == 8 && fclose(fp) == 0);
    assert(mlxserve_ds4_kv_cache_physical_bytes(c) == 8);
    assert(mlxserve_ds4_kv_cache_prune(c, err, sizeof(err)) == 0);
    assert(mlxserve_ds4_kv_cache_physical_bytes(c) == 0);
    assert(mlxserve_ds4_kv_cache_set_budget(c, saved / 2, err, sizeof(err)) == 0);
    assert(mlxserve_ds4_kv_cache_store(c, &source, prompt, 512, "cold", err, sizeof(err)) == 0);
    assert(mlxserve_ds4_kv_cache_set_budget(c, budget, err, sizeof(err)) == 0);
    assert(mlxserve_ds4_kv_cache_store(c, &source, prompt, 512, "cold", err, sizeof(err)) == 1);
    assert(mlxserve_ds4_kv_cache_set_budget(c, 0, err, sizeof(err)) == 0);
    assert(mlxserve_ds4_kv_cache_physical_bytes(c) == saved);
    assert(mlxserve_ds4_kv_cache_restore(c, &dest, prompt, 513, err, sizeof(err)) == 512);
    int before = staged_count;
    assert(mlxserve_ds4_kv_cache_store(c, &source, prompt, 512, "cold", err, sizeof(err)) == 0);
    assert(staged_count == before);
    assert(mlxserve_ds4_kv_cache_prune(c, err, sizeof(err)) == 0);
    assert(mlxserve_ds4_kv_cache_physical_bytes(c) == 0);
    free(path);
    mlxserve_ds4_kv_cache_close(c);
    ds4_tokens_free(&source.tokens); ds4_tokens_free(&dest.tokens);
    puts("ds4 native kvstore integration: passed");
    return 0;
}
#endif
