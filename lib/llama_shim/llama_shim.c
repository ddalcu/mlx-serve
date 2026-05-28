// Implementation of the mlx-serve llama.cpp shim. See llama_shim.h.
//
// Compiled by build.zig against the staged headers in lib/llama/include and
// linked against lib/llama/lib/libllama.dylib (scripts/fetch-llama.sh stages
// both). This is the only place that touches llama.cpp's real structs.
#include "llama_shim.h"

#include "llama.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct mlx_llama_engine {
    struct llama_model *model;
    const struct llama_vocab *vocab;
};

struct mlx_llama_session {
    struct llama_context *ctx;
    struct mlx_llama_engine *engine;
    int32_t pos; // tokens decoded into the KV cache so far
};

// Prefill chunk size. The default logical batch (n_batch) is 2048, so 512-token
// chunks are always accepted by llama_decode regardless of context params.
#define MLX_LLAMA_PREFILL_CHUNK 512

static pthread_once_t g_backend_once = PTHREAD_ONCE_INIT;

static void backend_init_once(void) {
    llama_backend_init();
}

static void copy_err(char *err, size_t errlen, const char *msg) {
    if (err && errlen > 0) {
        strncpy(err, msg, errlen - 1);
        err[errlen - 1] = '\0';
    }
}

mlx_llama_engine *mlx_llama_open(const char *gguf_path, int32_t n_gpu_layers, char *err, size_t errlen) {
    pthread_once(&g_backend_once, backend_init_once);

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;

    struct llama_model *model = llama_model_load_from_file(gguf_path, mp);
    if (!model) {
        copy_err(err, errlen, "llama_model_load_from_file failed");
        return NULL;
    }

    mlx_llama_engine *e = (mlx_llama_engine *)calloc(1, sizeof(*e));
    if (!e) {
        llama_model_free(model);
        copy_err(err, errlen, "out of memory allocating engine");
        return NULL;
    }
    e->model = model;
    e->vocab = llama_model_get_vocab(model);
    return e;
}

void mlx_llama_close(mlx_llama_engine *e) {
    if (!e) return;
    if (e->model) llama_model_free(e->model);
    free(e);
}

int32_t mlx_llama_eos_token(mlx_llama_engine *e) {
    return (int32_t)llama_vocab_eos(e->vocab);
}

bool mlx_llama_is_eog(mlx_llama_engine *e, int32_t token) {
    return llama_vocab_is_eog(e->vocab, (llama_token)token);
}

int32_t mlx_llama_n_vocab(mlx_llama_engine *e) {
    return llama_vocab_n_tokens(e->vocab);
}

int32_t mlx_llama_tokenize(mlx_llama_engine *e, const char *text, int32_t text_len,
                           bool add_special, bool parse_special,
                           int32_t *out, int32_t out_cap) {
    return llama_tokenize(e->vocab, text, text_len, (llama_token *)out, out_cap,
                          add_special, parse_special);
}

int32_t mlx_llama_token_to_piece(mlx_llama_engine *e, int32_t token, char *buf, int32_t buf_cap) {
    // lstrip=0, special=false: render the literal piece bytes.
    return llama_token_to_piece(e->vocab, (llama_token)token, buf, buf_cap, 0, false);
}

const char *mlx_llama_chat_template(mlx_llama_engine *e) {
    return llama_model_chat_template(e->model, NULL);
}

int32_t mlx_llama_apply_chat_template(mlx_llama_engine *e,
                                      const char **roles, const char **contents, int32_t n_msgs,
                                      bool add_assistant, char *buf, int32_t buf_cap) {
    const char *tmpl = llama_model_chat_template(e->model, NULL);
    size_t n = (size_t)(n_msgs > 0 ? n_msgs : 0);
    struct llama_chat_message *msgs =
        (struct llama_chat_message *)calloc(n ? n : 1, sizeof(struct llama_chat_message));
    if (!msgs) return -1;
    for (size_t i = 0; i < n; i++) {
        msgs[i].role = roles[i];
        msgs[i].content = contents[i];
    }
    int32_t r = llama_chat_apply_template(tmpl, msgs, n, add_assistant, buf, buf_cap);
    free(msgs);
    return r;
}

mlx_llama_session *mlx_llama_session_create(mlx_llama_engine *e, int32_t n_ctx, char *err, size_t errlen) {
    return mlx_llama_session_create_kv_quant(e, n_ctx, 0, 0, err, errlen);
}

mlx_llama_session *mlx_llama_session_create_kv_quant(mlx_llama_engine *e,
                                                    int32_t n_ctx,
                                                    int32_t type_k,
                                                    int32_t type_v,
                                                    char *err, size_t errlen) {
    struct llama_context_params cp = llama_context_default_params();
    if (n_ctx > 0) cp.n_ctx = (uint32_t)n_ctx;
    // Force the full-size SWA cache. With swa_full=false (the libllama default
    // since b73xx) sliding-window-attention layers only expose `window`-many KV
    // slots per sequence; after `llama_memory_seq_rm` trims a divergent tail in
    // a persistent prompt-prefix-reuse session, the next `llama_decode` can
    // fail to find a contiguous block of free slots and abort the prefill with
    //   init_batch: failed to prepare attention ubatches
    //   decode: failed to find a memory slot for batch of size 512
    // mlx-serve owns its own ctx-size cap up the stack, so the extra KV that
    // swa_full=true costs (window→full per SWA layer) is exactly what we
    // already accounted for. Matches `llama-server --swa-full` and addresses
    // llama.cpp issues #19794 / #21831 / #17196 for hybrid/SWA GGUFs.
    cp.swa_full = true;
    // Flash-attention is required when K/V are quantized — llama.cpp's plain
    // SDPA path only supports F16/F32 KV. flash_attn defaults vary by version;
    // turn it on whenever the caller asks for a non-default KV type so we
    // don't fall over inside llama_decode.
    if (type_k != 0 || type_v != 0) {
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    if (type_k != 0) cp.type_k = (enum ggml_type)type_k;
    if (type_v != 0) cp.type_v = (enum ggml_type)type_v;

    struct llama_context *ctx = llama_init_from_model(e->model, cp);
    if (!ctx) {
        copy_err(err, errlen, "llama_init_from_model failed");
        return NULL;
    }
    mlx_llama_session *s = (mlx_llama_session *)calloc(1, sizeof(*s));
    if (!s) {
        llama_free(ctx);
        copy_err(err, errlen, "out of memory allocating session");
        return NULL;
    }
    s->ctx = ctx;
    s->engine = e;
    s->pos = 0;
    return s;
}

void mlx_llama_session_free(mlx_llama_session *s) {
    if (!s) return;
    if (s->ctx) llama_free(s->ctx);
    free(s);
}

// Decode a contiguous run of tokens. llama_batch_get_one tracks positions
// automatically from the KV state and (logits == NULL) outputs logits for the
// last token only. Advances s->pos on success.
static int32_t decode_run(mlx_llama_session *s, const int32_t *tokens, int32_t n) {
    struct llama_batch batch = llama_batch_get_one((llama_token *)tokens, n);
    int32_t rc = llama_decode(s->ctx, batch);
    if (rc != 0) return rc;
    s->pos += n;
    return 0;
}

int32_t mlx_llama_session_sync(mlx_llama_session *s, const int32_t *tokens, int32_t n_tokens,
                               char *err, size_t errlen) {
    int32_t off = 0;
    while (off < n_tokens) {
        int32_t n = n_tokens - off;
        if (n > MLX_LLAMA_PREFILL_CHUNK) n = MLX_LLAMA_PREFILL_CHUNK;
        if (decode_run(s, tokens + off, n) != 0) {
            copy_err(err, errlen, "llama_decode failed during prefill");
            return -1;
        }
        off += n;
    }
    return 0;
}

int32_t mlx_llama_session_trim(mlx_llama_session *s, int32_t n_keep) {
    if (n_keep < 0) n_keep = 0;
    if (n_keep >= s->pos) return 0; // nothing resident beyond n_keep
    // Single-sequence (seq 0) usage: remove positions [n_keep, inf). Removing a
    // whole tail of one sequence never returns false (see llama.h seq_rm doc).
    llama_memory_seq_rm(llama_get_memory(s->ctx), 0, n_keep, -1);
    s->pos = n_keep;
    return 0;
}

void mlx_llama_session_reset(mlx_llama_session *s) {
    llama_memory_clear(llama_get_memory(s->ctx), true);
    s->pos = 0;
}

int32_t mlx_llama_session_eval(mlx_llama_session *s, int32_t token, char *err, size_t errlen) {
    int32_t t = token;
    if (decode_run(s, &t, 1) != 0) {
        copy_err(err, errlen, "llama_decode failed");
        return -1;
    }
    return 0;
}

int32_t mlx_llama_session_argmax(mlx_llama_session *s) {
    const float *logits = llama_get_logits_ith(s->ctx, -1);
    if (!logits) return -1;
    int32_t n = llama_vocab_n_tokens(s->engine->vocab);
    int32_t best = 0;
    float best_v = logits[0];
    for (int32_t i = 1; i < n; i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}

int32_t mlx_llama_session_sample(mlx_llama_session *s, float temperature, int32_t top_k,
                                 float top_p, float min_p, uint64_t *rng) {
    if (temperature <= 0.0f) return mlx_llama_session_argmax(s);

    struct llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    struct llama_sampler *chain = llama_sampler_chain_init(sp);
    if (top_k > 0) llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    if (top_p > 0.0f && top_p < 1.0f) llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
    if (min_p > 0.0f) llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));

    uint64_t state = (rng && *rng) ? *rng : 0x106689D45497FDB5ULL;
    uint32_t seed = (uint32_t)(state ^ ((uint64_t)s->pos * 0x9E3779B97F4A7C15ULL));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));

    int32_t tok = (int32_t)llama_sampler_sample(chain, s->ctx, -1);
    llama_sampler_free(chain);

    if (rng) {
        // xorshift64 so the next draw uses a fresh seed (reproducible chain).
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *rng = state;
    }
    return tok;
}

int32_t mlx_llama_session_pos(mlx_llama_session *s) {
    return s->pos;
}
