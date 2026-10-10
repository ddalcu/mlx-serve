// Implementation of the mlx-serve llama.cpp shim. See llama_shim.h.
//
// Compiled by build.zig against the staged headers in lib/llama/include and
// linked against lib/llama/lib/libllama.dylib (scripts/fetch-llama.sh stages
// both). This is the only place that touches llama.cpp's real structs. C++ only
// for the NextN staging calls below; the exported surface is the C header.
#include "llama_shim.h"

#include "llama.h"

#include <dlfcn.h>
#include <pthread.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// llama.cpp's staging API for MTP (src/llama-ext.h, not in the xcframework's
// headers). Exported with C++ linkage, so the mangled names pin these signatures
// at link time.
void    llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);
float * llama_get_embeddings_nextn(struct llama_context * ctx);
float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

struct mlx_llama_engine {
    struct llama_model *model;
    struct llama_model *mtp_model; // the sidecar head; NULL when the trunk carries its own
    const struct llama_vocab *vocab;
    bool has_mtp;
};

namespace {
struct Row {
    llama_token tok;
    llama_pos pos;
    llama_seq_id seq;
};
} // namespace

struct mlx_llama_ctx {
    mlx_llama_engine *e;
    struct llama_context *tgt;
    struct llama_context *dft; // MTP head, NULL without MTP
    struct llama_batch_ext *batch;
    struct llama_batch_ext *dbatch;
    int32_t n_seq;
    int32_t n_batch;
    int32_t n_vocab;
    int32_t mtp_drafts;
    size_t n_embd_h;              // width of a NextN hidden row
    std::vector<llama_pos> pos;   // tokens in each sequence's KV
    std::vector<int32_t> out_idx; // batch row holding each sequence's logits, -1 none
    std::vector<float> pending_h; // per sequence: the target hidden at its last KV position
    std::vector<float> draft_h;   // the head's hidden, fed to its next draft step
    std::vector<llama_token> drafts;
    std::vector<Row> rows;
};

static pthread_once_t g_backend_once = PTHREAD_ONCE_INIT;

// llama.cpp and ggml log to stderr by default (the loader dumps every metadata key);
// route them through mlx-serve's leveled log (src/arch/llama.zig). CONT continues a line.
extern "C" void mlx_llama_log(bool warn, const char *text);
static void log_to_mlx_serve(enum ggml_log_level level, const char *text, void *) {
    static enum ggml_log_level last = GGML_LOG_LEVEL_INFO;
    if (level != GGML_LOG_LEVEL_CONT) last = level;
    mlx_llama_log(last >= GGML_LOG_LEVEL_WARN, text);
}

static void backend_init_once(void) {
    llama_log_set(log_to_mlx_serve, nullptr);
    llama_backend_init();
    // The Linux release loads its backends (CUDA, CPU) as plugins, which
    // ggml looks for beside the executable; ours sit beside libllama.
    Dl_info info;
    if (ggml_backend_reg_count() == 0 && dladdr((void *)llama_backend_init, &info)) {
        std::string dir(info.dli_fname);
        ggml_backend_load_all_from_path(dir.substr(0, dir.rfind('/')).c_str());
    }
}

static void copy_err(char *err, size_t errlen, const char *msg) {
    if (err && errlen > 0) {
        strncpy(err, msg, errlen - 1);
        err[errlen - 1] = '\0';
    }
}

extern "C" {

mlx_llama_engine *mlx_llama_open(const char *gguf_path, int32_t n_gpu_layers, const char *mtp_path,
                                 bool load_mtp, char *err, size_t errlen) {
    pthread_once(&g_backend_once, backend_init_once);

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    mp.load_mtp = load_mtp && mtp_path == NULL;

    struct llama_model *model = llama_model_load_from_file(gguf_path, mp);
    if (!model) {
        copy_err(err, errlen, "llama_model_load_from_file failed");
        return NULL;
    }
    // A head that does not load costs the drafts, never the model.
    struct llama_model *mtp_model = NULL;
    if (mtp_path) {
        struct llama_model_params hp = llama_model_default_params();
        hp.n_gpu_layers = n_gpu_layers;
        hp.load_mtp = true; // the head IS the NextN tensors, skipped otherwise
        mtp_model = llama_model_load_from_file(mtp_path, hp);
    }

    mlx_llama_engine *e = (mlx_llama_engine *)calloc(1, sizeof(*e));
    if (!e) {
        if (mtp_model) llama_model_free(mtp_model);
        llama_model_free(model);
        copy_err(err, errlen, "out of memory allocating engine");
        return NULL;
    }
    e->model = model;
    e->mtp_model = mtp_model;
    e->vocab = llama_model_get_vocab(model);
    e->has_mtp = mtp_model ? llama_model_n_layer_nextn(mtp_model) > 0
                           : mp.load_mtp && llama_model_n_layer_nextn(model) > 0;
    return e;
}

void mlx_llama_close(mlx_llama_engine *e) {
    if (!e) return;
    if (e->mtp_model) llama_model_free(e->mtp_model);
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

bool mlx_llama_has_mtp(mlx_llama_engine *e) {
    return e->has_mtp;
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

} // extern "C"

// Build the MTP head context beside the target, or say why not in `err`.
static bool attach_mtp(mlx_llama_ctx *c, struct llama_context_params cp, int32_t drafts, char *err, size_t errlen) {
    mlx_llama_engine *e = c->e;
    struct llama_model *head = e->mtp_model ? e->mtp_model : e->model;
    if (llama_model_n_embd_out(head) != llama_model_n_embd_out(e->model) ||
        llama_vocab_n_tokens(llama_model_get_vocab(head)) != c->n_vocab) {
        copy_err(err, errlen, "the MTP head does not match the model (hidden width or vocab)");
        return false;
    }
    // A recurrent trunk rolls rejected drafts back only within its snapshot window.
    const bool recurrent = llama_model_is_recurrent(e->model) || llama_model_is_hybrid(e->model);
    if (recurrent && (int32_t)llama_n_rs_seq(c->tgt) < drafts) {
        copy_err(err, errlen, "the model's recurrent state cannot roll back rejected drafts");
        return false;
    }
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp.n_rs_seq = 0;
    c->dft = llama_init_from_model(head, cp);
    if (!c->dft) {
        copy_err(err, errlen, "llama_init_from_model failed for the MTP head");
        return false;
    }
    c->dbatch = llama_batch_ext_init(c->dft);
    llama_set_embeddings_nextn(c->tgt, true, /*masked*/ false);
    llama_set_embeddings_nextn(c->dft, true, /*masked*/ true);
    c->mtp_drafts = drafts;
    c->n_embd_h = (size_t)llama_model_n_embd_out(head);
    c->pending_h.assign((size_t)c->n_seq * c->n_embd_h, 0.0f);
    c->draft_h.assign(c->n_embd_h, 0.0f);
    c->drafts.assign((size_t)drafts, 0);
    return true;
}

extern "C" mlx_llama_ctx *mlx_llama_ctx_create(mlx_llama_engine *e, const mlx_llama_ctx_params *p,
                                               char *err, size_t errlen) {
    const int32_t n_seq = p->n_seq > 0 ? p->n_seq : 1;
    const int32_t n_ctx = p->n_ctx > 0 ? p->n_ctx : llama_model_n_ctx_train(e->model);

    struct llama_context_params cp = llama_context_default_params();
    // Each sequence gets its own KV stream of n_ctx cells and attends only to it.
    cp.n_seq_max = (uint32_t)n_seq;
    cp.n_ctx = (uint32_t)n_ctx * (uint32_t)n_seq;
    cp.kv_unified = false;
    // Force the full-size SWA cache. With swa_full=false sliding-window layers
    // only expose `window`-many KV slots per sequence; after a prefix-reuse trim
    // the next decode can fail to find a contiguous block and abort with
    //   decode: failed to find a memory slot for batch of size 512
    // mlx-serve owns its own ctx-size cap up the stack, so the extra KV is
    // already accounted for. Matches `llama-server --swa-full`.
    cp.swa_full = true;
    if (p->n_ubatch > 0) {
        cp.n_ubatch = (uint32_t)p->n_ubatch;
        if (cp.n_batch < cp.n_ubatch) cp.n_batch = cp.n_ubatch;
    }
    // Flash attention stays AUTO (on Metal it enables FA where the head dim is
    // supported and falls back otherwise). Quantized K/V *requires* FA — the
    // plain SDPA path is F16/F32 only — so force it on for a non-default type.
    if (p->type_k != 0 || p->type_v != 0) {
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    if (p->type_k != 0) cp.type_k = (enum ggml_type)p->type_k;
    if (p->type_v != 0) cp.type_v = (enum ggml_type)p->type_v;
    const bool want_mtp = p->mtp_drafts > 0 && e->has_mtp;
    if (want_mtp) cp.n_rs_seq = (uint32_t)p->mtp_drafts;

    struct llama_context *tgt = llama_init_from_model(e->model, cp);
    if (!tgt) {
        copy_err(err, errlen, "llama_init_from_model failed");
        return NULL;
    }
    mlx_llama_ctx *c = new (std::nothrow) mlx_llama_ctx();
    if (!c) {
        llama_free(tgt);
        copy_err(err, errlen, "out of memory allocating context");
        return NULL;
    }
    c->e = e;
    c->tgt = tgt;
    c->dft = NULL;
    c->batch = llama_batch_ext_init(tgt);
    c->dbatch = NULL;
    c->n_seq = n_seq;
    c->n_batch = (int32_t)llama_n_batch(tgt);
    c->n_vocab = llama_vocab_n_tokens(e->vocab);
    c->mtp_drafts = 0;
    c->n_embd_h = 0;
    c->pos.assign((size_t)n_seq, 0);
    c->out_idx.assign((size_t)n_seq, -1);
    if (want_mtp) attach_mtp(c, cp, p->mtp_drafts, err, errlen);
    return c;
}

extern "C" void mlx_llama_ctx_free(mlx_llama_ctx *c) {
    if (!c) return;
    if (c->dbatch) llama_batch_ext_free(c->dbatch);
    if (c->dft) llama_free(c->dft);
    if (c->batch) llama_batch_ext_free(c->batch);
    if (c->tgt) llama_free(c->tgt);
    delete c;
}

extern "C" int32_t mlx_llama_ctx_mtp_drafts(mlx_llama_ctx *c) {
    return c->mtp_drafts;
}

static bool valid_seq(const mlx_llama_ctx *c, int32_t seq) {
    return seq >= 0 && seq < c->n_seq;
}

static void seq_clear(mlx_llama_ctx *c, int32_t seq) {
    llama_memory_seq_rm(llama_get_memory(c->tgt), seq, -1, -1);
    if (c->dft) {
        llama_memory_seq_rm(llama_get_memory(c->dft), seq, -1, -1);
        std::fill_n(c->pending_h.begin() + (ptrdiff_t)((size_t)seq * c->n_embd_h), c->n_embd_h, 0.0f);
    }
    c->pos[seq] = 0;
    c->out_idx[seq] = -1;
}

// Drop the head's KV from `p0` on, all of it when the head cannot trim partially.
static void dft_trim(mlx_llama_ctx *c, int32_t seq, llama_pos p0) {
    if (!c->dft) return;
    llama_memory_t mem = llama_get_memory(c->dft);
    if (!llama_memory_seq_rm(mem, seq, p0, -1)) llama_memory_seq_rm(mem, seq, -1, -1);
}

// A head that fails costs the drafts, never the request: decode on without it.
static void drop_mtp(mlx_llama_ctx *c) {
    fprintf(stderr, "[llama] MTP head failed; decoding without drafts\n");
    llama_set_embeddings_nextn(c->tgt, false, false);
    llama_batch_ext_free(c->dbatch);
    llama_free(c->dft);
    c->dbatch = NULL;
    c->dft = NULL;
    c->mtp_drafts = 0;
}

// Pair each target row with the target hidden one position earlier (a
// sequence's first row with the one carried from its previous batch) and decode
// the pairs on the MTP head, so its KV covers every position the target holds.
static int32_t mtp_catch_up(mlx_llama_ctx *c, const Row *rows, int32_t n) {
    const float *h = llama_get_embeddings_nextn(c->tgt);
    if (!h) return -1;
    const size_t w = c->n_embd_h;
    llama_batch_ext_clear(c->dbatch);
    for (int32_t i = 0; i < n; i++) {
        const float *hrow = i > 0 && rows[i - 1].seq == rows[i].seq ? h + (size_t)(i - 1) * w
                                                                    : &c->pending_h[(size_t)rows[i].seq * w];
        const int32_t idx = llama_batch_ext_add_token(c->dbatch, rows[i].seq, rows[i].tok);
        if (idx < 0) return -1;
        llama_batch_ext_set_pos(c->dbatch, idx, &rows[i].pos);
        if (!llama_batch_ext_set_embd_token(c->dbatch, idx, { hrow, 1, w })) return -1;
    }
    const int32_t rc = llama_process(c->dft, LLAMA_PROCESS_TYPE_DECODE, c->dbatch);
    if (rc != 0) return rc;
    for (int32_t i = 0; i < n; i++) {
        if (i + 1 < n && rows[i + 1].seq == rows[i].seq) continue;
        memcpy(&c->pending_h[(size_t)rows[i].seq * w], h + (size_t)i * w, w * sizeof(float));
    }
    return 0;
}

// Decode `rows` (each sequence's rows contiguous, in position order) on the
// target; rows from `out_from` on get logits.
static int32_t decode_rows(mlx_llama_ctx *c, const Row *rows, int32_t n, int32_t out_from) {
    llama_batch_ext_clear(c->batch);
    for (int32_t i = 0; i < n; i++) {
        const int32_t idx = llama_batch_ext_add_token(c->batch, rows[i].seq, rows[i].tok);
        if (idx < 0) return -1;
        llama_batch_ext_set_pos(c->batch, idx, &rows[i].pos);
        if (i >= out_from) llama_batch_ext_set_output_logits(c->batch, idx, true);
    }
    std::fill(c->out_idx.begin(), c->out_idx.end(), -1);
    const int32_t rc = llama_process(c->tgt, LLAMA_PROCESS_TYPE_DECODE, c->batch);
    if (rc != 0) return rc;
    for (int32_t i = 0; i < n; i++) {
        c->pos[rows[i].seq] = rows[i].pos + 1;
        if (i >= out_from) c->out_idx[rows[i].seq] = i;
    }
    if (c->dft && mtp_catch_up(c, rows, n) != 0) drop_mtp(c);
    return 0;
}

extern "C" int32_t mlx_llama_seq_prefill(mlx_llama_ctx *c, int32_t seq, const int32_t *tokens, int32_t n_tokens,
                                         char *err, size_t errlen) {
    if (!valid_seq(c, seq)) {
        copy_err(err, errlen, "invalid sequence");
        return -1;
    }
    int32_t off = 0;
    while (off < n_tokens) {
        const int32_t m = std::min(n_tokens - off, c->n_batch);
        c->rows.resize((size_t)m);
        for (int32_t i = 0; i < m; i++) c->rows[i] = { tokens[off + i], c->pos[seq] + i, seq };
        if (decode_rows(c, c->rows.data(), m, off + m == n_tokens ? m - 1 : m) != 0) {
            seq_clear(c, seq);
            copy_err(err, errlen, "llama_process failed during prefill");
            return -1;
        }
        off += m;
    }
    return 0;
}

extern "C" int32_t mlx_llama_step(mlx_llama_ctx *c, const int32_t *seqs, const int32_t *tokens, int32_t n,
                                  char *err, size_t errlen) {
    c->rows.resize((size_t)n);
    for (int32_t i = 0; i < n; i++) {
        if (!valid_seq(c, seqs[i])) {
            copy_err(err, errlen, "invalid sequence");
            return -1;
        }
        c->rows[i] = { tokens[i], c->pos[seqs[i]], seqs[i] };
    }
    if (decode_rows(c, c->rows.data(), n, 0) != 0) {
        copy_err(err, errlen, "llama_process failed");
        return -1;
    }
    return 0;
}

extern "C" int32_t mlx_llama_seq_trim(mlx_llama_ctx *c, int32_t seq, int32_t n_keep) {
    if (n_keep < 0) n_keep = 0;
    if (n_keep >= c->pos[seq]) return 0; // nothing resident beyond n_keep
    // Recurrent / hybrid memory (Mamba, GDN: qwen35, qwen3next, nemotron_h) can
    // only roll a tail back within its per-token snapshot window and returns
    // false past it, mutating nothing. Ignoring that left the previous request's
    // whole tail resident under the new suffix (#286): clear instead, 1 = the
    // caller must cold-prefill.
    if (!llama_memory_seq_rm(llama_get_memory(c->tgt), seq, n_keep, -1)) {
        seq_clear(c, seq);
        return 1;
    }
    if (c->dft) dft_trim(c, seq, n_keep);
    c->pos[seq] = n_keep;
    c->out_idx[seq] = -1;
    return 0;
}

extern "C" void mlx_llama_seq_reset(mlx_llama_ctx *c, int32_t seq) {
    seq_clear(c, seq);
}

extern "C" int32_t mlx_llama_seq_pos(mlx_llama_ctx *c, int32_t seq) {
    return c->pos[seq];
}

static int32_t argmax_row(const float *logits, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

// Sample batch row `idx` of the target. `pos` (where the sampled token lands)
// salts the seed, so the same seed draws the same tokens drafted or not.
static int32_t sample_row(mlx_llama_ctx *c, int32_t idx, llama_pos pos, float temperature, int32_t top_k,
                          float top_p, float min_p, uint64_t *rng) {
    if (temperature <= 0.0f) {
        const float *logits = llama_get_logits_ith(c->tgt, idx);
        return logits ? argmax_row(logits, c->n_vocab) : -1;
    }

    struct llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    struct llama_sampler *chain = llama_sampler_chain_init(sp);
    if (top_k > 0) llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    if (top_p > 0.0f && top_p < 1.0f) llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
    if (min_p > 0.0f) llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));

    uint64_t state = (rng && *rng) ? *rng : 0x106689D45497FDB5ULL;
    uint32_t seed = (uint32_t)(state ^ ((uint64_t)pos * 0x9E3779B97F4A7C15ULL));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));

    int32_t tok = (int32_t)llama_sampler_sample(chain, c->tgt, idx);
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

extern "C" int32_t mlx_llama_seq_sample(mlx_llama_ctx *c, int32_t seq, float temperature, int32_t top_k,
                                        float top_p, float min_p, uint64_t *rng) {
    if (!valid_seq(c, seq) || c->out_idx[seq] < 0) return -1;
    return sample_row(c, c->out_idx[seq], c->pos[seq], temperature, top_k, top_p, min_p, rng);
}

// Greedy drafts from the head: each step reads (token, hidden one position
// earlier) and predicts the next token and the hidden that goes with it.
static int32_t draft(mlx_llama_ctx *c, int32_t seq, llama_token id_last, llama_pos pos0, int32_t k) {
    const size_t w = c->n_embd_h;
    memcpy(c->draft_h.data(), &c->pending_h[(size_t)seq * w], w * sizeof(float));
    llama_token tok = id_last;
    int32_t n = 0;
    for (; n < k; n++) {
        llama_batch_ext_clear(c->dbatch);
        const int32_t idx = llama_batch_ext_add_token(c->dbatch, seq, tok);
        if (idx < 0) break;
        const llama_pos p = pos0 + n;
        llama_batch_ext_set_pos(c->dbatch, idx, &p);
        if (!llama_batch_ext_set_embd_token(c->dbatch, idx, { c->draft_h.data(), 1, w })) break;
        llama_batch_ext_set_output_logits(c->dbatch, idx, true);
        if (llama_process(c->dft, LLAMA_PROCESS_TYPE_DECODE, c->dbatch) != 0) break;
        const float *logits = llama_get_logits_ith(c->dft, idx);
        const float *h = llama_get_embeddings_nextn_ith(c->dft, idx);
        if (!logits || !h) break;
        tok = argmax_row(logits, c->n_vocab);
        c->drafts[n] = tok;
        memcpy(c->draft_h.data(), h, w * sizeof(float));
    }
    // The verify replay rebuilds these positions from the target's own hidden.
    dft_trim(c, seq, pos0);
    return n;
}

extern "C" int32_t mlx_llama_seq_spec_step(mlx_llama_ctx *c, int32_t seq, int32_t id_last, int32_t max_drafts,
                                           float temperature, int32_t top_k, float top_p, float min_p,
                                           uint64_t *rng, int32_t *out, char *err, size_t errlen) {
    if (!valid_seq(c, seq)) {
        copy_err(err, errlen, "invalid sequence");
        return -1;
    }
    const llama_pos pos0 = c->pos[seq];
    int32_t k = std::min(max_drafts, c->mtp_drafts);
    k = std::min(k, (int32_t)llama_n_ctx_seq(c->tgt) - pos0 - 1);
    const int32_t n_draft = c->dft && k > 0 ? draft(c, seq, id_last, pos0, k) : 0;

    // Verify: id_last and the drafts in one target pass, logits for every row.
    c->rows.resize((size_t)n_draft + 1);
    c->rows[0] = { id_last, pos0, seq };
    for (int32_t i = 0; i < n_draft; i++) c->rows[i + 1] = { c->drafts[i], pos0 + 1 + i, seq };
    if (decode_rows(c, c->rows.data(), n_draft + 1, 0) != 0) {
        copy_err(err, errlen, "llama_process failed during MTP verify");
        return -1;
    }

    // Row i predicts the token after drafts[i - 1]: keep going while it agrees.
    int32_t accepted = 0;
    for (;;) {
        const int32_t tok = sample_row(c, accepted, pos0 + accepted + 1, temperature, top_k, top_p, min_p, rng);
        if (tok < 0) {
            copy_err(err, errlen, "sampling failed during MTP verify");
            return -1;
        }
        out[accepted] = tok;
        if (accepted == n_draft || tok != c->drafts[accepted]) break;
        accepted++;
    }

    // Drop the rejected drafts from both contexts.
    const llama_pos keep = pos0 + accepted + 1;
    if (keep < c->pos[seq]) {
        if (!llama_memory_seq_rm(llama_get_memory(c->tgt), seq, keep, -1)) {
            seq_clear(c, seq);
            copy_err(err, errlen, "could not roll back rejected MTP drafts");
            return -1;
        }
        dft_trim(c, seq, keep);
        c->pos[seq] = keep;
    }
    if (c->dft) {
        const size_t w = c->n_embd_h;
        memcpy(&c->pending_h[(size_t)seq * w], llama_get_embeddings_nextn(c->tgt) + (size_t)accepted * w,
               w * sizeof(float));
    }
    c->out_idx[seq] = -1;
    return accepted + 1;
}
