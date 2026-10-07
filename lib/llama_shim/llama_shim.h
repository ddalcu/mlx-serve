// Clean C API over llama.cpp's libllama, in the spirit of ds4.h.
//
// libllama exposes large, ABI-fragile structs (llama_model_params,
// llama_context_params) and a verbose decode/sample protocol. Rather than mirror
// those structs in Zig (where a field drift silently corrupts memory), this shim
// compiles against the real llama.h (so the ABI is always correct) and exports a
// small, stable surface that src/llama_ffi.zig mirrors 1:1 — exactly how
// src/ds4_ffi.zig mirrors ds4.h.
//
// Lifetimes & threading: the engine wraps a loaded model + vocab (and an
// optional MTP draft head); a context wraps ONE llama_context holding n_seq
// sequences, each with its own KV stream, decoded together. The llama backend is
// initialized once per process (pthread_once). A context is single-threaded:
// the scheduler's inference thread is its only caller.
//
// Logits contract: a decode leaves logits for the sequences it produced a row
// for, and the NEXT decode on the context discards them. Sample a sequence right
// after the call that produced its row.
#ifndef MLX_LLAMA_SHIM_H
#define MLX_LLAMA_SHIM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlx_llama_engine mlx_llama_engine;
typedef struct mlx_llama_ctx mlx_llama_ctx;

// Load a GGUF model. n_gpu_layers: layers to offload to Metal (999 = all).
// mtp_path: an MTP draft-head GGUF to load beside the trunk, or NULL. load_mtp:
// with no mtp_path, load the trunk's own NextN heads when it ships them.
// Returns NULL on failure with `err` filled.
mlx_llama_engine *mlx_llama_open(const char *gguf_path, int32_t n_gpu_layers, const char *mtp_path,
                                 bool load_mtp, char *err, size_t errlen);
void mlx_llama_close(mlx_llama_engine *e);

int32_t mlx_llama_eos_token(mlx_llama_engine *e);
bool    mlx_llama_is_eog(mlx_llama_engine *e, int32_t token); // EOS or any end-of-generation token
int32_t mlx_llama_n_vocab(mlx_llama_engine *e);
// True when an MTP head is loaded (the sidecar, or the trunk's own heads).
bool    mlx_llama_has_mtp(mlx_llama_engine *e);

// Tokenize `text` (length `text_len`). Writes up to `out_cap` ids into `out`.
// Returns the number of tokens written, or a negative value (= -required) when
// `out_cap` is too small (caller re-allocates and retries). Mirrors llama_tokenize.
int32_t mlx_llama_tokenize(mlx_llama_engine *e, const char *text, int32_t text_len,
                           bool add_special, bool parse_special,
                           int32_t *out, int32_t out_cap);

// One token -> bytes (NOT NUL-terminated). Returns #bytes written, or a negative
// value (= -required) when `buf_cap` is too small. Mirrors llama_token_to_piece.
int32_t mlx_llama_token_to_piece(mlx_llama_engine *e, int32_t token, char *buf, int32_t buf_cap);

// Raw GGUF chat-template string (jinja source), or NULL if the model has none.
// Borrowed pointer owned by the model; valid for the engine's lifetime. Callers
// that want robust rendering should feed this to mlx-serve's own jinja engine.
const char *mlx_llama_chat_template(mlx_llama_engine *e);

// Apply the model's built-in chat template via llama_chat_apply_template.
// NOTE: this is NOT a full jinja parser — it only recognizes a fixed set of
// known template formats. Prefer rendering mlx_llama_chat_template() through
// mlx-serve's jinja engine; this is a fallback. Returns the formatted byte count
// (may exceed buf_cap -> grow and retry), or a negative value on error.
int32_t mlx_llama_apply_chat_template(mlx_llama_engine *e,
                                      const char **roles, const char **contents, int32_t n_msgs,
                                      bool add_assistant, char *buf, int32_t buf_cap);

typedef struct {
    int32_t n_ctx;      // context per sequence; 0 = the model's trained context
    int32_t n_seq;      // sequences decoded together, each with its own KV (>= 1)
    int32_t type_k;     // ggml_type of the K cache (F16=1, Q8_0=8, Q4_0=2); 0 = libllama default
    int32_t type_v;     // same for V
    int32_t n_ubatch;   // physical prefill batch; 0 = libllama default
    int32_t mtp_drafts; // draft tokens per MTP round; 0 = no MTP context
} mlx_llama_ctx_params;

// Create a context. NULL on failure. An MTP context that cannot be built is
// dropped with a log line and mlx_llama_ctx_mtp_drafts() reports 0.
mlx_llama_ctx *mlx_llama_ctx_create(mlx_llama_engine *e, const mlx_llama_ctx_params *p, char *err, size_t errlen);
void mlx_llama_ctx_free(mlx_llama_ctx *c);
// Draft tokens per MTP round this context runs, 0 when it has no MTP context.
int32_t mlx_llama_ctx_mtp_drafts(mlx_llama_ctx *c);

// Decode `tokens` into `seq` from its current position, chunked to the batch
// size, with logits for the last token. 0 ok, -1 on failure (the sequence is
// left cleared: nothing resident).
int32_t mlx_llama_seq_prefill(mlx_llama_ctx *c, int32_t seq, const int32_t *tokens, int32_t n_tokens,
                              char *err, size_t errlen);

// One decode step for `n` sequences in ONE batch: tokens[i] goes to seqs[i] at
// its next position, with logits for every row. 0 ok, -1 on failure.
int32_t mlx_llama_step(mlx_llama_ctx *c, const int32_t *seqs, const int32_t *tokens, int32_t n,
                       char *err, size_t errlen);

// Keep the first n_keep tokens of `seq`, dropping the rest. 0 on success, 1 when
// the memory could not trim partially (recurrent/hybrid state past its rollback
// window) and the sequence was cleared instead: cold-prefill everything.
int32_t mlx_llama_seq_trim(mlx_llama_ctx *c, int32_t seq, int32_t n_keep);
// Clear `seq` (position -> 0).
void    mlx_llama_seq_reset(mlx_llama_ctx *c, int32_t seq);
// Tokens currently in `seq`'s KV.
int32_t mlx_llama_seq_pos(mlx_llama_ctx *c, int32_t seq);

// Sample `seq`'s next token from the logits its last decode left. temperature
// <= 0 => greedy argmax. `rng` is advanced so repeated draws differ while staying
// reproducible from the seed. -1 when `seq` has no logits row.
int32_t mlx_llama_seq_sample(mlx_llama_ctx *c, int32_t seq, float temperature, int32_t top_k,
                             float top_p, float min_p, uint64_t *rng);

// One MTP round for `seq`: draft up to `max_drafts` tokens from the MTP head,
// verify them after `id_last` (sampled, not yet in the KV) in one target pass,
// and keep the run the target agrees with. Writes the accepted drafts followed
// by the next token (also not yet in the KV) to `out` (capacity max_drafts + 1);
// returns that count (>= 1), or -1 on failure. Sampling as mlx_llama_seq_sample.
// A head that fails is dropped (mlx_llama_ctx_mtp_drafts() then reports 0) and
// the round decodes plain.
int32_t mlx_llama_seq_spec_step(mlx_llama_ctx *c, int32_t seq, int32_t id_last, int32_t max_drafts,
                                float temperature, int32_t top_k, float top_p, float min_p, uint64_t *rng,
                                int32_t *out, char *err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif // MLX_LLAMA_SHIM_H
