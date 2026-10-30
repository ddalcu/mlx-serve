// Generates src/sdk/quant/gguf/fixtures/<type>.bin: random quant blocks + ggml's own dequant
// of them, the oracle for the quants.zig tests. Run via scripts/gen_fixtures.sh.
// File layout: N_BLOCKS raw blocks, then N_BLOCKS * block_elems f32 (little endian).
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_BLOCKS 4

typedef void (*dequant_fn)(const void *x, float *y, int64_t k);

#define DECL(name) void dequantize_row_##name(const void *x, float *y, int64_t k);
DECL(q4_0) DECL(q4_1) DECL(q5_0) DECL(q5_1) DECL(q8_0) DECL(mxfp4) DECL(q2_K) DECL(q3_K) DECL(q4_K) DECL(q5_K) DECL(q6_K)
DECL(iq2_xxs) DECL(iq2_xs) DECL(iq2_s) DECL(iq3_xxs) DECL(iq3_s) DECL(iq4_nl) DECL(iq4_xs) DECL(iq1_s) DECL(iq1_m)

struct spec {
    const char *name;
    dequant_fn fn;
    int block_bytes, block_elems;
    int half_offsets[2]; // byte offsets of the f16 scales inside a block, -1 = none
};

#define S(n, bytes, elems, h0, h1) {#n, (dequant_fn)dequantize_row_##n, bytes, elems, {h0, h1}}
static const struct spec specs[] = {
    S(q4_0, 18, 32, 0, -1),      S(q4_1, 20, 32, 0, 2),       S(q5_0, 22, 32, 0, -1),      S(q5_1, 24, 32, 0, 2),
    S(q8_0, 34, 32, 0, -1),      S(mxfp4, 17, 32, -1, -1),      S(q2_K, 84, 256, 80, 82),    S(q3_K, 110, 256, 108, -1),
    S(q4_K, 144, 256, 0, 2),     S(q5_K, 176, 256, 0, 2),     S(q6_K, 210, 256, 208, -1),
    S(iq2_xxs, 66, 256, 0, -1),  S(iq2_s, 82, 256, 0, -1),    S(iq3_xxs, 98, 256, 0, -1),
    S(iq3_s, 110, 256, 0, -1),   S(iq4_nl, 18, 32, 0, -1),    S(iq4_xs, 136, 256, 0, -1),
    S(iq2_xs, 74, 256, 0, -1),   S(iq1_s, 50, 256, 0, -1),    S(iq1_m, 56, 256, -1, -1),
};

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: gen_fixtures <out_dir>\n"); return 1; }
    srand(1234);
    for (size_t t = 0; t < sizeof(specs) / sizeof(specs[0]); t++) {
        const struct spec *sp = &specs[t];
        uint8_t blocks[N_BLOCKS * 256];
        float out[N_BLOCKS * 256];
        for (int i = 0; i < N_BLOCKS * sp->block_bytes; i++) blocks[i] = (uint8_t)rand();
        // Random f16 scales would hit inf/nan; pin them to small finite values.
        for (int b = 0; b < N_BLOCKS; b++)
            for (int h = 0; h < 2; h++)
                if (sp->half_offsets[h] >= 0) {
                    uint16_t half = (uint16_t)(0x2800 + (rand() & 0x7ff)) | (uint16_t)((rand() & 1) << 15);
                    memcpy(blocks + b * sp->block_bytes + sp->half_offsets[h], &half, 2);
                }
        // MXFP4's scale is an E8M0 exponent byte: keep it near 2^0.
        if (!strcmp(sp->name, "mxfp4"))
            for (int b = 0; b < N_BLOCKS; b++) blocks[b * sp->block_bytes] = (uint8_t)(120 + rand() % 16);
        // IQ1_M has no f16 field: the scale's 4 nibbles ride in the top nibble of its 4 u16 scale words.
        if (!strcmp(sp->name, "iq1_m"))
            for (int b = 0; b < N_BLOCKS; b++) {
                const uint16_t half = (uint16_t)(0x2800 + (rand() & 0x7ff)) | (uint16_t)((rand() & 1) << 15);
                for (int n = 0; n < 4; n++) {
                    uint8_t *hi = blocks + b * sp->block_bytes + 48 + 2 * n + 1;
                    *hi = (uint8_t)((*hi & 0x0f) | (((half >> (4 * n)) & 0xf) << 4));
                }
            }
        sp->fn(blocks, out, (int64_t)N_BLOCKS * sp->block_elems);

        char path[1024];
        // File names are the lowercase type tags (q2_K -> q2_k.bin).
        int n = snprintf(path, sizeof(path), "%s/", argv[1]);
        for (const char *c = sp->name; *c; c++) path[n++] = (char)tolower(*c);
        strcpy(path + n, ".bin");
        FILE *f = fopen(path, "wb");
        if (!f) { perror(path); return 1; }
        fwrite(blocks, 1, (size_t)N_BLOCKS * sp->block_bytes, f);
        fwrite(out, sizeof(float), (size_t)N_BLOCKS * sp->block_elems, f);
        fclose(f);
    }
    return 0;
}
