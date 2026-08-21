#import "ane_mlp.h"

#import "ane_bridge.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* The MIL emitter mirrors the validated Stage-A harness
 * (~/claude-tmp/perf-aug17/p5-ane-mlp): one fp16 input tensor
 * [1, hidden, 1, rows], K-chunked down conv, fp16 datapath. Since the
 * procedure-bank round every layer's slice is one `procedureNNN` function
 * inside ONE program (the private runtime accepts only ~121 resident model
 * handles), so every emitted name carries its procedure index — MIL scopes
 * are per-function, but a bank is not worth betting that on.
 * I/O planes are fp16 (bf16→fp16 is exact in fp16's normal range and the
 * graph computed in fp16 anyway — the fp32 planes only doubled seam bytes). */

#define ANE_MLP_MAX_DOWN_K 4608u

struct msv_ane_mlp {
    uint32_t hidden;
    uint32_t rows;
    uint32_t procedures;
    IOSurfaceRef input_surface;
    IOSurfaceRef output_surface;
    __fp16 *input_base;
    __fp16 *output_base;
    msv_ane_model *model;
};

static void mlp_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

int msv_ane_available(void) {
    return msv_ane_bridge_available();
}

uint64_t msv_ane_internal_free_disk(void) {
    NSDictionary *attrs = [[NSFileManager defaultManager]
        attributesOfFileSystemForPath:@"/private/tmp" error:nil];
    NSNumber *free = attrs[NSFileSystemFreeSize];
    return free ? free.unsignedLongLongValue : 0;
}

/* msv_ane_plane is an IOSurfaceRef in a trench coat: the opaque typedef
 * keeps IOSurface types out of the public header (and Zig's FFI). */
msv_ane_plane *msv_ane_plane_create(size_t bytes) {
    return (msv_ane_plane *)msv_ane_bridge_surface(bytes);
}

void msv_ane_plane_free(msv_ane_plane *p) {
    if (p) CFRelease((IOSurfaceRef)p);
}

__fp16 *msv_ane_plane_base(msv_ane_plane *p) {
    return p ? IOSurfaceGetBaseAddress((IOSurfaceRef)p) : NULL;
}

/* Blob assembly: 64-byte file header, then 64-byte-aligned chunks. Grows on
 * demand — a bank's size is not known until its last procedure lands, and a
 * hand-computed bound is a buffer overrun waiting for the next op. */
typedef struct { uint8_t *data; size_t cursor; size_t cap; bool oom; } mlp_blob;

#define BLOB_BAD ((size_t)-1)

static bool blob_reserve(mlp_blob *b, size_t extra) {
    if (b->oom) return false;
    size_t need = b->cursor + extra;
    if (need <= b->cap) return true;
    size_t cap = b->cap ? b->cap : 4096;
    while (cap < need) cap *= 2;
    uint8_t *grown = realloc(b->data, cap);
    if (!grown) { b->oom = true; return false; }
    memset(grown + b->cap, 0, cap - b->cap);
    b->data = grown;
    b->cap = cap;
    return true;
}

static size_t blob_add(mlp_blob *b, const void *payload, size_t bytes) {
    size_t padded = 64 + ((bytes + 63) & ~(size_t)63);
    if (!blob_reserve(b, padded)) return BLOB_BAD;
    size_t header = b->cursor;
    uint8_t *chunk = b->data + header;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
    chunk[4] = 0x01;
    uint32_t size32 = (uint32_t)bytes;
    uint32_t offset32 = (uint32_t)(header + 64);
    memcpy(chunk + 8, &size32, sizeof(size32));
    memcpy(chunk + 16, &offset32, sizeof(offset32));
    if (payload) memcpy(chunk + 64, payload, bytes);
    b->cursor = header + padded;
    return header;
}

static size_t blob_add_scales_f16(mlp_blob *b, const float *scales,
                                  uint32_t n) {
    size_t header = blob_add(b, NULL, (size_t)n * 2);
    if (header == BLOB_BAD) return BLOB_BAD;
    __fp16 *dst = (__fp16 *)(void *)(b->data + header + 64);
    for (uint32_t i = 0; i < n; i++) dst[i] = (__fp16)scales[i];
    return header;
}

/* Smallest chunk count that divides ffn with chunks <= ANE_MLP_MAX_DOWN_K
 * wide; 0 when no divisor works (caller declines the arch). */
static uint32_t down_chunks_for(uint32_t ffn) {
    for (uint32_t n = 1; n <= 16; n++)
        if (ffn % n == 0 && ffn / n <= ANE_MLP_MAX_DOWN_K) return n;
    return 0;
}

static void emit_int8_weight(NSMutableString *text, const char *name,
                             uint32_t n, uint32_t k, size_t q_off,
                             size_t s_off) {
    [text appendFormat:@"        tensor<fp16, [%u,%u,1,1]> %s = "
        "constexpr_affine_dequantize()[axis=int32(0), name=string(\"%s\"), "
        "quantized_data=tensor<int8, [%u,%u,1,1]>(BLOBFILE("
        "path=string(\"@model_path/weights/weight.bin\"), "
        "offset=uint64(%zu))), scale=tensor<fp16, [%u]>(BLOBFILE("
        "path=string(\"@model_path/weights/weight.bin\"), "
        "offset=uint64(%zu))), zero_point=int8(0)];\n",
        n, k, name, name, n, k, q_off, n, s_off];
}

/* ── Bank builder ── */

struct msv_ane_bank {
    void *text;      /* NSMutableString, +1 retained */
    mlp_blob blob;
    uint32_t procedures;
    uint32_t hidden;
    uint32_t rows;
    uint32_t out_width;  /* every procedure's output channel count */
    bool failed;
};

static NSString *bank_header(void) {
    return @"program(1.3)\n[buildInfo = dict<string, string>({{"
        "\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", "
        "\"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n";
}

msv_ane_bank *msv_ane_bank_create(void) {
    msv_ane_bank *b = calloc(1, sizeof(*b));
    if (!b) return NULL;
    NSMutableString *text = [NSMutableString stringWithString:bank_header()];
    b->text = (__bridge_retained void *)text;
    b->blob.cursor = 64;
    if (!blob_reserve(&b->blob, 4096)) {
        msv_ane_bank_free(b);
        return NULL;
    }
    b->blob.data[0] = 0x01;
    b->blob.data[4] = 0x02;
    return b;
}

void msv_ane_bank_free(msv_ane_bank *b) {
    if (!b) return;
    @autoreleasepool {
        if (b->text) {
            id text = (__bridge_transfer id)b->text;
            (void)text;
        }
    }
    free(b->blob.data);
    free(b);
}

uint32_t msv_ane_bank_count(const msv_ane_bank *b) {
    return b ? b->procedures : 0;
}

uint64_t msv_ane_bank_bytes(const msv_ane_bank *b) {
    return b ? (uint64_t)b->blob.cursor : 0;
}

/* Every procedure in a bank binds the SAME input and output surfaces, so
 * their shapes have to agree; the first procedure fixes them. */
static bool bank_shape_ok(msv_ane_bank *b, uint32_t hidden, uint32_t rows,
                          uint32_t out_width, const char *what, char *error,
                          size_t error_size) {
    /* rows % 32: an fp16 plane's per-channel pitch is rows * 2 bytes, and a
     * pitch off the 64-byte grid compiles fine but fails every eval with a
     * bare "Program Inference error" (measured, A3 probe 2026-08-18) — so
     * the refusal happens HERE, by name. */
    if (!hidden || !rows || !out_width || rows % 32) {
        mlp_fail(error, error_size, "ANE %s: rows must be a positive multiple "
                 "of 32 — the fp16 plane pitch (rows x 2 bytes) must sit on "
                 "the 64-byte grid (got %u)", what, rows);
        return false;
    }
    if (b->procedures == 0) {
        b->hidden = hidden;
        b->rows = rows;
        b->out_width = out_width;
        return true;
    }
    if (b->hidden != hidden || b->rows != rows || b->out_width != out_width) {
        mlp_fail(error, error_size, "ANE %s: bank holds [%u->%u]x%u "
                 "procedures, cannot mix [%u->%u]x%u", what, b->hidden,
                 b->out_width, b->rows, hidden, out_width, rows);
        return false;
    }
    return true;
}

int msv_ane_bank_add_mlp(msv_ane_bank *b, uint32_t hidden, uint32_t ffn,
                         uint32_t rows,
                         const int8_t *gate_q, const float *gate_s,
                         const int8_t *up_q, const float *up_s,
                         const int8_t *down_q, const float *down_s,
                         char *error, size_t error_size) {
    if (!b || b->failed) {
        mlp_fail(error, error_size, "no ANE bank to add to");
        return -1;
    }
    if (!bank_shape_ok(b, hidden, rows, hidden, "mlp", error, error_size))
        return -1;
    uint32_t nch = down_chunks_for(ffn);
    if (!ffn || !nch) {
        mlp_fail(error, error_size, "ANE mlp: ffn %u has no K-chunking <= %u",
                 ffn, ANE_MLP_MAX_DOWN_K);
        return -1;
    }

    const uint32_t p = b->procedures;
    size_t big = (size_t)ffn * hidden;
    size_t gq = blob_add(&b->blob, gate_q, big);
    size_t gs = blob_add_scales_f16(&b->blob, gate_s, ffn);
    size_t uq = blob_add(&b->blob, up_q, big);
    size_t us = blob_add_scales_f16(&b->blob, up_s, ffn);
    /* Down weight re-packed as nch contiguous [hidden, ffn/nch] K-slabs. */
    uint32_t kc = ffn / nch;
    size_t dq_offs[16];
    {
        int8_t *slab = malloc((size_t)hidden * kc);
        if (!slab) {
            b->failed = true;
            mlp_fail(error, error_size, "out of memory building ANE mlp down "
                     "slabs");
            return -1;
        }
        for (uint32_t c = 0; c < nch; c++) {
            for (uint32_t n = 0; n < hidden; n++)
                memcpy(slab + (size_t)n * kc,
                       down_q + (size_t)n * ffn + (size_t)c * kc, kc);
            dq_offs[c] = blob_add(&b->blob, slab, (size_t)hidden * kc);
        }
        free(slab);
    }
    size_t ds = blob_add_scales_f16(&b->blob, down_s, hidden);
    if (b->blob.oom) {
        b->failed = true;
        mlp_fail(error, error_size, "out of memory building the ANE bank blob");
        return -1;
    }

    NSMutableString *text = (__bridge NSMutableString *)b->text;
    [text appendFormat:@"    func procedure%03u<ios18>(tensor<fp16, [1, %u, 1, "
        "%u]> x0) {\n"
        "        string pt%u = const()[name=string(\"pt%u\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st%u = const()[name=string(\"st%u\"), val=tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd%u = const()[name=string(\"pd%u\"), val=tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl%u = const()[name=string(\"dl%u\"), val=tensor<int32, [2]>([1, 1])];\n"
        "        int32 gr%u = const()[name=string(\"gr%u\"), val=int32(1)];\n",
        p, hidden, rows, p, p, p, p, p, p, p, p, p, p];
    char wname[24];
    snprintf(wname, sizeof(wname), "Wg%u", p);
    emit_int8_weight(text, wname, ffn, hidden, gq, gs);
    snprintf(wname, sizeof(wname), "Wu%u", p);
    emit_int8_weight(text, wname, ffn, hidden, uq, us);
    [text appendFormat:
        @"        tensor<fp16, [1, %u, 1, %u]> g%u = conv(dilations=dl%u, groups=gr%u, "
        "pad=pd%u, pad_type=pt%u, strides=st%u, weight=Wg%u, x=x0)[name=string(\"g%u\")];\n"
        "        tensor<fp16, [1, %u, 1, %u]> u%u = conv(dilations=dl%u, groups=gr%u, "
        "pad=pd%u, pad_type=pt%u, strides=st%u, weight=Wu%u, x=x0)[name=string(\"u%u\")];\n"
        "        tensor<fp16, [1, %u, 1, %u]> sg%u = sigmoid(x=g%u)[name=string(\"sg%u\")];\n"
        "        tensor<fp16, [1, %u, 1, %u]> gs%u = mul(x=g%u, y=sg%u)[name=string(\"gs%u\")];\n"
        "        tensor<fp16, [1, %u, 1, %u]> act%u = mul(x=gs%u, y=u%u)[name=string(\"act%u\")];\n"
        "        fp16 inv16_%u = const()[name=string(\"inv16_%u\"), val=fp16(0.0625)];\n"
        "        tensor<fp16, [1, %u, 1, %u]> act_p%u = mul(x=act%u, y=inv16_%u)[name=string(\"act_p%u\")];\n",
        ffn, rows, p, p, p, p, p, p, p, p,
        ffn, rows, p, p, p, p, p, p, p, p,
        ffn, rows, p, p, p,
        ffn, rows, p, p, p, p,
        ffn, rows, p, p, p, p,
        p, p,
        ffn, rows, p, p, p, p];
    for (uint32_t c = 0; c < nch; c++) {
        [text appendFormat:
            @"        tensor<int32, [4]> db%u_%u = const()[name=string(\"db%u_%u\"), "
            "val=tensor<int32, [4]>([0,%u,0,0])];\n"
            "        tensor<int32, [4]> dsz%u_%u = const()[name=string(\"dsz%u_%u\"), "
            "val=tensor<int32, [4]>([1,%u,1,%u])];\n"
            "        tensor<fp16, [1, %u, 1, %u]> a%u_%u = slice_by_size(x=act_p%u, "
            "begin=db%u_%u, size=dsz%u_%u)[name=string(\"a%u_%u\")];\n",
            p, c, p, c, c * kc, p, c, p, c, kc, rows, kc, rows, p, c, p,
            p, c, p, c, p, c];
        snprintf(wname, sizeof(wname), "Wd%u_%u", p, c);
        emit_int8_weight(text, wname, hidden, kc, dq_offs[c], ds);
        [text appendFormat:
            @"        tensor<fp16, [1, %u, 1, %u]> p%u_%u = conv(dilations=dl%u, "
            "groups=gr%u, pad=pd%u, pad_type=pt%u, strides=st%u, weight=Wd%u_%u, "
            "x=a%u_%u)[name=string(\"p%u_%u\")];\n",
            hidden, rows, p, c, p, p, p, p, p, p, c, p, c, p, c];
    }
    NSString *previous = [NSString stringWithFormat:@"p%u_0", p];
    for (uint32_t c = 1; c < nch; c++) {
        NSString *sum = [NSString stringWithFormat:@"ds%u_%u", p, c];
        [text appendFormat:
            @"        tensor<fp16, [1, %u, 1, %u]> %@ = add(x=%@, y=p%u_%u)"
            "[name=string(\"%@\")];\n", hidden, rows, sum, previous, p, c, sum];
        previous = sum;
    }
    [text appendFormat:
        @"        fp16 sixteen%u = const()[name=string(\"sixteen%u\"), val=fp16(16.0)];\n"
        "        tensor<fp16, [1, %u, 1, %u]> y%u = mul(x=%@, y=sixteen%u)[name=string(\"y%u\")];\n"
        "    } -> (y%u);\n", p, p, hidden, rows, p, previous, p, p, p];
    b->procedures = p + 1;
    return (int)p;
}

/* GDN input projections as ONE fused conv over the stacked
 * [qkv_out + z_out, hidden] weight (qkv rows first): each output row is an
 * independent dot product, so stacking along the output-channel axis is
 * byte-equivalent to two convs + concat, with one op fewer. K = hidden
 * (5120 on the 27B) sits well under the K=17408 chunking cliff, so no
 * in-graph K-chunking; values are post-norm-scale small (|y| < 1 measured),
 * so no accumulator wrap either — the same regime as the gate/up convs. */
int msv_ane_bank_add_gdn(msv_ane_bank *b, uint32_t hidden, uint32_t qkv_out,
                         uint32_t z_out, uint32_t rows,
                         const int8_t *qkv_q, const float *qkv_s,
                         const int8_t *z_q, const float *z_s,
                         char *error, size_t error_size) {
    if (!b || b->failed) {
        mlp_fail(error, error_size, "no ANE bank to add to");
        return -1;
    }
    if (!qkv_out || !z_out) {
        mlp_fail(error, error_size, "ANE gdn: empty qkv/z projection");
        return -1;
    }
    uint32_t out_total = qkv_out + z_out;
    if (!bank_shape_ok(b, hidden, rows, out_total, "gdn", error, error_size))
        return -1;

    const uint32_t p = b->procedures;
    size_t qkv_bytes = (size_t)qkv_out * hidden;
    size_t z_bytes = (size_t)z_out * hidden;
    /* Stacked weight: qkv rows then z rows, one contiguous blob chunk. */
    size_t q_off = blob_add(&b->blob, NULL, qkv_bytes + z_bytes);
    if (q_off != BLOB_BAD) {
        memcpy(b->blob.data + q_off + 64, qkv_q, qkv_bytes);
        memcpy(b->blob.data + q_off + 64 + qkv_bytes, z_q, z_bytes);
    }
    size_t s_off = blob_add(&b->blob, NULL, (size_t)out_total * 2);
    if (b->blob.oom || q_off == BLOB_BAD || s_off == BLOB_BAD) {
        b->failed = true;
        mlp_fail(error, error_size, "out of memory building the ANE bank blob");
        return -1;
    }
    {
        __fp16 *dst = (__fp16 *)(void *)(b->blob.data + s_off + 64);
        for (uint32_t i = 0; i < qkv_out; i++) dst[i] = (__fp16)qkv_s[i];
        for (uint32_t i = 0; i < z_out; i++) dst[qkv_out + i] = (__fp16)z_s[i];
    }

    NSMutableString *text = (__bridge NSMutableString *)b->text;
    [text appendFormat:@"    func procedure%03u<ios18>(tensor<fp16, [1, %u, 1, "
        "%u]> x0) {\n"
        "        string pt%u = const()[name=string(\"pt%u\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st%u = const()[name=string(\"st%u\"), val=tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd%u = const()[name=string(\"pd%u\"), val=tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl%u = const()[name=string(\"dl%u\"), val=tensor<int32, [2]>([1, 1])];\n"
        "        int32 gr%u = const()[name=string(\"gr%u\"), val=int32(1)];\n",
        p, hidden, rows, p, p, p, p, p, p, p, p, p, p];
    char wname[24];
    snprintf(wname, sizeof(wname), "Wqz%u", p);
    emit_int8_weight(text, wname, out_total, hidden, q_off, s_off);
    [text appendFormat:
        @"        tensor<fp16, [1, %u, 1, %u]> y%u = conv(dilations=dl%u, groups=gr%u, "
        "pad=pd%u, pad_type=pt%u, strides=st%u, weight=Wqz%u, x=x0)[name=string(\"y%u\")];\n"
        "    } -> (y%u);\n",
        out_total, rows, p, p, p, p, p, p, p, p, p];
    b->procedures = p + 1;
    return (int)p;
}

/* Bind a bank's I/O surfaces: a shared plane is retained, NULL allocates a
 * per-bank surface of `bytes`. Returns false on allocation failure. */
static bool mlp_bind_planes(msv_ane_mlp *m, msv_ane_plane *input_plane,
                            size_t input_bytes, msv_ane_plane *output_plane,
                            size_t output_bytes) {
    m->input_surface = input_plane ?
        (IOSurfaceRef)CFRetain((IOSurfaceRef)input_plane) :
        msv_ane_bridge_surface(input_bytes);
    m->output_surface = output_plane ?
        (IOSurfaceRef)CFRetain((IOSurfaceRef)output_plane) :
        msv_ane_bridge_surface(output_bytes);
    if (!m->input_surface || !m->output_surface) return false;
    m->input_base = IOSurfaceGetBaseAddress(m->input_surface);
    m->output_base = IOSurfaceGetBaseAddress(m->output_surface);
    /* A shared plane is packed by every user before its kick; zeroing it
     * here would wipe another program's live contents, so only fresh
     * per-bank surfaces are cleared. */
    if (!input_plane) memset(m->input_base, 0, input_bytes);
    if (!output_plane) memset(m->output_base, 0, output_bytes);
    return true;
}

msv_ane_mlp *msv_ane_bank_finish(msv_ane_bank *b, const char *name,
                                 int ane_instance,
                                 msv_ane_plane *input_plane,
                                 msv_ane_plane *output_plane,
                                 char *error, size_t error_size) {
    if (!b) {
        mlp_fail(error, error_size, "no ANE bank to finish");
        return NULL;
    }
    if (!msv_ane_available() || b->failed || b->procedures == 0) {
        const char *why = b->failed ? "the bank failed while building" :
            (b->procedures == 0 ? "the bank is empty" :
             "the Neural Engine bridge is unavailable");
        mlp_fail(error, error_size, "ANE %s: %s", name, why);
        msv_ane_bank_free(b);
        return NULL;
    }

    msv_ane_mlp *m = calloc(1, sizeof(*m));
    if (!m) {
        msv_ane_bank_free(b);
        mlp_fail(error, error_size, "out of memory creating ANE %s", name);
        return NULL;
    }
    m->hidden = b->hidden;
    m->rows = b->rows;
    m->procedures = b->procedures;
    if (!mlp_bind_planes(m, input_plane,
                         (size_t)b->hidden * b->rows * sizeof(__fp16),
                         output_plane,
                         (size_t)b->out_width * b->rows * sizeof(__fp16))) {
        msv_ane_bank_free(b);
        msv_ane_mlp_free(m);
        mlp_fail(error, error_size, "ANE %s surface alloc failed", name);
        return NULL;
    }

    @autoreleasepool {
        NSMutableString *text = (__bridge NSMutableString *)b->text;
        [text appendString:@"}\n"];
        uint8_t *weights = b->blob.data;
        size_t weight_bytes = b->blob.cursor;
        b->blob.data = NULL; /* ownership moves to the bridge */
        IOSurfaceRef inputs[1] = { m->input_surface };
        m->model = msv_ane_model_create(name, text.UTF8String, weights,
                                        weight_bytes, inputs, 1,
                                        m->output_surface, b->procedures,
                                        ane_instance, error, error_size);
    }
    msv_ane_bank_free(b);
    if (!m->model) {
        msv_ane_mlp_free(m);
        return NULL;
    }
    return m;
}

void msv_ane_mlp_free(msv_ane_mlp *m) {
    if (!m) return;
    msv_ane_model_free(m->model);
    if (m->input_surface) CFRelease(m->input_surface);
    if (m->output_surface) CFRelease(m->output_surface);
    free(m);
}

__fp16 *msv_ane_mlp_input(msv_ane_mlp *m) {
    return m ? m->input_base : NULL;
}

__fp16 *msv_ane_mlp_output(msv_ane_mlp *m) {
    return m ? m->output_base : NULL;
}

int msv_ane_mlp_eval(msv_ane_mlp *m, uint32_t procedure, char *error,
                     size_t error_size) {
    if (!m) {
        mlp_fail(error, error_size, "the ANE bank is not loaded");
        return 0;
    }
    return msv_ane_model_eval(m->model, procedure, error, error_size);
}

double msv_ane_mlp_compile_seconds(const msv_ane_mlp *m) {
    return m ? msv_ane_model_compile_seconds(m->model) : 0.0;
}

int msv_ane_mlp_cache_hit(const msv_ane_mlp *m) {
    return m ? (int)msv_ane_model_cache_hit(m->model) : 0;
}
