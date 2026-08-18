#import "ane_mlp.h"

#import "ane_bridge.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* The MIL emitter mirrors the validated Stage-A harness
 * (~/claude-tmp/perf-aug17/p5-ane-mlp): one program per layer, single fp16
 * input tensor [1, hidden, 1, rows], K-chunked down conv, fp16 datapath.
 * I/O planes are fp16 (bf16→fp16 is exact in fp16's normal range and the
 * graph computed in fp16 anyway — the fp32 planes only doubled seam bytes). */

#define ANE_MLP_MAX_DOWN_K 4608u

struct msv_ane_mlp {
    uint32_t hidden;
    uint32_t ffn;
    uint32_t rows;
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

/* Blob assembly: 64-byte file header, then 64-byte-aligned chunks. */
typedef struct { uint8_t *data; size_t cursor; } mlp_blob;

static size_t blob_add(mlp_blob *b, const void *payload, size_t bytes) {
    size_t header = b->cursor;
    uint8_t *chunk = b->data + header;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
    chunk[4] = 0x01;
    uint32_t size32 = (uint32_t)bytes;
    uint32_t offset32 = (uint32_t)(header + 64);
    memcpy(chunk + 8, &size32, sizeof(size32));
    memcpy(chunk + 16, &offset32, sizeof(offset32));
    if (payload) memcpy(chunk + 64, payload, bytes);
    b->cursor = header + 64 + ((bytes + 63) & ~(size_t)63);
    return header;
}

static size_t blob_add_scales_f16(mlp_blob *b, const float *scales,
                                  uint32_t n) {
    size_t header = b->cursor;
    blob_add(b, NULL, (size_t)n * 2);
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

static NSString *mlp_program(uint32_t hidden, uint32_t ffn, uint32_t rows,
                             uint32_t nch, size_t gq, size_t gs, size_t uq,
                             size_t us, const size_t *dq_offs, size_t ds) {
    NSMutableString *text = [NSMutableString string];
    [text appendFormat:@"program(1.3)\n[buildInfo = dict<string, string>({{"
        "\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", "
        "\"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"
        "    func main<ios18>(tensor<fp16, [1, %u, 1, %u]> x0) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1, 1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n",
        hidden, rows];
    emit_int8_weight(text, "Wg", ffn, hidden, gq, gs);
    emit_int8_weight(text, "Wu", ffn, hidden, uq, us);
    [text appendFormat:
        @"        tensor<fp16, [1, %u, 1, %u]> g = conv(dilations=dl, groups=gr, "
        "pad=pd, pad_type=pt, strides=st, weight=Wg, x=x0)[name=string(\"g\")];\n"
        "        tensor<fp16, [1, %u, 1, %u]> u = conv(dilations=dl, groups=gr, "
        "pad=pd, pad_type=pt, strides=st, weight=Wu, x=x0)[name=string(\"u\")];\n"
        "        tensor<fp16, [1, %u, 1, %u]> sg = sigmoid(x=g)[name=string(\"sg\")];\n"
        "        tensor<fp16, [1, %u, 1, %u]> gs = mul(x=g, y=sg)[name=string(\"gs\")];\n"
        "        tensor<fp16, [1, %u, 1, %u]> act = mul(x=gs, y=u)[name=string(\"act\")];\n"
        "        fp16 inv16 = const()[name=string(\"inv16\"), val=fp16(0.0625)];\n"
        "        tensor<fp16, [1, %u, 1, %u]> act_p = mul(x=act, y=inv16)[name=string(\"act_p\")];\n",
        ffn, rows, ffn, rows, ffn, rows, ffn, rows, ffn, rows, ffn, rows];
    uint32_t kc = ffn / nch;
    for (uint32_t c = 0; c < nch; c++) {
        [text appendFormat:
            @"        tensor<int32, [4]> db%u = const()[name=string(\"db%u\"), "
            "val=tensor<int32, [4]>([0,%u,0,0])];\n"
            "        tensor<int32, [4]> dsz%u = const()[name=string(\"dsz%u\"), "
            "val=tensor<int32, [4]>([1,%u,1,%u])];\n"
            "        tensor<fp16, [1, %u, 1, %u]> a%u = slice_by_size(x=act_p, "
            "begin=db%u, size=dsz%u)[name=string(\"a%u\")];\n",
            c, c, c * kc, c, c, kc, rows, kc, rows, c, c, c, c];
        char wname[16];
        snprintf(wname, sizeof(wname), "Wd%u", c);
        emit_int8_weight(text, wname, hidden, kc, dq_offs[c], ds);
        [text appendFormat:
            @"        tensor<fp16, [1, %u, 1, %u]> p%u = conv(dilations=dl, "
            "groups=gr, pad=pd, pad_type=pt, strides=st, weight=Wd%u, x=a%u)"
            "[name=string(\"p%u\")];\n",
            hidden, rows, c, c, c, c];
    }
    NSString *previous = @"p0";
    for (uint32_t c = 1; c < nch; c++) {
        NSString *sum = [NSString stringWithFormat:@"ds%u", c];
        [text appendFormat:
            @"        tensor<fp16, [1, %u, 1, %u]> %@ = add(x=%@, y=p%u)"
            "[name=string(\"%@\")];\n", hidden, rows, sum, previous, c, sum];
        previous = sum;
    }
    [text appendFormat:
        @"        fp16 sixteen = const()[name=string(\"sixteen\"), val=fp16(16.0)];\n"
        "        tensor<fp16, [1, %u, 1, %u]> y = mul(x=%@, y=sixteen)[name=string(\"y\")];\n"
        "        } -> (y);\n}\n", hidden, rows, previous];
    return text;
}

/* Bind a program's I/O surfaces: a shared plane is retained, NULL allocates
 * a per-program surface of `bytes`. Returns false on allocation failure. */
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
     * per-program surfaces are cleared. */
    if (!input_plane) memset(m->input_base, 0, input_bytes);
    if (!output_plane) memset(m->output_base, 0, output_bytes);
    return true;
}

msv_ane_mlp *msv_ane_mlp_create(const char *name,
                                uint32_t hidden, uint32_t ffn, uint32_t rows,
                                const int8_t *gate_q, const float *gate_s,
                                const int8_t *up_q, const float *up_s,
                                const int8_t *down_q, const float *down_s,
                                msv_ane_plane *input_plane,
                                msv_ane_plane *output_plane,
                                char *error, size_t error_size) {
    if (!msv_ane_available()) {
        mlp_fail(error, error_size, "the Neural Engine bridge is unavailable");
        return NULL;
    }
    /* rows % 32: an fp16 plane's per-channel pitch is rows * 2 bytes, and a
     * pitch off the 64-byte grid compiles fine but fails every eval with a
     * bare "Program Inference error" (measured, A3 probe 2026-08-18) — so
     * the refusal happens HERE, by name. */
    if (!hidden || !ffn || !rows || rows % 32) {
        mlp_fail(error, error_size, "ANE %s: rows must be a positive multiple "
                 "of 32 — the fp16 plane pitch (rows x 2 bytes) must sit on "
                 "the 64-byte grid (got %u)", name, rows);
        return NULL;
    }
    uint32_t nch = down_chunks_for(ffn);
    if (!nch || nch > 16) {
        mlp_fail(error, error_size, "ANE %s: ffn %u has no K-chunking <= %u",
                 name, ffn, ANE_MLP_MAX_DOWN_K);
        return NULL;
    }

    size_t big = (size_t)ffn * hidden;
    size_t bound = 64 + 2 * (64 + ((big + 63) & ~(size_t)63)) +
        (size_t)(nch + 3) * (64 + 64) + (size_t)hidden * ffn +
        2 * ((size_t)ffn * 2) + (size_t)hidden * 2 + 4096;
    mlp_blob blob = { .data = calloc(bound, 1), .cursor = 64 };
    if (!blob.data) {
        mlp_fail(error, error_size, "out of memory building ANE %s blob", name);
        return NULL;
    }
    blob.data[0] = 0x01;
    blob.data[4] = 0x02;
    size_t gq_off = blob_add(&blob, gate_q, big);
    size_t gs_off = blob_add_scales_f16(&blob, gate_s, ffn);
    size_t uq_off = blob_add(&blob, up_q, big);
    size_t us_off = blob_add_scales_f16(&blob, up_s, ffn);
    /* Down weight re-packed as nch contiguous [hidden, ffn/nch] K-slabs. */
    uint32_t kc = ffn / nch;
    size_t dq_offs[16];
    {
        int8_t *slab = malloc((size_t)hidden * kc);
        if (!slab) {
            free(blob.data);
            mlp_fail(error, error_size, "out of memory building ANE %s down "
                     "slabs", name);
            return NULL;
        }
        for (uint32_t c = 0; c < nch; c++) {
            for (uint32_t n = 0; n < hidden; n++)
                memcpy(slab + (size_t)n * kc,
                       down_q + (size_t)n * ffn + (size_t)c * kc, kc);
            dq_offs[c] = blob_add(&blob, slab, (size_t)hidden * kc);
        }
        free(slab);
    }
    size_t ds_off = blob_add_scales_f16(&blob, down_s, hidden);
    size_t blob_bytes = blob.cursor;

    msv_ane_mlp *m = calloc(1, sizeof(*m));
    if (!m) {
        free(blob.data);
        mlp_fail(error, error_size, "out of memory creating ANE %s", name);
        return NULL;
    }
    m->hidden = hidden;
    m->ffn = ffn;
    m->rows = rows;
    size_t plane_bytes = (size_t)hidden * rows * sizeof(__fp16);
    if (!mlp_bind_planes(m, input_plane, plane_bytes, output_plane,
                         plane_bytes)) {
        free(blob.data);
        msv_ane_mlp_free(m);
        mlp_fail(error, error_size, "ANE %s surface alloc failed", name);
        return NULL;
    }

    @autoreleasepool {
        NSString *text = mlp_program(hidden, ffn, rows, nch, gq_off, gs_off,
                                     uq_off, us_off, dq_offs, ds_off);
        IOSurfaceRef inputs[1] = { m->input_surface };
        m->model = msv_ane_model_create(name, text.UTF8String, blob.data,
                                       blob_bytes, inputs, 1,
                                       m->output_surface, error, error_size);
    }
    if (!m->model) {
        msv_ane_mlp_free(m);
        return NULL;
    }
    return m;
}

/* GDN input projections as ONE fused conv over the stacked
 * [qkv_out + z_out, hidden] weight (qkv rows first): each output row is an
 * independent dot product, so stacking along the output-channel axis is
 * byte-equivalent to two convs + concat, with one op fewer. K = hidden
 * (5120 on the 27B) sits well under the K=17408 chunking cliff, so no
 * in-graph K-chunking; values are post-norm-scale small (|y| < 1 measured),
 * so no accumulator wrap either — the same regime as the gate/up convs. */
static NSString *gdn_program(uint32_t hidden, uint32_t out_total,
                             uint32_t rows, size_t q_off, size_t s_off) {
    NSMutableString *text = [NSMutableString string];
    [text appendFormat:@"program(1.3)\n[buildInfo = dict<string, string>({{"
        "\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", "
        "\"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"
        "    func main<ios18>(tensor<fp16, [1, %u, 1, %u]> x0) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1, 1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n",
        hidden, rows];
    emit_int8_weight(text, "Wqz", out_total, hidden, q_off, s_off);
    [text appendFormat:
        @"        tensor<fp16, [1, %u, 1, %u]> y = conv(dilations=dl, groups=gr, "
        "pad=pd, pad_type=pt, strides=st, weight=Wqz, x=x0)[name=string(\"y\")];\n"
        "        } -> (y);\n}\n", out_total, rows];
    return text;
}

msv_ane_mlp *msv_ane_gdn_create(const char *name,
                                uint32_t hidden, uint32_t qkv_out,
                                uint32_t z_out, uint32_t rows,
                                const int8_t *qkv_q, const float *qkv_s,
                                const int8_t *z_q, const float *z_s,
                                msv_ane_plane *input_plane,
                                msv_ane_plane *output_plane,
                                char *error, size_t error_size) {
    if (!msv_ane_available()) {
        mlp_fail(error, error_size, "the Neural Engine bridge is unavailable");
        return NULL;
    }
    /* Same 64-byte fp16 plane-pitch contract as msv_ane_mlp_create. */
    if (!hidden || !qkv_out || !z_out || !rows || rows % 32) {
        mlp_fail(error, error_size, "ANE %s: rows must be a positive multiple "
                 "of 32 — the fp16 plane pitch (rows x 2 bytes) must sit on "
                 "the 64-byte grid (got %u)", name, rows);
        return NULL;
    }
    uint32_t out_total = qkv_out + z_out;
    size_t qkv_bytes = (size_t)qkv_out * hidden;
    size_t z_bytes = (size_t)z_out * hidden;
    size_t bound = 64 + (64 + ((qkv_bytes + z_bytes + 63) & ~(size_t)63)) +
        (64 + (size_t)out_total * 2) + 4096;
    mlp_blob blob = { .data = calloc(bound, 1), .cursor = 64 };
    if (!blob.data) {
        mlp_fail(error, error_size, "out of memory building ANE %s blob", name);
        return NULL;
    }
    blob.data[0] = 0x01;
    blob.data[4] = 0x02;
    /* Stacked weight: qkv rows then z rows, one contiguous blob chunk. */
    size_t q_off = blob_add(&blob, NULL, qkv_bytes + z_bytes);
    memcpy(blob.data + q_off + 64, qkv_q, qkv_bytes);
    memcpy(blob.data + q_off + 64 + qkv_bytes, z_q, z_bytes);
    size_t s_off = blob.cursor;
    blob_add(&blob, NULL, (size_t)out_total * 2);
    {
        __fp16 *dst = (__fp16 *)(void *)(blob.data + s_off + 64);
        for (uint32_t i = 0; i < qkv_out; i++) dst[i] = (__fp16)qkv_s[i];
        for (uint32_t i = 0; i < z_out; i++) dst[qkv_out + i] = (__fp16)z_s[i];
    }
    size_t blob_bytes = blob.cursor;

    msv_ane_mlp *m = calloc(1, sizeof(*m));
    if (!m) {
        free(blob.data);
        mlp_fail(error, error_size, "out of memory creating ANE %s", name);
        return NULL;
    }
    m->hidden = hidden;
    m->ffn = out_total;
    m->rows = rows;
    if (!mlp_bind_planes(m, input_plane,
                         (size_t)hidden * rows * sizeof(__fp16),
                         output_plane,
                         (size_t)out_total * rows * sizeof(__fp16))) {
        free(blob.data);
        msv_ane_mlp_free(m);
        mlp_fail(error, error_size, "ANE %s surface alloc failed", name);
        return NULL;
    }

    @autoreleasepool {
        NSString *text = gdn_program(hidden, out_total, rows, q_off, s_off);
        IOSurfaceRef inputs[1] = { m->input_surface };
        m->model = msv_ane_model_create(name, text.UTF8String, blob.data,
                                       blob_bytes, inputs, 1,
                                       m->output_surface, error, error_size);
    }
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

int msv_ane_mlp_eval(msv_ane_mlp *m, char *error, size_t error_size) {
    if (!m) {
        mlp_fail(error, error_size, "the ANE MLP is not loaded");
        return 0;
    }
    return msv_ane_model_eval(m->model, error, error_size);
}

double msv_ane_mlp_compile_seconds(const msv_ane_mlp *m) {
    return m ? msv_ane_model_compile_seconds(m->model) : 0.0;
}

int msv_ane_mlp_cache_hit(const msv_ane_mlp *m) {
    return m ? (int)msv_ane_model_cache_hit(m->model) : 0;
}
