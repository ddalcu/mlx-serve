// Windows build: no libwebp or Bonjour yet. WebP decodes fail (callers treat NULL as
// undecodable) and every DNS-SD call returns kDNSServiceErr_Unsupported, so LAN sharing stays off.
#include <stddef.h>
#include <stdint.h>

uint8_t *WebPDecodeRGBA(const uint8_t *data, size_t size, int *w, int *h) { (void)data; (void)size; (void)w; (void)h; return NULL; }
uint8_t *WebPDecodeRGB(const uint8_t *data, size_t size, int *w, int *h) { (void)data; (void)size; (void)w; (void)h; return NULL; }
void WebPFree(void *ptr) { (void)ptr; }

#define DNS_UNSUPPORTED (-65544)
int32_t DNSServiceRegister(void) { return DNS_UNSUPPORTED; }
int32_t DNSServiceBrowse(void) { return DNS_UNSUPPORTED; }
int32_t DNSServiceResolve(void) { return DNS_UNSUPPORTED; }
int32_t DNSServiceRefSockFD(void *ref) { (void)ref; return -1; }
int32_t DNSServiceProcessResult(void *ref) { (void)ref; return DNS_UNSUPPORTED; }
void DNSServiceRefDeallocate(void *ref) { (void)ref; }

// mlx-c compiles its float16/bfloat16 accessors out under MSVC (no _Float16). Both types are
// 16-bit storage, so they alias the uint16 accessors, as patches/mlxc-x86-f16-data-alias.patch
// does on Linux x86.
typedef struct mlx_array_ { void *ctx; } mlx_array;
const uint16_t *mlx_array_data_uint16(const mlx_array arr);
int mlx_array_item_uint16(uint16_t *res, const mlx_array arr);
const uint16_t *mlx_array_data_float16(const mlx_array arr) { return mlx_array_data_uint16(arr); }
const uint16_t *mlx_array_data_bfloat16(const mlx_array arr) { return mlx_array_data_uint16(arr); }
int mlx_array_item_float16(uint16_t *res, const mlx_array arr) { return mlx_array_item_uint16(res, arr); }
int mlx_array_item_bfloat16(uint16_t *res, const mlx_array arr) { return mlx_array_item_uint16(res, arr); }
