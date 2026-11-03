
#include <metal_stdlib>
using namespace metal;

// Order-preserving float32 -> uint32 key, DESCENDING (smaller key = larger
// score).  -0.0 is canonicalised to +0.0 so it compares EQUAL to +0.0, exactly
// as the reference's float `==` does; the indexer really does emit -0.0 (a ReLU
// zero times a negative weights_proj head).
inline uint dsv41_kd(float x) {
    uint u = as_type<uint>(x);
    if (u == 0x80000000u) { u = 0u; }
    uint key = (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
    return ~key;
}
