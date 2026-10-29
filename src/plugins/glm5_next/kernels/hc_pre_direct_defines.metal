#define HVAL(k) float(x[row * K + (k)])
#define STORE_H(k, v)
#define TAIL4(k) float4(*(const device vec<T, 4>*)(x + row * K + (k)))
