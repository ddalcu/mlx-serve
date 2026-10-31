const int KVH = v_shape[1];
const int L = v_shape[2];
const int r = int(thread_position_in_grid.x) / 32;
if (r >= KVH * L) return;
const uint lane = thread_index_in_simdgroup;
const float4 sc = max(*(const device float4*)(vm + (r / L) * 128 + lane * 4), float4(1e-20f)) / 127.0f;
const float4 x = float4(*(const device vec<T, 4>*)(v + (size_t)r * 128 + lane * 4));
*(device char4*)(vi + (size_t)r * 128 + lane * 4) = char4(rint(x / sc));