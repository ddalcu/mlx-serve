constexpr uint TGN   = (uint)TGS;
constexpr uint SIMDW = 32u;
constexpr uint NSIMD = TGN / SIMDW;
constexpr uint BINS  = (DIGITS == 8) ? 256u : 2048u;
constexpr uint NLEV  = (DIGITS == 8) ? 4u : 3u;
constexpr uint KTOP  = (uint)K;
constexpr int  SENTINEL = 2147483647;
static_assert(NSIMD <= SIMDW);

threadgroup metal::atomic_uint hist[BINS];
threadgroup uint sgs[2u * NSIMD];
threadgroup uint sh[4];

const uint row  = threadgroup_position_in_grid.y;
const uint tid  = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg   = simdgroup_index_in_threadgroup;
