const uint t = thread_position_in_grid.x;
const int kv = int(threads_per_grid.x);
const int blk = int(t) / RATIO;
bool visible = int(t) >= (kv / RATIO) * RATIO;
if (!visible) {
  int lo = 0;
  int hi = KB - 1;
  while (lo <= hi) {
    const int mid = (lo + hi) >> 1;
    const int v = blocks[mid];
    if (v == blk) { visible = true; break; }
    if (v < blk) lo = mid + 1; else hi = mid - 1;
  }
}
mask[t] = visible;