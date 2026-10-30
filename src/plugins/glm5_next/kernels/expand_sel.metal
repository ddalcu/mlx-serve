const int col = int(thread_position_in_grid.x);
const int row = int(thread_position_in_grid.y);
if (col >= OUT_W) {
  return;
}
const int qp = int(qpos[0]) + row;
const int pool_len = int(qpos[1]);
int v = -1;
if (col < SEL_K * KPOOL) {
  const int s = int(selected[row * SEL_K + col / KPOOL]);
  const bool valid = s < pool_len && (s + 1) * KPOOL - 1 <= qp;
  if (valid) {
    v = s * KPOOL + (col % KPOOL);
  }
} else if (col >= TOPK_W && col < TOPK_W + TAIL_W) {
  const int t = col - TOPK_W;
  const int tail_count = (qp + 1) % KPOOL;
  if (t < tail_count) {
    v = qp + 1 - tail_count + t;
  }
}
out[row * OUT_W + col] = v;