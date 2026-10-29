int t = thread_position_in_grid.x;
int c = thread_position_in_grid.y;
int j = thread_position_in_grid.z;
if (t >= S || c >= HC || j >= D) return;
const int PACK = 2 * HC + HC * HC;
float acc = 0.0f;
for (int q = 0; q < HC; ++q) {
  float p = pk[t * PACK + 2 * HC + c * HC + q] * stream_in[(t * HC + q) * D + j];
  acc = acc + p;
}
acc = acc + pk[t * PACK + HC + c] * out_v[t * D + j];
ns[(t * HC + c) * D + j] = acc;