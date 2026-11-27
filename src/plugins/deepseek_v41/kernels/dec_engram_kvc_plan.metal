// dec_engram_kvc_plan: the Engram contribution cache's look-up for one decode row (decode/engram_kvc.zig). Each of
// the NT tables (an Engram layer's slot) keeps S sets of W ways; way e's key is its table row id + 1 (0: empty), its
// age the step that last used it. Thread (t, s) owns set s of table t and writes it into the new keys and ages; where
// s is the set of row id ids[t] it looks the id up: a hit marks plan[t] (bit 31) and renews the way's age, a miss
// takes the oldest way (an empty one first) for the id. plan[t] is the way's entry, (t * S + s) * W + way. `after`
// (the last cached step's wkv, bound and never read) holds the look-up behind the writes of the entries it took.
const uint g = thread_position_in_grid.x;
if (g >= uint(NT * S)) return;
const uint t = g / uint(S);
const uint s = g % uint(S);
const uint base = g * uint(W);
uint k[W];
uint a[W];
for (int i = 0; i < W; ++i) {
  k[i] = keys[base + i];
  a[i] = ages[base + i];
}
const uint id = ids[t];
if (id % uint(S) == s) {
  const uint key = id + 1;
  const uint now = clk[0];
  int way = -1;
  for (int i = 0; i < W; ++i) way = (way < 0 && k[i] == key) ? i : way;
  uint hit = 0x80000000u;
  if (way < 0) {
    hit = 0;
    way = 0;
    for (int i = 1; i < W; ++i) way = a[i] < a[way] ? i : way;
    k[way] = key;
  }
  a[way] = now;
  plan[t] = hit | (base + uint(way));
}
for (int i = 0; i < W; ++i) {
  nkeys[base + i] = k[i];
  nages[base + i] = a[i];
}
