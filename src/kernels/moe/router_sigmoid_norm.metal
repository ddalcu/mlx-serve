  if (NORM != 0 && NK > 1) {
    w = w / (tot + 1e-20f);
  }
  // `scale` is a 0-dim array, which MLX binds as `const constant
  // float&` — a reference, not a subscriptable pointer.
  w = w * scale;