inline float msv_log1p(float x) {
    float xp1 = 1.0f + x;
    if (xp1 == metal::numeric_limits<float>::max()) { return metal::numeric_limits<float>::max(); }
    if (xp1 == 1.0f) { return x; }
    return x * (metal::log(xp1) / (xp1 - 1.0f));
}