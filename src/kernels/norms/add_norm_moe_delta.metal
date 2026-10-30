{
    float routed = 0.0f;
    for (int e = 0; e < K; e++) routed += float(Y[(size_t(r) * K + e) * D + c]) * WT[r * K + e];
    delta = float(T(routed));
    if (SHARED) delta = float(T(delta + float(XS[at])));
}