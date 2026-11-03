inline int ltx_na_start(int length, int kern, int i) {
    const int k = kern < length ? kern : length;
    const int lo = length - k;
    const int mid = k / 2;
    const int c = i >= mid ? (i - mid) : 0;
    return c < lo ? c : lo;
}
