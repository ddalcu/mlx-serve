const long kq_base = (long)bb * kqS[0] + (long)hk * kqS[1];
const long vq_base = (long)bb * vqS[0] + (long)hk * vqS[1];
const bool k_vec = (reinterpret_cast<ulong>(kq) & 15ul) == 0 && kqS[3] == 1 &&
    (kqS[2] & 7) == 0 && (kqS[1] & 7) == 0;
const bool v_vec = (reinterpret_cast<ulong>(vq) & 15ul) == 0 && vqS[3] == 1 &&
    (vqS[2] & 7) == 0 && (vqS[1] & 7) == 0;
