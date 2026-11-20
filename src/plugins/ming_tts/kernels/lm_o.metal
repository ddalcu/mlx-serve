for (int n = int(t); n < NO; n += 32 * SG) OUT[ob + n] = bfloat16_t(float(X[ob + n]) + rb(rsum(n, 0)));
