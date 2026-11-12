// What MLX's gemv kernel binds as constant buffers for GEMVKernel::run: K inputs (the row's length and the matrix's
// row stride) and N outputs; alpha, beta and the bias stride unread (no axpby).
constant int ro1_in = {d};
constant int ro1_out = {d};
constant int ro1_ld = {d};
constant float ro1_alpha = 1.0f;
constant float ro1_beta = 0.0f;
constant int ro1_bias_stride = 0;
