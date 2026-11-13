// kda_prologue.metal's sigmoid: unary_ops.h `Sigmoid` of the server's MLX, its body verbatim as a function, so the
// chain's `mlx_sigmoid` and the text take the same overloads (MLX's utils.h is before this header). It spells
// precise::exp as the op does: a bare metal::exp is the fast exp in a runtime-compiled kernel.
template <typename T>
inline T kda_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
