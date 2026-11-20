// deepseek_v41 exact DSpark verify (MLX_SERVE_DSV41_EXACT): x @ w.T for up to 16 rows on MLX's OWN one-row kernel.
//
// MLX's quantized_matmul runs one row (M = 1) on `qmv_fast` and two or more rows on `qmv_wide` (or `qmv_mma` /
// `qmm_splitk`), whose K reduction differs: a verify row's logits then differ from the plain step's in the low bits, and
// near-tie argmaxes flip. `qmv_fast` itself takes its input row from the threadgroup's x position (qmv_fast_impl:
// `x += tid.x * in_vec_size`, `y += tid.x * out_vec_size`), so launching the very pipeline MLX uses at M = 1 over a
// grid of M rows gives every row the one-row call's bits, the weights read once per output tile by the M threadgroups
// the GPU runs side by side. This primitive is that launch, bound as MLX's `qmv()` binds it (quantized.cpp), against
// the pinned MLX's metallib: no MLX change, no source compiled at run time. Batched (`[B, M, K]` against `[B, N, K']`)
// as MLX's `_batch_1` kernel (wo_a's groups). Included by mlx_alloc_shim.cpp (a fixed list of csrc files).
//
// A workaround for how the pinned MLX dispatches M > 1. It binds MLX internals: the `qmv_fast` kernel's name, its
// buffer order and its tile (`results_per_simdgroup = 4`). decode/exact_test.zig's row-qmv bit test (the default suite,
// on a real GPU) fails when an MLX bump changes them. It goes away once MLX's quantized_matmul is row-invariant (M rows
// give each row its M = 1 bits) or MLX dispatches `qmv_fast` over M rows itself.
#include <optional>
#include <string>
#include <vector>

#include "mlx/c/private/stream.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace {

class Dsv41RowQmv : public mx::UnaryPrimitive {
 public:
  Dsv41RowQmv(mx::Stream s, int group_size, int bits, std::string mode, bool has_biases)
      : mx::UnaryPrimitive(s), group_size_(group_size), bits_(bits), mode_(std::move(mode)), has_biases_(has_biases) {}

  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("dsv41_row_qmv: GPU only");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    const mx::array& x = inputs[0];
    const mx::array& w = inputs[1];
    const mx::array& scales = inputs[2];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    const int K = x.shape(-1);
    const int M = x.shape(-2);
    const int N = out.shape(-1);
    const int B = static_cast<int>(out.size() / M / N);
    // quantized.cpp qmv(): bn = 2 * results_per_simdgroup (4: the narrow tile is nvfp4 on an M5 Max only), bk 32.
    constexpr int results_per_simdgroup = 4;
    constexpr int bn = 2 * results_per_simdgroup;
    const bool partial_rows = N % bn != 0;
    std::string kname = mode_ + "_qmv_fast_bfloat16_t_gs_" + std::to_string(group_size_) + "_b_" + std::to_string(bits_) +
                        "_hgs_0" + (B > 1 ? "_batch_1" : "_batch_0") + (partial_rows ? "_pr_1" : "_pr_0") + "_r_" +
                        std::to_string(results_per_simdgroup);
    auto& d = mx::metal::device(stream().device);
    auto* kernel = d.get_kernel(kname);
    auto& enc = mx::metal::get_command_encoder(stream());
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(w, 0);
    enc.set_input_array(scales, 1);
    if (has_biases_) enc.set_input_array(inputs[3], 2);
    int c = 3;
    enc.set_input_array(x, c++);
    enc.set_output_array(out, c++);
    enc.set_bytes(K, c++);
    enc.set_bytes(N, c++);
    if (B > 1) {
      // quantized.cpp add_strides_and_shapes
      const int x_batch_ndims = x.ndim() - 2;
      const int w_batch_ndims = w.ndim() - 2;
      enc.set_bytes(x_batch_ndims, c++);
      enc.set_vector_bytes(x.shape(), c++);
      enc.set_vector_bytes(x.strides(), c++);
      enc.set_bytes(w_batch_ndims, c++);
      enc.set_vector_bytes(w.shape(), c++);
      enc.set_vector_bytes(w.strides(), c++);
      enc.set_vector_bytes(scales.strides(), c++);
      if (has_biases_) enc.set_vector_bytes(inputs[3].strides(), c++);
    }
    enc.dispatch_threadgroups(MTL::Size(M, (N + bn - 1) / bn, B), MTL::Size(32, 2, 1));
  }

  const char* name() const override {
    return "Dsv41RowQmv";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& o = static_cast<const Dsv41RowQmv&>(other);
    return group_size_ == o.group_size_ && bits_ == o.bits_ && mode_ == o.mode_ && has_biases_ == o.has_biases_;
  }

 private:
  int group_size_;
  int bits_;
  std::string mode_;
  bool has_biases_;
};

}  // namespace

extern "C" {

// *res (an mlx_array_new() handle) = x @ w.T, x [..., M, K] bf16 with M <= 16, w [N, K'] (or [B, N, K'] with x
// [B, M, K]) quantized in `mode` ("affine": biases required; "mxfp8"/"mxfp4": none) with transposed weights, on MLX's
// one-row qmv_fast pipeline: each row bit-equal to its own M = 1 quantized_matmul. K must suit qmv_fast
// (K % (32 * values per thread) == 0: 256 for 8-bit affine and mxfp8). 0, or -1 (dsv41_alloc_last_error).
int dsv41_row_qmv(mlx_array* res, mlx_array x_, mlx_array w_, mlx_array scales_, mlx_array biases_, int group_size,
                  int bits, const char* mode, mlx_stream s_) {
  try {
    const mx::Stream s = mlx_stream_get_(s_);
    const std::string md(mode);
    const bool has_biases = biases_.ctx != nullptr;
    if ((md == "affine") != has_biases) throw std::invalid_argument("dsv41_row_qmv: affine takes biases, fp modes none");
    const mx::array& x0 = mlx_array_get_(x_);
    const mx::array& w0 = mlx_array_get_(w_);
    if (x0.dtype() != mx::bfloat16) throw std::invalid_argument("dsv41_row_qmv: bf16 rows only");
    if (x0.ndim() < 2 || x0.ndim() > 3 || w0.ndim() != x0.ndim()) {
      if (!(x0.ndim() == 2 && w0.ndim() == 2)) throw std::invalid_argument("dsv41_row_qmv: x [M, K] / w [N, K'] or x [B, M, K] / w [B, N, K']");
    }
    if (x0.shape(-2) < 1 || x0.shape(-2) > 16) throw std::invalid_argument("dsv41_row_qmv: 1..16 rows");
    const int K = x0.shape(-1);
    // quantized.cpp qmv_fast_k_alignment: get_pack_factor(bits, 32) * (bits == 2 ? 1 : 2) * 32.
    const int pack_factor = (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : 32 / bits);
    if (K % (pack_factor * (bits == 2 ? 1 : 2) * 32) != 0) throw std::invalid_argument("dsv41_row_qmv: K off qmv_fast's block");
    const mx::array& s0 = mlx_array_get_(scales_);
    if (w0.dtype() != mx::uint32) throw std::invalid_argument("dsv41_row_qmv: packed weights are uint32");
    if (md == "affine" ? (s0.dtype() != mx::bfloat16 || mlx_array_get_(biases_).dtype() != mx::bfloat16) : s0.dtype() != mx::uint8)
      throw std::invalid_argument("dsv41_row_qmv: the bf16 kernel reads bf16 affine scales/biases, uint8 fp scales");
    const int N = w0.shape(-2);
    if (w0.ndim() == 3 && w0.shape(0) != x0.shape(0)) throw std::invalid_argument("dsv41_row_qmv: batch mismatch");
    // The rows made row-contiguous (the grouped wo_a input is a transposed view); a loaded, row-contiguous weight goes
    // as it is (a Contiguous node would copy one that is a view into a larger buffer on every call), as MLX's own
    // QuantizedMatmul takes it.
    auto as_is = [&](const mx::array& a) {
      return (a.is_available() && a.flags().row_contiguous) ? a : mx::contiguous(a, false, s);
    };
    std::vector<mx::array> ins = {mx::contiguous(x0, false, s), as_is(w0), as_is(s0)};
    if (has_biases) ins.push_back(as_is(mlx_array_get_(biases_)));
    mx::Shape shape = x0.shape();
    shape.back() = N;
    mlx_array_set_(*res, mx::array(shape, mx::bfloat16, std::make_shared<Dsv41RowQmv>(s, group_size, bits, md, has_biases), std::move(ins)));
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  } catch (...) {
    g_error = "dsv41_row_qmv failed";
    return -1;
  }
}

}  // extern "C"
