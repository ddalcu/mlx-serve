// SPDX-License-Identifier: Apache-2.0
// Ported from oMLX (jundot/omlx) omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/qsa_nax.py @ d6b2b92.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
#define UNROLL _Pragma("clang loop unroll(full)")
constant int GQA = 12;
constant int TOPK = 512;
constant int PV_TERMS = 2;
using PT = half;
