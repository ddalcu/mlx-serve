//! The KV-cache quantization scheme lives in the SDK (`sdk/blocks/attention/kv_quant.zig`); the engine reaches it here.

const kv_quant = @import("../../../sdk/blocks/attention/kv_quant.zig");
pub const Scheme = kv_quant.Scheme;
pub const KVQuantConfig = kv_quant.KVQuantConfig;
pub const QuantizedKV = kv_quant.QuantizedKV;
pub const quantizeAffine = kv_quant.quantizeAffine;
pub const dequantizeAffine = kv_quant.dequantizeAffine;
pub const BorrowedTriple = kv_quant.BorrowedTriple;
pub const quantAttention = kv_quant.quantAttention;
