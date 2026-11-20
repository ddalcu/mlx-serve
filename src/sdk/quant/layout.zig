//! `Weight.layout`: dense (pre-transposed `[in, out]`), an MLX mode, row-scaled e4m3 (`sushi_pack.fp8Linear`), or
//! e4m3 on a 64 x 128 scale grid (`fp8_block128.linear`). It imports nothing, so a base names it without a format.

pub const Layout = enum(u8) { dense, affine, nvfp4, mxfp4, mxfp8, fp8_rows, fp8_block128 };
