// PROFILE builds only (see dsv41_cb_timeline.h): MLX's current command buffer on a stream, through the metal
// backend's CommandEncoder (the handle conversion is mlx_event_shim.cpp's).
#include <cmath>

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/c/private/stream.h"
#include "dsv41_cb_timeline.h"

namespace mx = mlx::core;

extern "C" void *dsv41tl_mlx_buffer(mlx_stream s) {
  try {
    const mx::Stream &stream = mlx_stream_get_(s);
    if (stream.device.type != mx::Device::DeviceType::gpu) return nullptr;
    return mx::metal::get_command_encoder(stream).get_command_buffer();
  } catch (...) {
    return nullptr;
  }
}
