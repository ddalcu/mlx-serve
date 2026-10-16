// The libwebp decode API mlx-serve calls, for the Windows build, which has no libwebp yet:
// src/windows_stubs.c answers every call with NULL (undecodable).
#pragma once
#include <stddef.h>
#include <stdint.h>

uint8_t *WebPDecodeRGBA(const uint8_t *data, size_t data_size, int *width, int *height);
uint8_t *WebPDecodeRGB(const uint8_t *data, size_t data_size, int *width, int *height);
void WebPFree(void *ptr);
