// DSV41_LOOKAHEAD4 (q3-lookahead4-20260925.md): the pool's only Objective-C++.  One C-callable entry that hands a
// value to an MTLSharedEvent from a pool pthread (no Python, no GIL): the event object is created on MLX's own
// MTLDevice by the extension (ext/q3ev4.cpp, q3ev4.create_metal_event) and kept alive by it for the process lifetime.
// Compiled without ARC: `event` is the id<MTLSharedEvent> itself (metal-cpp objects are the Objective-C objects).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>

extern "C" void q3ld_mtl_signal(void *event, uint64_t value) {
    @autoreleasepool {
        [(id<MTLSharedEvent>)event setSignaledValue:value];
    }
}
