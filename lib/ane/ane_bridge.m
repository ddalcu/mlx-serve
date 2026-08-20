#import "ane_bridge.h"

#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <dlfcn.h>
#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

struct msv_ane_model {
    void *model;
    void *requests; /* NSArray<_ANERequest *>, one per procedure */
    void *options;  /* NSDictionary passed at compile/load/eval */
    char *staging_directory;
    double compile_seconds;
    bool cache_hit;
};

/* Instance affinity (oMLX, qwen35_prefill/csrc/qwen35_ane.mm:381). BOTH
 * keys are required together: the scheduler only honours an instance hint
 * for its single-ANE procedure variant, so naming a die without the variant
 * hint is silently ignored. Instance 0 = no hint at all, byte-identical to
 * the empty dict every single-ANE build has always passed. */
static NSDictionary *bridge_options(int ane_instance) {
    if (ane_instance <= 0) return @{};
    return @{ @"kANEFProcedureVariantHint" : @1,
              @"kANEFAneInstanceHint" : @(ane_instance) };
}

static void bridge_fail(char *error, size_t error_size, const char *format,
                        ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static double bridge_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

int msv_ane_bridge_available(void) {
    static int state = -1;
    if (state >= 0) return state;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
           "AppleNeuralEngine", RTLD_NOW);
    state = NSClassFromString(@"_ANEInMemoryModelDescriptor") != nil &&
            NSClassFromString(@"_ANEInMemoryModel") != nil &&
            NSClassFromString(@"_ANERequest") != nil &&
            NSClassFromString(@"_ANEIOSurfaceObject") != nil;
    return state;
}

IOSurfaceRef msv_ane_bridge_surface(size_t bytes) {
    size_t aligned = (bytes + 16383u) & ~(size_t)16383u;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(aligned),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(aligned),
        (id)kIOSurfaceAllocSize: @(aligned),
        (id)kIOSurfacePixelFormat: @0});
}

static NSString *bridge_cache_root(void) {
    const char *override = getenv("MLX_SERVE_ANE_CACHE_DIR");
    NSString *root = override ? @(override) :
        [NSHomeDirectory() stringByAppendingPathComponent:@".mlx-serve/ane-cache"];
    [[NSFileManager defaultManager] createDirectoryAtPath:root
        withIntermediateDirectories:YES attributes:nil error:nil];
    return root;
}

bool msv_ane_cache_enabled(void) {
    const char *env = getenv("MLX_SERVE_ANE_CACHE");
    return !env || atoi(env) != 0;
}

static void bridge_write_sources(NSString *directory, NSData *program,
                                 NSData *weights) {
    NSFileManager *files = [NSFileManager defaultManager];
    [files createDirectoryAtPath:
        [directory stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [program writeToFile:
        [directory stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [weights writeToFile:
        [directory stringByAppendingPathComponent:@"weights/weight.bin"]
        atomically:YES];
}

static NSString *bridge_cache_entry(NSString *identifier) {
    return [[bridge_cache_root()
        stringByAppendingPathComponent:@"entries"]
        stringByAppendingPathComponent:identifier];
}

/* Hardlink a directory tree (same volume, so links are free); copy as the
 * fallback. The destination is replaced. */
static bool bridge_mirror(NSString *from, NSString *to) {
    NSFileManager *files = [NSFileManager defaultManager];
    [files removeItemAtPath:to error:nil];
    [files createDirectoryAtPath:[to stringByDeletingLastPathComponent]
        withIntermediateDirectories:YES attributes:nil error:nil];
    if ([files linkItemAtPath:from toPath:to error:nil]) return true;
    [files removeItemAtPath:to error:nil];
    return [files copyItemAtPath:from toPath:to error:nil];
}

/* The model unload deletes its staging directory, so compiled artifacts are
 * preserved in a content-addressed entry next to it and restored on reuse. */
static bool bridge_cache_restore(NSString *identifier, NSString *directory) {
    NSString *entry = bridge_cache_entry(identifier);
    NSFileManager *files = [NSFileManager defaultManager];
    if (![files fileExistsAtPath:
            [entry stringByAppendingPathComponent:@"compiled.ok"]])
        return false;
    [files setAttributes:@{NSFileModificationDate : [NSDate date]}
            ofItemAtPath:entry
                   error:nil];
    if (!bridge_mirror(entry, directory)) return false;
    [files removeItemAtPath:
        [directory stringByAppendingPathComponent:@"compiled.ok"] error:nil];
    return true;
}

/* Entries are content-addressed and never invalidated, so every distinct
 * (weights x shape) combination adds a ~250 MB entry FOREVER — a share sweep
 * grew the cache to 49 GB and ran the disk to 100%, after which ANECCompile
 * fails bare mid-build and the offload silently ships partial layer coverage
 * (measured 2026-08-18). Oldest-first prune past the byte cap, run on the
 * cold-compile path only; restores touch the entry mtime so the policy is
 * LRU, not insertion order. MLX_SERVE_ANE_CACHE_CAP_GB overrides (default
 * 40 — two full 27B program sets). */
static unsigned long long bridge_cache_cap_bytes(void) {
    const char *env = getenv("MLX_SERVE_ANE_CACHE_CAP_GB");
    long gb = env ? atol(env) : 40;
    if (gb <= 0) gb = 40;
    return (unsigned long long)gb << 30;
}

static void bridge_cache_prune(void) {
    unsigned long long cap = bridge_cache_cap_bytes();
    NSString *root = [bridge_cache_root() stringByAppendingPathComponent:@"entries"];
    NSFileManager *files = [NSFileManager defaultManager];
    NSArray<NSString *> *names = [files contentsOfDirectoryAtPath:root error:nil];
    if (!names.count) return;
    NSMutableArray<NSArray *> *entries = [NSMutableArray array];
    unsigned long long total = 0;
    for (NSString *name in names) {
        NSString *path = [root stringByAppendingPathComponent:name];
        unsigned long long bytes = 0;
        for (NSString *sub in [files enumeratorAtPath:path])
            bytes += [[files attributesOfItemAtPath:
                [path stringByAppendingPathComponent:sub] error:nil] fileSize];
        NSDate *date = [[files attributesOfItemAtPath:path error:nil]
            fileModificationDate] ?: [NSDate distantPast];
        total += bytes;
        [entries addObject:@[ date, name, @(bytes) ]];
    }
    if (total <= cap) return;
    [entries sortUsingComparator:^NSComparisonResult(NSArray *a, NSArray *b) {
        return [a[0] compare:b[0]];
    }];
    for (NSArray *entry in entries) {
        if (total <= cap) break;
        [files removeItemAtPath:[root stringByAppendingPathComponent:entry[1]]
                          error:nil];
        total -= [entry[2] unsignedLongLongValue];
    }
}

/* Only the compiled artifacts are kept: `data` embeds the constants, so the
 * weights copy and the MIL text are dead weight in an entry. */
static void bridge_cache_store(NSString *identifier, NSString *directory) {
    NSString *entry = bridge_cache_entry(identifier);
    NSFileManager *files = [NSFileManager defaultManager];
    [files removeItemAtPath:entry error:nil];
    if (![files createDirectoryAtPath:entry withIntermediateDirectories:YES
                           attributes:nil error:nil]) return;
    for (NSString *name in
         [files contentsOfDirectoryAtPath:directory error:nil]) {
        if ([name isEqualToString:@"weights"] ||
            [name isEqualToString:@"model.mil"] ||
            [name isEqualToString:@"msv-ane.pid"]) continue;
        NSString *from = [directory stringByAppendingPathComponent:name];
        NSString *to = [entry stringByAppendingPathComponent:name];
        if (![files linkItemAtPath:from toPath:to error:nil]) {
            [files removeItemAtPath:to error:nil];
            if (![files copyItemAtPath:from toPath:to error:nil]) return;
        }
    }
    [[NSData data] writeToFile:
        [entry stringByAppendingPathComponent:@"compiled.ok"] atomically:YES];
}

static void bridge_cache_evict(const char *identifier) {
    if (!identifier) return;
    [[NSFileManager defaultManager]
        removeItemAtPath:bridge_cache_entry(@(identifier)) error:nil];
}

/* Symbol indices for one procedure: oMLX query the model rather than
 * assuming 0..count-1 (qwen35_ane.mm:724-731), because a bank's procedures
 * do NOT share one symbol numbering — the identity mapping serves procedure
 * 0 and fails EVERY other one with a bare "Program Inference error"
 * (measured 2026-08-20: 200 of 252 evals dead on a 42-procedure bank).
 *
 * The selectors live on `_ANEModel`, and `_ANEInMemoryModel` is NOT a
 * subclass of it (checked: `instancesRespondToSelector:` is NO) — it OWNS
 * one, reachable through its `-model` accessor once the net is loaded. So
 * the lookup unwraps first and only then falls back to identity. */
static id bridge_inner_model(id in_memory_model) {
    if (![in_memory_model respondsToSelector:@selector(model)]) return nil;
    return ((id(*)(id, SEL))objc_msgSend)(in_memory_model, @selector(model));
}

static NSArray *bridge_symbol_indices(id in_memory_model, bool want_input,
                                      uint32_t proc, uint32_t count) {
    id model = bridge_inner_model(in_memory_model) ?: in_memory_model;
    NSString *key = want_input ? @"ANEFModelInputSymbolIndexArray"
                               : @"ANEFModelOutputSymbolIndexArray";
    if ([model respondsToSelector:@selector(procedureInfoForProcedureIndex:)]) {
        id info = ((id(*)(id, SEL, unsigned int))objc_msgSend)(
            model, @selector(procedureInfoForProcedureIndex:), proc);
        if ([info isKindOfClass:[NSDictionary class]]) {
            id indices = ((NSDictionary *)info)[key];
            if ([indices isKindOfClass:[NSArray class]] &&
                [(NSArray *)indices count] == count)
                return indices;
        }
    }
    NSMutableArray *identity = [NSMutableArray array];
    for (uint32_t i = 0; i < count; i++) [identity addObject:@(i)];
    return identity;
}

/* Whether the loaded model can answer the symbol query at all. A bank of
 * more than one procedure CANNOT be served without it — identity indices
 * make every procedure past the first fail at eval, which reads as a slow
 * offload rather than a broken one — so the create is refused by name and
 * the caller's split ladder walks down to banks of one. */
static bool bridge_has_symbol_api(id in_memory_model) {
    id model = bridge_inner_model(in_memory_model) ?: in_memory_model;
    return model &&
        [model respondsToSelector:@selector(procedureInfoForProcedureIndex:)];
}

msv_ane_model *msv_ane_model_create(const char *name, const char *mil,
                                  void *weight_bytes_owned, size_t weight_bytes,
                                  IOSurfaceRef *input_surfaces,
                                  uint32_t input_count, IOSurfaceRef output,
                                  uint32_t procedure_count, int ane_instance,
                                  char *error, size_t error_size) {
    if (!procedure_count) procedure_count = 1;
    if (!msv_ane_bridge_available()) {
        free(weight_bytes_owned);
        bridge_fail(error, error_size, "the Neural Engine bridge is "
                    "unavailable");
        return NULL;
    }
    msv_ane_model *handle = calloc(1, sizeof(*handle));
    if (!handle) {
        free(weight_bytes_owned);
        bridge_fail(error, error_size, "out of memory creating ANE %s", name);
        return NULL;
    }
    @autoreleasepool {
        NSError *failure = nil;
        NSData *weights = [NSData dataWithBytesNoCopy:weight_bytes_owned
                                               length:weight_bytes
                                         freeWhenDone:YES];
        NSData *program = [NSData dataWithBytes:mil length:strlen(mil)];
        Class descriptorClass =
            NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        id descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:),
            program, @{@"@model_path/weights/weight.bin":
                       @{@"offset": @0, @"data": weights}}, nil);
        if (!descriptor) {
            bridge_fail(error, error_size, "ANE %s descriptor rejected", name);
            free(handle);
            return NULL;
        }
        id model = ((id(*)(Class, SEL, id))objc_msgSend)(
            modelClass, @selector(inMemoryModelWithDescriptor:), descriptor);
        if (!model) {
            bridge_fail(error, error_size, "ANE %s model rejected", name);
            free(handle);
            return NULL;
        }
        NSString *identifier = ((id(*)(id, SEL))objc_msgSend)(
            model, @selector(hexStringIdentifier));
        // Staging MUST stay in $TMPDIR: the compile/load runs in aned, a
        // separate daemon that cannot read this user's home directory
        // (ANECCompile() fails bare on a ~/.mlx-serve staging path —
        // measured 2026-08-17). Only the persistent cache entries live
        // under ~/.mlx-serve/ane-cache; $TMPDIR and home share the APFS
        // Data volume, so the hardlink mirror stays free.
        //
        // The staging LOCATION is the framework's contract, not ours:
        // _ANEInMemoryModel saves and compiles at $TMPDIR/<identifier>
        // exactly (moving our writes under a subdirectory fails EVERY
        // compile with a bare ANECCompile() FAILED — measured). A killed or
        // crashed server never frees its staging, and the leaked dirs
        // (8-20 GB per big model) marched the disk to 0 across restarts —
        // at which point the framework's own model save fails ("Write
        // weightsFilePath failed" in the unified log) and surfaces as a
        // bare "compile failed: ?" with silent partial layer coverage
        // (live 2026-08-18: 112 -> 63 -> 36 -> 29 programs as the orphans
        // accumulated). So each dir gets a pid MARKER file, and the first
        // create of a process reaps any marked dir whose owner is dead.
        static dispatch_once_t reap_once;
        dispatch_once(&reap_once, ^{
            NSFileManager *fm = [NSFileManager defaultManager];
            NSString *tmp = NSTemporaryDirectory();
            for (NSString *entry in
                 [fm contentsOfDirectoryAtPath:tmp error:nil]) {
                NSString *dir = [tmp stringByAppendingPathComponent:entry];
                NSString *marker =
                    [dir stringByAppendingPathComponent:@"msv-ane.pid"];
                NSString *owner_text =
                    [NSString stringWithContentsOfFile:marker
                                              encoding:NSUTF8StringEncoding
                                                 error:nil];
                if (!owner_text) continue;
                pid_t owner = (pid_t)owner_text.integerValue;
                if (owner <= 0 || owner == getpid()) continue;
                // ESRCH = the owning process is gone; EPERM = alive but
                // not ours to signal — keep those.
                if (kill(owner, 0) != 0 && errno == ESRCH)
                    [fm removeItemAtPath:dir error:nil];
            }
        });
        NSString *directory = [NSTemporaryDirectory()
            stringByAppendingPathComponent:identifier];
        NSFileManager *files = [NSFileManager defaultManager];
        bool cache = msv_ane_cache_enabled();
        bool cached = cache && bridge_cache_restore(identifier, directory);
        if (!cached) bridge_write_sources(directory, program, weights);
        [[NSString stringWithFormat:@"%d", getpid()]
            writeToFile:[directory stringByAppendingPathComponent:@"msv-ane.pid"]
             atomically:YES
               encoding:NSUTF8StringEncoding
                  error:nil];
        handle->staging_directory = strdup(directory.UTF8String);
        NSDictionary *options = bridge_options(ane_instance);
        double started = bridge_seconds();
        bool loaded = cached &&
            ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                model, @selector(loadWithQoS:options:error:), 21, options,
                &failure);
        if (cached && !loaded) {
            failure = nil;
            bridge_write_sources(directory, program, weights);
        }
        if (!loaded) {
            if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                    model, @selector(compileWithQoS:options:error:), 21,
                    options, &failure)) {
                bridge_fail(error, error_size, "ANE %s compile failed: %s",
                            name, failure ?
                            failure.localizedDescription.UTF8String : "?");
                [files removeItemAtPath:directory error:nil];
                free(handle->staging_directory);
                free(handle);
                return NULL;
            }
            if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                    model, @selector(loadWithQoS:options:error:), 21, options,
                    &failure)) {
                bridge_fail(error, error_size, "ANE %s load failed: %s", name,
                            failure ?
                            failure.localizedDescription.UTF8String : "?");
                [files removeItemAtPath:directory error:nil];
                free(handle->staging_directory);
                free(handle);
                return NULL;
            }
            if (cache) {
                bridge_cache_prune();
                bridge_cache_store(identifier, directory);
            }
        }
        /* The staging weights blob and MIL text are COMPILE inputs only —
         * the warm-restore path loads AND SERVES from a mirror that never
         * had them (bridge_cache_store skips both) — so they are deleted
         * once the net is loaded, halving the cold build's staging burst
         * (the ~2x-compiled-bytes burst that ran the disk out mid-build,
         * bare "compile failed" from layer ~50 on). The COMPILED artifacts
         * must stay for the handle's life: aned demand-reads the program
         * from staging during serving, so deleting the whole directory
         * passes a single immediate eval and then fails later evals with
         * "ANEProgramProcessRequestDirect ... Program Inference error"
         * (measured, 352 failures over one 32k benchmark serve). */
        [files removeItemAtPath:
            [directory stringByAppendingPathComponent:@"weights"]
                          error:nil];
        [files removeItemAtPath:
            [directory stringByAppendingPathComponent:@"model.mil"]
                          error:nil];
        handle->cache_hit = loaded;
        handle->compile_seconds = bridge_seconds() - started;
        NSMutableArray *inputs = [NSMutableArray array];
        for (uint32_t index = 0; index < input_count; index++)
            [inputs addObject:((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:),
                input_surfaces[index])];
        id wrapped = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceClass, @selector(objectWithIOSurface:), output);
        /* One request per procedure: a bank's procedures share the surfaces
         * but each carries its own procedureIndex. */
        if (procedure_count > 1 && !bridge_has_symbol_api(model)) {
            bridge_fail(error, error_size, "ANE %s: this framework build "
                        "exposes no per-procedure symbol indices, so a bank "
                        "of %u cannot be dispatched", name, procedure_count);
            handle->model = (__bridge_retained void *)model;
            msv_ane_model_free(handle);
            return NULL;
        }
        NSMutableArray *requests = [NSMutableArray array];
        for (uint32_t proc = 0; proc < procedure_count; proc++) {
            NSArray *in_indices =
                bridge_symbol_indices(model, true, proc, input_count);
            NSArray *out_indices =
                bridge_symbol_indices(model, false, proc, 1);
            id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))
                          objc_msgSend)(
                requestClass,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                          weightsBuffer:perfStats:procedureIndex:),
                inputs, in_indices, @[wrapped], out_indices, nil, nil,
                @(proc));
            if (!request) {
                bridge_fail(error, error_size,
                            "ANE %s request rejected for procedure %u", name,
                            proc);
                handle->model = (__bridge_retained void *)model;
                msv_ane_model_free(handle);
                return NULL;
            }
            [requests addObject:request];
        }
        handle->model = (__bridge_retained void *)model;
        handle->requests = (__bridge_retained void *)requests;
        handle->options = (__bridge_retained void *)options;
    }
    return handle;
}

int msv_ane_model_eval(msv_ane_model *handle, uint32_t procedure, char *error,
                       size_t error_size) {
    if (!handle || !handle->model || !handle->requests) {
        bridge_fail(error, error_size, "the ANE model is not loaded");
        return 0;
    }
    int ok = 0;
    @autoreleasepool {
        NSArray *requests = (__bridge NSArray *)handle->requests;
        if (procedure >= requests.count) {
            bridge_fail(error, error_size, "ANE procedure %u is outside the "
                        "bank's %lu procedures", procedure,
                        (unsigned long)requests.count);
            return 0;
        }
        NSError *failure = nil;
        ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            (__bridge id)handle->model,
            @selector(evaluateWithQoS:options:request:error:), 21,
            (__bridge id)handle->options, requests[procedure],
            &failure) ? 1 : 0;
        if (!ok)
            bridge_fail(error, error_size, "ANE evaluation failed: %s",
                        failure ?
                        failure.localizedDescription.UTF8String : "?");
    }
    return ok;
}

int msv_ane_model_unload(msv_ane_model *handle, char *error, size_t error_size) {
    if (!handle || !handle->model) {
        bridge_fail(error, error_size, "no ANE model to unload");
        return 0;
    }
    @autoreleasepool {
        NSError *failure = nil;
        if (!((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                (__bridge id)handle->model, @selector(unloadWithQoS:error:), 21,
                &failure)) {
            bridge_fail(error, error_size, "ANE unload failed: %s",
                        failure ?
                        failure.localizedDescription.UTF8String : "?");
            return 0;
        }
    }
    return 1;
}

int msv_ane_model_reload(msv_ane_model *handle, char *error, size_t error_size) {
    if (!handle || !handle->model || !handle->staging_directory) {
        bridge_fail(error, error_size, "no ANE model to reload");
        return 0;
    }
    @autoreleasepool {
        NSString *directory = @(handle->staging_directory);
        bridge_cache_restore(directory.lastPathComponent, directory);
        NSError *failure = nil;
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                (__bridge id)handle->model,
                @selector(loadWithQoS:options:error:), 21,
                handle->options ? (__bridge id)handle->options : @{},
                &failure)) {
            bridge_fail(error, error_size, "ANE reload failed: %s",
                        failure ?
                        failure.localizedDescription.UTF8String : "?");
            return 0;
        }
    }
    return 1;
}

void msv_ane_model_free(msv_ane_model *handle) {
    if (!handle) return;
    @autoreleasepool {
        if (handle->model) {
            id model = (__bridge_transfer id)handle->model;
            NSError *failure = nil;
            ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &failure);
        }
        if (handle->requests) {
            id requests = (__bridge_transfer id)handle->requests;
            (void)requests;
        }
        if (handle->options) {
            id options = (__bridge_transfer id)handle->options;
            (void)options;
        }
        if (handle->staging_directory) {
            [[NSFileManager defaultManager]
                removeItemAtPath:@(handle->staging_directory) error:nil];
            if (!msv_ane_cache_enabled())
                bridge_cache_evict(strrchr(handle->staging_directory, '/') ?
                    strrchr(handle->staging_directory, '/') + 1 :
                    handle->staging_directory);
            free(handle->staging_directory);
        }
    }
    free(handle);
}

double msv_ane_model_compile_seconds(const msv_ane_model *handle) {
    return handle ? handle->compile_seconds : 0.0;
}

bool msv_ane_model_cache_hit(const msv_ane_model *handle) {
    return handle ? handle->cache_hit : false;
}
