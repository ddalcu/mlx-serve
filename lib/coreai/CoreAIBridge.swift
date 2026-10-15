// libcoreai_bridge.dylib: a C ABI over the native Core AI Swift runtime (macOS 27), for hosts that cannot use the Swift
// API directly. From ANEMLL Forge (MIT, see NOTICE); mlx-serve dlopens it (coreai_bridge.h, src/coreai_ffi.zig).
// Built by build.zig when the SDK ships CoreAI.framework.
//
// Objects are opaque retained handles (release with cai_release):
//   model     cai_model_load(path, compute, check_version)           .aimodel (specialized + cached) or .aimodelc
//   function  cai_function_load(model, name); cai_function_describe -> JSON (inputs / outputs / states:
//             name, dtype, shape, strides (elements, the model's preferred layout), nbytes)
//   buffer    cai_buffer_create(dtype, rank, shape, strides|NULL)    IOSurface-backed, zero-filled; cai_buffer_address /
//             cai_buffer_nbytes give the CPU view (valid for the buffer's lifetime, no copy-on-write: one buffer can be an
//             output of one call and an input of the next)
//   binding   cai_binding_create(function, inputs, outputs, states)  names -> buffers, validated against the descriptor
//   run       cai_run(n, bindings, times_ms|NULL)                    runs the bindings in order inside one
//             Task.detached(priority: .userInitiated) and blocks the caller until all are done; outputs are written
//             into the bound buffers through outputViews (outputs left unbound are allocated by the runtime and dropped)
// Every call that can fail takes (char *err, int32 errcap) and returns NULL / nonzero with a message.
//
// Safety: shape / dtype / name mismatches and too-new compiled packages are reported as errors here, because the
// runtime treats several of them as fatal errors (process abort / SIGSEGV), which would take down the host.

import CoreAI
import Darwin
import Foundation
import IOSurface

// MARK: - errors, strings, blocking ------------------------------------------------------------------------------

struct BridgeError: Error, CustomStringConvertible {
    let description: String
    init(_ s: String) { description = s }
}

func writeCString(_ s: String, _ buf: UnsafeMutablePointer<CChar>?, _ cap: Int32) -> Int32 {
    let bytes = Array(s.utf8)
    guard let buf, cap > 0 else { return Int32(bytes.count + 1) }
    let n = min(bytes.count, Int(cap) - 1)
    buf.withMemoryRebound(to: UInt8.self, capacity: Int(cap)) { dst in
        for i in 0..<n { dst[i] = bytes[i] }
        dst[n] = 0
    }
    return Int32(bytes.count + 1)
}

func fail(_ err: UnsafeMutablePointer<CChar>?, _ cap: Int32, _ e: Error) {
    _ = writeCString((e as? BridgeError)?.description ?? "\(e)", err, cap)
}

final class ResultBox<T>: @unchecked Sendable {
    var value: T?
    var error: Error?
}

/// Runs `body` in a detached user-initiated task and blocks the calling (non-Swift-concurrency) thread until it ends.
func blocking<T>(_ body: @escaping () async throws -> T) throws -> T {
    let sem = DispatchSemaphore(value: 0)
    let box = ResultBox<T>()
    Task.detached(priority: .userInitiated) {
        do { box.value = try await body() } catch { box.error = error }
        sem.signal()
    }
    sem.wait()
    if let e = box.error { throw e }
    return box.value!
}

func retained(_ o: AnyObject) -> UnsafeMutableRawPointer { Unmanaged.passRetained(o).toOpaque() }
func object<T: AnyObject>(_ p: UnsafeMutableRawPointer?, _ t: T.Type) throws -> T {
    guard let p else { throw BridgeError("null handle") }
    guard let o = Unmanaged<AnyObject>.fromOpaque(p).takeUnretainedValue() as? T else {
        throw BridgeError("handle is not a \(T.self)")
    }
    return o
}

// MARK: - dtypes -----------------------------------------------------------------------------------------------

let dtypeTable: [(code: Int32, name: String, type: NDArray.ScalarType, size: Int)] = [
    (0, "float16", .float16, 2), (1, "float32", .float32, 4), (2, "int32", .int32, 4), (3, "int8", .int8, 1),
    (4, "uint8", .uint8, 1), (5, "bfloat16", .bfloat16, 2), (6, "int16", .int16, 2), (7, "uint16", .uint16, 2),
    (8, "int64", .int64, 8), (9, "float64", .float64, 8), (10, "bool", .bool, 1),
]
func dtypeName(_ t: NDArray.ScalarType) -> String { dtypeTable.first { $0.type == t }?.name ?? "\(t)" }
func dtypeSize(_ t: NDArray.ScalarType) -> Int { dtypeTable.first { $0.type == t }?.size ?? 1 }

// MARK: - MPSGraph package version guard ---------------------------------------------------------------------

func versionLE(_ a: String, _ b: String) -> Bool {
    let x = a.split(separator: ".").map { Int($0) ?? 0 }, y = b.split(separator: ".").map { Int($0) ?? 0 }
    for i in 0..<max(x.count, y.count) {
        let u = i < x.count ? x[i] : 0, v = i < y.count ? y[i] : 0
        if u != v { return u < v }
    }
    return true
}

/// The MPSGraph package version this OS reads (the framework's short version, e.g. 7.0.80 on macOS 27.0 26A428).
func osMPSGraphVersion() -> String? {
    if let e = ProcessInfo.processInfo.environment["COREAI_MPSGRAPH_MAX"] { return e }
    let p = "/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/Versions/A/Resources/version.plist"
    return (NSDictionary(contentsOfFile: p)?["CFBundleShortVersionString"] as? String)
}

/// Highest MPSGraph package version inside a compiled package (nil: none, e.g. an .aimodel source).
func packageMPSGraphVersion(_ url: URL) -> String? {
    guard let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil) else { return nil }
    var best: String? = nil
    for case let u as URL in e where u.lastPathComponent == "manifest.plist" && u.path.contains(".mpsgraphpackage") {
        guard let d = NSDictionary(contentsOf: u), let pv = d["Package Version"] as? [String: Any] else { continue }
        for k in pv.keys where best == nil || !versionLE(k, best!) { best = k }
    }
    return best
}

// MARK: - handles ----------------------------------------------------------------------------------------------

final class ModelH {
    let model: AIModel
    let path: String
    init(model: AIModel, path: String) { self.model = model; self.path = path }
}

struct ValueSpec {
    let name: String
    let type: NDArray.ScalarType
    let shape: [Int]
    let strides: [Int]
    let nbytes: Int
    let dynamic: Bool
    var json: [String: Any] {
        ["name": name, "dtype": dtypeName(type), "shape": shape, "strides": strides, "nbytes": nbytes, "dynamic": dynamic]
    }
}

func spec(_ name: String, _ v: InferenceValue.Descriptor?) -> ValueSpec? {
    guard let v else { return nil }
    switch v {
    case .ndArray(let d):
        return ValueSpec(name: name, type: d.scalarType, shape: d.shape, strides: d.preferredStrides,
                         nbytes: d.minimumByteCount, dynamic: d.hasDynamicShape)
    case .image:
        return nil
    @unknown default:
        return nil
    }
}

final class FunctionH {
    let fn: InferenceFunction
    let model: ModelH
    let name: String
    let inputs: [String: ValueSpec], outputs: [String: ValueSpec], states: [String: ValueSpec]
    let inputOrder: [String], outputOrder: [String], stateOrder: [String]
    init(fn: InferenceFunction, model: ModelH, name: String) {
        self.fn = fn; self.model = model; self.name = name
        let d = fn.descriptor
        inputOrder = d.inputNames; outputOrder = d.outputNames; stateOrder = d.stateNames
        var i: [String: ValueSpec] = [:], o: [String: ValueSpec] = [:], s: [String: ValueSpec] = [:]
        for n in d.inputNames { i[n] = spec(n, d.inputDescriptor(of: n)) }
        for n in d.outputNames { o[n] = spec(n, d.outputDescriptor(of: n)) }
        for n in d.stateNames { s[n] = spec(n, d.stateDescriptor(of: n)) }
        inputs = i; outputs = o; states = s
    }
    func describe() -> String {
        let obj: [String: Any] = [
            "name": name,
            "inputs": inputOrder.map { inputs[$0]?.json ?? ["name": $0, "dtype": "unsupported"] },
            "outputs": outputOrder.map { outputs[$0]?.json ?? ["name": $0, "dtype": "unsupported"] },
            "states": stateOrder.map { states[$0]?.json ?? ["name": $0, "dtype": "unsupported"] },
        ]
        let data = (try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])) ?? Data("{}".utf8)
        return String(decoding: data, as: UTF8.self)
    }
}

/// An IOSurface laid out like the tensor: width = last dimension, rows = the leading dimensions with the row stride
/// strides[rank-2] (Core AI's IOSurface views require width == last dimension).
final class BufferH {
    let surface: IOSurface
    let type: NDArray.ScalarType
    let shape: [Int]
    let strides: [Int]
    init(type: NDArray.ScalarType, shape: [Int], strides given: [Int]?) throws {
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else { throw BridgeError("bad shape \(shape)") }
        let es = dtypeSize(type)
        var st = given ?? []
        if st.isEmpty {
            st = [Int](repeating: 1, count: shape.count)
            for j in stride(from: shape.count - 2, through: 0, by: -1) { st[j] = st[j + 1] * shape[j + 1] }
        }
        guard st.count == shape.count, st.allSatisfy({ $0 > 0 }) else { throw BridgeError("bad strides \(st) for \(shape)") }
        guard st.last == 1 else { throw BridgeError("last-dimension stride must be 1 (got \(st))") }
        let width = shape.last!
        let rowStride = shape.count >= 2 ? st[shape.count - 2] : width
        guard rowStride >= width else { throw BridgeError("row stride \(rowStride) < width \(width)") }
        let spanElems = zip(shape, st).map { ($0 - 1) * $1 }.reduce(0, +) + 1
        let rowBytes = rowStride * es
        let height = max(1, shape.dropLast().reduce(1, *), (spanElems * es + rowBytes - 1) / rowBytes)
        let alloc = max(rowBytes * height, spanElems * es, 16)
        let props: [IOSurfacePropertyKey: Any] = [.width: width, .height: height, .bytesPerElement: es,
                                                  .bytesPerRow: rowBytes, .allocSize: alloc, .pixelFormat: 0]
        guard let s = IOSurface(properties: props) else {
            throw BridgeError("IOSurface(width \(width), height \(height), bytesPerRow \(rowBytes)) failed")
        }
        guard s.allocationSize >= spanElems * es else {
            throw BridgeError("IOSurface too small: \(s.allocationSize) < \(spanElems * es) bytes")
        }
        s.lock(options: [], seed: nil)
        memset(s.baseAddress, 0, s.allocationSize)
        s.unlock(options: [], seed: nil)
        surface = s
        self.type = type
        self.shape = shape
        strides = st
    }
}

final class BindingH {
    let f: FunctionH
    let inNames: [String], ins: [BufferH]
    let outNames: [String], outs: [BufferH]
    let stNames: [String], sts: [BufferH]
    init(f: FunctionH, inNames: [String], ins: [BufferH], outNames: [String], outs: [BufferH], stNames: [String],
         sts: [BufferH]) throws {
        func check(_ kind: String, _ names: [String], _ bufs: [BufferH], _ specs: [String: ValueSpec]) throws {
            guard Set(names).count == names.count else { throw BridgeError("\(f.name): duplicate \(kind) names") }
            for (n, b) in zip(names, bufs) {
                guard let s = specs[n] else {
                    throw BridgeError("\(f.name): no \(kind) named '\(n)' (have \(specs.keys.sorted()))")
                }
                guard s.type == b.type else {
                    throw BridgeError("\(f.name): \(kind) '\(n)' is \(dtypeName(s.type)), buffer is \(dtypeName(b.type))")
                }
                if !s.dynamic, s.shape != b.shape {
                    throw BridgeError("\(f.name): \(kind) '\(n)' has shape \(s.shape), buffer has \(b.shape)")
                }
            }
        }
        try check("input", inNames, ins, f.inputs)
        try check("output", outNames, outs, f.outputs)
        try check("state", stNames, sts, f.states)
        let missing = f.inputOrder.filter { !inNames.contains($0) }
        guard missing.isEmpty else { throw BridgeError("\(f.name): inputs not bound: \(missing)") }
        let missingSt = f.stateOrder.filter { !stNames.contains($0) }
        guard missingSt.isEmpty else { throw BridgeError("\(f.name): states not bound: \(missingSt)") }
        let written = Set((outs + sts).map { ObjectIdentifier($0) })
        if let clash = zip(inNames, ins).first(where: { written.contains(ObjectIdentifier($0.1)) }) {
            throw BridgeError("\(f.name): input '\(clash.0)' aliases an output / state buffer of the same call")
        }
        self.f = f
        self.inNames = inNames; self.ins = ins
        self.outNames = outNames; self.outs = outs
        self.stNames = stNames; self.sts = sts
    }

    func run() async throws {
        var inputs = InferenceFunction.Inputs()
        for i in 0..<ins.count {
            let b = ins[i], s = b.surface
            inputs.insert(_overrideLifetime(NDArray.RawView(ioSurface: s, scalarType: b.type, shape: b.shape,
                                                            strides: b.strides), borrowing: self), for: inNames[i])
        }
        var states = InferenceFunction.MutableViews()
        for i in 0..<sts.count {
            let b = sts[i], s = b.surface
            states.insert(_overrideLifetime(NDArray.MutableRawView(ioSurface: s, scalarType: b.type, shape: b.shape,
                                                                   strides: b.strides), borrowing: self), for: stNames[i])
        }
        var views = InferenceFunction.MutableViews()
        for i in 0..<outs.count {
            let b = outs[i], s = b.surface
            views.insert(_overrideLifetime(NDArray.MutableRawView(ioSurface: s, scalarType: b.type, shape: b.shape,
                                                                  strides: b.strides), borrowing: self), for: outNames[i])
        }
        _ = try await f.fn.run(inputs: inputs, states: states, outputViews: views)
    }
}

func names(_ n: Int32, _ p: UnsafePointer<UnsafePointer<CChar>?>?) throws -> [String] {
    guard n > 0 else { return [] }
    guard let p else { throw BridgeError("null name array") }
    return try (0..<Int(n)).map {
        guard let c = p[$0] else { throw BridgeError("null name") }
        return String(cString: c)
    }
}

func buffers(_ n: Int32, _ p: UnsafePointer<UnsafeMutableRawPointer?>?) throws -> [BufferH] {
    guard n > 0 else { return [] }
    guard let p else { throw BridgeError("null buffer array") }
    return try (0..<Int(n)).map { try object(p[$0], BufferH.self) }
}

// MARK: - C ABI ------------------------------------------------------------------------------------------------

@_cdecl("cai_abi_version")
public func cai_abi_version() -> Int32 { 1 }

@_cdecl("cai_release")
public func cai_release(_ h: UnsafeMutableRawPointer?) {
    guard let h else { return }
    Unmanaged<AnyObject>.fromOpaque(h).release()
}

/// OS MPSGraph package version (what compiled packages may use at most); returns the needed buffer size.
@_cdecl("cai_os_mpsgraph_version")
public func cai_os_mpsgraph_version(_ buf: UnsafeMutablePointer<CChar>?, _ cap: Int32) -> Int32 {
    writeCString(osMPSGraphVersion() ?? "", buf, cap)
}

/// Highest MPSGraph package version inside a package ("" for none).
@_cdecl("cai_package_mpsgraph_version")
public func cai_package_mpsgraph_version(_ path: UnsafePointer<CChar>, _ buf: UnsafeMutablePointer<CChar>?,
                                         _ cap: Int32) -> Int32 {
    writeCString(packageMPSGraphVersion(URL(fileURLWithPath: String(cString: path))) ?? "", buf, cap)
}

/// compute: 0 neural engine preferred, 1 GPU preferred, 2 CPU only, 3 default.
/// check_version != 0: refuse a compiled package whose MPSGraph package is newer than the OS reads (it would crash).
@_cdecl("cai_model_load")
public func cai_model_load(_ path: UnsafePointer<CChar>, _ compute: Int32, _ checkVersion: Int32,
                           _ err: UnsafeMutablePointer<CChar>?, _ errcap: Int32) -> UnsafeMutableRawPointer? {
    let p = String(cString: path)
    do {
        let url = URL(fileURLWithPath: p)
        guard FileManager.default.fileExists(atPath: p) else { throw BridgeError("no such package: \(p)") }
        if checkVersion != 0, let pv = packageMPSGraphVersion(url), let ov = osMPSGraphVersion(), !versionLE(pv, ov) {
            throw BridgeError("\(url.lastPathComponent): MPSGraph package \(pv) is newer than this OS reads (\(ov)); "
                              + "load the .aimodel source or recompile with a matching toolchain")
        }
        let opts: SpecializationOptions
        switch compute {
        case 0: opts = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        case 1: opts = SpecializationOptions(preferredComputeUnitKind: .gpu)
        case 2: opts = .cpuOnly
        default: opts = .default
        }
        let m = try blocking { try await AIModel(contentsOf: url, options: opts) }
        return retained(ModelH(model: m, path: p))
    } catch {
        fail(err, errcap, error)
        return nil
    }
}

/// Newline-separated function names; returns the needed buffer size.
@_cdecl("cai_model_function_names")
public func cai_model_function_names(_ model: UnsafeMutableRawPointer?, _ buf: UnsafeMutablePointer<CChar>?,
                                     _ cap: Int32) -> Int32 {
    guard let m = try? object(model, ModelH.self) else { return writeCString("", buf, cap) }
    return writeCString(m.model.functionNames.joined(separator: "\n"), buf, cap)
}

@_cdecl("cai_function_load")
public func cai_function_load(_ model: UnsafeMutableRawPointer?, _ name: UnsafePointer<CChar>,
                              _ err: UnsafeMutablePointer<CChar>?, _ errcap: Int32) -> UnsafeMutableRawPointer? {
    do {
        let m = try object(model, ModelH.self)
        let n = String(cString: name)
        guard let f = try m.model.loadFunction(named: n) else {
            throw BridgeError("no function '\(n)' in \(m.path) (have \(m.model.functionNames))")
        }
        return retained(FunctionH(fn: f, model: m, name: n))
    } catch {
        fail(err, errcap, error)
        return nil
    }
}

/// JSON description (inputs / outputs / states with dtype, shape, preferred strides, nbytes); returns needed size.
@_cdecl("cai_function_describe")
public func cai_function_describe(_ fn: UnsafeMutableRawPointer?, _ buf: UnsafeMutablePointer<CChar>?,
                                  _ cap: Int32) -> Int32 {
    guard let f = try? object(fn, FunctionH.self) else { return writeCString("{}", buf, cap) }
    return writeCString(f.describe(), buf, cap)
}

/// strides: element strides (NULL = packed). Zero-filled IOSurface storage.
@_cdecl("cai_buffer_create")
public func cai_buffer_create(_ dtype: Int32, _ rank: Int32, _ shape: UnsafePointer<Int64>?,
                              _ strides: UnsafePointer<Int64>?, _ err: UnsafeMutablePointer<CChar>?,
                              _ errcap: Int32) -> UnsafeMutableRawPointer? {
    return autoreleasepool {
        do {
            guard let t = dtypeTable.first(where: { $0.code == dtype }) else { throw BridgeError("unknown dtype code \(dtype)") }
            guard rank > 0, let shape else { throw BridgeError("rank must be > 0") }
            let sh = (0..<Int(rank)).map { Int(shape[$0]) }
            let st = strides.map { s in (0..<Int(rank)).map { Int(s[$0]) } }
            return retained(try BufferH(type: t.type, shape: sh, strides: st))
        } catch {
            fail(err, errcap, error)
            return nil
        }
    }
}

@_cdecl("cai_buffer_address")
public func cai_buffer_address(_ b: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    autoreleasepool { (try? object(b, BufferH.self))?.surface.baseAddress }
}

@_cdecl("cai_buffer_nbytes")
public func cai_buffer_nbytes(_ b: UnsafeMutableRawPointer?) -> Int64 {
    Int64((try? object(b, BufferH.self))?.surface.allocationSize ?? 0)
}

@_cdecl("cai_binding_create")
public func cai_binding_create(_ fn: UnsafeMutableRawPointer?,
                               _ nIn: Int32, _ inNames: UnsafePointer<UnsafePointer<CChar>?>?,
                               _ inBufs: UnsafePointer<UnsafeMutableRawPointer?>?,
                               _ nOut: Int32, _ outNames: UnsafePointer<UnsafePointer<CChar>?>?,
                               _ outBufs: UnsafePointer<UnsafeMutableRawPointer?>?,
                               _ nSt: Int32, _ stNames: UnsafePointer<UnsafePointer<CChar>?>?,
                               _ stBufs: UnsafePointer<UnsafeMutableRawPointer?>?,
                               _ err: UnsafeMutablePointer<CChar>?, _ errcap: Int32) -> UnsafeMutableRawPointer? {
    do {
        let f = try object(fn, FunctionH.self)
        let b = try BindingH(f: f, inNames: try names(nIn, inNames), ins: try buffers(nIn, inBufs),
                             outNames: try names(nOut, outNames), outs: try buffers(nOut, outBufs),
                             stNames: try names(nSt, stNames), sts: try buffers(nSt, stBufs))
        return retained(b)
    } catch {
        fail(err, errcap, error)
        return nil
    }
}

/// Runs n bindings in order (one detached user-initiated task), blocking until done. times_ms (optional, n doubles):
/// wall time of each binding's call. Returns 0, or the 1-based index of the failing binding with the error message.
@_cdecl("cai_run")
public func cai_run(_ n: Int32, _ bindings: UnsafePointer<UnsafeMutableRawPointer?>?,
                    _ timesMs: UnsafeMutablePointer<Double>?, _ err: UnsafeMutablePointer<CChar>?,
                    _ errcap: Int32) -> Int32 {
    let bs: [BindingH]
    do {
        guard n > 0, let bindings else { return 0 }
        bs = try (0..<Int(n)).map { try object(bindings[$0], BindingH.self) }
    } catch {
        fail(err, errcap, error)
        return -1
    }
    final class Times: @unchecked Sendable { var t: [Double]; var failed = 0; init(_ n: Int) { t = [Double](repeating: 0, count: n) } }
    let times = Times(bs.count)
    do {
        try blocking {
            for (i, b) in bs.enumerated() {
                let t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                times.failed = i + 1
                try await b.run()
                times.t[i] = Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0) / 1e6
            }
            times.failed = 0
        }
    } catch {
        fail(err, errcap, BridgeError("binding \(times.failed) (\(bs[max(0, times.failed - 1)].f.name)): \(error)"))
        return Int32(max(1, times.failed))
    }
    if let timesMs { for i in 0..<bs.count { timesMs[i] = times.t[i] } }
    return 0
}
