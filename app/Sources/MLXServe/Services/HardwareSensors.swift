import Foundation

#if !MAS_BUILD
import IOKit

/// The sensors macOS answers only through private interfaces, read the way iTherm reads them (the author's own
/// menu-bar thermal monitor, same MIT licence). The Mac App Store build compiles this out and reads neither.
final class HardwareSensors {
    // Opened by the first read, which runs on the sampler's queue: the SMC key scan is thousands of calls.
    private lazy var smc = SMCGPUTemperature()
    private lazy var ane = ANELoad()

    /// Hottest GPU sensor, °C.
    func gpuTemperature() -> Double? { smc?.read() }

    /// Share of the time since the last call that the Neural Engine was above idle, %. nil until there is a span to measure.
    func aneLoad() -> Double? { ane?.read() }
}

/// The Apple SMC's `Tg*` sensors. Reading is unprivileged.
private final class SMCGPUTemperature {
    private var conn: io_connect_t = 0
    private var keys: [UInt32] = []

    init?() {
        let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSMC"))
        defer { IOObjectRelease(service) }
        guard service != 0, IOServiceOpen(service, mach_task_self_, 0, &conn) == KERN_SUCCESS else { return nil }
        for i in 0..<(call(Self.fourCC("#KEY"), 5, size: 4).map { UInt32(bigEndian: u32($0, 48)) } ?? 0) {
            guard let k = call(0, 8, index: i).map({ u32($0, 0) }), k >> 24 == 84 /* T */, k >> 16 & 255 == 103 /* g */,
                  info(k)?.1 == Self.fourCC("flt ") else { continue }
            keys.append(k)
        }
        if keys.isEmpty {
            IOServiceClose(conn)
            return nil
        }
    }

    deinit { IOServiceClose(conn) }

    func read() -> Double? {
        keys.compactMap { float($0) }.filter { $0 > 10 && $0 < 150 }.max().map(Double.init)
    }

    private static func fourCC(_ s: String) -> UInt32 { s.utf8.reduce(0) { $0 << 8 | UInt32($1) } }

    /// One AppleSMC user-client call. Its 80-byte struct is packed by offset: key @0, info.size @28,
    /// info.type @32, result @40, command @42, index @44, data @48.
    private func call(_ key: UInt32, _ cmd: UInt8, size: UInt32 = 0, index: UInt32 = 0) -> [UInt8]? {
        var input = [UInt8](repeating: 0, count: 80), output = input, n = 80
        for (v, at) in [(key, 0), (size, 28), (index, 44)] { withUnsafeBytes(of: v) { input.replaceSubrange(at..<at + 4, with: $0) } }
        input[42] = cmd
        guard IOConnectCallStructMethod(conn, 2, input, 80, &output, &n) == KERN_SUCCESS, output[40] == 0 else { return nil }
        return output
    }

    private func u32(_ b: [UInt8], _ at: Int) -> UInt32 { b.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: at, as: UInt32.self) } }
    private func info(_ key: UInt32) -> (UInt32, UInt32)? { call(key, 9).map { (u32($0, 28), u32($0, 32)) } }
    private func float(_ key: UInt32) -> Float? { call(key, 5, size: 4).map { Float(bitPattern: u32($0, 48)) } }
}

/// ANE load from IOReport, the private library behind powermetrics: the share of time the ANE's performance-floor
/// requests sat above their lowest state (its Energy Model counters stopped counting in macOS 27).
private final class ANELoad {
    private typealias Sample = @convention(c) (OpaquePointer, CFMutableDictionary, CFTypeRef?) -> Unmanaged<CFDictionary>?
    private typealias Delta = @convention(c) (CFDictionary, CFDictionary, CFTypeRef?) -> Unmanaged<CFDictionary>?
    private typealias Name = @convention(c) (CFDictionary) -> Unmanaged<CFString>?
    private typealias Count = @convention(c) (CFDictionary) -> Int32
    private typealias Residency = @convention(c) (CFDictionary, Int32) -> Int64
    private let sub: OpaquePointer, channels: CFMutableDictionary
    private let sample: Sample, delta: Delta, count: Count, residency: Residency
    private var last: CFDictionary?

    init?() {
        typealias Copy = @convention(c) (CFString, CFString?, UInt64, UInt64, UInt64) -> Unmanaged<CFDictionary>?
        typealias Subscribe = @convention(c) (UnsafeRawPointer?, CFMutableDictionary, UnsafeMutablePointer<Unmanaged<CFMutableDictionary>?>,
                                              UInt64, CFTypeRef?) -> OpaquePointer?
        guard let h = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW) else { return nil }
        func fn<T>(_ s: String, _: T.Type) -> T? { dlsym(h, s).map { unsafeBitCast($0, to: T.self) } }
        guard let copy = fn("IOReportCopyChannelsInGroup", Copy.self), let subscribe = fn("IOReportCreateSubscription", Subscribe.self),
              let sample = fn("IOReportCreateSamples", Sample.self), let delta = fn("IOReportCreateSamplesDelta", Delta.self),
              let name = fn("IOReportChannelGetChannelName", Name.self), let group = fn("IOReportChannelGetSubGroup", Name.self),
              let count = fn("IOReportStateGetCount", Count.self), let residency = fn("IOReportStateGetResidency", Residency.self)
        else { return nil }
        // "ANE0-AF-BW" in "SOC Floor", "ANE1-DCS-BW" in "DCS Floor", ...; Max and Ultra chips have a PMP group per die
        let wanted = ["PMP", "PMP0", "PMP1", "PMP2", "PMP3"].flatMap { g in
            (copy(g as CFString, nil, 0, 0, 0)?.takeRetainedValue() as NSDictionary?)?["IOReportChannels"] as? [NSDictionary] ?? []
        }.filter {
            (name($0)?.takeUnretainedValue() as String? ?? "").hasPrefix("ANE") && (group($0)?.takeUnretainedValue() as String? ?? "").contains("Floor")
        }
        var subscribed: Unmanaged<CFMutableDictionary>?
        guard !wanted.isEmpty, let sub = subscribe(nil, NSMutableDictionary(dictionary: ["IOReportChannels": wanted]), &subscribed, 0, nil),
              let subscribed else { return nil }
        self.sub = sub
        channels = subscribed.takeRetainedValue()
        self.sample = sample; self.delta = delta; self.count = count; self.residency = residency
    }

    func read() -> Double? {
        guard let now = sample(sub, channels, nil)?.takeRetainedValue() else { return nil }
        defer { last = now }
        guard let last, let d = delta(last, now, nil)?.takeRetainedValue() as NSDictionary? else { return nil }
        var busiest = 0.0
        for c in d["IOReportChannels"] as? [NSDictionary] ?? [] {
            let r = (0..<count(c)).map { residency(c, $0) }, total = r.reduce(0, +)
            guard total > 0 else { continue }
            busiest = max(busiest, Double(total - r[0]) / Double(total) * 100)
        }
        return busiest
    }
}

#else

final class HardwareSensors {
    func gpuTemperature() -> Double? { nil }
    func aneLoad() -> Double? { nil }
}

#endif
