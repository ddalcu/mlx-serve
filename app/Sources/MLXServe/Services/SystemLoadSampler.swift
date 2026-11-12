import Foundation

/// One look at the machine, for the chat window's status strip.
struct SystemLoad: Equatable {
    /// GPU busy, %.
    var gpu: Int
    /// Hottest GPU sensor, °C; nil where it cannot be read.
    var gpuTemperature: Double?
    /// Neural Engine busy, %; nil where it cannot be read.
    var ane: Double?
    var memoryUsed: UInt64
    var memoryTotal: UInt64

    var memoryPercent: Double { memoryTotal > 0 ? Double(memoryUsed) / Double(memoryTotal) * 100 : 0 }

    /// Whole GiB used over total, as Activity Monitor counts them.
    var memoryLabel: String { "\(memoryUsed >> 30)/\(memoryTotal >> 30)G" }

    enum Level { case normal, busy, saturated }

    /// A bar's tint: green, yellow from 60%, red from 85%.
    static func level(_ percent: Double) -> Level { percent >= 85 ? .saturated : percent >= 60 ? .busy : .normal }

    /// How much of a bar a percentage fills.
    static func fraction(_ percent: Double) -> Double { min(max(percent / 100, 0), 1) }
}

/// Samples the machine every two seconds, off the main thread, while something is looking.
@MainActor
final class SystemLoadSampler: ObservableObject {
    @Published private(set) var load: SystemLoad?

    /// Every sampler's: the shared sensors are not thread-safe.
    private static let queue = DispatchQueue(label: "mlx-serve.system-load", qos: .utility)
    private var timer: DispatchSourceTimer?

    /// The temperature and Neural Engine reads are many kernel calls each, the rest a few: they run on every third
    /// tick, after two at the start that give the Neural Engine its first span.
    nonisolated static func readsSensors(tick: Int) -> Bool { tick < 2 || tick % 3 == 0 }

    /// Its gauges are laid out, and their window can be seen: it samples while both hold. Each has its own
    /// reporter, and a sidebar that closes and opens again moves only the first.
    var onScreen = false { didSet { gate() } }
    var windowVisible = false { didSet { gate() } }
    var isSampling: Bool { timer != nil }

    private func gate() { onScreen && windowVisible ? start() : stop() }

    private func start() {
        guard timer == nil else { return }
        let sensors = HardwareSensors.shared   // opens its private interfaces on first use, on `queue`
        let source = DispatchSource.makeTimerSource(queue: Self.queue)
        source.schedule(deadline: .now(), repeating: 2, leeway: .milliseconds(500))
        var tick = 0, temperature: Double?, ane: Double?
        source.setEventHandler { [weak self] in
            if Self.readsSensors(tick: tick) {
                temperature = sensors.gpuTemperature()
                ane = sensors.aneLoad()
            }
            tick += 1
            let load = SystemLoad(gpu: Int(SystemMetrics.gpuUtilization()), gpuTemperature: temperature,
                                  ane: ane, memoryUsed: SystemMetrics.memoryUsedBytes(),
                                  memoryTotal: ProcessInfo.processInfo.physicalMemory)
            Task { @MainActor in self?.load = load }
        }
        source.resume()
        timer = source
    }

    private func stop() {
        timer?.cancel()
        timer = nil
    }
}
