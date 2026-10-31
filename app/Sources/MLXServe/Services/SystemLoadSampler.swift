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

    private let queue = DispatchQueue(label: "mlx-serve.system-load", qos: .utility)
    private var timer: DispatchSourceTimer?

    func start() {
        guard timer == nil else { return }
        let sensors = HardwareSensors()   // opens its private interfaces on first use, on `queue`
        let source = DispatchSource.makeTimerSource(queue: queue)
        source.schedule(deadline: .now(), repeating: 2, leeway: .milliseconds(500))
        source.setEventHandler { [weak self] in
            let load = SystemLoad(gpu: Int(SystemMetrics.gpuUtilization()), gpuTemperature: sensors.gpuTemperature(),
                                  ane: sensors.aneLoad(), memoryUsed: SystemMetrics.memoryUsedBytes(),
                                  memoryTotal: ProcessInfo.processInfo.physicalMemory)
            Task { @MainActor in self?.load = load }
        }
        source.resume()
        timer = source
    }

    func stop() {
        timer?.cancel()
        timer = nil
    }
}
