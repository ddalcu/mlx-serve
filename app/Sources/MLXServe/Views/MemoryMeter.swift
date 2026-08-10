import SwiftUI

/// The GPU-memory + available-RAM meter, shared by the menu bar, the Model
/// Browser's Recommended pane, and the welcome screen. Two labeled bars over a
/// shared total-RAM denominator (so they're directly comparable) plus a total
/// caption. The GPU bar hides when nothing is loaded (no live footprint).
struct MemoryMeter: View {
    /// Live MLX/GPU footprint; nil when no model is loaded (bar hidden).
    var gpuBytes: Int64?
    /// Richer GPU label ("3.2 GB (+1.1 GB cache)"); falls back to formatted bytes.
    var gpuLabel: String?
    /// Reclaimable memory available for a new load.
    var availableBytes: Int64
    /// Physical RAM — the shared denominator.
    var totalBytes: Int64

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let gpu = gpuBytes, totalBytes > 0 {
                bar("GPU Memory", gpuLabel ?? MemoryInfo.format(gpu), fraction(gpu), .blue)
            }
            if availableBytes > 0, totalBytes > 0 {
                bar("Available RAM", MemoryInfo.format(availableBytes), fraction(availableBytes), .green)
            }
            if totalBytes > 0 {
                Text("\(MemoryInfo.format(totalBytes)) total")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private func fraction(_ part: Int64) -> Double {
        guard totalBytes > 0 else { return 0 }
        return min(1, max(0, Double(part) / Double(totalBytes)))
    }

    @ViewBuilder private func bar(_ label: String, _ value: String, _ fill: Double, _ tint: Color) -> some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Spacer()
            Text(value).font(.caption.monospaced())
        }
        ProgressView(value: fill).tint(tint)
    }
}

extension MemoryMeter {
    /// Build from the live server memory when a model is loaded, else from the
    /// kernel directly (GPU bar hidden). `server` is `ServerManager.memoryInfo`,
    /// which is nil when no server/model is up. Keeps the meter identical across
    /// surfaces and honest with or without a running server.
    static func live(server: MemoryInfo?) -> MemoryMeter {
        let total = Int64(ProcessInfo.processInfo.physicalMemory)
        if let m = server {
            return MemoryMeter(gpuBytes: m.activeBytes, gpuLabel: m.gpuMemoryLabel,
                               availableBytes: m.availableBytes, totalBytes: total)
        }
        return MemoryMeter(gpuBytes: nil, gpuLabel: nil,
                           availableBytes: Int64(bitPattern: SystemMetrics.availableForModelBytes()),
                           totalBytes: total)
    }
}
