import SwiftUI

/// The memory meter, shared by the menu bar, the Model Browser's Recommended
/// pane, and the welcome screen. ONE bar over total physical RAM: the model's
/// GPU footprint, then everything else in use, then the reclaimable remainder.
struct MemoryMeter: View {
    /// Live MLX/GPU footprint; nil when no model is loaded.
    var gpuBytes: Int64?
    /// Richer GPU label ("3.2 GB (+1.1 GB cache)"); falls back to formatted bytes.
    var gpuLabel: String?
    /// Reclaimable memory available for a new load.
    var availableBytes: Int64
    /// Physical RAM — the bar's denominator.
    var totalBytes: Int64

    private var gpu: Int64 { max(0, gpuBytes ?? 0) }
    private var other: Int64 { max(0, totalBytes - availableBytes - gpu) }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            GeometryReader { geo in
                HStack(spacing: 0) {
                    Rectangle().fill(Color.accentColor).frame(width: width(gpu, geo.size.width))
                    Rectangle().fill(Color.secondary.opacity(0.45)).frame(width: width(other, geo.size.width))
                    Rectangle().fill(Color.green.opacity(0.35))
                }
            }
            .frame(height: 6)
            .clipShape(Capsule())

            HStack(spacing: 6) {
                if gpuBytes != nil {
                    key(.accentColor, "GPU \(gpuLabel ?? MemoryInfo.format(gpu))")
                }
                if availableBytes > 0 {
                    key(.green.opacity(0.6), "\(MemoryInfo.format(availableBytes)) free")
                }
                Spacer()
                Text("\(MemoryInfo.format(totalBytes)) total")
                    .foregroundStyle(.tertiary)
            }
            .font(.caption2)
        }
    }

    private func width(_ part: Int64, _ full: CGFloat) -> CGFloat {
        guard totalBytes > 0 else { return 0 }
        return full * min(1, max(0, CGFloat(part) / CGFloat(totalBytes)))
    }

    @ViewBuilder private func key(_ tint: Color, _ text: String) -> some View {
        HStack(spacing: 3) {
            Circle().fill(tint).frame(width: 5, height: 5)
            Text(text).foregroundStyle(.secondary)
        }
    }
}

extension MemoryMeter {
    /// Build from the live server memory when a model is loaded, else from the
    /// kernel directly. `server` is `ServerManager.memoryInfo`, nil when no
    /// server/model is up.
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
