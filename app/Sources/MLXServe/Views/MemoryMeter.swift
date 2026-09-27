import SwiftUI

/// The memory meter, shared by the menu bar, the Model Browser's Recommended
/// pane, and the welcome screen. ONE bar over total physical RAM: the model's
/// GPU footprint, then everything else in use, then the reclaimable remainder,
/// split at the GPU working-set cap into what a model can use and RAM it can't.
struct MemoryMeter: View {
    /// Live MLX/GPU footprint; nil when no model is loaded.
    var gpuBytes: Int64?
    /// Richer GPU label ("3.2 GB (+1.1 GB cache)"); falls back to formatted bytes.
    var gpuLabel: String?
    /// Reclaimable memory available for a new load.
    var availableBytes: Int64
    /// Physical RAM — the bar's denominator.
    var totalBytes: Int64
    /// The GPU working-set cap (`iogpu.wired_limit_mb`); nil = unknown.
    var gpuLimitBytes: Int64? = nil
    /// The GPU footprint as model / KV cache / working; nil draws it as one segment.
    var breakdown: MemoryInfo.GpuBreakdown? = nil

    struct Split {
        let gpu: Int64, other: Int64, gpuFree: Int64, ramOnlyFree: Int64

        init(gpu: Int64, available: Int64, total: Int64, gpuLimit: Int64?) {
            let gpu = max(0, gpu)
            self.gpu = gpu
            other = max(0, total - available - gpu)
            let room = gpuLimit.map { max(0, $0 - gpu) } ?? available
            gpuFree = max(0, min(available, room))
            ramOnlyFree = max(0, available - gpuFree)
        }
    }

    private var split: Split {
        Split(gpu: gpuBytes ?? 0, available: availableBytes, total: totalBytes, gpuLimit: gpuLimitBytes)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            let s = split
            GeometryReader { geo in
                HStack(spacing: 0) {
                    if let b = breakdown {
                        Rectangle().fill(Color.accentColor).frame(width: width(b.model, geo.size.width))
                        Rectangle().fill(Color.accentColor.opacity(0.6)).frame(width: width(b.kvCache, geo.size.width))
                        Rectangle().fill(Color.accentColor.opacity(0.3)).frame(width: width(b.working, geo.size.width))
                    } else {
                        Rectangle().fill(Color.accentColor).frame(width: width(s.gpu, geo.size.width))
                    }
                    Rectangle().fill(Color.secondary.opacity(0.45)).frame(width: width(s.other, geo.size.width))
                    Rectangle().fill(Color.green.opacity(0.35)).frame(width: width(s.gpuFree, geo.size.width))
                    Rectangle().fill(Color.orange.opacity(0.35))
                }
            }
            .frame(height: 6)
            .clipShape(Capsule())

            // Wraps instead of truncating: the tray is too narrow for one line.
            FlowLayout(spacing: 10, rowSpacing: 4) {
                if gpuBytes != nil {
                    key(.accentColor, "GPU \(gpuLabel ?? MemoryInfo.format(s.gpu))")
                }
                if s.ramOnlyFree > 0 {
                    key(.green.opacity(0.6), L10n.format("%@ free for GPU", MemoryInfo.format(s.gpuFree)))
                    key(.orange.opacity(0.6), L10n.format("%@ past GPU limit", MemoryInfo.format(s.ramOnlyFree)))
                } else if availableBytes > 0 {
                    key(.green.opacity(0.6), L10n.format("%@ free", MemoryInfo.format(availableBytes)))
                }
                if let b = breakdown {
                    key(.accentColor, L10n.format("Model %@", MemoryInfo.format(b.model)))
                    key(.accentColor.opacity(0.6), L10n.format("KV cache %@", MemoryInfo.format(b.kvCache)))
                    key(.accentColor.opacity(0.3), L10n.format("Working %@", MemoryInfo.format(b.working)))
                }
                Text("\(MemoryInfo.format(totalBytes)) total")
                    .foregroundStyle(.tertiary)
            }
            .font(.app(.caption2))
        }
        .help(gpuLimitBytes.map {
            L10n.format("GPU memory limit: %@ (iogpu.wired_limit_mb). Raise it with: sudo sysctl iogpu.wired_limit_mb=<MB>",
                        MemoryInfo.format($0))
        } ?? "")
    }

    private func width(_ part: Int64, _ full: CGFloat) -> CGFloat {
        guard totalBytes > 0 else { return 0 }
        return full * min(1, max(0, CGFloat(part) / CGFloat(totalBytes)))
    }

    @ViewBuilder private func key(_ tint: Color, _ text: String) -> some View {
        HStack(spacing: 3) {
            Circle().fill(tint).frame(width: 5, height: 5)
            Text(text).foregroundStyle(.secondary).font(.app(.body))
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
                               availableBytes: m.availableBytes, totalBytes: total,
                               gpuLimitBytes: m.gpuLimitBytes, breakdown: m.gpuBreakdown)
        }
        return MemoryMeter(gpuBytes: nil, gpuLabel: nil,
                           availableBytes: Int64(bitPattern: SystemMetrics.availableForModelBytes()),
                           totalBytes: total, gpuLimitBytes: SystemMetrics.gpuMemoryLimitBytes())
    }
}
