import Foundation

/// A canvas the model can actually sample: both sides on its grid and inside
/// its range.
struct AspectCanvas: Equatable, Identifiable {
    let width: Int
    let height: Int

    var id: String { "\(width)x\(height)" }
    var area: Int { width * height }
}

/// Canvases that match a starting frame's shape.
///
/// The model's own resolution list is a curated set of sizes, not the limit of
/// what it samples: anything on the grid inside the range generates. So the
/// candidates are enumerated FROM THE GRID, and the starting frame's aspect
/// ratio is what picks among them. Without this, a 16:9 photo could only be
/// generated at one of the shipped 16:9 rows, and everything else stretched
/// the picture (the server resizes a first frame to the canvas without
/// letterboxing or cropping).
enum AspectCanvases {
    /// How far a canvas may be off the source's aspect. At 3% the distortion
    /// is not noticeable; past it the shape visibly changes, which is the one
    /// thing choosing by aspect is meant to avoid.
    static let maxDeviation = 0.03

    /// Up to `count` canvases matching the source's shape, largest first.
    ///
    /// Candidates are every legal width with the height rounded to the NEAREST
    /// grid point (not up, as a typed correction is: nearest is what minimises
    /// the deviation, and it is still on the grid). Off-shape candidates are
    /// dropped before the spread is chosen, or the smallest legal canvas would
    /// be offered no matter how distorted it is: for 16:9 on LTX that is
    /// 480 x 256, off by 5.5%, while 512 x 288 is off by 0.03%.
    static func options(sourceWidth: Int, sourceHeight: Int,
                        grid: ResolutionGrid, count: Int = 5) -> [AspectCanvas] {
        guard sourceWidth > 0, sourceHeight > 0, grid.alignment > 0, count > 0 else { return [] }
        let ratio = Double(sourceWidth) / Double(sourceHeight)

        var scored: [(canvas: AspectCanvas, deviation: Double)] = []
        var seen = Set<String>()
        var w = grid.alignment * Int((Double(grid.minDim) / Double(grid.alignment)).rounded(.up))
        while w <= grid.maxDim {
            defer { w += grid.alignment }
            let ideal = Double(w) / ratio
            let h = grid.alignment * Int((ideal / Double(grid.alignment)).rounded())
            guard h >= grid.minDim, h <= grid.maxDim else { continue }
            let canvas = AspectCanvas(width: w, height: h)
            guard seen.insert(canvas.id).inserted else { continue }
            let deviation = abs(Double(w) / Double(h) - ratio) / ratio
            scored.append((canvas, deviation))
        }
        guard !scored.isEmpty else { return [] }

        var faithful = scored.filter { $0.deviation <= maxDeviation }.map(\.canvas)
        if faithful.isEmpty {
            // Nothing is faithful (a very unusual aspect): offer the closest
            // there is rather than an empty menu.
            faithful = scored.sorted { $0.deviation < $1.deviation }.prefix(count).map(\.canvas)
        }
        return spread(faithful, count: count)
    }

    /// The source's own size on the grid, when it fits. This is the one option
    /// that does not rescale the picture at all.
    static func sourceSize(sourceWidth: Int, sourceHeight: Int, grid: ResolutionGrid) -> AspectCanvas? {
        guard sourceWidth > 0, sourceHeight > 0, grid.alignment > 0 else { return nil }
        let snap = { (v: Int) in grid.alignment * Int((Double(v) / Double(grid.alignment)).rounded()) }
        let w = snap(sourceWidth), h = snap(sourceHeight)
        guard w >= grid.minDim, w <= grid.maxDim, h >= grid.minDim, h <= grid.maxDim else { return nil }
        return AspectCanvas(width: w, height: h)
    }

    /// `count` canvases spread by AREA, largest first: the two ends, then the
    /// ones nearest to evenly spaced areas between them. Area is what drives
    /// both time and memory, so it is the axis the choice is made on.
    static func spread(_ canvases: [AspectCanvas], count: Int) -> [AspectCanvas] {
        let sorted = canvases.sorted { $0.area > $1.area }
        guard sorted.count > count else { return sorted }
        guard count >= 2, let largest = sorted.first, let smallest = sorted.last else { return sorted }

        var picked = [largest, smallest]
        let steps = count - 1
        for i in 1..<steps {
            let target = Double(largest.area) + (Double(smallest.area) - Double(largest.area))
                * Double(i) / Double(steps)
            if let nearest = sorted
                .filter({ c in !picked.contains(c) })
                .min(by: { abs(Double($0.area) - target) < abs(Double($1.area) - target) }) {
                picked.append(nearest)
            }
        }
        return picked.sorted { $0.area > $1.area }
    }

    /// The shape as people write it: small whole numbers, never a decimal.
    ///
    /// The exact reduction is useless here — 1090 x 1070 reduces to 109:107,
    /// and nobody would read it as the square it is. So this looks for the
    /// closest fraction with a denominator up to 16 and prefers the simplest
    /// one when two are equally close. Deviations are measured on the real
    /// numbers elsewhere; this is only what the row says.
    static func ratioLabel(width: Int, height: Int) -> String {
        guard width > 0, height > 0 else { return "" }
        let ratio = Double(width) / Double(height)
        var best = (w: 1, h: 1, error: Double.infinity)
        for h in 1...16 {
            let w = max(1, Int((ratio * Double(h)).rounded()))
            let error = abs(Double(w) / Double(h) - ratio) / ratio
            // Strictly better, so the smallest denominator wins a tie and
            // 16:16 never stands in for 1:1.
            if error < best.error - 1e-9 { best = (w, h, error) }
        }
        let g = gcd(best.w, best.h)
        return "\(best.w / g):\(best.h / g)"
    }

    private static func gcd(_ a: Int, _ b: Int) -> Int {
        var x = abs(a), y = abs(b)
        while y != 0 { (x, y) = (y, x % y) }
        return max(x, 1)
    }
}
