import Foundation
import ImageIO

/// A canvas the model can actually sample: both sides on its grid and inside
/// its range.
struct AspectCanvas: Equatable, Identifiable {
    let width: Int
    let height: Int

    var id: String { "\(width)x\(height)" }
    var area: Int { width * height }
}

/// One row of a "Set by source" menu: a canvas and what to call it.
struct SourceCanvasChoice: Identifiable, Equatable {
    let canvas: AspectCanvas
    let name: String?
    var isSourceSize: Bool = false
    var id: String { canvas.id }
}

/// Canvases that match a starting frame's shape.
///
/// The model's own resolution list is a curated set of sizes, not the limit of
/// what it samples: anything on the grid inside the range generates. So the
/// candidates are enumerated FROM THE GRID, and the starting frame's aspect
/// ratio is what picks among them. Without this, a 16:9 photo could only be
/// generated at one of the shipped 16:9 rows, and everything else lost the
/// picture's edges (the server center-crops a first frame to the canvas).
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
        guard count > 0 else { return [] }
        return spread(faithful(sourceWidth: sourceWidth, sourceHeight: sourceHeight, grid: grid, fallback: count),
                      count: count)
    }

    /// The clip-size slider's steps under a first frame: every canvas faithful to
    /// its shape, smallest first, none larger than the picture (that only upscales it).
    static func ladder(sourceWidth: Int, sourceHeight: Int, grid: ResolutionGrid) -> [AspectCanvas] {
        let all = faithful(sourceWidth: sourceWidth, sourceHeight: sourceHeight, grid: grid, fallback: 5)
            .sorted { $0.area < $1.area }
        guard let smallest = all.first else { return [] }
        var steps = all.filter { $0.area <= sourceWidth * sourceHeight }
        if let own = sourceSize(sourceWidth: sourceWidth, sourceHeight: sourceHeight, grid: grid), !steps.contains(own) {
            steps.append(own)
            steps.sort { $0.area < $1.area }
        }
        return steps.isEmpty ? [smallest] : steps
    }

    /// How far `canvas` is off the source's shape, as a fraction of its aspect ratio.
    static func deviation(of canvas: AspectCanvas, from source: (width: Int, height: Int)) -> Double {
        let ratio = Double(source.width) / Double(max(source.height, 1))
        return abs(Double(canvas.width) / Double(max(canvas.height, 1)) - ratio) / ratio
    }

    /// The step closest in area to `area`: a new picture keeps the size the user had.
    static func nearest(_ ladder: [AspectCanvas], area: Int) -> AspectCanvas? {
        ladder.min { abs($0.area - area) < abs($1.area - area) }
    }

    /// Canvases within `maxDeviation` of the source's shape, or the `fallback`
    /// closest when none is.
    private static func faithful(sourceWidth: Int, sourceHeight: Int, grid: ResolutionGrid,
                                 fallback: Int) -> [AspectCanvas] {
        guard sourceWidth > 0, sourceHeight > 0, grid.alignment > 0 else { return [] }
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
            scored.append((canvas, deviation(of: canvas, from: (sourceWidth, sourceHeight))))
        }
        guard !scored.isEmpty else { return [] }

        let faithful = scored.filter { $0.deviation <= maxDeviation }.map(\.canvas)
        // Nothing is faithful (a very unusual aspect): offer the closest there
        // is rather than nothing.
        return faithful.isEmpty ? scored.sorted { $0.deviation < $1.deviation }.prefix(fallback).map(\.canvas) : faithful
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

    /// The menu rows for a source picture: its own size first (the one option
    /// that does not rescale it, when it fits the grid), then the spread named
    /// largest to smallest. Fewer candidates take the ends and the middle,
    /// because "large" among two says nothing.
    static func choices(sourceWidth: Int, sourceHeight: Int, grid: ResolutionGrid) -> [SourceCanvasChoice] {
        let spread = options(sourceWidth: sourceWidth, sourceHeight: sourceHeight, grid: grid)
        let names: [String?] = {
            switch spread.count {
            case 5:  return ["largest", "large", "medium", "small", "smallest"]
            case 4:  return ["largest", "large", "small", "smallest"]
            case 3:  return ["largest", "medium", "smallest"]
            case 2:  return ["largest", "smallest"]
            default: return [nil]
            }
        }()
        var out = spread.enumerated().map { i, c in
            SourceCanvasChoice(canvas: c, name: i < names.count ? names[i] : nil)
        }
        if let own = sourceSize(sourceWidth: sourceWidth, sourceHeight: sourceHeight, grid: grid) {
            out.removeAll { $0.canvas == own }
            out.insert(SourceCanvasChoice(canvas: own, name: "source size", isSourceSize: true), at: 0)
        }
        return out
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

    /// The file's pixel size from its metadata: ImageIO reads the header
    /// without decoding the picture, which a 4000px photo would make a
    /// noticeable pause.
    static func pixelSize(of url: URL) -> (width: Int, height: Int)? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int,
              let h = props[kCGImagePropertyPixelHeight] as? Int,
              w > 0, h > 0 else { return nil }
        return (w, h)
    }
}
