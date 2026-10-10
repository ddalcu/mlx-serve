import SwiftUI
import AppKit

// The MLX-Amp skin, drawn as pixel art. Every coordinate is a pixel of the
// classic main window. Everything is drawn at a fixed 1.5 points per skin pixel
// (whole device pixels on Retina) and the window stretches to the pane, so nothing
// is smoothed. What MOVES (clock, spectrum, scrolling titles, the seek thumb) is drawn
// by an AppKit view on its own timer, never by a SwiftUI timeline: a timeline ticks
// the whole window's layout, which is what kept the player at 20% CPU.

enum MlxAmpStyle {
    /// Narrowest window, in skin pixels, the controls fit in.
    static let minWidth: CGFloat = 236
    /// Points per skin pixel: 3 device pixels on Retina.
    static let scale: CGFloat = 1.5
    /// Point size of the ordinary text in the skin (playlist rows, the radio field):
    /// 8 skin pixels at the skin's scale.
    static let rowFont: CGFloat = 12

    static func rgb(_ hex: UInt32) -> Color {
        Color(red: Double((hex >> 16) & 255) / 255, green: Double((hex >> 8) & 255) / 255, blue: Double(hex & 255) / 255)
    }
    static let chassis = rgb(0x3A3760)
    static let chassisLight = rgb(0x8583B0)
    static let chassisDark = rgb(0x14132A)
    static let innerLight = rgb(0x514E80)
    static let innerDark = rgb(0x242245)
    static let lcd = rgb(0x00E000)
    static let lcdDim = rgb(0x0A3A0A)
    static let white = rgb(0xE8E8F4)
    static let goldLight = rgb(0xE6D08C)
    static let gold = rgb(0xC4A95A)
    static let goldDark = rgb(0x7A6430)
    static let keyFace = rgb(0xB7BFD0)
    static let keyHi = rgb(0xF4F6FC)
    static let keyHi2 = rgb(0xDDE2EE)
    static let keyLo = rgb(0x2A2C42)
    static let keyLo2 = rgb(0x7C849E)
    static let ink = rgb(0x1C1D2E)
}

// MARK: - Pixel drawing

/// Rectangles in skin pixels, drawn into a SwiftUI canvas or an AppKit view's
/// Core Graphics context. A pixel is a rect, so nothing needs anti-aliasing.
struct Px {
    private enum Backend { case ui(GraphicsContext), cg(CGContext) }
    private let backend: Backend

    init(ctx: GraphicsContext) { backend = .ui(ctx) }
    init(cg: CGContext) { backend = .cg(cg) }

    private static let colorLock = NSLock()
    nonisolated(unsafe) private static var cgColors: [Color: CGColor] = [:]

    private static func cgColor(_ color: Color) -> CGColor {
        colorLock.lock(); defer { colorLock.unlock() }
        if let hit = cgColors[color] { return hit }
        let made = NSColor(color).cgColor
        cgColors[color] = made
        return made
    }

    func rect(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat, _ color: Color) {
        switch backend {
        case .ui(let ctx): ctx.fill(Path(CGRect(x: x, y: y, width: w, height: h)), with: .color(color))
        case .cg(let ctx): ctx.setFillColor(Self.cgColor(color)); ctx.fill(CGRect(x: x, y: y, width: w, height: h))
        }
    }

    /// Draw `body` with everything outside `rect` cut away.
    func clipped(to rect: CGRect, _ body: (Px) -> Void) {
        switch backend {
        case .ui(var ctx):
            ctx.clip(to: Path(rect))
            body(Px(ctx: ctx))
        case .cg(let ctx):
            ctx.saveGState()
            ctx.clip(to: rect)
            body(self)
            ctx.restoreGState()
        }
    }

    /// 1-px frame, light on the top/left edges and dark on the bottom/right.
    func bevel(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat, light: Color, dark: Color) {
        rect(x, y, w, 1, light); rect(x, y, 1, h, light)
        rect(x, y + h - 1, w, 1, dark); rect(x + w - 1, y, 1, h, dark)
    }

    /// A recessed black screen.
    func screen(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat) {
        rect(x, y, w, h, .black)
        bevel(x, y, w, h, light: MlxAmpStyle.chassisDark, dark: MlxAmpStyle.chassisLight.opacity(0.8))
    }

    func text(_ s: String, _ x: CGFloat, _ y: CGFloat, _ color: Color, advance: CGFloat = 6) {
        let path = PixelFont.path(for: s, advance: advance)
        switch backend {
        case .ui(var ctx):
            ctx.translateBy(x: x, y: y)
            ctx.fill(path, with: .color(color))
        case .cg(let ctx):
            ctx.saveGState()
            ctx.translateBy(x: x, y: y)
            ctx.setFillColor(Self.cgColor(color))
            ctx.addPath(path.cgPath)
            ctx.fillPath()
            ctx.restoreGState()
        }
    }

    /// 3 x 5 capitals for the small labels and the vertical O-A-I-D-U-V strip.
    func mini(_ c: Character, _ x: CGFloat, _ y: CGFloat, _ color: Color) {
        let rows: [Character: [UInt8]] = [
            "O": [7, 5, 5, 5, 7], "A": [2, 5, 7, 5, 5], "I": [7, 2, 2, 2, 7], "D": [6, 5, 5, 5, 6], "U": [5, 5, 5, 5, 7],
            "V": [5, 5, 5, 5, 2], "K": [5, 5, 6, 5, 5], "B": [6, 5, 6, 5, 6], "P": [6, 5, 6, 4, 4], "S": [3, 4, 2, 1, 6],
            "H": [5, 5, 7, 5, 5], "Z": [7, 1, 2, 4, 7], "M": [5, 7, 7, 5, 5], "N": [6, 5, 5, 5, 5], "T": [7, 2, 2, 2, 2],
            "E": [7, 4, 6, 4, 7], "R": [6, 5, 6, 5, 5],
        ]
        for (row, bits) in (rows[c] ?? []).enumerated() {
            for col in 0..<3 where bits & (0b100 >> col) != 0 { rect(x + CGFloat(col), y + CGFloat(row), 1, 1, color) }
        }
    }

    func miniText(_ s: String, _ x: CGFloat, _ y: CGFloat, _ color: Color) {
        for (i, c) in s.enumerated() { mini(c, x + CGFloat(i) * 4, y, color) }
    }

    static func miniWidth(_ s: String) -> CGFloat { CGFloat(s.count) * 4 - 1 }

    static func textWidth(_ s: String, advance: CGFloat = 6) -> CGFloat { CGFloat(PixelFont.printable(s).count) * advance - 1 }

    /// Seven-segment numeral in a 9 x 13 cell.
    func digit(_ d: Int, _ x: CGFloat, _ y: CGFloat, _ color: Color) {
        let on = PixelFont.segments[d]
        let seg: [(CGFloat, CGFloat, CGFloat, CGFloat)] = [
            (1, 0, 7, 2), (7, 1, 2, 5), (7, 7, 2, 5), (1, 11, 7, 2), (0, 7, 2, 5), (0, 1, 2, 5), (1, 5, 7, 2),
        ]
        for (i, s) in seg.enumerated() where on & (1 << i) != 0 { rect(x + s.0, y + s.1, s.2, s.3, color) }
    }
}

/// A fixed-size drawing, one unit per skin pixel.
struct PixelCanvas: View {
    let width: CGFloat
    let height: CGFloat
    let draw: (Px) -> Void

    var body: some View {
        Canvas { ctx, _ in
            var scaled = ctx
            scaled.scaleBy(x: MlxAmpStyle.scale, y: MlxAmpStyle.scale)
            draw(Px(ctx: scaled))
        }
        .frame(width: width * MlxAmpStyle.scale, height: height * MlxAmpStyle.scale)
    }
}

/// 5 x 7 dot-matrix capitals, rows top to bottom, bit 4 the left column.
enum PixelFont {
    static let glyphs: [Character: [UInt8]] = {
        let rows: [Character: [UInt8]] = [
            "A": [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11], "B": [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
            "C": [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E], "D": [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
            "E": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F], "F": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
            "G": [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F], "H": [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
            "I": [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E], "J": [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
            "K": [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11], "L": [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
            "M": [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11], "N": [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
            "O": [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E], "P": [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
            "Q": [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D], "R": [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
            "S": [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E], "T": [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
            "U": [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E], "V": [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
            "W": [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A], "X": [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
            "Y": [0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04], "Z": [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
            "0": [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E], "1": [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
            "2": [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F], "3": [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
            "4": [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02], "5": [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
            "6": [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E], "7": [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
            "8": [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E], "9": [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
            ".": [0, 0, 0, 0, 0, 0x0C, 0x0C], ",": [0, 0, 0, 0, 0x0C, 0x04, 0x08], "-": [0, 0, 0, 0x1F, 0, 0, 0],
            "_": [0, 0, 0, 0, 0, 0, 0x1F], ":": [0, 0x0C, 0x0C, 0, 0x0C, 0x0C, 0], "(": [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
            ")": [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08], "*": [0, 0x04, 0x15, 0x0E, 0x15, 0x04, 0],
            "/": [0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10], "'": [0x04, 0x04, 0x08, 0, 0, 0, 0], " ": [0, 0, 0, 0, 0, 0, 0],
            "+": [0, 0x04, 0x04, 0x1F, 0x04, 0x04, 0], "!": [0x04, 0x04, 0x04, 0x04, 0x04, 0, 0x04],
            "?": [0x0E, 0x11, 0x01, 0x02, 0x04, 0, 0x04], "\"": [0x0A, 0x0A, 0x0A, 0, 0, 0, 0],
        ]
        return rows
    }()

    private static let pathLock = NSLock()
    nonisolated(unsafe) private static var paths: [String: Path] = [:]

    /// The text as ONE path of pixel rects, built once per string: a scrolling title
    /// redraws every frame, and a fill per lit pixel was most of what it cost.
    static func path(for s: String, advance: CGFloat) -> Path {
        let key = "\(advance)|\(s)"
        pathLock.lock(); defer { pathLock.unlock() }
        if let hit = paths[key] { return hit }
        var path = Path()
        var x: CGFloat = 0
        for ch in printable(s) {
            for (row, bits) in (glyphs[ch] ?? glyphs["?"]!).enumerated() {
                for col in 0..<5 where bits & (0b10000 >> col) != 0 {
                    path.addRect(CGRect(x: x + CGFloat(col), y: CGFloat(row), width: 1, height: 1))
                }
            }
            x += advance
        }
        if paths.count > 96 { paths.removeAll() }
        paths[key] = path
        return path
    }

    /// What the font can draw: capitals, with the characters it lacks folded to their
    /// plain look (an ellipsis becomes dots, an accent goes) so no stray "?" shows.
    static func printable(_ s: String) -> String {
        var out = s
        for (from, to) in [("…", "..."), ("–", "-"), ("—", "-"), ("‘", "'"), ("’", "'"), ("“", "\""), ("”", "\"")] {
            out = out.replacingOccurrences(of: from, with: to)
        }
        return out.folding(options: .diacriticInsensitive, locale: nil).uppercased()
    }

    /// Lit segments of a numeral: bit i = a b c d e f g.
    static let segments: [Int] = [0b0111111, 0b0000110, 0b1011011, 0b1001111, 0b1100110, 0b1101101, 0b1111101, 0b0000111, 0b1111111, 0b1101111]
}

// MARK: - Motion

/// How fast the moving parts move, and how they behave when the machine is short of
/// power: a weak laptop gets a slower spectrum and a still title.
enum MlxAmpMotion {
    struct Rates: Equatable {
        /// Frames per second; 0 = hold still.
        var spectrum: Double
        var title: Double
        var seek: Double
    }

    static func rates(lowPower: Bool, thermal: ProcessInfo.ThermalState) -> Rates {
        let reduced = lowPower || thermal == .serious || thermal == .critical
        return reduced ? Rates(spectrum: 6, title: 0, seek: 2) : Rates(spectrum: 12, title: 6, seek: 4)
    }

    /// The rates for this machine right now.
    static var current: Rates {
        rates(lowPower: ProcessInfo.processInfo.isLowPowerModeEnabled, thermal: ProcessInfo.processInfo.thermalState)
    }

    static let titleGap: CGFloat = 36
    /// Skin pixels per second.
    static let titleSpeed: Double = 14

    /// How far a title has scrolled `time` seconds in: whole pixels, 0 while it fits.
    static func titleOffset(textWidth: CGFloat, boxWidth: CGFloat, time: Double) -> CGFloat {
        guard textWidth > boxWidth else { return 0 }
        return CGFloat(Int(time * titleSpeed) % Int(textWidth + titleGap))
    }
}

// MARK: - Window chrome

/// Hands `content` the pane's size in skin pixels; the skin stretches and the scale is fixed.
/// A view places itself with `px` and draws with `PixelCanvas`, which apply the scale.
struct MlxAmpStretched<Content: View>: View {
    @ViewBuilder let content: (_ width: CGFloat, _ height: CGFloat) -> Content

    var body: some View {
        GeometryReader { geo in
            let scale = MlxAmpStyle.scale
            content(max(MlxAmpStyle.minWidth, (geo.size.width / scale).rounded(.down)), (geo.size.height / scale).rounded(.down))
                .frame(width: geo.size.width, height: geo.size.height, alignment: .topLeading)
                .clipped()
        }
    }
}

/// A window: bevelled chassis with the gold-railed title bar.
private struct MlxAmpWindow<Content: View>: View {
    let title: String
    let width: CGFloat
    let height: CGFloat
    @ViewBuilder let content: Content

    var body: some View {
        ZStack(alignment: .topLeading) {
            PixelCanvas(width: width, height: height) { p in
                let w = width
                p.rect(0, 0, w, height, MlxAmpStyle.chassis)
                p.bevel(0, 0, w, height, light: MlxAmpStyle.chassisLight, dark: MlxAmpStyle.chassisDark)
                p.bevel(1, 1, w - 2, height - 2, light: MlxAmpStyle.innerLight, dark: MlxAmpStyle.innerDark)
                // Logo: three gold bars.
                for (i, h) in [3, 7, 5, 7, 3].enumerated() as EnumeratedSequence<[CGFloat]> {
                    p.rect(5 + CGFloat(i) * 2, 3 + (7 - h) / 2, 1, h, MlxAmpStyle.gold)
                }
                let textW = Px.textWidth(title)
                let textX = ((w - textW) / 2).rounded()
                for (x0, x1) in [(CGFloat(18), textX - 5), (textX + textW + 5, w - 6)] {
                    p.rect(x0, 4, x1 - x0, 2, MlxAmpStyle.goldLight)
                    p.rect(x0, 6, x1 - x0, 1, MlxAmpStyle.gold)
                    p.rect(x0, 8, x1 - x0, 2, MlxAmpStyle.gold)
                    p.rect(x0, 10, x1 - x0, 1, MlxAmpStyle.goldDark)
                }
                p.text(title, textX, 4, MlxAmpStyle.white)
            }
            content
        }
        .frame(width: width * MlxAmpStyle.scale, height: height * MlxAmpStyle.scale, alignment: .topLeading)
    }
}

private extension View {
    /// A control the running radio takes over: dimmed and dead.
    func locked(_ on: Bool) -> some View { disabled(on).opacity(on ? 0.4 : 1) }

    /// Place at a pixel rectangle of the window.
    func px(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat) -> some View {
        let s = MlxAmpStyle.scale
        return frame(width: w * s, height: h * s).offset(x: x * s, y: y * s)
    }
}

// MARK: - Controls

private struct MlxAmpKeyStyle: ButtonStyle {
    let width: CGFloat
    let height: CGFloat
    let draw: (Px, CGFloat) -> Void   // ink, given the 1-px press shift

    func makeBody(configuration: Configuration) -> some View {
        let down = configuration.isPressed
        return PixelCanvas(width: width, height: height) { p in
            p.rect(0, 0, width, height, MlxAmpStyle.keyFace)
            if down {
                p.bevel(0, 0, width, height, light: MlxAmpStyle.keyLo, dark: MlxAmpStyle.keyHi)
                p.bevel(1, 1, width - 2, height - 2, light: MlxAmpStyle.keyLo2, dark: MlxAmpStyle.keyHi2)
            } else {
                p.bevel(0, 0, width, height, light: MlxAmpStyle.keyHi, dark: MlxAmpStyle.keyLo)
                p.bevel(1, 1, width - 2, height - 2, light: MlxAmpStyle.keyHi2, dark: MlxAmpStyle.keyLo2)
            }
            draw(p, down ? 1 : 0)
        }
    }
}

private enum MlxAmpGlyph { case prev, play, pause, stop, next, eject }

private struct MlxAmpGlyphKey: View {
    let glyph: MlxAmpGlyph
    let width: CGFloat
    let height: CGFloat
    let help: String
    let action: () -> Void

    var body: some View {
        Button(action: action) { EmptyView() }
            .buttonStyle(MlxAmpKeyStyle(width: width, height: height) { p, shift in draw(p, shift) })
            .help(help)
    }

    private func draw(_ p: Px, _ s: CGFloat) {
        let ink = MlxAmpStyle.ink
        let cx = (width / 2).rounded() + s, cy = (height / 2).rounded() + s
        func triangle(_ left: CGFloat, forward: Bool) {
            for y in 0..<9 {
                let w = (Double(min(y, 8 - y) + 1) * 1.6).rounded()
                p.rect(forward ? left : left + 8 - w, cy - 4 + CGFloat(y), CGFloat(w), 1, ink)
            }
        }
        switch glyph {
        case .play: triangle(cx - 4, forward: true)
        case .pause: p.rect(cx - 4, cy - 4, 3, 9, ink); p.rect(cx + 1, cy - 4, 3, 9, ink)
        case .stop: p.rect(cx - 4, cy - 4, 8, 8, ink)
        case .prev: p.rect(cx - 5, cy - 4, 2, 9, ink); triangle(cx - 3, forward: false)
        case .next: triangle(cx - 5, forward: true); p.rect(cx + 3, cy - 4, 2, 9, ink)
        case .eject:
            for y in 0..<5 { p.rect(cx - CGFloat(y) - 1, cy - 5 + CGFloat(y), CGFloat(y) * 2 + 2, 1, ink) }
            p.rect(cx - 6, cy + 1, 12, 2, ink)
        }
    }
}

/// A key with a label and, for toggles, an LED.
private struct MlxAmpTextKey: View {
    let text: String
    var led: Bool?
    let width: CGFloat
    let height: CGFloat
    let help: String
    let action: () -> Void

    var body: some View {
        Button(action: action) { EmptyView() }
            .buttonStyle(MlxAmpKeyStyle(width: width, height: height) { p, s in
                let ledW: CGFloat = led == nil ? 0 : 6
                let x = ((width - ledW - Px.textWidth(text)) / 2).rounded() + ledW + s
                let y = ((height - 7) / 2).rounded() + s
                if let led { p.rect(x - ledW + 1, y + 2, 3, 3, led ? MlxAmpStyle.lcd : MlxAmpStyle.lcdDim) }
                p.text(text, x, y, MlxAmpStyle.ink)
            })
            .help(help)
    }
}

private struct MlxAmpRepeatKey: View {
    let lit: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) { EmptyView() }
            .buttonStyle(MlxAmpKeyStyle(width: 28, height: 15) { p, s in
                p.rect(4 + s, 6 + s, 3, 3, lit ? MlxAmpStyle.lcd : MlxAmpStyle.lcdDim)
                // A loop arrow: box with a break and an arrowhead.
                let x = 10 + s, y = 4 + s, ink = MlxAmpStyle.ink
                p.rect(x, y, 10, 1, ink); p.rect(x, y + 6, 10, 1, ink)
                p.rect(x, y + 1, 1, 5, ink); p.rect(x + 9, y + 1, 1, 2, ink)
                p.rect(x + 7, y + 3, 5, 1, ink); p.rect(x + 8, y + 4, 3, 1, ink); p.rect(x + 9, y + 5, 1, 1, ink)
            })
            .help("Play the next track when one ends")
    }
}

/// The volume slider: a groove, a value, a thumb. (The seek bar moves, so it is a live view.)
struct MlxAmpSlider: View {
    let length: CGFloat
    let fraction: Double
    var onDrag: ((Double) -> Void)?

    static let thumbWidth: CGFloat = 14

    /// Where a pointer at `x` (skin pixels from the slider's left edge) puts the thumb, 0...1.
    static func fraction(x: CGFloat, length: CGFloat, thumb: CGFloat) -> Double {
        min(1, max(0, Double((x - thumb / 2) / max(1, length - thumb))))
    }

    var body: some View {
        let thumb = Self.thumbWidth
        let span = length - thumb
        PixelCanvas(width: length, height: 13) { p in
            let tx = (span * fraction).rounded()
            p.rect(0, 3, length, 7, .black)
            p.bevel(0, 3, length, 7, light: MlxAmpStyle.chassisDark, dark: MlxAmpStyle.chassisLight.opacity(0.7))
            let filled = tx + thumb / 2
            var x: CGFloat = 1
            while x < min(length - 1, filled) {
                let t = Double(x / length)
                let color = Color(red: min(1, 0.1 + t * 1.6), green: min(0.85, 1.4 - t * 1.2), blue: 0.05)
                p.rect(x, 4, min(2, filled - x), 5, color)
                x += 2
            }
            p.rect(tx, 1, thumb, 11, MlxAmpStyle.keyFace)
            p.bevel(tx, 1, thumb, 11, light: MlxAmpStyle.keyHi, dark: MlxAmpStyle.keyLo)
            p.bevel(tx + 1, 2, thumb - 2, 9, light: MlxAmpStyle.keyHi2, dark: MlxAmpStyle.keyLo2)
            for g in 0..<3 { p.rect(tx + 4 + CGFloat(g) * 2, 4, 1, 5, MlxAmpStyle.keyLo2) }
        }
        .contentShape(Rectangle())
        .gesture(DragGesture(minimumDistance: 0).onChanged { drag in
            onDrag?(Self.fraction(x: drag.location.x / MlxAmpStyle.scale, length: length, thumb: thumb))
        })
    }
}

// MARK: - Live (animated) drawing

/// An AppKit view that draws a region of the skin and redraws itself on its own timer, so a
/// moving spectrum or title costs a draw of that one view and not a layout of the window.
/// The timer runs only while `shouldRun` says something is moving and the window is on screen.
final class MlxAmpLiveView: NSView {
    var draw: @MainActor (Px) -> Void = { _ in }
    var shouldRun: @MainActor () -> Bool = { false }
    var fps: @MainActor () -> Double = { 0 }
    /// Pointer x in skin pixels, for a view that takes drags (the seek bar).
    var onDrag: (@MainActor (CGFloat) -> Void)?

    private var timer: Timer?
    private var observers: [NSObjectProtocol] = []

    override var isFlipped: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layerContentsRedrawPolicy = .onSetNeedsDisplay
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    deinit {
        timer?.invalidate()
        observers.forEach(NotificationCenter.default.removeObserver)
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let cg = NSGraphicsContext.current?.cgContext else { return }
        cg.setShouldAntialias(false)
        cg.scaleBy(x: MlxAmpStyle.scale, y: MlxAmpStyle.scale)
        draw(Px(cg: cg))
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        observers.forEach(NotificationCenter.default.removeObserver)
        observers = []
        let center = NotificationCenter.default
        // A slower machine state, or a window that is hidden or covered, changes what is worth drawing.
        for name in [Notification.Name.NSProcessInfoPowerStateDidChange, ProcessInfo.thermalStateDidChangeNotification] {
            observers.append(center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in self?.reschedule() })
        }
        if let window {
            observers.append(center.addObserver(forName: NSWindow.didChangeOcclusionStateNotification, object: window, queue: .main) { [weak self] _ in
                self?.reschedule()
            })
        }
        reschedule()
    }

    /// Redraw once, then run the timer only if something is moving and the window can be seen.
    func reschedule() {
        timer?.invalidate()
        timer = nil
        needsDisplay = true
        guard let window, window.occlusionState.contains(.visible), shouldRun() else { return }
        let rate = fps()
        guard rate > 0 else { return }
        let t = Timer(timeInterval: 1 / rate, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.needsDisplay = true
            if !self.shouldRun() { self.reschedule() }
        }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    private func skinX(_ event: NSEvent) -> CGFloat { convert(event.locationInWindow, from: nil).x / MlxAmpStyle.scale }
    override func mouseDown(with event: NSEvent) { onDrag?(skinX(event)); needsDisplay = true }
    override func mouseDragged(with event: NSEvent) { onDrag?(skinX(event)); needsDisplay = true }
}

/// SwiftUI's door to `MlxAmpLiveView`. SwiftUI hands it new closures when the player's state
/// changes (a track starts, pauses, stops); between those, only the view's own timer draws.
struct MlxAmpLive: NSViewRepresentable {
    let width: CGFloat
    let height: CGFloat
    let shouldRun: @MainActor () -> Bool
    let fps: @MainActor () -> Double
    var onDrag: (@MainActor (CGFloat) -> Void)?
    let draw: @MainActor (Px) -> Void

    func makeNSView(context: Context) -> MlxAmpLiveView { configured(MlxAmpLiveView(frame: .zero)) }

    func updateNSView(_ view: MlxAmpLiveView, context: Context) {
        _ = configured(view)
        view.reschedule()
    }

    func sizeThatFits(_ proposal: ProposedViewSize, nsView: MlxAmpLiveView, context: Context) -> CGSize? {
        CGSize(width: width * MlxAmpStyle.scale, height: height * MlxAmpStyle.scale)
    }

    private func configured(_ view: MlxAmpLiveView) -> MlxAmpLiveView {
        view.draw = draw
        view.shouldRun = shouldRun
        view.fps = fps
        view.onDrag = onDrag
        return view
    }
}

/// What the live regions draw, as plain functions of the player's state.
@MainActor
private enum MlxAmpLiveDrawing {
    static func levelColor(_ t: Double) -> Color {
        t < 0.45 ? MlxAmpStyle.rgb(0x20D020) : t < 0.7 ? MlxAmpStyle.rgb(0xC8D020) : t < 0.85 ? MlxAmpStyle.rgb(0xE09020) : MlxAmpStyle.rgb(0xE02010)
    }

    /// The clock + spectrum screen (98 x 38).
    static func clockScreen(_ p: Px, player: AudioClipPlayer, analyzer: SpectrumAnalyzer, path: String, enabled: Bool) {
        let active = enabled && player.playingPath == path
        let playing = active && !player.isPaused
        p.screen(0, 0, 98, 38)
        for (i, c) in "OAIDUV".enumerated() { p.mini(c, 3, 3 + CGFloat(i) * 6, MlxAmpStyle.rgb(0x5A5A70)) }
        // Play-state glyph.
        if playing { for y in 0..<9 { p.rect(15, 4 + CGFloat(y), CGFloat(min(y, 8 - y) + 1), 1, MlxAmpStyle.lcd) } }
        else if active { p.rect(15, 4, 3, 9, MlxAmpStyle.lcd); p.rect(20, 4, 3, 9, MlxAmpStyle.lcd) }
        else { p.rect(15, 5, 7, 7, MlxAmpStyle.lcd) }
        p.rect(15, 14, 3, 2, playing ? MlxAmpStyle.rgb(0xC03020) : MlxAmpStyle.rgb(0x501810))
        // Clock.
        let t = Int(max(0, active ? player.elapsed : 0))
        let digits = [t / 600 % 10, t / 60 % 10, t % 60 / 10, t % 60 % 10]
        p.digit(digits[0], 29, 4, MlxAmpStyle.lcd); p.digit(digits[1], 41, 4, MlxAmpStyle.lcd)
        p.rect(55, 7, 2, 2, MlxAmpStyle.lcd); p.rect(55, 12, 2, 2, MlxAmpStyle.lcd)
        p.digit(digits[2], 60, 4, MlxAmpStyle.lcd); p.digit(digits[3], 72, 4, MlxAmpStyle.lcd)
        // Spectrum: 19 bars, 3 px wide with 1 px between, a grey cap hanging above each.
        let frame = analyzer.step(playing: playing, time: player.elapsed)
        for b in 0..<SpectrumAnalyzer.bands {
            let x = 13 + CGFloat(b) * 4
            for r in 0..<Int(16 * frame.levels[b]) { p.rect(x, 36 - CGFloat(r), 3, 1, levelColor(Double(r) / 16)) }
            let cap = Int(16 * frame.peaks[b])
            if cap > 0 { p.rect(x, 36 - CGFloat(cap), 3, 1, MlxAmpStyle.rgb(0xB0B0B8)) }
        }
        for x in stride(from: CGFloat(13), to: 89, by: 2) { p.rect(x, 36, 1, 1, MlxAmpStyle.rgb(0x2E48A0)) }
    }

    /// A screen with one line of text; it scrolls when it does not fit and `scrolling` is true.
    static func titleScreen(_ p: Px, text: String, color: Color, width: CGFloat, height: CGFloat, scrolling: Bool) {
        p.screen(0, 0, width, height)
        let boxWidth = width - 6
        let textWidth = Px.textWidth(text)
        let offset = scrolling ? MlxAmpMotion.titleOffset(textWidth: textWidth, boxWidth: boxWidth,
                                                          time: Date().timeIntervalSinceReferenceDate) : 0
        let y = ((height - 7) / 2).rounded(.down)
        // Clipped to the screen's inside so the text slides under its frame.
        p.clipped(to: CGRect(x: 3, y: 1, width: boxWidth, height: height - 2)) { c in
            c.text(text, 3 - offset, y, color)
            if textWidth > boxWidth { c.text(text, 3 + textWidth + MlxAmpMotion.titleGap - offset, y, color) }
        }
    }

    /// The radio strip: the status line over a progress bar. A dim fill grows behind the text as the
    /// track is composed; with no number yet, a block sweeps back and forth instead.
    static func radioStrip(_ p: Px, text: String, progress: Double?, width: CGFloat, height: CGFloat, scrolling: Bool) {
        p.screen(0, 0, width, height)
        let inner = width - 2, barColor = MlxAmpStyle.rgb(0x15601C)
        if let progress {
            p.rect(1, 1, RadioStrip.fillWidth(progress: progress, width: inner), height - 2, barColor)
        } else {
            p.rect(1 + RadioStrip.blockOffset(time: Date().timeIntervalSinceReferenceDate, width: inner), 1,
                   RadioStrip.blockWidth, height - 2, barColor)
        }
        let boxWidth = width - 6
        let textWidth = Px.textWidth(text)
        let offset = scrolling ? MlxAmpMotion.titleOffset(textWidth: textWidth, boxWidth: boxWidth,
                                                          time: Date().timeIntervalSinceReferenceDate) : 0
        let y = ((height - 7) / 2).rounded(.down)
        p.clipped(to: CGRect(x: 3, y: 1, width: boxWidth, height: height - 2)) { c in
            c.text(text, 3 - offset, y, MlxAmpStyle.lcd)
            if textWidth > boxWidth { c.text(text, 3 + textWidth + MlxAmpMotion.titleGap - offset, y, MlxAmpStyle.lcd) }
        }
    }

    /// The seek bar (width x 10): a groove and, while something is loaded, the gold thumb.
    static func seekBar(_ p: Px, width: CGFloat, fraction: Double, showsThumb: Bool) {
        p.rect(0, 0, width, 10, .black)
        p.bevel(0, 0, width, 10, light: MlxAmpStyle.chassisDark, dark: MlxAmpStyle.chassisLight.opacity(0.7))
        guard showsThumb else { return }
        let tx = ((width - 29) * fraction).rounded()
        p.rect(tx, 0, 29, 10, MlxAmpStyle.gold)
        p.bevel(tx, 0, 29, 10, light: MlxAmpStyle.goldLight, dark: MlxAmpStyle.goldDark)
        p.bevel(tx + 3, 2, 23, 6, light: MlxAmpStyle.goldDark, dark: MlxAmpStyle.goldLight)
    }
}

// MARK: - Main window

struct MlxAmpPlayer: View {
    enum Status { case busy(String, progress: Double?), failed(String) }

    static let height: CGFloat = 132

    @ObservedObject var player: AudioClipPlayer
    @ObservedObject var radio: AIRadio
    @StateObject private var analyzer = SpectrumAnalyzer()
    let width: CGFloat
    let path: String
    @Binding var showPlaylist: Bool
    /// The AI Radio theme typed into the strip, and what pressing RADIO / Return does.
    @Binding var radioTheme: String
    var onRadioStart: () -> Void
    /// Stopping a running station is asked first: it discards the track being composed.
    var onRadioStop: () -> Void
    /// Overrides the title with what the pane is doing (generating, failed, empty).
    var status: Status?
    var onShowLog: (() -> Void)?

    var body: some View {
        let info = status == nil ? AudioClipPlayer.info(path) : nil
        let active = status == nil && player.playingPath == path
        let playing = active && !player.isPaused
        let total = info?.seconds ?? 0
        // The right-hand cluster starts a gap clear of the clock/spectrum screen.
        let rx: CGFloat = 115, rw = width - rx - 9
        let title = titleText()
        let seekWidth = width - 27
        let stripWidth = width - 66 - 11
        // Read at draw time: the live views must never wait for SwiftUI to hand them state.
        let playingNow: @MainActor () -> Bool = { status == nil && player.playingPath == path && !player.isPaused }
        MlxAmpWindow(title: "MLX-AMP", width: width, height: Self.height) {
            ZStack(alignment: .topLeading) {
                MlxAmpLive(width: 98, height: 38,
                           shouldRun: { playingNow() || !analyzer.isSettled },
                           fps: { MlxAmpMotion.current.spectrum },
                           draw: { MlxAmpLiveDrawing.clockScreen($0, player: player, analyzer: analyzer, path: path, enabled: status == nil) })
                    .px(11, 22, 98, 38)

                // Title: scrolls only while something plays (or the pane reports progress).
                let scrolls: @MainActor () -> Bool = {
                    MlxAmpMotion.current.title > 0 && Px.textWidth(title.text) > rw - 6 && (playingNow() || status != nil)
                }
                MlxAmpLive(width: rw, height: 13, shouldRun: scrolls, fps: { MlxAmpMotion.current.title },
                           draw: { MlxAmpLiveDrawing.titleScreen($0, text: title.text, color: title.color, width: rw, height: 13, scrolling: scrolls()) })
                    .px(rx, 24, rw, 13)

                PixelCanvas(width: width - rx - 4, height: 11) { p in
                    let rate = Int(((info?.sampleRate ?? 0) / 1000).rounded())
                    let kbps = "\(info?.kbps ?? 0)", khz = "\(rate)"
                    p.screen(0, 0, 25, 11); p.text(kbps, 24 - Px.textWidth(kbps), 2, MlxAmpStyle.lcd)
                    p.miniText("KBPS", 28, 3, MlxAmpStyle.white)
                    p.screen(47, 0, 15, 11); p.text(khz, 61 - Px.textWidth(khz), 2, MlxAmpStyle.lcd)
                    p.miniText("KHZ", 65, 3, MlxAmpStyle.white)
                    let stereo = (info?.channels ?? 2) > 1
                    let right = width - rx - 4
                    let dim = MlxAmpStyle.rgb(0x77778F)
                    // Both words when they fit beside the labels, else only the active one.
                    let monoX = right - Px.miniWidth("STEREO") - 4 - Px.miniWidth("MONO")
                    let both = monoX >= 79
                    if both || stereo { p.miniText("STEREO", right - Px.miniWidth("STEREO"), 3, stereo ? MlxAmpStyle.lcd : dim) }
                    if both || !stereo { p.miniText("MONO", both ? monoX : right - Px.miniWidth("MONO"), 3, stereo ? dim : MlxAmpStyle.lcd) }
                }
                .px(rx, 41, width - rx - 4, 11)

                MlxAmpSlider(length: 58, fraction: player.volume) { player.volume = $0 }
                    .px(rx, 57, 58, 13).help("Volume")
                MlxAmpTextKey(text: "PL", led: showPlaylist, width: 23, height: 12, help: "Playlist") { showPlaylist.toggle() }.px(width - 33, 58, 23, 12)
                if case .failed = status, let onShowLog {
                    MlxAmpTextKey(text: "LOG", width: 24, height: 12, help: "Show log", action: onShowLog).px(width - 59, 58, 24, 12)
                } else {
                    MlxAmpTextKey(text: "M4A", width: 24, height: 12, help: "Export as M4A") { if !path.isEmpty { AudioExport.exportAndReveal(path) } }
                        .px(width - 59, 58, 24, 12)
                }

                MlxAmpLive(width: seekWidth, height: 10, shouldRun: playingNow, fps: { MlxAmpMotion.current.seek },
                           onDrag: { x in
                               guard total > 0 else { return }
                               player.seek(to: total * MlxAmpSlider.fraction(x: x, length: seekWidth, thumb: 29))
                           },
                           draw: { p in
                               let moving = player.playingPath == path && status == nil && total > 0
                               MlxAmpLiveDrawing.seekBar(p, width: seekWidth,
                                                         fraction: moving ? min(1, player.elapsed / total) : progressFraction,
                                                         showsThumb: moving || progressFraction > 0)
                           })
                    .px(16, 72, seekWidth, 10)

                // Transport.
                MlxAmpGlyphKey(glyph: .prev, width: 23, height: 18, help: "Previous") { player.advance(-1, from: path) }
                    .locked(radio.isOn).px(16, 88, 23, 18)
                MlxAmpGlyphKey(glyph: playing ? .pause : .play, width: 23, height: 18, help: playing ? "Pause" : "Play") {
                    if active { player.togglePause() } else if !path.isEmpty { player.play(path) }
                }.px(39, 88, 23, 18)
                MlxAmpGlyphKey(glyph: .stop, width: 23, height: 18, help: "Stop") { if radio.isOn { onRadioStop() } else { player.stop() } }.px(62, 88, 23, 18)
                MlxAmpGlyphKey(glyph: .next, width: 22, height: 18, help: "Next") { if radio.isOn { radio.skip() } else { player.advance(1, from: path) } }.px(85, 88, 22, 18)
                MlxAmpGlyphKey(glyph: .eject, width: 22, height: 16, help: "Reveal in Finder") {
                    if !path.isEmpty { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)]) }
                }.px(113, 89, 22, 16)
                MlxAmpTextKey(text: "SHUFFLE", led: player.shuffle, width: 56, height: 15, help: "Shuffle") { player.shuffle.toggle() }
                    .locked(radio.isOn).px(138, 89, 56, 15)
                MlxAmpRepeatKey(lit: player.autoNext) { player.autoNext.toggle() }
                    .locked(radio.isOn).px(194, 89, 28, 15)

                // AI Radio: the key starts and stops the station; the screen takes its theme.
                MlxAmpTextKey(text: "RADIO", led: radio.isOn, width: 46, height: 15,
                              help: radio.isOn ? "Stop AI Radio" : "Start AI Radio") {
                    if radio.isOn { onRadioStop() } else { onRadioStart() }
                }.px(16, 111, 46, 15)
                if radio.isOn {
                    // The strip is the progress bar: it redraws on its own timer, never through SwiftUI.
                    MlxAmpLive(width: stripWidth, height: 15, shouldRun: { radio.isOn },
                               fps: { max(MlxAmpMotion.current.title, 4) },
                               draw: { p in
                                   MlxAmpLiveDrawing.radioStrip(p, text: radio.status, progress: radio.progress, width: stripWidth,
                                                                height: 15, scrolling: MlxAmpMotion.current.title > 0)
                               })
                        .px(66, 111, stripWidth, 15)
                } else {
                    PixelCanvas(width: stripWidth, height: 15) { $0.screen(0, 0, stripWidth, 15) }.px(66, 111, stripWidth, 15)
                    TextField("", text: $radioTheme,
                              prompt: Text("AI Radio theme, e.g. mellow lo-fi for coding").foregroundStyle(MlxAmpStyle.lcdDim))
                        .textFieldStyle(.plain)
                        .font(.system(size: MlxAmpStyle.rowFont, weight: .bold, design: .monospaced))
                        .foregroundStyle(MlxAmpStyle.lcd)
                        .tint(MlxAmpStyle.lcd)
                        .onSubmit(onRadioStart)
                        .px(70, 113, stripWidth - 8, 11)
                }
            }
        }
        .task(id: path) { await analyzer.load(path) }
    }

    private var progressFraction: Double {
        if case .busy(_, let progress) = status { return progress ?? 0 }
        return 0
    }

    private func titleText() -> (text: String, color: Color) {
        switch status {
        case .busy(let message, _): return (message, MlxAmpStyle.lcd)
        case .failed(let message): return (message, MlxAmpStyle.rgb(0xF04030))
        case nil:
            let number = (player.queue.firstIndex(of: path) ?? 0) + 1
            let name = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
            let length = AudioClipPlayer.clock(AudioClipPlayer.info(path)?.seconds ?? 0)
            return ("*** \(number). \(name) (\(length)) ***", MlxAmpStyle.lcd)
        }
    }
}

// MARK: - Playlist

/// The playlist window: green-on-black rows, the current track in white, a gold scroll thumb.
struct MlxAmpPlaylist: View {
    static let minHeight: CGFloat = 125

    private struct Metrics: Equatable { var offset: CGFloat = 0; var content: CGFloat = 0; var visible: CGFloat = 0 }

    let width: CGFloat
    let height: CGFloat
    let paths: [String]
    let current: String?
    /// A running AI Radio owns playback: rows only offer export and reveal.
    var locked = false
    let onPlay: (String) -> Void
    let onStop: () -> Void
    @State private var position = ScrollPosition(edge: .top)
    @State private var metrics = Metrics()

    var body: some View {
        let listW = width - 16, listH = height - 25
        MlxAmpWindow(title: "PLAYLIST", width: width, height: height) {
            ZStack(alignment: .topLeading) {
                PixelCanvas(width: listW, height: listH) { $0.screen(0, 0, listW, listH) }.px(8, 17, listW, listH)
                ScrollView {
                    VStack(spacing: 0) {
                        ForEach(Array(paths.enumerated()), id: \.element) { index, path in row(path, number: index + 1) }
                    }
                }
                .scrollPosition($position)
                .scrollIndicators(.never)
                .onScrollGeometryChange(for: Metrics.self) {
                    Metrics(offset: $0.contentOffset.y, content: $0.contentSize.height, visible: $0.visibleRect.height)
                } action: { _, new in metrics = new }
                .px(9, 18, listW - 19, listH - 2)
                scrollbar(height: listH - 2).px(width - 24, 18, 15, listH - 2)
            }
        }
    }

    /// Metrics are in points (the scroll view's), the thumb in skin pixels.
    private func scrollbar(height track: CGFloat) -> some View {
        let scale = MlxAmpStyle.scale
        let scrollable = max(1, metrics.content - metrics.visible)
        let thumb = max(16, track * min(1, metrics.visible / max(1, metrics.content))).rounded()
        let y = ((track - thumb) * min(1, max(0, metrics.offset / scrollable))).rounded()
        return PixelCanvas(width: 15, height: track) { p in
            guard metrics.content > metrics.visible + 1 else { return }
            p.rect(3, 0, 8, track, MlxAmpStyle.chassisDark)
            p.rect(3, y, 8, thumb, MlxAmpStyle.gold)
            p.bevel(3, y, 8, thumb, light: MlxAmpStyle.goldLight, dark: MlxAmpStyle.goldDark)
            p.rect(5, y + thumb / 2 - 2, 4, 1, MlxAmpStyle.goldDark); p.rect(5, y + thumb / 2, 4, 1, MlxAmpStyle.goldDark)
            p.rect(5, y + thumb / 2 + 2, 4, 1, MlxAmpStyle.goldDark)
        }
        .contentShape(Rectangle())
        .gesture(DragGesture(minimumDistance: 0).onChanged { drag in
            guard metrics.content > metrics.visible else { return }
            let f = min(1, max(0, (drag.location.y / scale - thumb / 2) / max(1, track - thumb)))
            position.scrollTo(y: f * scrollable)
        })
    }

    private func row(_ path: String, number: Int) -> some View {
        let playing = current == path
        let scale = MlxAmpStyle.scale
        return HStack(spacing: 4 * scale) {
            Text(verbatim: "\(number). \(URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent)")
                .lineLimit(1).truncationMode(.middle).help(path)
            Spacer(minLength: 2 * scale)
            Text(verbatim: AudioClipPlayer.info(path).map { AudioClipPlayer.clock($0.seconds) } ?? "")
            Button { AudioExport.exportAndReveal(path) } label: { Image(systemName: "arrow.down.doc") }
                .buttonStyle(.plain).help("Export as M4A")
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
            } label: { Image(systemName: "folder") }
                .buttonStyle(.plain).help("Reveal in Finder")
        }
        .font(.system(size: MlxAmpStyle.rowFont, weight: .medium))
        .foregroundStyle(playing ? MlxAmpStyle.white : MlxAmpStyle.lcd)
        .padding(.horizontal, 3 * scale)
        .frame(height: 11 * scale)
        .contentShape(Rectangle())
        .onTapGesture { guard !locked else { return }; playing ? onStop() : onPlay(path) }
    }
}
