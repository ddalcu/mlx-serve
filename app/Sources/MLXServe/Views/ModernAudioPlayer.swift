import SwiftUI
import AppKit

// The Modern look of the generated-audio player (Settings ▸ Interface ▸ Audio
// Player): the same player, AI Radio and actions as MLX-Amp, drawn with the
// system's controls on the panes' grey card, over a dark screen whose spectrum
// takes the app icon's night blue to violet. What moves (spectrum, seek bar,
// radio strip) draws in a `LiveCanvas` on MLX-Amp's rates, never a SwiftUI timeline.

/// What the Modern deck shows, as plain functions.
enum ModernAudioPlayer {

    /// "48 kHz · Stereo · 1536 kbps"; empty when the file cannot be read.
    static func infoLine(_ info: AudioClipPlayer.Info?) -> String {
        guard let info else { return "" }
        let khz = info.sampleRate / 1000
        let rate = khz == khz.rounded() ? String(Int(khz)) : String(format: "%.1f", khz)
        let channels = L10n.text(info.channels > 1 ? "Stereo" : "Mono")
        return "\(rate) kHz · \(channels) · \(info.kbps) kbps"
    }

    static func volumeSymbol(_ volume: Double) -> String {
        switch volume {
        case ...0: return "speaker.slash.fill"
        case ..<0.34: return "speaker.wave.1.fill"
        case ..<0.67: return "speaker.wave.2.fill"
        default: return "speaker.wave.3.fill"
        }
    }

    /// A deck key's ink: grey like the volume glyph, brighter while pressed,
    /// violet while switched on, faint while AI Radio locks it.
    static func keyInk(enabled: Bool, lit: Bool, pressed: Bool) -> NSColor {
        switch (enabled, lit) {
        case (false, false): return .quaternaryLabelColor
        case (false, true): return ModernSpectrum.violetNS.withAlphaComponent(0.4)
        case (true, true): return ModernSpectrum.violetNS
        case (true, false): return pressed ? .labelColor : .secondaryLabelColor
        }
    }

    /// Where a pointer at `x` puts the playhead along `track`, 0...1.
    static func seekFraction(x: Double, track: ClosedRange<Double>) -> Double {
        guard track.upperBound > track.lowerBound else { return 0 }
        return min(1, max(0, (x - track.lowerBound) / (track.upperBound - track.lowerBound)))
    }
}

/// The deck's colours and drawing. The palette is the app icon's: a night
/// ground lit blue at one corner and violet at the other, bars running from
/// indigo through violet to orchid.
enum ModernSpectrum {
    static let night = NSColor(srgbRed: 0.035, green: 0.035, blue: 0.10, alpha: 1)
    static let glowBlue = NSColor(srgbRed: 0.22, green: 0.30, blue: 0.68, alpha: 0.55)
    static let glowViolet = NSColor(srgbRed: 0.40, green: 0.16, blue: 0.50, alpha: 0.45)
    static let indigo = NSColor(srgbRed: 0.23, green: 0.18, blue: 0.84, alpha: 1)
    static let violetNS = NSColor(srgbRed: 0.48, green: 0.36, blue: 0.96, alpha: 1)
    static let orchid = NSColor(srgbRed: 0.80, green: 0.55, blue: 0.97, alpha: 1)
    static let blush = NSColor(srgbRed: 0.96, green: 0.80, blue: 0.98, alpha: 1)
    static let cap = NSColor(srgbRed: 0.88, green: 0.84, blue: 1.0, alpha: 0.9)
    /// The deck's one accent: seek fill, lit toggles, the playing row.
    static let violet = Color(nsColor: violetNS)

    /// Room the title takes at the top of the screen, and the floor under the bars.
    static let titleBand: CGFloat = 46
    static let floor: CGFloat = 12

    /// Each band, with the midpoint of every neighbouring pair between: twice
    /// the bars from the same analysis, so a wide screen reads as one spectrum.
    static func bars(_ levels: [Float]) -> [Float] {
        levels.indices.flatMap { i in
            i + 1 < levels.count ? [levels[i], (levels[i] + levels[i + 1]) / 2] : [levels[i]]
        }
    }

    static func draw(_ cg: CGContext, size: CGSize, levels: [Float], peaks: [Float]) {
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        cg.setFillColor(night.cgColor)
        cg.fill(CGRect(origin: .zero, size: size))
        for (color, center) in [(glowBlue, CGPoint(x: size.width * 0.12, y: 0)),
                                (glowViolet, CGPoint(x: size.width * 0.95, y: size.height))] {
            let glow = CGGradient(colorsSpace: space, colors: [color.cgColor, color.withAlphaComponent(0).cgColor] as CFArray,
                                  locations: [0, 1])!
            cg.drawRadialGradient(glow, startCenter: center, startRadius: 0, endCenter: center,
                                  endRadius: max(size.width, size.height) * 0.75, options: [])
        }

        let values = bars(levels), caps = bars(peaks)
        guard !values.isEmpty else { return }
        let inset: CGFloat = 14
        let base = size.height - floor
        let tallest = max(8, base - titleBand)
        let slot = (size.width - 2 * inset) / CGFloat(values.count)
        let width = max(2, (slot * 0.62).rounded())
        let radius = min(width / 2, 3)
        let path = CGMutablePath()
        for (i, level) in values.enumerated() {
            let height = max(2, CGFloat(min(1, level)) * tallest)
            let x = inset + CGFloat(i) * slot + (slot - width) / 2
            path.addRoundedRect(in: CGRect(x: x, y: base - height, width: width, height: height),
                                cornerWidth: radius, cornerHeight: radius)
        }
        // A soft violet glow under the bars, then the bars in one gradient the
        // full height of the screen, so a loud band reaches the orchid.
        cg.saveGState()
        cg.setShadow(offset: .zero, blur: 10, color: violetNS.withAlphaComponent(0.6).cgColor)
        cg.addPath(path)
        cg.setFillColor(indigo.cgColor)
        cg.fillPath()
        cg.restoreGState()
        cg.saveGState()
        cg.addPath(path)
        cg.clip()
        let fill = CGGradient(colorsSpace: space, colors: [indigo.cgColor, violetNS.cgColor, orchid.cgColor, blush.cgColor] as CFArray,
                              locations: [0, 0.4, 0.8, 1])!
        cg.drawLinearGradient(fill, start: CGPoint(x: 0, y: base), end: CGPoint(x: 0, y: base - tallest),
                              options: [.drawsAfterEndLocation])
        cg.restoreGState()

        cg.setFillColor(cap.cgColor)
        for (i, peak) in caps.enumerated() where peak > values[i] + 0.02 {
            let x = inset + CGFloat(i) * slot + (slot - width) / 2
            cg.fill(CGRect(x: x, y: base - CGFloat(min(1, peak)) * tallest - 3, width: width, height: 1.5))
        }
    }
}

/// The seek row and the radio strip: text and bars drawn by a `LiveCanvas`.
@MainActor
enum ModernLiveDrawing {
    static let timeFont = NSFont.monospacedDigitSystemFont(ofSize: AppType.pointSize(for: .caption), weight: .regular)
    static let statusFont = AppType.system(.caption)
    static let labelWidth = ceil(("00:00" as NSString).size(withAttributes: [.font: timeFont]).width)
    static let labelGap: CGFloat = 8

    /// Where the seek bar's track runs in a row `width` wide.
    static func track(width: CGFloat) -> ClosedRange<Double> {
        Double(labelWidth + labelGap)...Double(max(labelWidth + labelGap + 1, width - labelWidth - labelGap))
    }

    /// `times` nil leaves the clocks out: while the pane generates, the bar is its progress.
    static func seekRow(_ cg: CGContext, size: CGSize, times: (elapsed: Double, total: Double)?, fraction: Double, showsThumb: Bool) {
        if let times {
            let attrs: [NSAttributedString.Key: Any] = [.font: timeFont, .foregroundColor: NSColor.secondaryLabelColor]
            let y = (size.height - timeFont.boundingRectForFont.height) / 2
            (AudioClipPlayer.clock(times.elapsed) as NSString).draw(at: CGPoint(x: 0, y: y), withAttributes: attrs)
            let total = AudioClipPlayer.clock(times.total) as NSString
            total.draw(at: CGPoint(x: size.width - total.size(withAttributes: attrs).width, y: y), withAttributes: attrs)
        }

        let range = track(width: size.width)
        let left = CGFloat(range.lowerBound), length = CGFloat(range.upperBound - range.lowerBound)
        let mid = (size.height / 2).rounded()
        cg.setFillColor(NSColor.tertiaryLabelColor.cgColor)
        cg.addPath(CGPath(roundedRect: CGRect(x: left, y: mid - 2, width: length, height: 4), cornerWidth: 2, cornerHeight: 2, transform: nil))
        cg.fillPath()
        let filled = length * CGFloat(min(1, max(0, fraction)))
        if filled > 0 {
            cg.setFillColor(ModernSpectrum.violetNS.cgColor)
            cg.addPath(CGPath(roundedRect: CGRect(x: left, y: mid - 2, width: filled, height: 4), cornerWidth: 2, cornerHeight: 2, transform: nil))
            cg.fillPath()
        }
        guard showsThumb else { return }
        cg.saveGState()
        cg.setShadow(offset: CGSize(width: 0, height: 1), blur: 3, color: NSColor.black.withAlphaComponent(0.35).cgColor)
        cg.setFillColor(NSColor.white.cgColor)
        cg.fillEllipse(in: CGRect(x: left + filled - 6, y: mid - 6, width: 12, height: 12))
        cg.restoreGState()
    }

    /// The running radio's line: a violet fill grows behind its status as the
    /// track is composed; with no number yet, a block sweeps to and fro.
    static func radioStrip(_ cg: CGContext, size: CGSize, text: String, progress: Double?) {
        let box = CGRect(origin: .zero, size: size)
        cg.saveGState()
        cg.addPath(CGPath(roundedRect: box, cornerWidth: 6, cornerHeight: 6, transform: nil))
        cg.clip()
        cg.setFillColor(NSColor.quaternaryLabelColor.cgColor)
        cg.fill(box)
        cg.setFillColor(ModernSpectrum.violetNS.withAlphaComponent(0.35).cgColor)
        if let progress {
            cg.fill(CGRect(x: 0, y: 0, width: RadioStrip.fillWidth(progress: progress, width: size.width), height: size.height))
        } else {
            let x = RadioStrip.blockOffset(time: Date().timeIntervalSinceReferenceDate, width: size.width)
            cg.fill(CGRect(x: x, y: 0, width: RadioStrip.blockWidth, height: size.height))
        }
        cg.restoreGState()
        let style = NSMutableParagraphStyle()
        style.lineBreakMode = .byTruncatingTail
        let attrs: [NSAttributedString.Key: Any] = [.font: statusFont, .foregroundColor: NSColor.labelColor, .paragraphStyle: style]
        let height = statusFont.boundingRectForFont.height
        (text as NSString).draw(with: CGRect(x: 8, y: (size.height - height) / 2, width: size.width - 16, height: height),
                                options: [.usesLineFragmentOrigin, .truncatesLastVisibleLine], attributes: attrs)
    }
}

/// The Music tab's deck in the Modern look: the screen with the spectrum and
/// the title, the seek row, the transport, and AI Radio. Every control does
/// what its MLX-Amp key does.
struct ModernAudioDeck: View {
    @ObservedObject var player: AudioClipPlayer
    @ObservedObject var radio: AIRadio
    @StateObject private var analyzer = SpectrumAnalyzer()
    let path: String
    @Binding var showPlaylist: Bool
    /// The AI Radio theme typed into the radio row, and what Radio / Return does.
    @Binding var radioTheme: String
    var onRadioStart: () -> Void
    /// Stopping a running station is asked first: it discards the track being composed.
    var onRadioStop: () -> Void
    /// Overrides the title with what the pane is doing (generating, failed, empty).
    var status: MlxAmpPlayer.Status?
    var onShowLog: (() -> Void)?

    var body: some View {
        let info = status == nil ? AudioClipPlayer.info(path) : nil
        let active = status == nil && player.playingPath == path
        let playing = active && !player.isPaused
        VStack(spacing: 10) {
            screen(info: info)
            seekRow(total: info?.seconds ?? 0)
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 12) {
                    volume.frame(maxWidth: .infinity, alignment: .leading)
                    transport(active: active, playing: playing)
                    actions.frame(maxWidth: .infinity, alignment: .trailing)
                }
                VStack(spacing: 8) {
                    transport(active: active, playing: playing)
                    HStack { volume; Spacer(minLength: 8); actions }
                }
            }
            radioRow
        }
        .font(.app(.body))
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.15)))
        .task(id: path) { await analyzer.load(path) }
    }

    // MARK: - Screen

    private func screen(info: AudioClipPlayer.Info?) -> some View {
        let playingNow: @MainActor () -> Bool = { status == nil && player.playingPath == path && !player.isPaused }
        let title = titleText()
        let line = ModernAudioPlayer.infoLine(info)
        return ZStack(alignment: .topLeading) {
            LiveCanvas(shouldRun: { playingNow() || !analyzer.isSettled }, fps: { MlxAmpMotion.current.spectrum }) { cg, size in
                let frame = analyzer.step(playing: playingNow(), time: player.elapsed)
                ModernSpectrum.draw(cg, size: size, levels: frame.levels, peaks: frame.peaks)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(verbatim: title.text)
                    .font(.app(.body, weight: .semibold))
                    .foregroundStyle(title.color)
                    .lineLimit(status == nil ? 1 : 2).truncationMode(.middle)
                if !line.isEmpty {
                    Text(verbatim: line).font(.app(.caption)).foregroundStyle(.white.opacity(0.6))
                }
            }
            .padding(.horizontal, 12).padding(.top, 10)
            .help(path)
        }
        .frame(height: 132)
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func titleText() -> (text: String, color: Color) {
        switch status {
        case .busy(let message, _): return (message, .white)
        case .failed(let message): return (message, Color(red: 1, green: 0.45, blue: 0.42))
        case nil:
            let number = (player.queue.firstIndex(of: path) ?? 0) + 1
            return ("\(number). \(URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent)", .white)
        }
    }

    // MARK: - Seek

    private func seekRow(total: Double) -> some View {
        let progress: Double = { if case .busy(_, let p) = status { return p ?? 0 }; return 0 }()
        let moving: @MainActor () -> Bool = { player.playingPath == path && status == nil && total > 0 }
        return LiveCanvas(shouldRun: { moving() && !player.isPaused }, fps: { MlxAmpMotion.current.seek },
                          onPointer: { point, size in
                              guard total > 0 else { return }
                              let range = ModernLiveDrawing.track(width: size.width)
                              player.seek(to: total * ModernAudioPlayer.seekFraction(x: Double(point.x), track: range))
                          }) { cg, size in
            let now = moving()
            ModernLiveDrawing.seekRow(cg, size: size, times: status == nil ? (now ? player.elapsed : 0, total) : nil,
                                      fraction: now ? min(1, player.elapsed / total) : progress,
                                      showsThumb: now)
        }
        .frame(height: 20)
        .help(L10n.text("Seek"))
        .accessibilityElement()
        .accessibilityLabel(Text(L10n.text("Seek")))
    }

    // MARK: - Controls

    private var volume: some View {
        HStack(spacing: 6) {
            Image(systemName: ModernAudioPlayer.volumeSymbol(player.volume))
                .foregroundStyle(.secondary)
                .frame(width: 18)
            Slider(value: $player.volume, in: 0...1)
                .controlSize(.mini)
                .frame(minWidth: 40, idealWidth: 80, maxWidth: 80)
                .tint(ModernSpectrum.violet)
                .help(L10n.text("Volume"))
                .accessibilityLabel(Text(L10n.text("Volume")))
        }
    }

    private func transport(active: Bool, playing: Bool) -> some View {
        HStack(spacing: 14) {
            key("backward.fill", L10n.text("Previous")) { player.advance(-1, from: path) }
                .disabled(radio.isOn)
            key(playing ? "pause.fill" : "play.fill", L10n.text(playing ? "Pause" : "Play"), large: true) {
                if active { player.togglePause() } else if !path.isEmpty { player.play(path) }
            }
            key("stop.fill", L10n.text("Stop")) { if radio.isOn { onRadioStop() } else { player.stop() } }
            key("forward.fill", L10n.text("Next")) { if radio.isOn { radio.skip() } else { player.advance(1, from: path) } }
        }
    }

    /// The quieter keys beside the transport, the way IINA dims its side icons.
    private var actions: some View {
        HStack(spacing: 10) {
            key("shuffle", L10n.text("Shuffle"), side: true, lit: player.shuffle) { player.shuffle.toggle() }
                .disabled(radio.isOn)
            key("repeat", L10n.text("Repeat"), side: true, lit: player.autoNext) { player.autoNext.toggle() }
                .disabled(radio.isOn)
            if case .failed = status, let onShowLog {
                key("doc.text.magnifyingglass", L10n.text("Show log"), side: true, action: onShowLog)
            } else {
                key("square.and.arrow.down", L10n.text("Export as M4A"), side: true) { if !path.isEmpty { AudioExport.exportAndReveal(path) } }
            }
            key("folder", L10n.text("Reveal in Finder"), side: true) {
                if !path.isEmpty { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)]) }
            }
            key("list.bullet", L10n.text("Playlist"), side: true, lit: showPlaylist) { showPlaylist.toggle() }
        }
    }

    /// One glyph button: its title is its tooltip and what VoiceOver reads.
    private func key(_ symbol: String, _ title: String, large: Bool = false, side: Bool = false, lit: Bool = false,
                     action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label { Text(verbatim: title) } icon: {
                Image(systemName: symbol)
                    .font(large ? .app(.title, weight: .semibold) : side ? .app(.body) : .app(.title3))
            }
            .labelStyle(.iconOnly)
            .frame(minWidth: large ? 30 : 24, minHeight: 24)
            .contentShape(Rectangle())
        }
        .buttonStyle(ModernKeyStyle(lit: lit))
        .help(title)
    }

    // MARK: - AI Radio

    private var radioRow: some View {
        HStack(spacing: 8) {
            Button {
                if radio.isOn { onRadioStop() } else { onRadioStart() }
            } label: {
                Label { Text(L10n.text("Radio")).font(.app(.callout)) } icon: { Image(systemName: "dot.radiowaves.left.and.right") }
                    .foregroundStyle(radio.isOn ? AnyShapeStyle(ModernSpectrum.violet) : AnyShapeStyle(.primary))
            }
            .buttonStyle(.bordered)
            .help(L10n.text(radio.isOn ? "Stop AI Radio" : "Start AI Radio"))
            if radio.isOn {
                // The strip redraws on its own timer, never through SwiftUI.
                LiveCanvas(shouldRun: { radio.isOn }, fps: { max(MlxAmpMotion.current.title, 4) }) { cg, size in
                    ModernLiveDrawing.radioStrip(cg, size: size, text: radio.status, progress: radio.progress)
                }
                .frame(height: 22)
            } else {
                TextField("", text: $radioTheme, prompt: Text(L10n.text("AI Radio theme, e.g. mellow lo-fi for coding")))
                    .textFieldStyle(.roundedBorder)
                    .font(.app(.callout))
                    .onSubmit(onRadioStart)
            }
        }
    }
}

/// A deck key: its ink from `ModernAudioPlayer.keyInk`, on the playing row's
/// violet ground while switched on.
private struct ModernKeyStyle: ButtonStyle {
    var lit = false
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(Color(nsColor: ModernAudioPlayer.keyInk(enabled: isEnabled, lit: lit, pressed: configuration.isPressed)))
            .background(RoundedRectangle(cornerRadius: 5).fill(lit ? ModernSpectrum.violet.opacity(0.15) : Color.clear))
    }
}

/// The history list in the Modern look, for the Voice, Music and Sound tabs:
/// the rows of the shelf before MLX-Amp, numbered and timed like its playlist.
struct ModernPlaylist: View {
    let paths: [String]
    let current: String?
    /// A running AI Radio owns playback: rows only offer export and reveal.
    var locked = false
    let onPlay: (String) -> Void
    let onStop: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(L10n.text("History")).font(.app(.caption, weight: .semibold)).foregroundStyle(.secondary)
                Spacer()
                Text(verbatim: "\(paths.count)").font(.app(.caption).monospacedDigit()).foregroundStyle(.tertiary)
            }
            ScrollView {
                LazyVStack(spacing: 2) {
                    ForEach(Array(paths.enumerated()), id: \.element) { index, path in row(path, number: index + 1) }
                }
            }
        }
        .padding(10)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    /// The row plays or stops; export and reveal sit beside it, never inside a
    /// tap gesture that would swallow their clicks.
    private func row(_ path: String, number: Int) -> some View {
        let playing = current == path
        return HStack(spacing: 6) {
            Button { playing ? onStop() : onPlay(path) } label: {
                HStack(spacing: 8) {
                    // The same glyph throughout: what says it is playing is that it MOVES.
                    Image(systemName: "waveform")
                        .foregroundStyle(playing ? AnyShapeStyle(ModernSpectrum.violet) : AnyShapeStyle(.secondary))
                        .symbolEffect(.variableColor.iterative.dimInactiveLayers.nonReversing,
                                      options: .repeat(.continuous), isActive: playing)
                        .frame(width: 16)
                    Text(verbatim: "\(number).").foregroundStyle(.tertiary).monospacedDigit()
                    Text(verbatim: URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent)
                        .lineLimit(1).truncationMode(.middle)
                    Spacer(minLength: 6)
                    Text(verbatim: AudioClipPlayer.info(path).map { AudioClipPlayer.clock($0.seconds) } ?? "")
                        .foregroundStyle(.secondary).monospacedDigit()
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(locked)
            .help(path)
            Button { AudioExport.exportAndReveal(path) } label: {
                Label { Text(L10n.text("Export as M4A")) } icon: { Image(systemName: "square.and.arrow.down") }.labelStyle(.iconOnly)
            }
            .buttonStyle(.borderless)
            .help(L10n.text("Export as M4A"))
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
            } label: {
                Label { Text(L10n.text("Reveal in Finder")) } icon: { Image(systemName: "folder") }.labelStyle(.iconOnly)
            }
            .buttonStyle(.borderless)
            .help(L10n.text("Reveal in Finder"))
        }
        .font(.app(.caption))
        .padding(.vertical, 3)
        .padding(.horizontal, 6)
        .background(RoundedRectangle(cornerRadius: 5).fill(playing ? ModernSpectrum.violet.opacity(0.15) : Color.clear))
    }
}
