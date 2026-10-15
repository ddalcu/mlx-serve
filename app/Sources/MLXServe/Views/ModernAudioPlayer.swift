import SwiftUI
import AppKit

// The Modern look of the generated-audio player (Settings ▸ Interface ▸ Audio
// Player). What moves draws in a `LiveCanvas` at MLX-Amp's rates, never a
// SwiftUI timeline.

/// What the Modern deck shows, as plain functions.
enum ModernAudioPlayer {
    /// The ground of the deck and the history: the panes' preview card.
    static let ground = Color.black.opacity(0.15)
    static let coverSize: CGFloat = 92

    /// "48 kHz · Stereo · 1536 kbps"; empty when the file cannot be read.
    static func infoLine(_ info: AudioClipPlayer.Info?) -> String {
        guard let info else { return "" }
        let khz = info.sampleRate / 1000
        let rate = khz == khz.rounded() ? String(Int(khz)) : String(format: "%.1f", khz)
        let channels = L10n.text(info.channels > 1 ? "Stereo" : "Mono")
        return "\(rate) kHz · \(channels) · \(info.kbps) kbps"
    }

    /// "YuE2-3B-MLX-Serve-8bit · 15:42": what made the clip and when.
    static func detailLine(_ meta: TrackMeta) -> String {
        let time = meta.date?.formatted(date: .omitted, time: .shortened)
        return [meta.model, time ?? ""].filter { !$0.isEmpty }.joined(separator: " · ")
    }

    static func volumeSymbol(_ volume: Double) -> String {
        switch volume {
        case ...0: return "speaker.slash.fill"
        case ..<0.34: return "speaker.wave.1.fill"
        case ..<0.67: return "speaker.wave.2.fill"
        default: return "speaker.wave.3.fill"
        }
    }

    /// A deck key's ink: secondary like the volume glyph, the accent while
    /// switched on, faint while AI Radio locks it.
    static func keyInk(enabled: Bool, lit: Bool, pressed: Bool) -> NSColor {
        switch (enabled, lit) {
        case (false, false): return .quaternaryLabelColor
        case (false, true): return NSColor.controlAccentColor.withAlphaComponent(0.4)
        case (true, true): return .controlAccentColor
        case (true, false): return pressed ? .labelColor : .secondaryLabelColor
        }
    }

    /// Where a pointer at `x` puts the playhead along `track`, 0...1.
    static func seekFraction(x: Double, track: ClosedRange<Double>) -> Double {
        guard track.upperBound > track.lowerBound else { return 0 }
        return min(1, max(0, (x - track.lowerBound) / (track.upperBound - track.lowerBound)))
    }

    /// "-0:19": the right-hand clock counts what is left.
    @MainActor static func remaining(elapsed: Double, total: Double) -> String {
        "-" + AudioClipPlayer.clock(max(0, total - elapsed))
    }
}

/// The deck's drawing: the cover with its spectrum, the waveform seek bar and
/// the radio strip.
@MainActor
enum ModernDrawing {
    static let timeFont = NSFont.monospacedDigitSystemFont(ofSize: AppType.pointSize(for: .caption), weight: .regular)
    static let statusFont = AppType.system(.caption)
    /// The clocks' row under the waveform.
    static let clockRow: CGFloat = 14
    /// One waveform bar and the gap after it.
    static let barPitch: CGFloat = 3

    /// The cover: the clip's two hues, the spectrum in white along its foot,
    /// and a note while nothing sounds.
    static func cover(_ cg: CGContext, size: CGSize, colors: (NSColor, NSColor), levels: [Float], resting: Bool) {
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let radius = size.width * 0.1
        cg.saveGState()
        cg.addPath(CGPath(roundedRect: CGRect(origin: .zero, size: size), cornerWidth: radius, cornerHeight: radius, transform: nil))
        cg.clip()
        let wash = CGGradient(colorsSpace: space, colors: [colors.0.cgColor, colors.1.cgColor] as CFArray, locations: [0, 1])!
        cg.drawLinearGradient(wash, start: .zero, end: CGPoint(x: size.width, y: size.height), options: [])
        let shine = CGGradient(colorsSpace: space, colors: [NSColor.white.withAlphaComponent(0.28).cgColor,
                                                             NSColor.white.withAlphaComponent(0).cgColor] as CFArray,
                               locations: [0, 1])!
        let corner = CGPoint(x: size.width * 0.2, y: size.height * 0.1)
        cg.drawRadialGradient(shine, startCenter: corner, startRadius: 0, endCenter: corner, endRadius: size.width, options: [])

        if resting {
            let config = NSImage.SymbolConfiguration(pointSize: size.width * 0.34, weight: .medium)
                .applying(NSImage.SymbolConfiguration(paletteColors: [NSColor.white.withAlphaComponent(0.85)]))
            if let note = NSImage(systemSymbolName: "music.note", accessibilityDescription: nil)?.withSymbolConfiguration(config) {
                let s = note.size
                note.draw(in: CGRect(x: (size.width - s.width) / 2, y: (size.height - s.height) / 2, width: s.width, height: s.height),
                          from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: true, hints: nil)
            }
        } else if !levels.isEmpty {
            let inset = size.width * 0.12
            let slot = (size.width - 2 * inset) / CGFloat(levels.count)
            let width = max(1.5, slot * 0.62)
            let base = size.height - inset, tallest = size.height * 0.6
            let path = CGMutablePath()
            for (i, level) in levels.enumerated() {
                let height = max(2, CGFloat(min(1, level)) * tallest)
                path.addRoundedRect(in: CGRect(x: inset + CGFloat(i) * slot + (slot - width) / 2, y: base - height, width: width, height: height),
                                    cornerWidth: width / 2, cornerHeight: width / 2)
            }
            cg.addPath(path)
            cg.setFillColor(NSColor.white.withAlphaComponent(0.9).cgColor)
            cg.fillPath()
        }
        cg.restoreGState()
    }

    /// Where a click seeks: the waveform, not the clocks under it.
    static func seekHitRect(size: CGSize) -> CGRect {
        CGRect(x: 0, y: 0, width: size.width, height: max(1, size.height - clockRow))
    }

    static func barCount(width: CGFloat) -> Int { max(1, Int((width + 1) / barPitch)) }

    /// The clip's outline as the seek bar: played bars in the accent, the rest
    /// faint, a flat track until the outline is read. `times` nil leaves the
    /// clocks out.
    static func waveform(_ cg: CGContext, size: CGSize, envelope: [Float], fraction: Double, times: (elapsed: Double, total: Double)?) {
        let band = seekHitRect(size: size).height
        let mid = (band / 2).rounded()
        let bars = SpectrumAnalyzer.bars(envelope, count: barCount(width: size.width))
        if bars.isEmpty {
            cg.setFillColor(NSColor.tertiaryLabelColor.cgColor)
            cg.fill(CGRect(x: 0, y: mid - 1, width: size.width, height: 2))
            cg.setFillColor(NSColor.controlAccentColor.cgColor)
            cg.fill(CGRect(x: 0, y: mid - 1, width: size.width * CGFloat(fraction), height: 2))
        } else {
            let lit = CGMutablePath(), dim = CGMutablePath()
            let width = barPitch - 1
            for (i, level) in bars.enumerated() {
                let height = max(2, CGFloat(level) * band)
                ((Double(i) + 0.5) / Double(bars.count) <= fraction ? lit : dim)
                    .addRoundedRect(in: CGRect(x: CGFloat(i) * barPitch, y: mid - height / 2, width: width, height: height),
                                    cornerWidth: width / 2, cornerHeight: width / 2)
            }
            for (path, color) in [(dim, NSColor.tertiaryLabelColor), (lit, NSColor.controlAccentColor)] {
                cg.addPath(path)
                cg.setFillColor(color.cgColor)
                cg.fillPath()
            }
        }

        guard let times else { return }
        let attrs: [NSAttributedString.Key: Any] = [.font: timeFont, .foregroundColor: NSColor.secondaryLabelColor]
        let y = size.height - timeFont.boundingRectForFont.height
        (AudioClipPlayer.clock(times.elapsed) as NSString).draw(at: CGPoint(x: 0, y: y), withAttributes: attrs)
        let left = ModernAudioPlayer.remaining(elapsed: times.elapsed, total: times.total) as NSString
        left.draw(at: CGPoint(x: size.width - left.size(withAttributes: attrs).width, y: y), withAttributes: attrs)
    }

    /// The pane's progress in the waveform's place while it generates; with no
    /// number yet, a block sweeps to and fro.
    static func progress(_ cg: CGContext, size: CGSize, fraction: Double?) {
        let mid = (seekHitRect(size: size).height / 2).rounded()
        cg.setFillColor(NSColor.quaternaryLabelColor.cgColor)
        cg.addPath(CGPath(roundedRect: CGRect(x: 0, y: mid - 2, width: size.width, height: 4), cornerWidth: 2, cornerHeight: 2, transform: nil))
        cg.fillPath()
        let bar = fraction.map { CGRect(x: 0, y: mid - 2, width: size.width * CGFloat(min(1, max(0, $0))), height: 4) }
            ?? CGRect(x: RadioStrip.blockOffset(time: Date().timeIntervalSinceReferenceDate, width: size.width), y: mid - 2,
                      width: RadioStrip.blockWidth, height: 4)
        cg.setFillColor(NSColor.controlAccentColor.cgColor)
        cg.addPath(CGPath(roundedRect: bar, cornerWidth: 2, cornerHeight: 2, transform: nil))
        cg.fillPath()
    }

    /// The running radio's line: an accent fill grows behind its status as the
    /// track is composed; with no number yet, a block sweeps to and fro.
    static func radioStrip(_ cg: CGContext, size: CGSize, text: String, progress: Double?) {
        let box = CGRect(origin: .zero, size: size)
        cg.saveGState()
        cg.addPath(CGPath(roundedRect: box, cornerWidth: 6, cornerHeight: 6, transform: nil))
        cg.clip()
        cg.setFillColor(NSColor.quaternaryLabelColor.cgColor)
        cg.fill(box)
        cg.setFillColor(NSColor.controlAccentColor.withAlphaComponent(0.3).cgColor)
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

/// The Music tab's deck in the Modern look: the cover and the title, the
/// waveform seek bar, the transport, and AI Radio. Every control does what its
/// MLX-Amp key does, through `DeckActions`.
struct ModernAudioDeck: View {
    @ObservedObject var player: AudioClipPlayer
    @ObservedObject var radio: AIRadio
    @StateObject private var analyzer = SpectrumAnalyzer()
    /// The path whose outline the analyzer holds; setting it redraws the waveform.
    @State private var outlined = ""
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
        let deck = DeckActions(player: player, radio: radio, path: path, status: status,
                               onRadioStart: onRadioStart, onRadioStop: onRadioStop)
        VStack(spacing: 0) {
            VStack(spacing: 14) {
                HStack(alignment: .top, spacing: 14) {
                    cover
                    header(info: info, deck: deck)
                }
                seekBar(total: info?.seconds ?? 0)
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 12) {
                        volume.frame(maxWidth: .infinity, alignment: .leading)
                        transport(deck)
                        playlistKey.frame(maxWidth: .infinity, alignment: .trailing)
                    }
                    VStack(spacing: 10) {
                        transport(deck)
                        HStack { volume; Spacer(minLength: 0); playlistKey }
                    }
                }
            }
            .padding(14)
            Divider()
            radioRow(deck).padding(.horizontal, 14).padding(.vertical, 10)
        }
        .font(.app(.body))
        .background(RoundedRectangle(cornerRadius: 10).fill(ModernAudioPlayer.ground))
        .task(id: path) {
            await analyzer.load(path)
            outlined = path
        }
    }

    private var playingNow: @MainActor () -> Bool {
        { [player, path, status] in status == nil && player.playingPath == path && !player.isPaused }
    }

    // MARK: - Cover and title

    private var cover: some View {
        let colors = status == nil && !path.isEmpty ? TrackMeta.coverColors(path)
            : (NSColor.controlAccentColor, NSColor.controlAccentColor.blended(withFraction: 0.5, of: .black) ?? .black)
        let playing = playingNow
        return LiveCanvas(shouldRun: { playing() || !analyzer.isSettled }, fps: { MlxAmpMotion.current.spectrum }) { cg, size in
            let frame = analyzer.step(playing: playing(), time: player.elapsed)
            ModernDrawing.cover(cg, size: size, colors: colors, levels: frame.levels, resting: analyzer.isSettled && !playing())
        }
        .frame(width: ModernAudioPlayer.coverSize, height: ModernAudioPlayer.coverSize)
        .shadow(color: .black.opacity(0.25), radius: 6, y: 3)
        .accessibilityHidden(true)
    }

    private func header(info: AudioClipPlayer.Info?, deck: DeckActions) -> some View {
        let meta = status == nil && !path.isEmpty ? TrackMeta.of(path) : nil
        let format = ModernAudioPlayer.infoLine(info)
        return HStack(alignment: .top, spacing: 8) {
            VStack(alignment: .leading, spacing: 4) {
                switch status {
                case .busy(let message, _):
                    Text(verbatim: message).font(.app(.title3, weight: .semibold)).lineLimit(2)
                case .failed(let message):
                    Text(verbatim: message).font(.app(.callout)).foregroundStyle(.red).lineLimit(3)
                    if let onShowLog {
                        Button(L10n.text("Show log"), action: onShowLog).controlSize(.small)
                    }
                case nil:
                    Text(verbatim: meta?.title ?? "").font(.app(.title3, weight: .semibold)).lineLimit(2)
                    if let meta, !ModernAudioPlayer.detailLine(meta).isEmpty {
                        Text(verbatim: ModernAudioPlayer.detailLine(meta)).font(.app(.callout)).foregroundStyle(.secondary).lineLimit(1)
                    }
                    if !format.isEmpty {
                        Text(verbatim: format).font(.app(.footnote)).foregroundStyle(.tertiary).lineLimit(1)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .help(status == nil ? path : "")
            moreMenu(deck)
        }
    }

    /// The file's own actions, and Stop, which a deck with Pause rarely needs.
    private func moreMenu(_ deck: DeckActions) -> some View {
        let noFile = path.isEmpty || status != nil
        return Menu {
            Button(L10n.text("Export as M4A")) { deck.press(.export) }.disabled(noFile)
            Button(L10n.text("Reveal in Finder")) { deck.press(.reveal) }.disabled(noFile)
            Divider()
            Button(L10n.text(radio.isOn ? "Stop AI Radio" : "Stop")) { deck.press(.stop) }
                .disabled(!radio.isOn && !deck.active)
        } label: {
            Label { Text(L10n.text("More")) } icon: { Image(systemName: "ellipsis.circle") }.labelStyle(.iconOnly)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help(L10n.text("More"))
    }

    // MARK: - Seek

    private func seekBar(total: Double) -> some View {
        let generating: Double?? = { if case .busy(_, let p) = status { return .some(p) }; return .none }()
        let empty = path.isEmpty || status != nil
        let envelope = outlined == path && !empty ? analyzer.envelope : []
        let moving: @MainActor () -> Bool = { [player, path, status] in player.playingPath == path && status == nil && total > 0 }
        return LiveCanvas(shouldRun: { (moving() && !player.isPaused) || generating == .some(nil) },
                          fps: { MlxAmpMotion.current.seek * 2 },
                          onPointer: { point, size in
                              guard total > 0, !empty else { return }
                              player.seek(to: total * ModernAudioPlayer.seekFraction(x: Double(point.x), track: 0...Double(size.width)))
                          },
                          pointerRegion: { ModernDrawing.seekHitRect(size: $0) }) { cg, size in
            if let generating {
                ModernDrawing.progress(cg, size: size, fraction: generating)
                return
            }
            let now = moving()
            ModernDrawing.waveform(cg, size: size, envelope: envelope, fraction: now ? min(1, player.elapsed / total) : 0,
                                   times: empty ? nil : (now ? player.elapsed : 0, total))
        }
        .frame(height: 30 + ModernDrawing.clockRow)
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
                .frame(minWidth: 40, idealWidth: 90, maxWidth: 90)
                .help(L10n.text("Volume"))
                .accessibilityLabel(Text(L10n.text("Volume")))
        }
    }

    private func transport(_ deck: DeckActions) -> some View {
        HStack(spacing: 16) {
            key(.shuffle, deck, "shuffle", L10n.text("Shuffle"), lit: player.shuffle)
            key(.previous, deck, "backward.fill", L10n.text("Previous"), large: true)
            playButton(deck)
            key(.next, deck, "forward.fill", L10n.text("Next"), large: true)
            key(.repeatAll, deck, "repeat", L10n.text("Repeat"), help: L10n.text("Play the next track when one ends"),
                lit: player.autoNext)
        }
    }

    private func playButton(_ deck: DeckActions) -> some View {
        let title = L10n.text(deck.playing ? "Pause" : "Play")
        let idle = path.isEmpty && !deck.active
        return Button { deck.press(.playPause) } label: {
            Label { Text(verbatim: title) } icon: {
                Image(systemName: deck.playing ? "pause.fill" : "play.fill")
                    .font(.app(.title2))
                    .foregroundStyle(Color(nsColor: .windowBackgroundColor))
                    .frame(width: 40, height: 40)
                    .background(Circle().fill(Color.primary))
            }
            .labelStyle(.iconOnly)
            .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .disabled(idle)
        .opacity(idle ? 0.4 : 1)
        .help(title)
    }

    private var playlistKey: some View {
        key("list.bullet", L10n.text("Playlist"), lit: showPlaylist) { showPlaylist.toggle() }
    }

    /// A deck key: what it does and whether the radio locks it come from `DeckActions`.
    private func key(_ deckKey: DeckKey, _ deck: DeckActions, _ symbol: String, _ title: String, help: String? = nil,
                     large: Bool = false, lit: Bool = false) -> some View {
        key(symbol, title, help: help, large: large, lit: lit) { deck.press(deckKey) }
            .disabled(deck.isLocked(deckKey))
    }

    /// One glyph button: its title is what VoiceOver reads and, unless `help` says more, its tooltip.
    private func key(_ symbol: String, _ title: String, help: String? = nil, large: Bool = false,
                     lit: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label { Text(verbatim: title) } icon: {
                Image(systemName: symbol).font(large ? .app(.title2) : .app(.body, weight: .medium))
            }
            .labelStyle(.iconOnly)
            .frame(minWidth: 28, minHeight: 28)
            .contentShape(Rectangle())
        }
        .buttonStyle(ModernKeyStyle(lit: lit))
        .help(help ?? title)
        .accessibilityAddTraits(lit ? .isSelected : [])
    }

    // MARK: - AI Radio

    private func radioRow(_ deck: DeckActions) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "dot.radiowaves.left.and.right")
                .foregroundStyle(radio.isOn ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.secondary))
                .symbolEffect(.variableColor.iterative.dimInactiveLayers.nonReversing, options: .repeat(.continuous), isActive: radio.isOn)
                .accessibilityHidden(true)
            if radio.isOn {
                // The strip redraws on its own timer, never through SwiftUI.
                LiveCanvas(shouldRun: { radio.isOn }, fps: { max(MlxAmpMotion.current.title, 4) }) { cg, size in
                    ModernDrawing.radioStrip(cg, size: size, text: radio.status, progress: radio.progress)
                }
                .frame(height: 22)
            } else {
                TextField("", text: $radioTheme, prompt: Text(L10n.text("AI Radio theme, e.g. mellow lo-fi for coding")))
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(onRadioStart)
            }
            Button(L10n.text(radio.isOn ? "Stop AI Radio" : "Start AI Radio")) { deck.press(.radio) }
                .buttonStyle(.bordered)
                .fixedSize()
        }
        .font(.app(.callout))
    }
}

/// A deck key: its ink from `ModernAudioPlayer.keyInk`, on an accent ground while switched on.
private struct ModernKeyStyle: ButtonStyle {
    var lit = false
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(Color(nsColor: ModernAudioPlayer.keyInk(enabled: isEnabled, lit: lit, pressed: configuration.isPressed)))
            .background(RoundedRectangle(cornerRadius: 6).fill(lit ? Color.accentColor.opacity(0.15) : Color.clear))
    }
}

/// A clip's cover in the history: its two hues, and over them what a click
/// does or that it is playing.
private struct CoverTile: View {
    let path: String
    var glyph: String?
    var moving = false

    var body: some View {
        let (a, b) = TrackMeta.coverColors(path)
        RoundedRectangle(cornerRadius: 6)
            .fill(LinearGradient(colors: [Color(nsColor: a), Color(nsColor: b)], startPoint: .topLeading, endPoint: .bottomTrailing))
            .overlay {
                if let glyph {
                    ZStack {
                        RoundedRectangle(cornerRadius: 6).fill(.black.opacity(0.3))
                        Image(systemName: glyph)
                            .font(.app(.callout, weight: .semibold))
                            .foregroundStyle(.white)
                            .symbolEffect(.variableColor.iterative.dimInactiveLayers.nonReversing,
                                          options: .repeat(.continuous), isActive: moving)
                    }
                }
            }
            .frame(width: 32, height: 32)
    }
}

/// The history list in the Modern look, for the Voice, Music and Sound tabs:
/// clips grouped by day, each with its cover, title, model, time and length.
struct ModernPlaylist: View {
    let paths: [String]
    let current: String?
    var paused = false
    /// A running AI Radio owns playback: rows only offer export and reveal.
    var locked = false
    let onPlay: (String) -> Void
    let onStop: () -> Void
    @State private var hovered: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .firstTextBaseline) {
                Text(L10n.text("History")).font(.app(.headline))
                Text(verbatim: "\(paths.count)").font(.app(.callout).monospacedDigit()).foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.horizontal, 14).padding(.top, 12).padding(.bottom, 4)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(TrackDay.group(paths, date: { TrackMeta.date(stem: Self.stem($0)) })) { day in
                        Text(verbatim: day.label)
                            .font(.app(.caption, weight: .semibold))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 8).padding(.top, 10).padding(.bottom, 4)
                        ForEach(day.paths, id: \.self) { row($0) }
                    }
                }
                .padding(.horizontal, 6).padding(.bottom, 8)
            }
        }
        .background(RoundedRectangle(cornerRadius: 10).fill(ModernAudioPlayer.ground))
    }

    static func stem(_ path: String) -> String {
        ((path as NSString).lastPathComponent as NSString).deletingPathExtension
    }

    /// What a row's cover shows: what a click would do while hovered, else
    /// that it plays. The waveform moves only while its track plays.
    static func glyph(playing: Bool, paused: Bool, hovered: Bool, locked: Bool) -> (symbol: String?, moving: Bool) {
        if hovered && !locked { return (playing ? "stop.fill" : "play.fill", false) }
        if playing { return paused ? ("pause.fill", false) : ("waveform", true) }
        return (nil, false)
    }

    /// The row plays or stops; export and reveal sit beside it, never inside a
    /// tap gesture that would swallow their clicks.
    private func row(_ path: String) -> some View {
        let meta = TrackMeta.of(path)
        let playing = current == path
        let hover = hovered == path
        let glyph = Self.glyph(playing: playing, paused: paused, hovered: hover, locked: locked)
        let detail = ModernAudioPlayer.detailLine(meta)
        return HStack(spacing: 8) {
            Button { playing ? onStop() : onPlay(path) } label: {
                HStack(spacing: 10) {
                    CoverTile(path: path, glyph: glyph.symbol, moving: glyph.moving)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(verbatim: meta.title)
                            .font(.app(.body, weight: playing ? .semibold : .regular))
                            .foregroundStyle(playing ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.primary))
                            .lineLimit(1)
                        if !detail.isEmpty {
                            Text(verbatim: detail).font(.app(.footnote)).foregroundStyle(.secondary).lineLimit(1)
                        }
                    }
                    Spacer(minLength: 4)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(locked)
            .accessibilityLabel(Text(verbatim: meta.title))
            if hover || playing {
                rowAction("square.and.arrow.down", L10n.text("Export as M4A")) { AudioExport.exportAndReveal(path) }
                rowAction("folder", L10n.text("Reveal in Finder")) {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                }
            }
            Text(verbatim: AudioClipPlayer.info(path).map { AudioClipPlayer.clock($0.seconds) } ?? "")
                .font(.app(.callout).monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 5).padding(.horizontal, 8)
        .background(RoundedRectangle(cornerRadius: 7)
            .fill(playing ? Color.accentColor.opacity(0.14) : hover ? Color.primary.opacity(0.06) : Color.clear))
        .onHover { inside in
            if inside { hovered = path } else if hovered == path { hovered = nil }
        }
        .help(path)
    }

    private func rowAction(_ symbol: String, _ title: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label { Text(verbatim: title) } icon: { Image(systemName: symbol) }.labelStyle(.iconOnly)
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help(title)
    }
}
