import XCTest
import SwiftUI
import AppKit
@testable import MLXCore

/// The Modern look of the generated-audio player: the setting that picks it, and what it draws.
final class ModernAudioPlayerTests: XCTestCase {

    private var suite: String!
    private var defaults: UserDefaults!

    override func setUp() {
        suite = "mlx-core-player-style-\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suite)
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: suite)
    }

    // MARK: - The setting

    func testModernIsTheDefault() {
        XCTAssertEqual(AudioPlayerStyle.defaultStyle, .modern)
        XCTAssertEqual(stored().wrappedValue, .modern)
    }

    func testTheStoredLookIsTheOneShown() {
        defaults.set(AudioPlayerStyle.mlxAmp.rawValue, forKey: InterfacePrefKey.audioPlayerStyle)
        XCTAssertEqual(stored().wrappedValue, .mlxAmp)
    }

    /// A value a later build wrote falls back to the default rather than to nothing.
    func testAnUnknownValueFallsBackToTheDefault() {
        defaults.set("cassette", forKey: InterfacePrefKey.audioPlayerStyle)
        XCTAssertEqual(stored().wrappedValue, .modern)
    }

    /// The setting the way the three views read it.
    private func stored() -> AppStorage<AudioPlayerStyle> {
        AppStorage(wrappedValue: AudioPlayerStyle.defaultStyle, InterfacePrefKey.audioPlayerStyle, store: defaults)
    }

    // MARK: - What the deck shows

    func testTheInfoLineNamesRateChannelsAndBitrate() {
        let stereo = AudioClipPlayer.Info(seconds: 120, sampleRate: 48_000, channels: 2, kbps: 1536)
        XCTAssertEqual(ModernAudioPlayer.infoLine(stereo), "48 kHz · Stereo · 1536 kbps")
        let mono = AudioClipPlayer.Info(seconds: 4, sampleRate: 44_100, channels: 1, kbps: 705)
        XCTAssertEqual(ModernAudioPlayer.infoLine(mono), "44.1 kHz · Mono · 705 kbps")
        XCTAssertEqual(ModernAudioPlayer.infoLine(nil), "")
    }

    func testTheVolumeGlyphFollowsTheLevel() {
        XCTAssertEqual(ModernAudioPlayer.volumeSymbol(0), "speaker.slash.fill")
        XCTAssertEqual(ModernAudioPlayer.volumeSymbol(0.2), "speaker.wave.1.fill")
        XCTAssertEqual(ModernAudioPlayer.volumeSymbol(0.5), "speaker.wave.2.fill")
        XCTAssertEqual(ModernAudioPlayer.volumeSymbol(0.9), "speaker.wave.3.fill")
    }

    func testADeckKeyIsGreyAccentWhenOnAndFaintWhileLocked() {
        XCTAssertEqual(ModernAudioPlayer.keyInk(enabled: true, lit: false, pressed: false), .secondaryLabelColor)
        XCTAssertEqual(ModernAudioPlayer.keyInk(enabled: true, lit: false, pressed: true), .labelColor)
        XCTAssertEqual(ModernAudioPlayer.keyInk(enabled: true, lit: true, pressed: false), .controlAccentColor)
        XCTAssertEqual(ModernAudioPlayer.keyInk(enabled: false, lit: false, pressed: false), .quaternaryLabelColor)
        XCTAssertEqual(ModernAudioPlayer.keyInk(enabled: false, lit: true, pressed: false).alphaComponent, 0.4, accuracy: 0.001)
    }

    func testAPointerOnTheSeekBarClampsToTheTrack() {
        let track = 40.0...240.0
        XCTAssertEqual(ModernAudioPlayer.seekFraction(x: 140, track: track), 0.5)
        XCTAssertEqual(ModernAudioPlayer.seekFraction(x: 10, track: track), 0)
        XCTAssertEqual(ModernAudioPlayer.seekFraction(x: 400, track: track), 1)
    }

    /// A click on the clocks under the waveform is not a seek; one anywhere on the waveform is.
    @MainActor
    func testTheSeekBarTakesClicksOnTheWaveformOnly() {
        let size = CGSize(width: 300, height: 30 + ModernDrawing.clockRow)
        let hit = ModernDrawing.seekHitRect(size: size)
        XCTAssertTrue(hit.contains(CGPoint(x: 1, y: 15)), "the start of the waveform")
        XCTAssertTrue(hit.contains(CGPoint(x: 299, y: 15)), "the end of the waveform")
        XCTAssertFalse(hit.contains(CGPoint(x: 150, y: size.height - 4)), "the clocks")
    }

    @MainActor
    func testTheRightClockCountsWhatIsLeft() {
        XCTAssertEqual(ModernAudioPlayer.remaining(elapsed: 13, total: 32), "-0:19")
        XCTAssertEqual(ModernAudioPlayer.remaining(elapsed: 40, total: 32), "-0:00")
    }

    /// A click outside the canvas's pointer region does nothing, drag included; one inside drags anywhere.
    @MainActor
    func testACanvasClickOutsideItsPointerRegionIsIgnoredWithItsDrag() throws {
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 200, height: 20), styleMask: [.borderless],
                              backing: .buffered, defer: false)
        let view = LiveCanvasView(frame: NSRect(x: 0, y: 0, width: 200, height: 20))
        window.contentView = view
        var hits: [CGFloat] = []
        view.onPointer = { point, _ in hits.append(point.x) }
        view.pointerRegion = { size in CGRect(x: 50, y: 0, width: 100, height: size.height) }
        func event(_ type: NSEvent.EventType, _ x: CGFloat) throws -> NSEvent {
            try XCTUnwrap(NSEvent.mouseEvent(with: type, location: NSPoint(x: x, y: 10), modifierFlags: [], timestamp: 0,
                                             windowNumber: window.windowNumber, context: nil, eventNumber: 0,
                                             clickCount: 1, pressure: 1))
        }
        view.mouseDown(with: try event(.leftMouseDown, 20))
        view.mouseDragged(with: try event(.leftMouseDragged, 100))
        XCTAssertEqual(hits, [])
        view.mouseDown(with: try event(.leftMouseDown, 100))
        view.mouseDragged(with: try event(.leftMouseDragged, 190))
        XCTAssertEqual(hits, [100, 190])
    }

    /// A row's cover says what a click does while hovered; otherwise it moves only while its track plays.
    func testAHistoryRowCoverGlyph() {
        let playing = ModernPlaylist.glyph(playing: true, paused: false, hovered: false, locked: false)
        XCTAssertEqual(playing.symbol, "waveform")
        XCTAssertTrue(playing.moving)
        let paused = ModernPlaylist.glyph(playing: true, paused: true, hovered: false, locked: false)
        XCTAssertEqual(paused.symbol, "pause.fill")
        XCTAssertFalse(paused.moving)
        XCTAssertEqual(ModernPlaylist.glyph(playing: true, paused: false, hovered: true, locked: false).symbol, "stop.fill")
        XCTAssertEqual(ModernPlaylist.glyph(playing: false, paused: false, hovered: true, locked: false).symbol, "play.fill")
        XCTAssertNil(ModernPlaylist.glyph(playing: false, paused: false, hovered: true, locked: true).symbol, "AI Radio locks the rows")
        XCTAssertNil(ModernPlaylist.glyph(playing: false, paused: false, hovered: false, locked: false).symbol)
    }

    // MARK: - The waveform

    func testTheEnvelopeScalesTheLoudestSliceToOne() {
        XCTAssertEqual(SpectrumAnalyzer.envelope([0, 0.25, -1, 0.5], slices: 4), [0, 0.25, 1, 0.5])
        XCTAssertEqual(SpectrumAnalyzer.envelope([3, -3, 0, 0], slices: 2), [1, 0], "a slice's loudness, not its sign")
        XCTAssertEqual(SpectrumAnalyzer.envelope([0, 0], slices: 2), [0, 0])
        XCTAssertEqual(SpectrumAnalyzer.envelope([], slices: 8), [])
    }

    /// Fewer bars than slices: each bar keeps the loudest slice it covers, so a hit never vanishes.
    func testWaveformBarsKeepTheLoudestSlice() {
        XCTAssertEqual(SpectrumAnalyzer.bars([0.1, 0.9, 0.2, 0.3], count: 2), [0.9, 0.3])
        XCTAssertEqual(SpectrumAnalyzer.bars([0.4], count: 3), [0.4, 0.4, 0.4])
        XCTAssertEqual(SpectrumAnalyzer.bars([], count: 3), [])
    }
}
