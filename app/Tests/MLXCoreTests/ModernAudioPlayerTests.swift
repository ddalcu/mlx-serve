import XCTest
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

    func testMlxAmpIsTheDefault() {
        XCTAssertEqual(AudioPlayerStyle.current(defaults), .mlxAmp)
    }

    func testTheStoredLookIsTheOneShown() {
        defaults.set(AudioPlayerStyle.modern.rawValue, forKey: InterfacePrefKey.audioPlayerStyle)
        XCTAssertEqual(AudioPlayerStyle.current(defaults), .modern)
    }

    /// A value a later build wrote falls back to the default rather than to nothing.
    func testAnUnknownValueFallsBackToMlxAmp() {
        defaults.set("cassette", forKey: InterfacePrefKey.audioPlayerStyle)
        XCTAssertEqual(AudioPlayerStyle.current(defaults), .mlxAmp)
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

    func testADeckKeyIsGreyVioletWhenOnAndFaintWhileLocked() {
        XCTAssertEqual(ModernAudioPlayer.keyInk(enabled: true, lit: false, pressed: false), .secondaryLabelColor)
        XCTAssertEqual(ModernAudioPlayer.keyInk(enabled: true, lit: false, pressed: true), .labelColor)
        XCTAssertEqual(ModernAudioPlayer.keyInk(enabled: true, lit: true, pressed: false), ModernSpectrum.violetNS)
        XCTAssertEqual(ModernAudioPlayer.keyInk(enabled: false, lit: false, pressed: false), .quaternaryLabelColor)
        XCTAssertEqual(ModernAudioPlayer.keyInk(enabled: false, lit: true, pressed: false).alphaComponent, 0.4, accuracy: 0.001)
    }

    func testAPointerOnTheSeekBarClampsToTheTrack() {
        let track = 40.0...240.0
        XCTAssertEqual(ModernAudioPlayer.seekFraction(x: 140, track: track), 0.5)
        XCTAssertEqual(ModernAudioPlayer.seekFraction(x: 10, track: track), 0)
        XCTAssertEqual(ModernAudioPlayer.seekFraction(x: 400, track: track), 1)
    }

    /// The analyzer's bands, with each neighbour pair's midpoint between them: a wide screen reads as one smooth spectrum.
    func testTheSpectrumDrawsEachBandAndTheMidpointsBetween() {
        XCTAssertEqual(ModernSpectrum.bars([0, 1, 0.5]), [0, 0.5, 1, 0.75, 0.5])
        XCTAssertEqual(ModernSpectrum.bars([0.3]), [0.3])
        XCTAssertEqual(ModernSpectrum.bars([]), [])
    }
}
