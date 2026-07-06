import Foundation
import AVFoundation

/// A short "I'm listening" earcon played when the wake word ("Hey Loki") is
/// heard, so the user gets audible confirmation the assistant is now engaged —
/// the way Siri/Google chime on their hotword. Behind a protocol so the
/// controller's trigger logic is unit-testable with a fake (no real audio).
protocol WakeChime: AnyObject {
    func play()
}

/// Two short ascending sine notes (E5 → B5) with click-free Hann envelopes —
/// a friendly rising "bdp" distinct from the loading cue's single low blip.
/// Owns its own `AVAudioEngine`, independent of mic capture.
///
/// The WHOLE audio graph is built lazily on first `play()` — never at init.
/// The chime is constructed inside `VoiceModeController`, which the menu-bar
/// tray forces at APP LAUNCH; building an engine graph there brings up the
/// CoreAudio HAL, whose voice-isolation evaluation consults the microphone
/// TCC service → the app prompted for the mic at launch instead of at
/// voice-mode enable (live 2026-07-05; see the CLAUDE.md gotcha). Once built,
/// the engine is kept warm so repeats are instant. Laziness pinned by
/// `VoiceActivityTests.testWakeChimeBuildsNoEngineAtInit`.
final class SystemWakeChime: WakeChime {
    private(set) var engine: AVAudioEngine?
    private var player: AVAudioPlayerNode?
    private var buffer: AVAudioPCMBuffer?

    func play() {
        let (engine, player, buffer) = ensureEngine()
        do {
            if !engine.isRunning { try engine.start() }
        } catch { return }
        if !player.isPlaying { player.play() }
        // `.interrupts` so a rapid re-trigger restarts cleanly rather than queueing.
        player.scheduleBuffer(buffer, at: nil, options: .interrupts, completionHandler: nil)
    }

    private func ensureEngine() -> (AVAudioEngine, AVAudioPlayerNode, AVAudioPCMBuffer) {
        if let engine, let player, let buffer { return (engine, player, buffer) }
        let sampleRate = 44_100.0
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        let buf = SystemWakeChime.makeChime(format: format, sampleRate: sampleRate)
        let eng = AVAudioEngine()
        let node = AVAudioPlayerNode()
        eng.attach(node)
        eng.connect(node, to: eng.mainMixerNode, format: format)
        eng.mainMixerNode.outputVolume = 0.3
        engine = eng
        player = node
        buffer = buf
        return (eng, node, buf)
    }

    /// Two 85 ms notes (E5 → B5) under Hann windows so each fades in/out clickless.
    private static func makeChime(format: AVAudioFormat, sampleRate: Double) -> AVAudioPCMBuffer {
        let noteDur = 0.085
        let notes: [Double] = [659.25, 987.77]      // E5 → B5
        let perNote = Int(sampleRate * noteDur)
        let total = perNote * notes.count
        let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(total))!
        buf.frameLength = AVAudioFrameCount(total)
        let ch = buf.floatChannelData![0]
        for (noteIdx, freq) in notes.enumerated() {
            for i in 0..<perNote {
                let t = Double(i) / sampleRate
                let env = 0.5 - 0.5 * cos(2 * .pi * Double(i) / Double(max(1, perNote - 1)))
                ch[noteIdx * perNote + i] = Float(sin(2 * .pi * freq * t) * env * 0.5)
            }
        }
        return buf
    }
}
