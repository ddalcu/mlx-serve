import Foundation
import AVFoundation

/// A subtle audible "I'm working" cue played while the model is busy but nothing
/// is being spoken (generating the first tokens, or waiting on a tool call mid
/// agent loop). Behind a protocol so the controller's start/stop logic is
/// unit-testable with a fake.
protocol LoadingCue: AnyObject {
    func start()
    func stop()
}

/// Soft sine "blip" with a smooth (click-free) envelope, low volume, repeated
/// every ~1.6 s. Uses its own `AVAudioEngine` for output — independent of the
/// mic capture engine, and the mic is off while we're loading anyway.
///
/// The audio graph is built lazily on first `start()` — never at init. Like
/// the wake chime, this object is constructed inside `VoiceModeController`
/// at APP LAUNCH, and building an engine graph there brings up the CoreAudio
/// HAL whose voice-isolation evaluation consults the mic TCC service → a
/// launch-time microphone prompt (live 2026-07-05). Laziness pinned by
/// `VoiceActivityTests.testLoadingCueBuildsNoEngineAtInitOrStop`.
final class SystemLoadingCue: LoadingCue {
    private(set) var engine: AVAudioEngine?
    private var player: AVAudioPlayerNode?
    private var buffer: AVAudioPCMBuffer?
    private var timer: Timer?
    private var running = false

    func start() {
        guard !running else { return }
        running = true
        let (engine, player, _) = ensureEngine()
        do {
            if !engine.isRunning { try engine.start() }
        } catch { running = false; return }
        player.play()
        playOnce()
        let t = Timer(timeInterval: 1.6, repeats: true) { [weak self] _ in self?.playOnce() }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    func stop() {
        guard running else { return }
        running = false
        timer?.invalidate(); timer = nil
        player?.stop()
        engine?.pause()
    }

    private func playOnce() {
        guard running, let player, let buffer else { return }
        player.scheduleBuffer(buffer, at: nil, options: [], completionHandler: nil)
    }

    private func ensureEngine() -> (AVAudioEngine, AVAudioPlayerNode, AVAudioPCMBuffer) {
        if let engine, let player, let buffer { return (engine, player, buffer) }
        let sampleRate = 44_100.0
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        let buf = SystemLoadingCue.makeBlip(format: format, sampleRate: sampleRate)
        let eng = AVAudioEngine()
        let node = AVAudioPlayerNode()
        eng.attach(node)
        eng.connect(node, to: eng.mainMixerNode, format: format)
        eng.mainMixerNode.outputVolume = 0.16   // keep it subtle
        engine = eng
        player = node
        buffer = buf
        return (eng, node, buf)
    }

    /// ~160 ms, 396 Hz sine under a Hann window so it fades in/out without clicks.
    private static func makeBlip(format: AVAudioFormat, sampleRate: Double) -> AVAudioPCMBuffer {
        let duration = 0.16, freq = 396.0
        let n = AVAudioFrameCount(sampleRate * duration)
        let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: n)!
        buf.frameLength = n
        let ch = buf.floatChannelData![0]
        let count = Int(n)
        for i in 0..<count {
            let t = Double(i) / sampleRate
            let env = 0.5 - 0.5 * cos(2 * .pi * Double(i) / Double(max(1, count - 1)))
            ch[i] = Float(sin(2 * .pi * freq * t) * env * 0.5)
        }
        return buf
    }
}
