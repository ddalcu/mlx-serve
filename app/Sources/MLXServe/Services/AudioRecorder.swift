import Foundation
import AVFoundation

/// Captures microphone audio and resamples it on the fly to the exact PCM the
/// Gemma 4 12B unified audio embedder expects: float32-LE 16 kHz mono. Stopping
/// yields a `Data` blob ready to wrap in a `ChatAudio` — the same format
/// `AudioPreprocessor` produces from files, so the rest of the pipeline is shared.
final class AudioRecorder: ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var duration: TimeInterval = 0   // seconds captured
    @Published private(set) var level: Float = 0             // 0…1 RMS, for a meter

    static let targetSampleRate: Double = 16_000
    static let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: AudioRecorder.targetSampleRate,
        channels: 1, interleaved: false
    )!

    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private let lock = NSLock()
    private var samples: [Float] = []   // accumulated 16 kHz mono float32
    private var captured = 0
    /// When set before `start()`, samples go here instead of into memory, and
    /// `stop()` returns nil: a long recording that streams to disk.
    var sink: ((UnsafeBufferPointer<Float>) -> Void)?

    /// Ask for mic permission (returns the granted state). Safe to call repeatedly.
    static func requestPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return true
        case .notDetermined:
            return await withCheckedContinuation { cont in
                AVCaptureDevice.requestAccess(for: .audio) { cont.resume(returning: $0) }
            }
        default: return false
        }
    }

    /// Begin capturing. Throws if the engine can't start (no input device, etc.).
    func start() throws {
        guard !isRecording else { return }
        lock.lock(); samples.removeAll(keepingCapacity: true); captured = 0; lock.unlock()
        DispatchQueue.main.async { self.duration = 0; self.level = 0 }

        let input = engine.inputNode
        let inFormat = input.inputFormat(forBus: 0)
        guard inFormat.sampleRate > 0, inFormat.channelCount > 0 else {
            throw NSError(domain: "AudioRecorder", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No usable audio input device."])
        }
        converter = AVAudioConverter(from: inFormat, to: Self.targetFormat)

        input.installTap(onBus: 0, bufferSize: 4096, format: inFormat) { [weak self] buffer, _ in
            self?.appendBuffer(buffer)
        }
        engine.prepare()
        try engine.start()
        DispatchQueue.main.async { self.isRecording = true }
    }

    /// Stop capturing and return the recorded PCM (float32-LE 16 kHz mono), or
    /// nil if nothing was captured.
    func stop() -> Data? {
        guard isRecording || engine.isRunning else { return nil }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        converter = nil
        DispatchQueue.main.async { self.isRecording = false; self.level = 0 }

        lock.lock(); let snapshot = samples; samples.removeAll(); lock.unlock()
        guard !snapshot.isEmpty else { return nil }
        return Self.pcmData(from: snapshot)
    }

    /// Discard an in-progress recording.
    func cancel() { _ = stop() }

    // Audio render thread: resample → accumulate → publish meter/duration.
    private func appendBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let converter, let out = Self.resample(buffer, with: converter), let ch = out.floatChannelData else { return }
        let n = Int(out.frameLength)

        lock.lock()
        let chunk = UnsafeBufferPointer(start: ch[0], count: n)
        if let sink { sink(chunk) } else { samples.append(contentsOf: chunk) }
        captured += n
        let total = captured
        lock.unlock()

        var sumSq: Float = 0
        for i in 0..<n { sumSq += ch[0][i] * ch[0][i] }
        let rms = (n > 0) ? (sumSq / Float(n)).squareRoot() : 0
        DispatchQueue.main.async {
            self.level = min(1, rms * 6)
            self.duration = Double(total) / Self.targetSampleRate
        }
    }

    /// One buffer through `converter` into `targetFormat`; nil when nothing came out.
    static func resample(_ buffer: AVAudioPCMBuffer, with converter: AVAudioConverter) -> AVAudioPCMBuffer? {
        guard buffer.frameLength > 0 else { return nil }
        let ratio = targetSampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1024
        guard let out = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else { return nil }

        // Streaming convert: feed this buffer once, then report no-data-now so the
        // stateful converter keeps its resampler continuity across taps.
        var fed = false
        var err: NSError?
        converter.convert(to: out, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true; status.pointee = .haveData; return buffer
        }
        return err == nil && out.frameLength > 0 ? out : nil
    }

    /// Serialize float samples to little-endian float32 bytes (the wire format).
    /// Pure + testable; shared by `stop()`.
    static func pcmData(from samples: [Float]) -> Data {
        samples.withUnsafeBufferPointer { Data(buffer: $0) }
    }
}
