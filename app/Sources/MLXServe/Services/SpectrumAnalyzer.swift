import AVFoundation
import Accelerate

/// The player's spectrum, computed from the clip's own samples at the playback
/// position. `NSSound` offers no audio tap (and the mic-prompt note in
/// `AudioClipPlayer` rules out the engine that does), but a generated clip is
/// a file we can read, so "what is playing now" is an FFT of the file at
/// `currentTime`.
@MainActor
final class SpectrumAnalyzer: ObservableObject {
    static let bands = 19

    /// True once nothing is moving (stopped, bars and caps at rest): the live view
    /// stops its timer. Plain, not published: the view reads it when it draws.
    private(set) var isSettled = true
    private static let log2n = vDSP_Length(10)
    private static let n = 1 << 10

    private var samples: [Float] = []
    private var rate = 44_100.0
    private var levels = [Float](repeating: 0, count: bands)
    private var peaks = [Float](repeating: 0, count: bands)
    private let fft = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))!
    private let hann: [Float] = vDSP.window(ofType: Float.self, usingSequence: .hanningDenormalized, count: n, isHalfWindow: false)
    /// Log-spaced bin edges: low bands are single bins, high ones span many.
    private let edges: [Int] = {
        var e = [1]
        for k in 1...bands { e.append(max(e[k - 1] + 1, Int(pow(Double(n / 2 - 1), Double(k) / Double(bands)).rounded()))) }
        return e
    }()

    deinit { vDSP_destroy_fftsetup(fft) }

    func load(_ path: String) async {
        let read = await Task.detached { Self.readMono(path) }.value
        samples = read?.samples ?? []
        rate = read?.rate ?? 44_100
    }

    nonisolated private static func readMono(_ path: String) -> (samples: [Float], rate: Double)? {
        guard let file = try? AVAudioFile(forReading: URL(fileURLWithPath: path)),
              let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)),
              (try? file.read(into: buffer)) != nil, let ch = buffer.floatChannelData else { return nil }
        let frames = Int(buffer.frameLength)
        var mono = Array(UnsafeBufferPointer(start: ch[0], count: frames))
        if buffer.format.channelCount > 1 {
            mono = vDSP.multiply(0.5, vDSP.add(mono, Array(UnsafeBufferPointer(start: ch[1], count: frames))))
        }
        return (mono, file.processingFormat.sampleRate)
    }

    /// One animation frame: bars rise to the new level and fall at a fixed
    /// speed, peak caps hang a little longer.
    func step(playing: Bool, time: TimeInterval) -> (levels: [Float], peaks: [Float]) {
        let target = playing ? bandLevels(at: time) : [Float](repeating: 0, count: Self.bands)
        for i in 0..<Self.bands {
            levels[i] = max(target[i], levels[i] - 0.08)
            peaks[i] = max(levels[i], peaks[i] - 0.015)
        }
        let atRest = !playing && !levels.contains { $0 > 0 } && !peaks.contains { $0 > 0 }
        isSettled = atRest
        return (levels, peaks)
    }

    private func bandLevels(at time: TimeInterval) -> [Float] {
        let n = Self.n
        let start = Int(time * rate)
        guard start >= 0, start + n <= samples.count else { return [Float](repeating: 0, count: Self.bands) }
        let windowed = vDSP.multiply(hann, samples[start..<(start + n)])
        var real = [Float](repeating: 0, count: n / 2)
        var imag = [Float](repeating: 0, count: n / 2)
        var mags = [Float](repeating: 0, count: n / 2)
        real.withUnsafeMutableBufferPointer { rp in
            imag.withUnsafeMutableBufferPointer { ip in
                var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                windowed.withUnsafeBufferPointer { w in
                    w.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: n / 2) {
                        vDSP_ctoz($0, 2, &split, 1, vDSP_Length(n / 2))
                    }
                }
                vDSP_fft_zrip(fft, &split, 1, Self.log2n, FFTDirection(FFT_FORWARD))
                vDSP_zvmags(&split, 1, &mags, 1, vDSP_Length(n / 2))
            }
        }
        return (0..<Self.bands).map { k in
            let peak = mags[edges[k]..<max(edges[k] + 1, edges[k + 1])].max() ?? 0
            // A full-scale sine is ~54 dB here; the tilt lifts the highs, which
            // carry far less energy than the lows in music.
            let db = 10 * log10(peak + 1e-9) + Float(k) * 0.6
            return min(1, max(0, (db - 8) / 46))
        }
    }
}
