import Foundation
import AVFoundation

/// Turns a reference voice clip — either recorded in-app or picked from a file
/// — into the exact format the neural TTS models expect to clone from: a
/// **24 kHz mono 16-bit WAV**. Doing the decode/resample/encode in Swift keeps
/// AudioGen free of any extra system dependency (no ffmpeg).
///
/// `wavData(fromMonoFloat:sampleRate:)` is pure and unit-tested; the two
/// `normalizedReferenceWav` entry points wrap AVFoundation decode/resample
/// around it.
enum AudioReference {

    /// The sample rate the TTS reference path standardizes on. F5/MOSS/Qwen3
    /// reference encoders are trained at 24 kHz.
    static let targetSampleRate: Double = 24_000

    enum RefError: LocalizedError {
        case emptyRecording
        case decodeFailed(String)
        case resampleFailed

        var errorDescription: String? {
            switch self {
            case .emptyRecording: return "The reference recording was empty."
            case .decodeFailed(let f): return "Couldn't read the reference audio file: \(f)"
            case .resampleFailed: return "Couldn't convert the reference audio to 24 kHz mono."
            }
        }
    }

    // MARK: - Public entry points

    /// Normalize a clip recorded by `AudioRecorder` (float32-LE 16 kHz mono
    /// `Data`) into a temp 24 kHz mono WAV. Returns the file URL.
    static func normalizedReferenceWav(
        fromRecordedPCM data: Data,
        sourceSampleRate: Double = AudioRecorder.targetSampleRate
    ) throws -> URL {
        let samples = floatSamples(from: data)
        guard !samples.isEmpty else { throw RefError.emptyRecording }
        let resampled = try resample(samples, from: sourceSampleRate, to: targetSampleRate)
        let wav = wavData(fromMonoFloat: resampled, sampleRate: Int(targetSampleRate))
        return try writeTempWav(wav)
    }

    /// Normalize an arbitrary audio file (any container/rate AVFoundation can
    /// decode) into a temp 24 kHz mono WAV. Returns the file URL.
    static func normalizedReferenceWav(fromFile url: URL) throws -> URL {
        let mono = try decodeToMonoFloat(url)
        guard !mono.samples.isEmpty else { throw RefError.emptyRecording }
        let resampled = mono.sampleRate == targetSampleRate
            ? mono.samples
            : try resample(mono.samples, from: mono.sampleRate, to: targetSampleRate)
        let wav = wavData(fromMonoFloat: resampled, sampleRate: Int(targetSampleRate))
        return try writeTempWav(wav)
    }

    /// Music reference (ACE-Step `ref_audio`): the engine wants 48 kHz STEREO,
    /// not the 24 kHz mono voice-clone shape. Decodes any file AVFoundation
    /// reads, keeps/duplicates to two channels, resamples, crops to
    /// `maxSeconds` (the server only ever uses 30 s) and writes a temp WAV.
    static func referenceWav(fromFile url: URL, sampleRate: Double = 48_000, maxSeconds: Double = 30) throws -> URL {
        let dec = try decodeToFloat(url)
        guard !dec.channels.isEmpty, !dec.channels[0].isEmpty else { throw RefError.emptyRecording }
        let stereo = dec.channels.count >= 2 ? Array(dec.channels[0..<2]) : [dec.channels[0], dec.channels[0]]
        var out: [[Float]] = []
        for ch in stereo {
            let r = dec.sampleRate == sampleRate ? ch : try resample(ch, from: dec.sampleRate, to: sampleRate)
            out.append(r)
        }
        let interleaved = interleaveAndCrop(out, sampleRate: sampleRate, maxSeconds: maxSeconds)
        let wav = wavData(fromInterleaved: interleaved, channels: 2, sampleRate: Int(sampleRate))
        return try writeTempWav(wav)
    }

    /// Interleave equal-length channel buffers, cropped to `maxSeconds`. Pure.
    static func interleaveAndCrop(_ channels: [[Float]], sampleRate: Double, maxSeconds: Double) -> [Float] {
        let n = min(channels.map(\.count).min() ?? 0, Int(maxSeconds * sampleRate))
        var out = [Float](repeating: 0, count: n * channels.count)
        for i in 0..<n { for (c, ch) in channels.enumerated() { out[i * channels.count + c] = ch[i] } }
        return out
    }

    // MARK: - Pure WAV writer

    /// Encode mono float samples (`-1...1`) as a canonical 16-bit PCM WAV blob.
    /// Pure + deterministic so it can be unit-tested without AVFoundation.
    static func wavData(fromMonoFloat samples: [Float], sampleRate: Int) -> Data {
        wavData(fromInterleaved: samples, channels: 1, sampleRate: sampleRate)
    }

    /// Interleaved `channels`-channel float samples → canonical 16-bit PCM WAV.
    static func wavData(fromInterleaved samples: [Float], channels: UInt16, sampleRate: Int) -> Data {
        let bitsPerSample: UInt16 = 16
        let bytesPerSample = Int(bitsPerSample / 8)
        let dataBytes = samples.count * bytesPerSample
        let byteRate = UInt32(sampleRate) * UInt32(channels) * UInt32(bytesPerSample)
        let blockAlign = channels * UInt16(bytesPerSample)

        var data = Data(capacity: 44 + dataBytes)
        func append(_ s: String) { data.append(contentsOf: s.utf8) }
        func append32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func append16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }

        // RIFF chunk descriptor
        append("RIFF")
        append32(UInt32(36 + dataBytes))   // ChunkSize = 36 + Subchunk2Size
        append("WAVE")
        // "fmt " subchunk
        append("fmt ")
        append32(16)                       // Subchunk1Size for PCM
        append16(1)                        // AudioFormat = 1 (PCM)
        append16(channels)
        append32(UInt32(sampleRate))
        append32(byteRate)
        append16(blockAlign)
        append16(bitsPerSample)
        // "data" subchunk
        append("data")
        append32(UInt32(dataBytes))
        // One pass to Int16 + one append: the per-sample `Data.append` this
        // replaced froze the UI for seconds on a 10-minute source clip.
        let pcm = samples.map { Int16(clamping: Int((max(-1.0, min(1.0, $0)) * 32767.0).rounded())).littleEndian }
        pcm.withUnsafeBufferPointer { data.append(UnsafeRawBufferPointer($0).bindMemory(to: UInt8.self)) }
        return data
    }

    // MARK: - Private helpers

    /// Reinterpret raw float32-LE bytes as `[Float]` (the wire format
    /// `AudioRecorder.pcmData` produces).
    static func floatSamples(from data: Data) -> [Float] {
        let count = data.count / MemoryLayout<Float>.size
        guard count > 0 else { return [] }
        return data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self).prefix(count))
        }
    }

    /// Resample mono float samples between two rates via `AVAudioConverter`.
    private static func resample(_ samples: [Float], from src: Double, to dst: Double) throws -> [Float] {
        guard src != dst else { return samples }
        guard
            let inFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: src, channels: 1, interleaved: false),
            let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: dst, channels: 1, interleaved: false),
            let converter = AVAudioConverter(from: inFormat, to: outFormat),
            let inBuffer = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(samples.count))
        else { throw RefError.resampleFailed }

        inBuffer.frameLength = AVAudioFrameCount(samples.count)
        if let ch = inBuffer.floatChannelData { samples.withUnsafeBufferPointer { ch[0].update(from: $0.baseAddress!, count: samples.count) } }

        let ratio = dst / src
        let outCapacity = AVAudioFrameCount(Double(samples.count) * ratio) + 4096
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outCapacity) else {
            throw RefError.resampleFailed
        }

        var fed = false
        var err: NSError?
        converter.convert(to: outBuffer, error: &err) { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            fed = true; status.pointee = .haveData; return inBuffer
        }
        if err != nil { throw RefError.resampleFailed }
        let n = Int(outBuffer.frameLength)
        guard n > 0, let ch = outBuffer.floatChannelData else { throw RefError.resampleFailed }
        return Array(UnsafeBufferPointer(start: ch[0], count: n))
    }

    /// Decode a file to mono float samples at its native rate (downmixing
    /// stereo). Returns samples + the file's sample rate for a later resample.
    private static func decodeToMonoFloat(_ url: URL) throws -> (samples: [Float], sampleRate: Double) {
        let dec = try decodeToFloat(url)
        let n = dec.channels[0].count
        if dec.channels.count == 1 { return (dec.channels[0], dec.sampleRate) }
        var mono = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var sum: Float = 0
            for ch in dec.channels { sum += ch[i] }
            mono[i] = sum / Float(dec.channels.count)
        }
        return (mono, dec.sampleRate)
    }

    /// Decode a file to per-channel float samples at its native rate.
    private static func decodeToFloat(_ url: URL) throws -> (channels: [[Float]], sampleRate: Double) {
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) }
        catch { throw RefError.decodeFailed(error.localizedDescription) }

        let processing = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard frames > 0, let buffer = AVAudioPCMBuffer(pcmFormat: processing, frameCapacity: frames) else {
            throw RefError.decodeFailed("empty file")
        }
        do { try file.read(into: buffer) }
        catch { throw RefError.decodeFailed(error.localizedDescription) }

        let n = Int(buffer.frameLength)
        guard n > 0, let chData = buffer.floatChannelData else { throw RefError.decodeFailed("no PCM data") }
        let channels = max(1, Int(processing.channelCount))
        let out = (0..<channels).map { Array(UnsafeBufferPointer(start: chData[$0], count: n)) }
        return (out, processing.sampleRate)
    }

    private static func writeTempWav(_ data: Data) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-serve-tts-ref", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("ref-\(UUID().uuidString).wav")
        try data.write(to: url)
        return url
    }
}
