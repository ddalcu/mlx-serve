import XCTest
import Foundation

// =============================================================================
// MARK: - Replicas from MediaGen.swift / AudioGenService.swift / AudioReference.swift
//
// MLXCore is an executable target — tests can't @testable-import it, so the
// pure-logic pieces are copied here verbatim and asserted on (same pattern as
// MediaGenTests / HFSearchTests). Keep these in sync with the shipping code;
// the canonical source is the app file.
// =============================================================================

// --- AudioModelPreset catalog replica ---

private struct AudioPresetReplica {
    let id: String
    let repo: String
    let approxRAMGB: Int
}

private let audioPresetsReplica: [AudioPresetReplica] = [
    .init(id: "mlx-audio/qwen3-tts-0.6b-base-8bit", repo: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit", approxRAMGB: 3),
    .init(id: "mlx-audio/qwen3-tts-0.6b-base",      repo: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16", approxRAMGB: 4),
    .init(id: "mlx-audio/qwen3-tts-1.7b-base-8bit", repo: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit", approxRAMGB: 5),
    .init(id: "mlx-audio/qwen3-tts-1.7b-base",      repo: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16", approxRAMGB: 8),
]

// --- AudioGenService.buildArgs replica ---

private struct AudioRequestReplica {
    var repo: String
    var text: String
    var refAudioPath: String?
    var refText: String = ""
    var speed: Double = 1.0
    var temperature: Double = 0.7
}

private func audioBuildArgsReplica(_ r: AudioRequestReplica, outputPath: String) -> [String] {
    var args = [
        "--repo", r.repo,
        "--text", r.text,
        "--speed", String(r.speed),
        "--temperature", String(r.temperature),
        "--output", outputPath,
    ]
    if let ref = r.refAudioPath, !ref.isEmpty {
        args.append(contentsOf: ["--ref-audio", ref])
        let t = r.refText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.isEmpty { args.append(contentsOf: ["--ref-text", t]) }
    }
    return args
}

// --- makeOutputPath slug replica ---

private func audioSlugReplica(_ text: String) -> String {
    String(text
        .lowercased()
        .replacingOccurrences(of: #"[^a-z0-9]+"#, with: "-", options: .regularExpression)
        .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        .prefix(40))
}

// --- AudioReference.wavData replica ---

private func wavDataReplica(fromMonoFloat samples: [Float], sampleRate: Int) -> Data {
    let channels: UInt16 = 1
    let bitsPerSample: UInt16 = 16
    let bytesPerSample = Int(bitsPerSample / 8)
    let dataBytes = samples.count * bytesPerSample
    let byteRate = UInt32(sampleRate) * UInt32(channels) * UInt32(bytesPerSample)
    let blockAlign = channels * UInt16(bytesPerSample)

    var data = Data(capacity: 44 + dataBytes)
    func append(_ s: String) { data.append(contentsOf: s.utf8) }
    func append32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
    func append16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }

    append("RIFF"); append32(UInt32(36 + dataBytes)); append("WAVE")
    append("fmt "); append32(16); append16(1); append16(channels)
    append32(UInt32(sampleRate)); append32(byteRate); append16(blockAlign); append16(bitsPerSample)
    append("data"); append32(UInt32(dataBytes))
    for s in samples {
        let clamped = max(-1.0, min(1.0, s))
        let v = Int16(clamping: Int((clamped * 32767.0).rounded()))
        append16(UInt16(bitPattern: v))
    }
    return data
}

private func floatSamplesReplica(from data: Data) -> [Float] {
    let count = data.count / MemoryLayout<Float>.size
    guard count > 0 else { return [] }
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self).prefix(count)) }
}

final class AudioGenTests: XCTestCase {

    // MARK: - Preset catalog

    func testAudioPresetCatalogIsCloningFirstAndWellFormed() {
        XCTAssertFalse(audioPresetsReplica.isEmpty, "Audio catalog must not be empty")
        // Unique ids.
        XCTAssertEqual(Set(audioPresetsReplica.map(\.id)).count, audioPresetsReplica.count,
                       "Preset ids must be unique")
        // Default (first) is the lightest model — the 8-bit 0.6B.
        XCTAssertEqual(audioPresetsReplica.first?.id, "mlx-audio/qwen3-tts-0.6b-base-8bit",
                       "Qwen3-TTS 0.6B 8-bit should be the default (first) preset")
        // Every repo is an open mlx-community mirror (downloads without a login).
        for p in audioPresetsReplica {
            XCTAssertTrue(p.repo.hasPrefix("mlx-community/"),
                          "Preset \(p.id) must use an open mlx-community repo, got \(p.repo)")
        }
        // Catalog ordered lightest → heaviest by RAM.
        XCTAssertEqual(audioPresetsReplica.map(\.approxRAMGB), audioPresetsReplica.map(\.approxRAMGB).sorted(),
                       "Catalog should be ordered lightest → heaviest")
    }

    // MARK: - buildArgs

    func testBuildArgsAlwaysCarriesCoreFlags() {
        let args = audioBuildArgsReplica(
            .init(repo: "mlx-community/MOSS-TTS-Nano-100M", text: "Hello there"),
            outputPath: "/tmp/out.wav")
        for flag in ["--repo", "--text", "--speed", "--temperature", "--output"] {
            XCTAssertTrue(args.contains(flag), "buildArgs must always include \(flag)")
        }
        // --output value is the wav path.
        let i = args.firstIndex(of: "--output")!
        XCTAssertEqual(args[args.index(after: i)], "/tmp/out.wav")
    }

    func testBuildArgsOmitsCloningFlagsWithoutReference() {
        let args = audioBuildArgsReplica(
            .init(repo: "r", text: "hi", refAudioPath: nil), outputPath: "/tmp/o.wav")
        XCTAssertFalse(args.contains("--ref-audio"),
                       "No reference → no --ref-audio (model uses its default voice)")
        XCTAssertFalse(args.contains("--ref-text"))
    }

    func testBuildArgsIncludesReferenceWhenCloning() {
        let args = audioBuildArgsReplica(
            .init(repo: "r", text: "hi", refAudioPath: "/tmp/ref.wav", refText: "the caption"),
            outputPath: "/tmp/o.wav")
        let ra = args.firstIndex(of: "--ref-audio")
        XCTAssertNotNil(ra, "Cloning request must pass --ref-audio")
        XCTAssertEqual(args[args.index(after: ra!)], "/tmp/ref.wav")
        let rt = args.firstIndex(of: "--ref-text")
        XCTAssertNotNil(rt, "A supplied transcript must pass --ref-text")
        XCTAssertEqual(args[args.index(after: rt!)], "the caption")
    }

    func testBuildArgsOmitsRefTextWhenBlank() {
        // Reference present but no transcript → mlx-audio should auto-transcribe,
        // so we must NOT pass an empty --ref-text.
        let args = audioBuildArgsReplica(
            .init(repo: "r", text: "hi", refAudioPath: "/tmp/ref.wav", refText: "   "),
            outputPath: "/tmp/o.wav")
        XCTAssertTrue(args.contains("--ref-audio"))
        XCTAssertFalse(args.contains("--ref-text"),
                       "Blank transcript must be omitted so Whisper auto-transcribes")
    }

    // MARK: - Output path slug

    func testSlugSanitizesAndCaps() {
        XCTAssertEqual(audioSlugReplica("Hello, World! 123"), "hello-world-123")
        XCTAssertEqual(audioSlugReplica("  leading/trailing  "), "leading-trailing")
        XCTAssertLessThanOrEqual(audioSlugReplica(String(repeating: "a", count: 100)).count, 40)
    }

    // MARK: - WAV writer

    func testWavHeaderIsCanonical24kMono16bit() {
        let samples: [Float] = [0, 0.5, -0.5, 1.0, -1.0]
        let data = wavDataReplica(fromMonoFloat: samples, sampleRate: 24_000)

        // Total length: 44-byte header + 2 bytes/sample.
        XCTAssertEqual(data.count, 44 + samples.count * 2)

        func str(_ r: Range<Int>) -> String { String(bytes: data[r], encoding: .ascii)! }
        func u32(_ off: Int) -> UInt32 {
            data.subdata(in: off..<off+4).withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }
        }
        func u16(_ off: Int) -> UInt16 {
            data.subdata(in: off..<off+2).withUnsafeBytes { $0.load(as: UInt16.self).littleEndian }
        }

        XCTAssertEqual(str(0..<4), "RIFF")
        XCTAssertEqual(str(8..<12), "WAVE")
        XCTAssertEqual(str(12..<16), "fmt ")
        XCTAssertEqual(u32(16), 16, "PCM fmt subchunk size")
        XCTAssertEqual(u16(20), 1, "AudioFormat = PCM")
        XCTAssertEqual(u16(22), 1, "mono")
        XCTAssertEqual(u32(24), 24_000, "sample rate")
        XCTAssertEqual(u16(34), 16, "bits per sample")
        XCTAssertEqual(str(36..<40), "data")
        XCTAssertEqual(u32(40), UInt32(samples.count * 2), "data chunk size")
        XCTAssertEqual(u32(4), UInt32(36 + samples.count * 2), "RIFF chunk size")
    }

    func testWavFullScaleSamplesClampCorrectly() {
        let data = wavDataReplica(fromMonoFloat: [1.0, -1.0], sampleRate: 24_000)
        func s16(_ off: Int) -> Int16 {
            Int16(bitPattern: data.subdata(in: off..<off+2).withUnsafeBytes { $0.load(as: UInt16.self).littleEndian })
        }
        XCTAssertEqual(s16(44), 32767, "+1.0 → +32767")
        XCTAssertEqual(s16(46), -32767, "-1.0 → -32767")
    }

    // MARK: - float32 PCM round-trip (AudioRecorder.pcmData ↔ floatSamples)

    func testFloatSamplesRoundTrip() {
        let original: [Float] = [0, 0.25, -0.75, 0.999, -1.0]
        let blob = original.withUnsafeBufferPointer { Data(buffer: $0) }
        XCTAssertEqual(floatSamplesReplica(from: blob), original)
    }
}

