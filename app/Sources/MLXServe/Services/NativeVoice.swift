import SwiftUI

/// A model that hears the recording itself (LFM3-Audio, Omni; `ModelInfo.speaks`): voice mode sends the utterance's
/// own audio, and the voice chosen in Settings (or the agent's) reads the answer, as for any other model.

/// The utterance the mic heard, as 16 kHz mono f32 (what the server's speech towers read): everything since speech
/// began plus a short pre-roll, so the first syllable that tripped the VAD is not clipped. Capped, keeping the latest.
struct UtteranceAudio {
    static let sampleRate = 16_000
    var preRollSeconds: Double = 0.4
    var maxSeconds: Double = 60
    private var samples: [Float] = []
    /// Where speech began in `samples`; nil = no utterance (only the pre-roll window is kept).
    private var start: Int?

    init(preRollSeconds: Double = 0.4, maxSeconds: Double = 60) {
        self.preRollSeconds = preRollSeconds
        self.maxSeconds = maxSeconds
    }

    private var preRoll: Int { Int(preRollSeconds * Double(Self.sampleRate)) }
    private var cap: Int { Int(maxSeconds * Double(Self.sampleRate)) }

    mutating func append(_ chunk: [Float]) {
        samples.append(contentsOf: chunk)
        // Between utterances only the pre-roll matters; inside one, the cap.
        let keep = start == nil ? preRoll : preRoll + cap
        if samples.count > keep {
            let drop = samples.count - keep
            samples.removeFirst(drop)
            if let s = start { start = max(0, s - drop) }
        }
    }

    mutating func markSpeechStart() {
        if start == nil { start = samples.count }
    }

    /// The utterance (pre-roll + speech, at most `maxSeconds` of the latest), ending it.
    mutating func take() -> [Float] {
        guard let s = start else { return [] }
        let from = max(0, s - preRoll)
        var out = Array(samples[from...])
        if out.count > cap { out.removeFirst(out.count - cap) }
        samples.removeAll(keepingCapacity: true)
        start = nil
        return out
    }

    /// Channels averaged, then resampled (linear) to 16 kHz.
    static func downmix16k(channels: [[Float]], sampleRate: Double) -> [Float] {
        guard let n = channels.first?.count, n > 0, sampleRate > 0 else { return [] }
        var mono = [Float](repeating: 0, count: n)
        for ch in channels { for i in 0..<n { mono[i] += ch[i] } }
        let k = Float(channels.count)
        for i in 0..<n { mono[i] /= k }
        let ratio = Double(Self.sampleRate) / sampleRate
        let outCount = Int((Double(n) * ratio).rounded(.down))
        return (0..<outCount).map { j in
            let x = Double(j) / ratio
            let i = Int(x)
            let f = Float(x - Double(i))
            return i + 1 < n ? mono[i] * (1 - f) + mono[i + 1] * f : mono[min(i, n - 1)]
        }
    }
}

/// The chat window's voice button: when the model hears you itself (LFM3 Audio, Omni) it has its own glyph and colour,
/// because the model gets your recording rather than a transcript.
struct VoiceToggleLook: Equatable {
    let symbol: String
    let onTint: Color
    let help: String

    static func `for`(hears: Bool) -> VoiceToggleLook {
        hears
            ? VoiceToggleLook(symbol: "waveform.and.person.filled", onTint: .purple,
                              help: "Voice mode — the model hears you itself, and the voice chosen in Settings reads its answer.")
            : VoiceToggleLook(symbol: "waveform", onTint: .accentColor,
                              help: "Voice mode — talk to the model hands-free. What you say is transcribed on this Mac and the answer spoken in your chosen voice.")
    }
}

/// LFM3 as a voice (Settings ▸ Voice, an agent's voice): which pack speaks the answers.
@MainActor
enum LFM3Voice {
    /// The pack Settings offers to download: the smallest that speaks.
    static let repo = "LiquidAI/LFM3-Audio-2B-MLX-8bit"

    /// The chat model when it speaks (no second model resident), else a whole LFM3 speech pack on disk, Audio first.
    nonisolated static func modelPath(_ models: [LocalModel], chatModelPath: String?, chatSpeaks: Bool) -> String? {
        if chatSpeaks, let chatModelPath { return chatModelPath }
        for type in ["lfm3_audio", "lfm3_omni"] {
            if let m = models.first(where: { $0.modelType == type && $0.defect == nil && !$0.isDownloading }) { return m.path }
        }
        return nil
    }

    /// The voices to offer: the first row that lists any (every LFM3 pack lists the same), else just `current`.
    nonisolated static func voices(_ rows: [ModelInfo], current: String) -> [String] {
        rows.first { !$0.voices.isEmpty }?.voices ?? [current]
    }

    /// `british-english-male` → "British English (male)".
    nonisolated static func displayName(_ id: String) -> String {
        var words = id.split(separator: "-").map(String.init)
        guard words.count >= 2, let gender = words.popLast() else { return id }
        return words.map(\.capitalized).joined(separator: " ") + " (\(gender))"
    }

    static func modelPath(app: AppState?) -> String? {
        guard let app else { return nil }
        return modelPath(app.localModels, chatModelPath: app.selectedModelPath, chatSpeaks: app.server.chatModelInfo?.speaks == true)
    }
}
