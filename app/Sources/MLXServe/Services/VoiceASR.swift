import Foundation

/// Speech-to-text on our own server: Qwen3-ASR through `/v1/audio/transcriptions`. Voice mode, the Audio pane's
/// dictation and Telegram voice notes all transcribe here; the audio never leaves this Mac.
@MainActor
enum VoiceASR {
    static let repo = TrayTranscript.asrRepo
    /// The app whose server and models transcribe; set at launch, nil in tests.
    static weak var app: AppState?

    /// The pack to transcribe with (Qwen3-ASR, or Nemotron Transcribe when Settings ▸ Voice picks it): on disk, whole,
    /// not still arriving.
    nonisolated static func modelPath(_ models: [LocalModel], engine: ServerOptions.TranscribeEngine = .combo) -> String? {
        TrayTranscript.asrPath(models, engine: engine)
    }

    static var ready: Bool { app.flatMap { modelPath($0.localModels, engine: $0.serverOptions.transcribeEngine) } != nil }

    /// What `pcm` (16 kHz mono) says; nil for no words, no speech model, or a failed request.
    static func transcribe(_ pcm: [Float]) async -> String? {
        guard let app, let model = modelPath(app.localModels, engine: app.serverOptions.transcribeEngine), !pcm.isEmpty else { return nil }
        let wav = AudioReference.wavData(fromMonoFloat: pcm, sampleRate: UtteranceAudio.sampleRate)
        let server = app.server.baseURL
        guard let data = try? await TrayTranscript.post(wav, model: model, server: server, format: "json") else { return nil }
        return text(fromResponse: data)
    }

    /// The `json` response's `text`, trimmed; nil when it holds no words.
    nonisolated static func text(fromResponse data: Data) -> String? {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let t = (obj["text"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines), !t.isEmpty else { return nil }
        return t
    }
}
