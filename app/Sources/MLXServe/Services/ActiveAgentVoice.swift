import Foundation

/// The voice the active agent speaks with, consulted before the global settings.
///
/// `ClonedVoiceSynthesizer`'s production `voice:` closure already re-reads the
/// saved settings ONCE PER UTTERANCE (so a Settings change applies to the very
/// next sentence with no restart). That re-read is the whole seam: an agent voice
/// is a value this holder publishes and the same closure prefers — no new
/// plumbing, no synthesizer rebuild, and it applies mid-answer.
///
/// A lock rather than a main-actor property because the closure runs wherever the
/// synthesizer pumps.
enum ActiveAgentVoice {
    private static let lock = NSLock()
    private static var value: AgentVoice?

    /// Set when an agent with a voice becomes active; nil when none is (which is
    /// what "follow Settings" means).
    static func set(_ voice: AgentVoice?) {
        lock.lock(); value = voice; lock.unlock()
    }

    static var current: AgentVoice? {
        lock.lock(); defer { lock.unlock() }
        return value
    }

    /// The neural voice to synthesize with, or nil for "speak with the Apple
    /// system synthesizer" — the same contract the synthesizer's closure has
    /// always had.
    ///
    /// An agent voice wins, INCLUDING when it pins the system synthesizer (nil
    /// must not fall through to a global Kokoro setting, or an agent asking for
    /// the plain voice would still sound neural). An agent voice with an empty
    /// value is a half-saved setting, not a request for silence, so it defers.
    static func neuralVoice(agent: AgentVoice?, options: ServerOptions) -> NeuralVoice? {
        if let agent {
            switch agent {
            case .system:
                return nil
            case .kokoro(let v):
                let name = v.trimmingCharacters(in: .whitespaces)
                if !name.isEmpty { return .kokoro(voice: name) }
            case .clone(let path):
                if !path.trimmingCharacters(in: .whitespaces).isEmpty {
                    return .clone(clipPath: path)
                }
            }
        }
        switch options.voiceEngine {
        case .system:
            return nil
        case .kokoro:
            let v = options.kokoroVoice.trimmingCharacters(in: .whitespaces)
            return .kokoro(voice: v.isEmpty ? "af_heart" : v)
        case .clone:
            return options.voiceClonePath.isEmpty ? nil : .clone(clipPath: options.voiceClonePath)
        }
    }

    /// Live read for the synthesizer's per-utterance closure.
    static func currentNeuralVoice(options: ServerOptions) -> NeuralVoice? {
        neuralVoice(agent: current, options: options)
    }
}
