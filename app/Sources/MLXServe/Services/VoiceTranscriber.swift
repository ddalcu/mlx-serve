import Foundation
import Speech

/// One-shot, on-device speech-to-text for a finished audio file (the `.m4a`
/// `VoicePreprocessor` produces). Used by the Telegram bridge to make voice
/// notes work on models that can't hear audio: decode → transcribe → send the
/// words as a normal text turn.
///
/// Privacy contract matches Voice mode: recognition is forced **on-device**
/// (`requiresOnDeviceRecognition = true`) so the clip never leaves the Mac. If
/// the on-device model isn't installed for the locale we refuse with the same
/// actionable guidance (`OnDeviceSpeech.unavailableMessage`) rather than
/// silently falling back to Apple's servers.
enum VoiceTranscriber {
    enum Failure: Error, Equatable {
        /// On-device recognition is unavailable; `String` is a user-facing fix.
        case unavailable(String)
        /// The clip decoded but produced no recognizable speech.
        case noSpeech
        /// The recognizer reported an error mid-task.
        case failed(String)
    }

    static func transcribe(fileURL: URL, locale: Locale? = nil) async -> Result<String, Failure> {
        let resolvedLocale: Locale
        if let locale {
            resolvedLocale = locale
        } else {
            switch SpeechLocale.legacyResolution() {
            case .available(let locale), .unavailable(let locale):
                resolvedLocale = locale
            case .unsupported(let reported):
                return .failure(.unavailable(
                    "Speech recognition isn't available for \(reported.identifier)."))
            }
        }

        guard let recognizer = SFSpeechRecognizer(locale: resolvedLocale), recognizer.isAvailable else {
            return .failure(.unavailable("Speech recognition isn't available for \(resolvedLocale.identifier)."))
        }
        if let message = OnDeviceSpeech.unavailableMessage(
            supportsOnDevice: recognizer.supportsOnDeviceRecognition,
            locale: resolvedLocale.identifier) {
            return .failure(.unavailable(message))
        }

        let authorized = await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
            SFSpeechRecognizer.requestAuthorization { cont.resume(returning: $0 == .authorized) }
        }
        guard authorized else {
            return .failure(.unavailable(
                "Speech-recognition permission is off. Enable it in System Settings → Privacy & Security → Speech Recognition."))
        }

        let request = SFSpeechURLRecognitionRequest(url: fileURL)
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = false

        return await withCheckedContinuation { (cont: CheckedContinuation<Result<String, Failure>, Never>) in
            // The recognition task drives a single callback to completion; the
            // `done` guard makes resuming the continuation exactly-once safe even
            // if the framework emits both a result and a trailing error. The task
            // retains itself while running, so we don't need to hold a reference.
            var done = false
            _ = recognizer.recognitionTask(with: request) { result, error in
                if done { return }
                if let result, result.isFinal {
                    done = true
                    let text = result.bestTranscription.formattedString
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    cont.resume(returning: text.isEmpty ? .failure(.noSpeech) : .success(text))
                    return
                }
                if let error {
                    done = true
                    cont.resume(returning: .failure(.failed(error.localizedDescription)))
                }
            }
        }
    }
}
