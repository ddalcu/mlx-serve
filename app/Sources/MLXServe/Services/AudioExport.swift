import AVFoundation
import AppKit

/// Compressed copies of generated audio. Apple ships an AAC encoder and no MP3
/// one, so M4A is the compressed format the app can write with no dependency.
/// Encoded through `AVAudioFile`: `AVAssetExportSession` makes macOS ask for
/// microphone access, a file encode does not.
enum AudioExport {
    /// `track.wav` -> `track.m4a` beside it; returns the new file.
    static func m4a(from wav: URL) async throws -> URL {
        try await Task.detached {
            let out = wav.deletingPathExtension().appendingPathExtension("m4a")
            try? FileManager.default.removeItem(at: out)
            let src = try AVAudioFile(forReading: wav)
            let format = src.processingFormat
            let dst = try AVAudioFile(forWriting: out, settings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: format.sampleRate,
                AVNumberOfChannelsKey: format.channelCount, AVEncoderBitRateKey: 192_000,
            ])
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 16_384) else { throw CocoaError(.fileWriteUnknown) }
            while src.framePosition < src.length {
                try src.read(into: buffer)
                try dst.write(from: buffer)
            }
            return out
        }.value
    }

    /// The export button's action: convert, then show the result.
    static func exportAndReveal(_ path: String) {
        Task {
            if let m4a = try? await m4a(from: URL(fileURLWithPath: path)) {
                NSWorkspace.shared.activateFileViewerSelecting([m4a])
            }
        }
    }
}
