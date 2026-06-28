import Foundation
import AppKit

/// Helpers for showing an agent-generated image inline in chat.
///
/// `ChatImage` stores JPEG bytes and the inline-image extractor keys on the
/// `data:image/jpeg;base64,` marker, but `ImageGenService` writes PNG — so the
/// agent path (a) transcodes the PNG output to a JPEG data URI, and (b) splits
/// the tool output into the model-facing caption (path/summary) and the image
/// bytes, keeping the multi-KB base64 OUT of the model's context.
enum AgentMediaInline {

    /// The marker the `generate_image`/`browse` tool outputs use to carry an
    /// inline image, and which `splitInlineImage` / the chat row builder key on.
    static let jpegDataURIMarker = "data:image/jpeg;base64,"

    /// Split a tool output that may carry a trailing
    /// `data:image/jpeg;base64,<b64>` payload into the leading caption text and
    /// the decoded JPEG bytes. The caption is everything BEFORE the marker
    /// (trimmed) — never the base64 — so the model-facing tool content stays
    /// small. Returns the whole string as caption + `nil` image when no marker
    /// is present. Pure → unit-tested.
    static func splitInlineImage(_ output: String) -> (caption: String, jpeg: Data?) {
        guard let range = output.range(of: jpegDataURIMarker) else {
            return (output, nil)
        }
        let caption = String(output[..<range.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let remainder = output[range.upperBound...]
        let b64End = remainder.firstIndex(of: "\n") ?? remainder.endIndex
        let b64 = String(remainder[..<b64End])
        return (caption, Data(base64Encoded: b64))
    }

    /// Transcode a PNG file on disk to a `data:image/jpeg;base64,<b64>` URI for
    /// inline display (`ChatImage` is JPEG). nil when the file can't be read or
    /// re-encoded.
    static func pngFileToJpegDataURI(_ path: String) -> String? {
        guard let image = NSImage(contentsOfFile: path),
              let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let jpeg = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.9])
        else { return nil }
        return "\(jpegDataURIMarker)\(jpeg.base64EncodedString())"
    }
}
