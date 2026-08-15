import Foundation
import AVFoundation
import AppKit

/// Extracts JPEG frames from a video file for Qwen3-VL-family video input.
/// Frames are sampled evenly across the WHOLE clip duration — unlike
/// MiniMax-H3's reference-video ingestion (`VideoGenService.videoFileToRefPayload`,
/// first-N-seconds at a fixed 24 fps, snapped to H3's own frame ladder), a chat
/// attachment is meant to summarize the whole clip, not condition generation on
/// its opening beats. The server groups every 2 consecutive frames into one
/// temporal patch (Qwen's `temporal_patch_size`), so the frame COUNT — not the
/// clip's raw length — is what sizes the prompt-token cost.
enum VideoPreprocessor {
    /// Frames requested per attachment. Qwen pays roughly one image's worth of
    /// merged tokens per PAIR of frames (temporal_patch_size=2), so this costs
    /// about what 8 images would — a reasonable budget for one attachment.
    static let maxFrames = 16
    static let frameMaxEdge: CGFloat = 768

    /// `count` timestamps evenly spaced across `[0, duration]` (inclusive of
    /// both ends when count > 1). Pure — factored out so the sampling math is
    /// testable without decoding a real video file.
    nonisolated static func frameTimes(duration: Double, count: Int) -> [Double] {
        guard duration > 0, count > 0 else { return [] }
        guard count > 1 else { return [0] }
        return (0..<count).map { duration * Double($0) / Double(count - 1) }
    }

    /// Returns JPEG frame bytes (one per sampled frame, in playback order), or
    /// nil if the file can't be read or has no readable duration.
    static func extractFrames(url: URL, maxFrames: Int = maxFrames) -> [Data]? {
        let asset = AVURLAsset(url: url)
        let seconds = CMTimeGetSeconds(asset.duration)
        guard seconds.isFinite, seconds > 0, maxFrames > 0 else { return nil }

        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.requestedTimeToleranceBefore = .zero
        gen.requestedTimeToleranceAfter = .zero
        gen.maximumSize = CGSize(width: frameMaxEdge, height: frameMaxEdge)

        var frames: [Data] = []
        frames.reserveCapacity(maxFrames)
        for t in frameTimes(duration: seconds, count: maxFrames) {
            let time = CMTime(seconds: t, preferredTimescale: 600)
            guard let cg = try? gen.copyCGImage(at: time, actualTime: nil) else { return nil }
            let rep = NSBitmapImageRep(cgImage: cg)
            guard let jpeg = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.85])
            else { return nil }
            frames.append(jpeg)
        }
        return frames.isEmpty ? nil : frames
    }
}
