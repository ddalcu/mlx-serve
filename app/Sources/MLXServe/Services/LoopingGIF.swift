import AVFoundation
import ImageIO
import UniformTypeIdentifiers

/// A clip as a GIF that loops without a jump: it plays forward, then back.
enum LoopingGIF {
    static func make(from video: URL, to out: URL, maxSide: CGFloat = 320, fps: Double = 10) async throws {
        let asset = AVURLAsset(url: video)
        let seconds = try await asset.load(.duration).seconds
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: maxSide, height: maxSide)
        gen.requestedTimeToleranceBefore = .zero
        gen.requestedTimeToleranceAfter = .zero
        var forward: [CGImage] = []
        for t in stride(from: 0, to: seconds, by: 1 / fps) {
            forward.append(try await gen.image(at: CMTime(seconds: t, preferredTimescale: 600)).image)
        }
        let frames = forward + forward.reversed().dropFirst().dropLast()
        try? FileManager.default.removeItem(at: out)
        guard !forward.isEmpty,
              let dest = CGImageDestinationCreateWithURL(out as CFURL, UTType.gif.identifier as CFString, frames.count, nil)
        else { throw MediaGenError.server(L10n.text("Couldn't make the GIF.")) }
        CGImageDestinationSetProperties(dest, [kCGImagePropertyGIFDictionary: [kCGImagePropertyGIFLoopCount: 0]] as CFDictionary)
        let delay = [kCGImagePropertyGIFDictionary: [kCGImagePropertyGIFDelayTime: 1 / fps]] as CFDictionary
        for frame in frames { CGImageDestinationAddImage(dest, frame, delay) }
        guard CGImageDestinationFinalize(dest) else { throw MediaGenError.server(L10n.text("Couldn't make the GIF.")) }
    }
}
