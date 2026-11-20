import AVFoundation
import ImageIO

/// A song as a video anyone can post: its cover photo with the title on it,
/// held for as long as the song plays.
enum SongVideo {
    private static let rate = 44_100

    static func make(cover: URL, title: String, song: URL, shape: ShareShape, to out: URL) async throws {
        let w = shape.width, h = shape.height
        guard let frame = coverFrame(cover, title: title, width: w, height: h) else {
            throw MediaGenError.server(L10n.text("Couldn't read the cover photo."))
        }
        let pcm = try await stereo16(song)
        let frames = max(1, Int((Double(pcm.count) / Double(rate * 4)).rounded(.up)))
        try await Task.detached(priority: .userInitiated) {
            var rgb = Data(capacity: frame.count * frames)
            for _ in 0..<frames { rgb.append(frame) }
            try VideoGenService.writeMP4(rgb: rgb, frames: frames, width: w, height: h, fps: 1, to: out,
                                         audioPCM: pcm, audioSampleRate: rate, audioChannels: 2)
        }.value
    }

    /// The photo filling the canvas, the title across its foot, as packed RGB.
    private static func coverFrame(_ url: URL, title: String, width w: Int, height h: Int) -> Data? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let photo = CGImageSourceCreateImageAtIndex(source, 0, nil),
              let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        let W = CGFloat(w), H = CGFloat(h), pw = CGFloat(photo.width), ph = CGFloat(photo.height)
        let scale = max(W / pw, H / ph)
        ctx.draw(photo, in: CGRect(x: (W - pw * scale) / 2, y: (H - ph * scale) / 2, width: pw * scale, height: ph * scale))
        if !title.isEmpty, let t = CaptionBurner.memeText(title, width: W) {
            let tw = CGFloat(t.width), th = CGFloat(t.height)
            ctx.draw(t, in: CGRect(x: ((W - tw) / 2).rounded(), y: (H * 0.04).rounded(), width: tw, height: th))
        }
        guard let base = ctx.data else { return nil }
        // A bitmap context's memory starts at the top row, the order the video writer reads.
        let rgba = base.bindMemory(to: UInt8.self, capacity: w * h * 4)
        var rgb = Data(count: w * h * 3)
        rgb.withUnsafeMutableBytes { dst in
            let d = dst.bindMemory(to: UInt8.self)
            for i in 0..<(w * h) {
                d[i * 3] = rgba[i * 4]; d[i * 3 + 1] = rgba[i * 4 + 1]; d[i * 3 + 2] = rgba[i * 4 + 2]
            }
        }
        return rgb
    }

    /// The song as 44.1 kHz stereo 16-bit interleaved PCM.
    private static func stereo16(_ url: URL) async throws -> Data {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
            throw MediaGenError.server(L10n.text("The song has no sound."))
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: rate, AVNumberOfChannelsKey: 2,
            AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false, AVLinearPCMIsNonInterleaved: false,
        ])
        reader.add(output)
        reader.startReading()
        var pcm = Data()
        while let buffer = output.copyNextSampleBuffer(), let block = CMSampleBufferGetDataBuffer(buffer) {
            let n = CMBlockBufferGetDataLength(block)
            var chunk = Data(count: n)
            chunk.withUnsafeMutableBytes { _ = CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: n, destination: $0.baseAddress!) }
            pcm.append(chunk)
        }
        guard reader.status == .completed, !pcm.isEmpty else { throw MediaGenError.server(L10n.text("Couldn't read the song.")) }
        return pcm
    }
}
