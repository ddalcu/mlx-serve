import AppKit
import AVFoundation
import UniformTypeIdentifiers
import ImageIO
import CoreImage

/// One word the ASR heard, timed from the start of the clip.
struct CaptionWord: Equatable {
    let text: String
    let start: Double
    let end: Double
}

/// What the screen shows over one span: a group of words, one of them live.
struct CaptionState: Equatable {
    let words: [String]
    let active: Int
    let start: Double
    let end: Double
}

/// Short-video captions: up to three words at a time, the word being said lit.
enum CaptionPlan {
    /// The words of a `verbose_json` transcript asked for word timestamps.
    static func words(fromVerboseJSON data: Data) -> [CaptionWord] {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [] }
        let raw = (obj["words"] as? [[String: Any]])
            ?? (obj["segments"] as? [[String: Any]] ?? []).flatMap { $0["words"] as? [[String: Any]] ?? [] }
        return raw.compactMap { w in
            guard let text = (w["word"] as? String)?.trimmingCharacters(in: .whitespaces), !text.isEmpty,
                  let start = (w["start"] as? NSNumber)?.doubleValue,
                  let end = (w["end"] as? NSNumber)?.doubleValue else { return nil }
            return CaptionWord(text: text, start: start, end: end)
        }
    }

    /// Words in caption groups: a group ends at three words, after punctuation,
    /// or at a pause longer than `gap`.
    static func groups(_ words: [CaptionWord], most: Int = 3, gap: Double = 0.45) -> [[CaptionWord]] {
        var out: [[CaptionWord]] = [], cur: [CaptionWord] = []
        for word in words {
            if let last = cur.last,
               cur.count == most || word.start - last.end > gap || ".,!?".contains(last.text.last ?? " ") {
                out.append(cur)
                cur = []
            }
            cur.append(word)
        }
        return cur.isEmpty ? out : out + [cur]
    }

    /// One state per word. A group stays up a beat after its last word, or
    /// until the next group when the pause before it is short.
    static func states(_ words: [CaptionWord]) -> [CaptionState] {
        let gs = groups(words)
        var out: [CaptionState] = []
        for (gi, g) in gs.enumerated() {
            let next = gi + 1 < gs.count ? gs[gi + 1][0].start : nil
            let tail = g[g.count - 1].end
            let end = next.map { $0 - tail < 0.6 ? $0 : tail + 0.25 } ?? tail + 0.25
            let texts = g.map { $0.text.uppercased() }
            for (i, word) in g.enumerated() {
                out.append(CaptionState(words: texts, active: i, start: word.start,
                                        end: i + 1 < g.count ? g[i + 1].start : end))
            }
        }
        return out
    }
}

/// Burns caption states onto a video and keeps its audio, with Core Image
/// (never the Core Animation tool: `docs/gotchas/app.md`).
enum CaptionBurner {
    /// `top` is meme text: white Impact in a band across the top for the whole clip.
    static func burn(video: URL, words: [CaptionWord], top: String = "", to out: URL) async throws {
        let asset = AVURLAsset(url: video)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw MediaGenError.server(L10n.text("Couldn't add the captions."))
        }
        let size = try await track.load(.naturalSize)
        // Core Image's origin is the bottom left: the band sits 30% up.
        let overlays: [(state: CaptionState, image: CIImage)] = CaptionPlan.states(words).compactMap { state in
            guard let cg = render(state, width: size.width) else { return nil }
            let image = CIImage(cgImage: cg)
            let x = ((size.width - image.extent.width) / 2).rounded()
            let y = (size.height * 0.30 - image.extent.height / 2).rounded()
            return (state, image.transformed(by: CGAffineTransform(translationX: x, y: y)))
        }
        let banner: CIImage? = top.isEmpty ? nil : render(CaptionState(words: [top.uppercased()], active: -1, start: 0, end: 0),
                                                          width: size.width, meme: true).map { cg in
            let image = CIImage(cgImage: cg)
            let x = ((size.width - image.extent.width) / 2).rounded()
            return image.transformed(by: CGAffineTransform(translationX: x, y: (size.height * 0.96 - image.extent.height).rounded()))
        }
        let composition = try await AVMutableVideoComposition.videoComposition(with: asset, applyingCIFiltersWithHandler: { request in
            let t = request.compositionTime.seconds
            var frame = banner?.composited(over: request.sourceImage) ?? request.sourceImage
            if let caption = overlays.first(where: { $0.state.start <= t && t < $0.state.end }) {
                frame = caption.image.composited(over: frame)
            }
            request.finish(with: frame, context: nil)
        })
        guard let export = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetHighestQuality) else {
            throw MediaGenError.server(L10n.text("Couldn't add the captions."))
        }
        export.videoComposition = composition
        try? FileManager.default.removeItem(at: out)
        try await export.export(to: out, as: .mp4)
    }

    /// Speech from a video's soundtrack as a 16 kHz mono WAV for the ASR.
    static func speechWAV(from video: URL) async throws -> Data {
        let asset = AVURLAsset(url: video)
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
            throw MediaGenError.server(L10n.text("The video has no soundtrack."))
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 16000, AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false, AVLinearPCMIsNonInterleaved: false,
        ])
        reader.add(output)
        reader.startReading()
        var samples: [Float] = []
        while let buffer = output.copyNextSampleBuffer(), let block = CMSampleBufferGetDataBuffer(buffer) {
            let n = CMBlockBufferGetDataLength(block)
            var chunk = [Float](repeating: 0, count: n / 4)
            chunk.withUnsafeMutableBytes { _ = CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: n, destination: $0.baseAddress!) }
            samples += chunk
        }
        guard reader.status == .completed, !samples.isEmpty else {
            throw MediaGenError.server(L10n.text("Couldn't read the video's soundtrack."))
        }
        return AudioReference.wavData(fromMonoFloat: samples, sampleRate: 16000)
    }

    /// Meme text as an image: white Impact, black outline, wrapped to `width`.
    static func memeText(_ text: String, width: CGFloat) -> CGImage? {
        render(CaptionState(words: [text.uppercased()], active: -1, start: 0, end: 0), width: width, meme: true)
    }

    /// A picture with meme text across its top and its bottom, as a PNG.
    static func burn(image src: URL, top: String, bottom: String, to out: URL) throws {
        guard let source = CGImageSourceCreateWithURL(src as CFURL, nil),
              let picture = CGImageSourceCreateImageAtIndex(source, 0, nil),
              let ctx = CGContext(data: nil, width: picture.width, height: picture.height, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw MediaGenError.server(L10n.text("Couldn't add the meme text.")) }
        let w = CGFloat(picture.width), h = CGFloat(picture.height)
        ctx.draw(picture, in: CGRect(x: 0, y: 0, width: w, height: h))
        // Core Graphics' origin is the bottom left.
        for (text, top) in [(top, true), (bottom, false)] where !text.isEmpty {
            guard let t = memeText(text, width: w) else { continue }
            let tw = CGFloat(t.width), th = CGFloat(t.height)
            ctx.draw(t, in: CGRect(x: ((w - tw) / 2).rounded(), y: (top ? h * 0.97 - th : h * 0.03).rounded(), width: tw, height: th))
        }
        guard let burned = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(out as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw MediaGenError.server(L10n.text("Couldn't add the meme text.")) }
        CGImageDestinationAddImage(dest, burned, nil)
        guard CGImageDestinationFinalize(dest) else { throw MediaGenError.server(L10n.text("Couldn't add the meme text.")) }
    }

    /// One caption state drawn as an image: white words, the live one yellow,
    /// a black outline so it reads on any frame. Meme text is Impact, wrapped
    /// over as many lines as it needs; a caption is one line, shrunk to fit.
    private static func render(_ state: CaptionState, width: CGFloat, meme: Bool = false) -> CGImage? {
        var size = (width * (meme ? 0.08 : 0.085)).rounded()
        let centered = NSMutableParagraphStyle()
        centered.alignment = .center
        func line(_ size: CGFloat) -> NSAttributedString {
            let font = NSFont(name: meme ? "Impact" : "ArialRoundedMTBold", size: size) ?? .systemFont(ofSize: size, weight: .heavy)
            let text = NSMutableAttributedString()
            for (i, word) in state.words.enumerated() {
                if i > 0 { text.append(NSAttributedString(string: " ", attributes: [.font: font])) }
                let color = i == state.active ? NSColor(srgbRed: 1, green: 0.867, blue: 0.2, alpha: 1) : .white
                text.append(NSAttributedString(string: word, attributes: [
                    .font: font, .foregroundColor: color, .paragraphStyle: centered,
                    .strokeColor: NSColor.black, .strokeWidth: -7.0,
                ]))
            }
            return text
        }
        var text = line(size)
        if !meme, text.size().width > width * 0.9 {
            size = (size * width * 0.9 / text.size().width).rounded(.down)
            text = line(size)
        }
        let box = meme ? text.boundingRect(with: CGSize(width: width * 0.9, height: 10_000),
                                           options: [.usesLineFragmentOrigin, .usesFontLeading]).size
                       : text.size()
        let pad = (size * 0.2).rounded(.up)
        let w = Int((box.width + 2 * pad).rounded(.up)), h = Int((box.height + 2 * pad).rounded(.up))
        guard w > 0, h > 0,
              let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: w, pixelsHigh: h, bitsPerSample: 8,
                                         samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                         colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0),
              let ctx = NSGraphicsContext(bitmapImageRep: rep) else { return nil }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = ctx
        text.draw(with: CGRect(x: pad, y: pad, width: box.width, height: box.height),
                  options: [.usesLineFragmentOrigin, .usesFontLeading])
        NSGraphicsContext.restoreGraphicsState()
        return rep.cgImage
    }
}

/// Whether this Mac can caption a video: the speech model and, for Qwen3-ASR,
/// the aligner that times its words (never the diarizer, which captions do not use).
enum CaptionModels: Equatable {
    case ready
    /// The repos to fetch; the ASR's download brings the aligner after it.
    case download([String])
    /// Nemotron Transcribe is chosen and is not on this Mac: it is converted by hand, not downloaded.
    case notConverted

    static func of(_ models: [LocalModel], engine: ServerOptions.TranscribeEngine) -> CaptionModels {
        guard engine == .combo else {
            return TrayTranscript.asrPath(models, engine: engine) == nil ? .notConverted : .ready
        }
        let missing = TrayTranscript.missingRepos(models, engine: engine).filter { $0 != DownloadManager.diarizerRepo }
        return missing.isEmpty ? .ready : .download(missing)
    }
}

