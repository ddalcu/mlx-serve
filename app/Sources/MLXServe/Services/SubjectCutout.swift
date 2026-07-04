import Foundation
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO
import Vision
import AppKit

/// Cuts the main subject out of a photo and composites it on an opaque white
/// background before the image is sent for 3D reconstruction — the native
/// analogue of the reference pipeline's rembg step. Isolating the subject on a
/// clean background gives the shape model a much better silhouette.
///
/// Everything is best-effort: a decode failure, no detected subject, or Vision
/// erroring all fall back to a white-flattened (or, worst case, the original)
/// image. The server composites alpha-on-white itself, so a passthrough is
/// always safe.
enum SubjectCutout {

    /// Cut the subject and return PNG bytes on white. Never throws; returns the
    /// ORIGINAL bytes on any failure so the caller can always send something.
    static func cutoutOnWhite(imageData: Data) -> Data {
        guard let source = decodeCGImage(imageData) else { return imageData }
        // Prefer the Vision-masked subject; fall back to the whole image (still
        // flattened on white so a transparent PNG never reaches the server).
        let subject = maskedForeground(source) ?? source
        guard let flat = flattenOnWhite(subject),
              let png = pngData(from: flat) else { return imageData }
        return png
    }

    // MARK: - Pure helpers (testable)

    /// Composite `source` (which may carry an alpha channel) over opaque white,
    /// returning an alpha-free image the same size. Pure + testable.
    static func flattenOnWhite(_ source: CGImage) -> CGImage? {
        let w = source.width, h = source.height
        guard w > 0, h > 0 else { return nil }
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
            space: cs, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.draw(source, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()
    }

    static func decodeCGImage(_ data: Data) -> CGImage? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }

    static func pngData(from cg: CGImage) -> Data? {
        let rep = NSBitmapImageRep(cgImage: cg)
        return rep.representation(using: .png, properties: [:])
    }

    // MARK: - Vision

    /// Run `VNGenerateForegroundInstanceMaskRequest` and return the subject cut
    /// out with an alpha channel (background transparent), or nil if Vision
    /// finds no subject / errors.
    private static func maskedForeground(_ cg: CGImage) -> CGImage? {
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        do {
            try handler.perform([request])
            guard let result = request.results?.first, !result.allInstances.isEmpty else { return nil }
            let buffer = try result.generateMaskedImage(
                ofInstances: result.allInstances, from: handler, croppedToInstancesExtent: false)
            return cgImage(fromPixelBuffer: buffer)
        } catch {
            return nil
        }
    }

    private static func cgImage(fromPixelBuffer buffer: CVPixelBuffer) -> CGImage? {
        let ci = CIImage(cvPixelBuffer: buffer)
        return CIContext(options: nil).createCGImage(ci, from: ci.extent)
    }
}
