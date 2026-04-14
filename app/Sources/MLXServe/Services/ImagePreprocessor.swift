import Foundation
import AppKit

enum ImagePreprocessor {

    /// Preprocess an image for the Gemma 4 SigLIP vision encoder.
    /// - Resize to 768x768 (bicubic) — dynamic resolution for Gemma 4
    ///   (max_patches = 280*9 = 2520, sqrt(2520) ≈ 50 → 48x48 grid → 768px)
    /// - Convert to float32 CHW [3, 768, 768]
    /// - Rescale by 1/255 (no normalization — mean=[0,0,0], std=[1,1,1])
    /// - Returns raw float32 bytes (3*768*768*4 = 7,077,888 bytes)
    static func preprocess(_ imageData: Data, targetSize: Int = 768) -> Data? {
        guard let image = NSImage(data: imageData) else { return nil }
        return preprocess(image, targetSize: targetSize)
    }

    static func preprocess(_ image: NSImage, targetSize: Int = 768) -> Data? {
        // Resize to targetSize x targetSize using high-quality interpolation
        guard let resized = resize(image, to: NSSize(width: targetSize, height: targetSize)),
              let bitmap = NSBitmapImageRep(data: resized.tiffRepresentation ?? Data()) else {
            return nil
        }

        let width = bitmap.pixelsWide
        let height = bitmap.pixelsHigh
        guard width == targetSize, height == targetSize else { return nil }
        guard let rawPtr = bitmap.bitmapData else { return nil }

        let bpp = bitmap.bitsPerPixel / 8   // bytes per pixel (3 for RGB, 4 for RGBA)
        let rowBytes = bitmap.bytesPerRow

        // Convert to float32 CHW format [3, H, W], rescaled by 1/255
        let channelSize = targetSize * targetSize
        var pixels = [Float](repeating: 0, count: 3 * channelSize)

        for y in 0..<height {
            let rowStart = y * rowBytes
            for x in 0..<width {
                let pixelOffset = rowStart + x * bpp
                let idx = y * width + x
                pixels[0 * channelSize + idx] = Float(rawPtr[pixelOffset])     / 255.0  // R
                pixels[1 * channelSize + idx] = Float(rawPtr[pixelOffset + 1]) / 255.0  // G
                pixels[2 * channelSize + idx] = Float(rawPtr[pixelOffset + 2]) / 255.0  // B
            }
        }

        return pixels.withUnsafeBufferPointer { buffer in
            Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.size)
        }
    }

    private static func resize(_ image: NSImage, to size: NSSize) -> NSImage? {
        let newImage = NSImage(size: size)
        newImage.lockFocus()
        NSGraphicsContext.current?.imageInterpolation = .high
        image.draw(in: NSRect(origin: .zero, size: size),
                   from: NSRect(origin: .zero, size: image.size),
                   operation: .copy,
                   fraction: 1.0)
        newImage.unlockFocus()
        return newImage
    }
}
