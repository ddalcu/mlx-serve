import XCTest
import CoreGraphics
@testable import MLXCore

/// The subject-cutout compositing step: `flattenOnWhite` places a (possibly
/// transparent) image over opaque white before the photo is sent for 3D
/// reconstruction. The Vision masking itself needs a real photo + on-device
/// model and isn't unit-tested; the pure geometry/compositing is.
final class SubjectCutoutTests: XCTestCase {

    func testFlattenTransparentBecomesOpaqueWhite() throws {
        // A fully transparent 8×8 image → flattened → every pixel opaque white.
        let src = try makeImage(width: 8, height: 8) { ctx, w, h in
            ctx.clear(CGRect(x: 0, y: 0, width: w, height: h))  // all (0,0,0,0)
        }
        let flat = try XCTUnwrap(SubjectCutout.flattenOnWhite(src))
        XCTAssertEqual(flat.width, 8)
        XCTAssertEqual(flat.height, 8)
        let (r, g, b, a) = try centerPixel(flat)
        XCTAssertEqual(r, 255); XCTAssertEqual(g, 255); XCTAssertEqual(b, 255)
        XCTAssertEqual(a, 255, "flattened image must be opaque")
    }

    func testFlattenPreservesOpaqueColor() throws {
        // An opaque red image stays red after compositing over white (it must
        // NOT wash out to white/gray). Assert red DOMINANCE rather than exact
        // channels — device↔working color-space matching nudges channels a few
        // levels, so exact equality is brittle, but the hue is unmistakable.
        let deviceRed = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [1, 0, 0, 1])!
        let src = try makeImage(width: 8, height: 8) { ctx, w, h in
            ctx.setFillColor(deviceRed)
            ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        }
        let flat = try XCTUnwrap(SubjectCutout.flattenOnWhite(src))
        let (r, g, b, _) = try centerPixel(flat)
        XCTAssertGreaterThan(r, 200, "red channel should stay strong")
        XCTAssertLessThan(g, 80, "green should stay low (not washed toward white)")
        XCTAssertLessThan(b, 80, "blue should stay low (not washed toward white)")
    }

    func testCutoutFallsBackToOriginalOnGarbage() {
        // Non-image bytes → passthrough (never a crash); the server composites
        // alpha-on-white itself, so returning the input is always safe.
        let junk = Data("not an image".utf8)
        XCTAssertEqual(SubjectCutout.cutoutOnWhite(imageData: junk), junk)
    }

    // MARK: - Helpers

    private func makeImage(width: Int, height: Int, draw: (CGContext, Int, Int) -> Void) throws -> CGImage {
        let ctx = try XCTUnwrap(CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        draw(ctx, width, height)
        return try XCTUnwrap(ctx.makeImage())
    }

    /// Read the center pixel of `cg` as RGBA8 (0–255). The draw happens inside
    /// the buffer-scoped closure so the backing bytes stay alive.
    private func centerPixel(_ cg: CGImage) throws -> (UInt8, UInt8, UInt8, UInt8) {
        var px: [UInt8] = [0, 0, 0, 0]
        px.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(
                data: raw.baseAddress, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
            // Offset the source so its center pixel lands at the 1×1 output origin.
            let cx = -CGFloat(cg.width / 2), cy = -CGFloat(cg.height / 2)
            ctx.draw(cg, in: CGRect(x: cx, y: cy, width: CGFloat(cg.width), height: CGFloat(cg.height)))
        }
        return (px[0], px[1], px[2], px[3])
    }
}
