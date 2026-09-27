import AppKit
import XCTest
@testable import MLXCore

final class MediaImageTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("MediaImageTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func writePNG(width: Int, height: Int, to url: URL) throws {
        let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                            bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        guard let cg = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)
        else { XCTFail("cannot synthesize PNG"); return }
        CGImageDestinationAddImage(dest, cg, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
    }

    private func replacePNG(width: Int, height: Int, at url: URL, epoch: TimeInterval) throws {
        try writePNG(width: width, height: height, to: url)
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: epoch)], ofItemAtPath: url.path)
    }

    func testDownsampleCapsTheLongEdgeAndKeepsTheRatio() throws {
        let url = dir.appendingPathComponent("wide.png")
        try writePNG(width: 40, height: 20, to: url)
        let img = try XCTUnwrap(MediaImage.downsample(url: url, maxPixel: 8))
        XCTAssertEqual(img.size, NSSize(width: 8, height: 4))
    }

    func testDownsampleDoesNotUpscale() throws {
        let url = dir.appendingPathComponent("small.png")
        try writePNG(width: 6, height: 6, to: url)
        let img = try XCTUnwrap(MediaImage.downsample(url: url, maxPixel: 512))
        XCTAssertEqual(img.size, NSSize(width: 6, height: 6))
    }

    func testUnreadableFilesDecodeToNil() throws {
        XCTAssertNil(MediaImage.downsample(url: dir.appendingPathComponent("nope.png"), maxPixel: 8))
        let junk = dir.appendingPathComponent("junk.png")
        try Data("not an image".utf8).write(to: junk)
        XCTAssertNil(MediaImage.downsample(url: junk, maxPixel: 8))
        XCTAssertNil(MediaImage.downsample(data: Data("not an image".utf8), maxPixel: 8))
    }

    func testDownsampleFromBytesCapsTheLongEdge() throws {
        let url = dir.appendingPathComponent("tall.png")
        try writePNG(width: 20, height: 40, to: url)
        let img = try XCTUnwrap(MediaImage.downsample(data: Data(contentsOf: url), maxPixel: 10))
        XCTAssertEqual(img.size, NSSize(width: 5, height: 10))
    }

    func testLoadCachesOneSharedDecodingPerFileAndSize() async throws {
        let url = dir.appendingPathComponent("cached.png")
        try writePNG(width: 64, height: 32, to: url)
        let first = await MediaImage.load(url: url, maxPixel: 16)
        let second = await MediaImage.load(url: url, maxPixel: 16)
        XCTAssertNotNil(first)
        XCTAssertTrue(first === second, "second load re-decoded instead of hitting the cache")
    }

    func testLoadReDecodesWhenTheFileIsReplaced() async throws {
        let url = dir.appendingPathComponent("swapped.png")
        try replacePNG(width: 64, height: 64, at: url, epoch: 1_700_000_000)
        let before = await MediaImage.load(url: url, maxPixel: 64)
        XCTAssertEqual(before?.size, NSSize(width: 64, height: 64))
        try replacePNG(width: 32, height: 32, at: url, epoch: 1_700_000_060)
        let after = await MediaImage.load(url: url, maxPixel: 64)
        XCTAssertEqual(after?.size, NSSize(width: 32, height: 32))
    }

    func testLoadFromBytesIsKeyedById() async throws {
        let url = dir.appendingPathComponent("bytes.png")
        try writePNG(width: 50, height: 25, to: url)
        let data = try Data(contentsOf: url)
        let id = UUID().uuidString
        let first = await MediaImage.load(data: data, id: id, maxPixel: 10)
        let second = await MediaImage.load(data: data, id: id, maxPixel: 10)
        XCTAssertTrue(first === second)
        let otherId = await MediaImage.load(data: data, id: UUID().uuidString, maxPixel: 10)
        XCTAssertNotNil(otherId)
        let junk = await MediaImage.load(data: Data("nope".utf8), id: UUID().uuidString, maxPixel: 10)
        XCTAssertNil(junk)
    }

    func testMissingFileIsNotCachedAsMissing() async throws {
        let url = dir.appendingPathComponent("late.png")
        let gone = await MediaImage.load(url: url, maxPixel: 16)
        XCTAssertNil(gone)
        try writePNG(width: 24, height: 24, to: url)
        let arrived = await MediaImage.load(url: url, maxPixel: 16)
        XCTAssertEqual(arrived?.size, NSSize(width: 16, height: 16))
    }
}
