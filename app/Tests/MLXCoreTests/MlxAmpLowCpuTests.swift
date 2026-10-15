import XCTest
import AppKit
@testable import MLXCore

/// The player must cost nothing while nothing moves and little while it does: the
/// animated parts live in an AppKit layer driven by its own timer, at rates chosen
/// by the machine's power state, and the form behind it stops hitting the disk.
final class MlxAmpLowCpuTests: XCTestCase {

    // MARK: motion rates

    func testNormalRatesAreModest() {
        let r = MlxAmpMotion.rates(lowPower: false, thermal: .nominal)
        XCTAssertEqual(r.spectrum, 12)
        XCTAssertEqual(r.title, 6)
        XCTAssertEqual(r.seek, 4)
    }

    func testLowPowerModeSlowsTheSpectrumAndStopsTheTitleScroll() {
        let r = MlxAmpMotion.rates(lowPower: true, thermal: .nominal)
        XCTAssertLessThanOrEqual(r.spectrum, 6)
        XCTAssertEqual(r.title, 0, "0 = a still title")
    }

    func testAHotMachineIsTreatedLikeLowPower() {
        XCTAssertEqual(MlxAmpMotion.rates(lowPower: false, thermal: .serious), MlxAmpMotion.rates(lowPower: true, thermal: .nominal))
        XCTAssertEqual(MlxAmpMotion.rates(lowPower: false, thermal: .fair), MlxAmpMotion.rates(lowPower: false, thermal: .nominal))
    }

    // MARK: title scroll

    func testATitleThatFitsNeverScrolls() {
        XCTAssertEqual(MlxAmpMotion.titleOffset(textWidth: 100, boxWidth: 150, time: 37), 0)
    }

    func testALongTitleScrollsWholePixelsAndWraps() {
        let a = MlxAmpMotion.titleOffset(textWidth: 400, boxWidth: 150, time: 1)
        let b = MlxAmpMotion.titleOffset(textWidth: 400, boxWidth: 150, time: 2)
        XCTAssertEqual(a, a.rounded())
        XCTAssertGreaterThan(b, a)
        let cycle = 400 + MlxAmpMotion.titleGap
        XCTAssertEqual(MlxAmpMotion.titleOffset(textWidth: 400, boxWidth: 150, time: Double(cycle) / MlxAmpMotion.titleSpeed + 0.01),
                       0, "it wraps after one text plus the gap")
    }

    // MARK: slider geometry

    func testSliderFractionClampsAndCentresTheThumb() {
        XCTAssertEqual(MlxAmpSlider.fraction(x: 0, length: 248, thumb: 29), 0)
        XCTAssertEqual(MlxAmpSlider.fraction(x: 1000, length: 248, thumb: 29), 1)
        XCTAssertEqual(MlxAmpSlider.fraction(x: 14.5 + (248 - 29) / 2, length: 248, thumb: 29), 0.5, accuracy: 0.001)
    }

    // MARK: the AppKit drawing backend

    func testThePixelBackendDrawsTheSameRectsIntoACoreGraphicsBitmap() throws {
        let w = 12, h = 10
        let ctx = try XCTUnwrap(CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                          space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        // An NSView that is flipped hands drawing a top-left origin; do the same here.
        ctx.translateBy(x: 0, y: CGFloat(h)); ctx.scaleBy(x: 1, y: -1)
        let p = Px(cg: ctx)
        p.rect(2, 3, 4, 2, MlxAmpStyle.lcd)
        let data = try XCTUnwrap(ctx.data).assumingMemoryBound(to: UInt8.self)
        func green(_ x: Int, _ y: Int) -> UInt8 { data[(y * w + x) * 4 + 1] }
        XCTAssertGreaterThan(green(2, 3), 200)
        XCTAssertGreaterThan(green(5, 4), 200)
        XCTAssertEqual(green(6, 3), 0)
        XCTAssertEqual(green(2, 5), 0)
    }

    func testPixelTextAndDigitsDrawThroughTheCoreGraphicsBackend() throws {
        let w = 80, h = 20
        let ctx = try XCTUnwrap(CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                          space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        ctx.translateBy(x: 0, y: CGFloat(h)); ctx.scaleBy(x: 1, y: -1)
        let p = Px(cg: ctx)
        p.text("MLX", 1, 1, MlxAmpStyle.lcd)
        p.digit(8, 40, 3, MlxAmpStyle.lcd)
        let data = try XCTUnwrap(ctx.data).assumingMemoryBound(to: UInt8.self)
        var lit = 0
        for i in stride(from: 1, to: w * h * 4, by: 4) where data[i] > 200 { lit += 1 }
        XCTAssertGreaterThan(lit, 60, "the text and the numeral must reach the bitmap")
    }

    // MARK: readiness checks stop hitting the disk

    @MainActor
    func testBundleReadinessIsRememberedUntilTheDownloadsChange() throws {
        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "ready-\(UUID().uuidString)"
        let dir = (root as NSString).appendingPathComponent("a/b")
        try fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(atPath: root) }
        let comp = MediaComponent(repo: "a/b", selection: .chatDefault, readyMarkers: ["config.json"])
        let bundle = MediaBundle(id: "t", displayName: "t", components: [comp], sizeEstimateGB: 1)
        let manager = DownloadManager(modelsRoot: root)

        XCTAssertFalse(manager.bundleReady(bundle))
        fm.createFile(atPath: (dir as NSString).appendingPathComponent("config.json"), contents: Data("{}".utf8))
        fm.createFile(atPath: (dir as NSString).appendingPathComponent("model.safetensors"), contents: Data([1]))
        XCTAssertFalse(manager.bundleReady(bundle), "a re-render within the window must not re-stat the folder")
        manager.downloads = [:]   // any download event drops what was remembered
        XCTAssertTrue(manager.bundleReady(bundle))
    }
}
