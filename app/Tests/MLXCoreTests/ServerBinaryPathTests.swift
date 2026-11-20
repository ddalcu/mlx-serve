import XCTest
@testable import MLXCore

/// A build from source (swift test, swift run) has no bundled mlx-serve, so the app falls back to the checkout's own
/// `zig-out/bin/mlx-serve`: the plugin install and every version probe run it.
final class ServerBinaryPathTests: XCTestCase {
    func testASourceBuildLooksForTheServerAtTheCheckoutRoot() {
        let bin = URL(fileURLWithPath: ServerManager.sourceBuildBinaryPath)
        XCTAssertEqual(bin.pathComponents.suffix(3), ["zig-out", "bin", "mlx-serve"])
        let root = bin.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("build.zig").path),
                      "\(root.path) is not the checkout root")
    }
}
