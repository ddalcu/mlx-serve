import XCTest
@testable import MLXCore

/// Every folder/file picker shows hidden files: dotfolders (`.config`,
/// `.claude`, a hidden workspace) are exactly the kind of thing people point
/// an agent at. One constructor, so a new picker cannot forget.
final class OpenPanelTests: XCTestCase {

    func testEveryOpenPanelIsBuiltThroughTheOneConstructor() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe")
        let fm = FileManager.default
        guard let en = fm.enumerator(at: root, includingPropertiesForKeys: nil) else { return XCTFail("no sources") }
        var offenders: [String] = []
        var seenConstructor = false
        for case let url as URL in en where url.pathExtension == "swift" {
            let text = SourceScan.strippingComments(try String(contentsOf: url, encoding: .utf8))
            if url.lastPathComponent == "OpenPanel.swift" {
                seenConstructor = text.contains("NSOpenPanel()")
                continue
            }
            if text.contains("NSOpenPanel()") { offenders.append(url.lastPathComponent) }
        }
        XCTAssertTrue(seenConstructor, "OpenPanel.make() must be the one place an NSOpenPanel is built")
        XCTAssertTrue(offenders.isEmpty, "build pickers with OpenPanel.make(): \(offenders)")
    }
}
