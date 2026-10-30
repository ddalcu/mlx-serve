import XCTest
import SwiftUI
@testable import MLXCore

final class SidebarSessionLabelTests: XCTestCase {
    @MainActor
    private func bitmap<V: View>(_ view: V) throws -> NSBitmapImageRep {
        let hosting = NSHostingView(rootView: view.environment(\.colorScheme, .light).accentColor(.red))
        hosting.appearance = NSAppearance(named: .aqua)
        hosting.frame.size = hosting.fittingSize
        hosting.layoutSubtreeIfNeeded()
        let bitmap = try XCTUnwrap(hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds))
        hosting.cacheDisplay(in: hosting.bounds, to: bitmap)
        return bitmap
    }

    @MainActor
    private func textLines(symbol: String?, bridge: Bool = false, subject: String? = "Heading") throws -> [CGRect] {
        let image = try bitmap(
            SidebarSessionLabel(title: "Heading", subject: subject, agentSymbol: symbol,
                                isExternalBridge: bridge, isSelected: false) {
                Text("Status").foregroundStyle(.blue)
            }.frame(width: 240, alignment: .leading).background(.white)
        )
        // Single-line comparisons isolate dark title pixels from the plain chat's gray icon.
        let darkness: CGFloat = subject == nil ? 0.3 : 0.85
        // Grayscale pixels belong to text, not the red symbols or blue status.
        var lines: [CGRect] = []
        var current: CGRect?
        for y in 0..<image.pixelsHigh {
            var row: CGRect?
            for x in 0..<image.pixelsWide {
                let color = try XCTUnwrap(image.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB))
                if color.redComponent < darkness,
                   abs(color.redComponent - color.greenComponent) < 0.02,
                   abs(color.greenComponent - color.blueComponent) < 0.02 {
                    let pixel = CGRect(x: x, y: y, width: 1, height: 1)
                    row = row.map { $0.union(pixel) } ?? pixel
                }
            }
            if let row {
                current = current.map { $0.union(row) } ?? row
            } else if let line = current {
                lines.append(line)
                current = nil
            }
        }
        if let current { lines.append(current) }
        if subject == nil, let first = lines.first {
            lines = [lines.dropFirst().reduce(first) { $0.union($1) }]
        }
        let scale = CGFloat(image.pixelsWide) / 240
        return lines.map { CGRect(x: $0.minX / scale, y: $0.minY / scale,
                                  width: $0.width / scale, height: $0.height / scale) }
    }

    @MainActor
    func testAgentTitlesIgnoreSymbolWidthAndSubjectsAlignWithTitles() throws {
        var leading: CGFloat?
        for bridge in [false, true] {
            for symbol in ["pencil", "brain.head.profile", "sparkles", "ellipsis"] {
                let lines = try textLines(symbol: symbol, bridge: bridge)
                XCTAssertEqual(lines.count, 2)
                guard lines.count == 2 else { continue }
                XCTAssertEqual(lines[0].minX, lines[1].minX, accuracy: 1)
                if let leading { XCTAssertEqual(lines[0].minX, leading, accuracy: 0.5) }
                leading = lines[0].minX
            }
        }
    }

    @MainActor
    func testSingleLineLabelsStayCompact() throws {
        for symbol in ["pencil", nil] as [String?] {
            let lines = try textLines(symbol: symbol, subject: nil)
            XCTAssertEqual(lines.count, 1)
            XCTAssertLessThan(try XCTUnwrap(lines.first).height, ChatMetrics.sidebarButtonHeight)
        }
    }

    @MainActor
    func testPlainAgentAndBridgeTitlesShareOneLeadingEdge() throws {
        var leading: CGFloat?
        for bridge in [false, true] {
            for symbol in ["pencil", "brain.head.profile", "sparkles", "ellipsis", nil] as [String?] {
                let lines = try textLines(symbol: symbol, bridge: bridge, subject: nil)
                XCTAssertEqual(lines.count, 1)
                let title = try XCTUnwrap(lines.first)
                XCTAssertLessThan(title.minX, 24, "One glyph slot must not reserve space for a second icon")
                if let leading { XCTAssertEqual(title.minX, leading, accuracy: 0.5) }
                leading = title.minX
            }
        }
    }
}
