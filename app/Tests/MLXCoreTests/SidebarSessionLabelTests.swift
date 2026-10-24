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
        // Grayscale pixels belong to text, not the red symbols or blue status.
        var lines: [CGRect] = []
        var current: CGRect?
        for y in 0..<image.pixelsHigh {
            var row: CGRect?
            for x in 0..<image.pixelsWide {
                let color = try XCTUnwrap(image.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB))
                if color.redComponent < 0.85,
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
        let scale = CGFloat(image.pixelsWide) / 240
        return lines.map { CGRect(x: $0.minX / scale, y: $0.minY / scale,
                                  width: $0.width / scale, height: $0.height / scale) }
    }

    @MainActor
    func testAgentTitlesIgnoreSymbolWidthAndSubjectsAlignWithTitles() throws {
        for bridge in [false, true] {
            var leading: CGFloat?
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
    func testSidebarSnapshot() throws {
        let symbols = ["pencil", "brain.head.profile", "sparkles", "ellipsis"]
        let view = VStack(alignment: .leading, spacing: 8) {
            ForEach(symbols, id: \.self) { symbol in
                SidebarSessionLabel(title: "Agent \(symbol)", subject: "Conversation summary",
                                    agentSymbol: symbol, isExternalBridge: false, isSelected: false) {
                    EmptyView()
                }
            }
            SidebarSessionLabel(title: "Telegram agent", subject: "Bridge conversation summary",
                                agentSymbol: "paperplane", isExternalBridge: true, isSelected: false) {
                Image(systemName: "checkmark.circle.fill")
            }
            SidebarSessionLabel(title: "New Chat", subject: nil, agentSymbol: nil,
                                isExternalBridge: false, isSelected: false) { EmptyView() }
        }.padding(12).frame(width: 280, alignment: .leading)
            .background(Color(nsColor: .windowBackgroundColor))
        let image = try bitmap(view)
        XCTAssertGreaterThan(image.pixelsHigh, 0)
        if let path = ProcessInfo.processInfo.environment["SIDEBAR_UI_SNAPSHOT"] {
            let png = try XCTUnwrap(image.representation(using: .png, properties: [:]))
            try png.write(to: URL(fileURLWithPath: path))
        }
    }
}
