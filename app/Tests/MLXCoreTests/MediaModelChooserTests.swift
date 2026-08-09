import XCTest
import SwiftUI
@testable import MLXCore

/// The Create panes persist their model pick on `.onChange(of: model)` — which
/// never fires when the local pick already IS the clicked preset. Deselecting a
/// LAN model back to the same local one cleared `lanModel` in memory only, so
/// a relaunch restored the LAN pick the user explicitly left. The shared
/// `MediaModelChooser.pane` factory persists in its own onSelect/onDownload
/// (like onSelectLan always did); one factory, so all four panes are covered.
@MainActor
final class MediaModelChooserTests: XCTestCase {

    private func makePane(model: Binding<ImageModelPreset>, lan: Binding<String?>,
                          persist: @escaping () -> Void) -> MediaModelChooser<ImageModelPreset> {
        MediaModelChooser.pane(
            all: ImageModelPreset.all, onThisMac: [], capability: "image",
            selected: model, lanModel: lan,
            capabilityOf: { _ in "" },
            bundleOf: { $0.bundle },
            downloads: DownloadManager(modelsRoot: NSTemporaryDirectory()),
            onDownloadFinished: {},
            persist: persist)
    }

    func testDeselectingLanBackToTheSameLocalPresetStillPersists() throws {
        let preset = try XCTUnwrap(ImageModelPreset.all.first)
        var model = preset
        var lan: String? = "\(preset.id)@studio"
        var persisted = 0
        let pane = makePane(model: Binding(get: { model }, set: { model = $0 }),
                            lan: Binding(get: { lan }, set: { lan = $0 }),
                            persist: { persisted += 1 })

        // Click the local row for the model that is ALREADY the local pick.
        pane.onSelect(preset)

        XCTAssertNil(lan, "the LAN pick is cleared in memory")
        XCTAssertEqual(model.id, preset.id)
        XCTAssertGreaterThan(persisted, 0,
                             "…and persisted — onChange(of: model) cannot fire for an unchanged value")
    }
    // onDownload carries the same persist() (it also selects); not exercised
    // here because it would kick off a real bundle transfer.
}
