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
                          resolveCustom: @escaping (String) -> ImageModelPreset? = { _ in nil },
                          persist: @escaping () -> Void) -> MediaModelChooser<ImageModelPreset> {
        MediaModelChooser.pane(
            all: ImageModelPreset.all, onThisMac: [], capability: "image",
            selected: model, lanModel: lan,
            capabilityOf: { _ in "" },
            resolveCustom: resolveCustom,
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

    /// A LAN pick adopts the preset the request will be SHAPED by — from the
    /// catalogue when the base id is known, and through the custom-family
    /// resolver when it isn't (a peer's own conversion). The fallback is the
    /// half that regressed in the one-row-chooser rewrite: `onSelectLan` only
    /// searched the catalogue, so a peer's custom H3 left `model` on whatever
    /// was picked before — the LTX-shapes-sent-to-H3 class.
    func testLanPickAdoptsCatalogThenCustomFamilyPreset() throws {
        let start = try XCTUnwrap(ImageModelPreset.all.first)
        let catalog = try XCTUnwrap(ImageModelPreset.all.last)
        let custom = ImageModelPreset.mageFlowTurbo.asCustom(id: "somebody/custom-mageflow")
        var model = start
        var lan: String? = nil
        let pane = makePane(model: Binding(get: { model }, set: { model = $0 }),
                            lan: Binding(get: { lan }, set: { lan = $0 }),
                            resolveCustom: { id in id == custom.id ? custom : nil },
                            persist: {})

        // Known base id → the catalogue preset.
        pane.onSelectLan("\(catalog.id)@studio")
        XCTAssertEqual(model.id, catalog.id)
        XCTAssertEqual(lan, "\(catalog.id)@studio")

        // Unknown to the catalogue → the custom-family resolver's answer.
        pane.onSelectLan("\(custom.id)@studio")
        XCTAssertEqual(model.id, custom.id,
                       "a peer's custom model must adopt its FAMILY preset — the request is shaped by `model`")

        // Nothing resolves → the previous pick stands (an unknown remote id
        // must not blank the pane).
        pane.onSelectLan("nobody/unknown@studio")
        XCTAssertEqual(model.id, custom.id)
    }

    /// Every Create pane's model control is the ONE shared chooser. The fifth
    /// copy (Music) was left on the old picker in the rewrite that promised
    /// "a fix to one copy left three behind" — this names the pane that drifts.
    /// A `BundleDownloadBar` in a gen view must be progress-only: the download
    /// BUTTON lives on the chooser's model row, and two stacked buttons
    /// (Download above Generate) was the pane's most confusing moment.
    func testEveryCreatePaneUsesTheSharedModelChooser() throws {
        let views = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent("Sources/MLXServe/Views")
        for pane in ["ImageGenView", "VideoGenView", "AudioGenView",
                     "MusicGenView", "Model3DGenView"] {
            let source = try String(
                contentsOf: views.appendingPathComponent("\(pane).swift"),
                encoding: .utf8)
            XCTAssertTrue(source.contains("MediaModelChooser.pane("),
                          "\(pane) must use the shared model chooser")
            XCTAssertFalse(source.contains("LanPick.selection("),
                           "\(pane) still wires the retired picker binding")
            for line in source.split(separator: "\n")
            where line.contains("BundleDownloadBar(") {
                XCTAssertTrue(line.contains("showsStartButton: false"),
                              "\(pane): a gen-view download bar is progress-only — the button is on the model row")
            }
        }
    }
}
