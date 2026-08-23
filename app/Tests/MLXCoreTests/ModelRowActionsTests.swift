import XCTest
@testable import MLXCore

/// The three things a My Models row could never do: show you where the file
/// is, let you delete it when it lives in a folder you own through another
/// app, or offer any of it on right-click.
///
/// The row had exactly two controls — "Use", and either a trash or a mystery
/// badge. The badge was `externaldrive.badge.icloud`, which reads as "external
/// drive" or "cloud" and meant neither; clicking it did nothing because it was
/// an `Image`, not a `Button`. So for every model outside `~/.mlx-serve` the
/// row's answer to "where is this and how do I get rid of it" was a glyph.
final class ModelRowActionsTests: XCTestCase {

    private func model(source: LocalModelSource,
                       defect: ModelDefect? = nil,
                       quantFile: String? = nil) -> LocalModel {
        LocalModel(id: "\(source.rawValue):org/m", name: "org/m", path: "/tmp/root/org/m",
                   sizeFormatted: "1 GB", modelType: "qwen3", source: source, kind: .base,
                   quantFile: quantFile, defect: defect)
    }

    /// Our own tree: trash from the start, no lock to click.
    func testAnOwnedModelNeedsNoUnlock() {
        let m = model(source: .mlxServe)
        XCTAssertTrue(ModelRowActions.showsTrash(m, unlocked: false))
        XCTAssertFalse(ModelRowActions.showsLock(m, unlocked: false))
    }

    /// A foreign tree starts locked — the caution is right, the dead end was not.
    func testAForeignModelStartsLockedAndUnlocksToATrash() {
        let m = model(source: .osaurus)
        XCTAssertTrue(ModelRowActions.showsLock(m, unlocked: false))
        XCTAssertFalse(ModelRowActions.showsTrash(m, unlocked: false))

        XCTAssertTrue(ModelRowActions.showsTrash(m, unlocked: true),
                      "unlocking must actually produce a trash, or the lock is the old dead badge")
        XCTAssertFalse(ModelRowActions.showsLock(m, unlocked: true))
    }

    /// Every locked row must say which app owns it. A lock you cannot explain
    /// is the mystery glyph again with a nicer shape.
    func testEveryLockedSourceExplainsItself() {
        for source in LocalModelSource.allCases where source != .mlxServe {
            let m = model(source: source)
            let help = ModelRowActions.lockHelp(m)
            XCTAssertFalse(help.isEmpty, "\(source) has no lock explanation")
            XCTAssertTrue(help.lowercased().contains("delete"),
                          "\(source)'s lock never says clicking it enables deleting: \(help)")
        }
    }

    /// A broken folder skips the lock entirely — it is junk, not someone's model.
    func testABrokenFolderIsTrashableWithoutUnlocking() {
        let m = model(source: .osaurus, defect: .missingWeights)
        XCTAssertTrue(ModelRowActions.showsTrash(m, unlocked: false))
        XCTAssertFalse(ModelRowActions.showsLock(m, unlocked: false))
    }

    /// Deleting into a foreign tree must name the PATH. "Delete org/m?" does not
    /// tell you which of four tools' copies is about to go.
    func testTheForeignDeleteConfirmationNamesTheFolder() {
        let m = model(source: .osaurus)
        XCTAssertTrue(ModelRowActions.deleteMessage(m).contains(m.path))
    }

    /// The HF cache is the one tree where deletion damages models we are not
    /// deleting: snapshots share blobs. It is still allowed — it is the user's
    /// disk — but the confirmation has to say so.
    func testTheHuggingFaceCacheWarnsAboutSharedBlobs() {
        let msg = ModelRowActions.deleteMessage(model(source: .huggingFace))
        XCTAssertTrue(msg.lowercased().contains("hugging face") || msg.lowercased().contains("cache"))
        XCTAssertTrue(msg.lowercased().contains("share") || msg.lowercased().contains("other models"))
    }

    /// A GGUF row is ONE quant. Deleting it must not promise its siblings.
    func testAQuantRowStillPromisesOnlyItself() {
        let msg = ModelRowActions.deleteMessage(model(source: .mlxServe, quantFile: "m-Q4_K_M.gguf"))
        XCTAssertTrue(msg.lowercased().contains("other quants"))
    }

    /// Reveal is the one action every row gets, including broken folders —
    /// especially those, since seeing the folder is how you confirm it is junk.
    func testRevealIsOfferedForEveryRow() {
        for source in LocalModelSource.allCases {
            XCTAssertFalse(ModelRowActions.revealHelp(model(source: source)).isEmpty)
        }
        XCTAssertFalse(ModelRowActions.revealHelp(model(source: .osaurus, defect: .missingWeights)).isEmpty)
    }

    /// The row is the only place these actions exist, so a control that is not
    /// wired into the view is a control the user does not have. Pins the four
    /// against the source, since none of them is reachable from a unit test.
    func testTheRowActuallyWiresEveryControl() throws {
        let src = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Views/ModelBrowserView.swift")
        let text = try String(contentsOf: src, encoding: .utf8)
        for needle in ["activateFileViewerSelecting", ".contextMenu", "ModelRowActions.showsLock",
                       "ModelRowActions.showsTrash", "systemName: \"folder\"", "systemName: \"lock\""] {
            XCTAssertTrue(text.contains(needle), "the row never wires \(needle)")
        }
        XCTAssertFalse(text.contains("externaldrive.badge.icloud"),
                       "the mystery badge is back")
    }
}
