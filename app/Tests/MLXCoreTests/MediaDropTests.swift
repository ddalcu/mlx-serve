import XCTest
@testable import MLXCore

/// The pure half of the Create panes' drag-and-drop: which files a drop keeps,
/// how much ROOM the slot it lands on has, and which list each file joins.
///
/// The room matters as much as the placement, because SwiftUI animates the
/// drop in on the strength of it: a target that advertises room the placement
/// then has no use for swallows the file with the accept animation playing,
/// which is the one outcome a drop must never have.
final class MediaDropTests: XCTestCase {

    private func u(_ name: String) -> URL { URL(fileURLWithPath: "/tmp/drop/\(name)") }

    /// The drag-time verdict on a readable drag — the case every drop from
    /// Finder takes.
    private func accepts(_ names: [String], kind: MediaDropKind?, limit: Int = 1) -> Bool {
        MediaDropValidation.accepts(names.map(u), carriesFileURLs: true,
                                    kind: kind, limit: limit)
    }

    // MARK: - What a drop keeps

    /// Providers resolve independently, so one failing must not shift the
    /// files after it: the reference lists are numbered and that numbering is
    /// a contract with the model.
    func testAcceptedKeepsDropOrderAcrossAFailedProvider() {
        let kept = MediaDrop.accepted([u("a.png"), nil, u("c.png")], as: .image, limit: 9)
        XCTAssertEqual(kept, [u("a.png"), u("c.png")])
    }

    /// The file type is the slot's own allow-list, and over the cap the
    /// EARLIEST files win — a drop of four onto one slot keeps the first.
    func testAcceptedDropsWhatTheSlotCannotOpenAndSpendsTheCapEarliestFirst() {
        let mixed = [u("a.txt"), u("b.png"), u("c.mov"), u("d.png")]
        XCTAssertEqual(MediaDrop.accepted(mixed, as: .image, limit: 9), [u("b.png"), u("d.png")])
        XCTAssertEqual(MediaDrop.accepted(mixed, as: .image, limit: 1), [u("b.png")])
        XCTAssertEqual(MediaDrop.accepted(mixed, as: .image, limit: 0), [])
    }

    // MARK: - The verdict the drag itself gets

    /// The whole point of validating: a file the pane cannot open is refused
    /// while the drag is still in the air, so it never lights the border.
    func testAFileThePaneCannotOpenIsRefusedBeforeItAnimatesIn() {
        XCTAssertFalse(accepts(["notes.txt"], kind: .image))
        XCTAssertFalse(accepts(["clip.mov"], kind: .image))
        XCTAssertTrue(accepts(["a.png"], kind: .image))
    }

    /// One usable file in a mixed drop is enough to accept it — the rest are
    /// dropped on the way in, which is what `accepted` above is for.
    func testAMixedDropIsAcceptedWhenAnySingleFileFits() {
        XCTAssertTrue(accepts(["notes.txt", "a.png"], kind: .image))
        XCTAssertFalse(accepts(["notes.txt", "readme.md"], kind: .image))
    }

    /// The mixed target (H3 references) takes any of the three, and nothing
    /// else.
    func testTheMixedTargetTakesAnyOfTheThreeTypesAndNothingElse() {
        for name in ["a.png", "b.mov", "c.wav"] {
            XCTAssertTrue(accepts([name], kind: nil, limit: 12), name)
        }
        XCTAssertFalse(accepts(["notes.txt"], kind: nil, limit: 12))
    }

    /// A full slot bounces rather than swallowing the file with the accept
    /// animation playing — the same `room` the target advertises.
    func testAFullSlotIsRefusedWhateverIsBeingDragged() {
        XCTAssertFalse(accepts(["a.png"], kind: .image, limit: 0))
        XCTAssertFalse(MediaDropValidation.accepts([], carriesFileURLs: true,
                                                   kind: .image, limit: 0))
    }

    /// A drag whose files we cannot read is NO INFORMATION: it is accepted and
    /// filtered after the providers resolve, exactly as before. Bouncing
    /// everything is the worse of the two failures — it is the one that makes
    /// the pane look broken.
    func testADragWeCannotReadFallsBackToFilteringAfterTheResolve() {
        XCTAssertTrue(MediaDropValidation.accepts([], carriesFileURLs: true,
                                                  kind: .image, limit: 1))
        XCTAssertFalse(MediaDropValidation.accepts([], carriesFileURLs: false,
                                                   kind: .image, limit: 1))
    }

    /// The verdict and the post-resolve filter are ONE allow-list: anything
    /// `accepted` would keep must survive the gate in front of it, or the
    /// picker opens a file its own drop target bounces.
    func testEveryAcceptedExtensionAlsoPassesTheDragTimeVerdict() {
        for kind in [MediaDropKind.image, .video, .audio] {
            for ext in kind.extensions {
                XCTAssertTrue(accepts(["f.\(ext)"], kind: kind),
                              "\(ext) is accepted by \(kind) but its drop target would bounce it")
                XCTAssertTrue(accepts(["f.\(ext)"], kind: nil),
                              "\(ext) is accepted by \(kind) but the mixed target would bounce it")
            }
        }
    }

    /// The verdict is only as good as the read behind it: a file drag puts
    /// `public.file-url` items on the pasteboard, and that is what the helper
    /// has to turn back into paths (with anything that isn't a file — a URL
    /// dragged out of a browser — left out of the count).
    func testTheDragPasteboardReadRecoversTheDroppedFilesAndOnlyThose() throws {
        let pasteboard = NSPasteboard(name: .init("com.mlxserve.tests.mediadrop"))
        pasteboard.clearContents()
        let file = NSPasteboardItem()
        file.setString(u("a.png").absoluteString, forType: .fileURL)
        let link = NSPasteboardItem()
        link.setString("https://example.com/b.png", forType: .URL)
        pasteboard.writeObjects([file, link])

        let read = MediaDropValidation.draggedFileURLs(pasteboard)
        XCTAssertEqual(read.map(\.lastPathComponent), ["a.png"])
        XCTAssertTrue(MediaDropValidation.accepts(read, carriesFileURLs: true,
                                                  kind: .image, limit: 1))
    }

    /// A Finder drag carries `public.file-url` and NOTHING else — the file's
    /// own content type is not on the pasteboard — so a target registered for
    /// `.image`/`.movie`/`.audio` is never offered the drag at all. That is
    /// what made every Create pane stop accepting files; the type filtering it
    /// was meant to buy lives in `validateDrop` instead.
    func testTheDropTargetRegistersTheFileURLTypeAndRefusesBeforeTheHighlight() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Views/MediaDropTarget.swift")
        let source = try String(contentsOf: url, encoding: .utf8)
        let body = source.components(separatedBy: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }
            .joined(separator: "\n")
        XCTAssertTrue(body.contains(".onDrop(of: [.fileURL]"),
                      "a Finder drag matches nothing but .fileURL")
        XCTAssertTrue(body.contains("func validateDrop"),
                      "without a synchronous verdict the type check can only run after the drop is accepted")
    }

    // MARK: - Where a dropped image lands on the Image pane

    /// Variation mode is ONE slot. Replacing the source once per file kept the
    /// LAST of a multi-file drop and silently discarded the rest, which reads
    /// as the pane picking a file at random.
    func testAVariationDropKeepsTheFirstFileNotTheLast() {
        let placed = ImageDropPlacement.place(
            [u("a.png"), u("b.png"), u("c.png")], source: u("source.png"),
            editing: false, refs: [], refLimit: 3)
        XCTAssertEqual(placed.source, u("a.png"))
        XCTAssertTrue(placed.refs.isEmpty)
    }

    /// The empty source is filled first, then the references — one drop of
    /// several files fills the whole pane in the order they were dropped.
    func testAnEditDropFillsTheSourceThenTheReferencesInDropOrder() {
        let placed = ImageDropPlacement.place(
            [u("a.png"), u("b.png"), u("c.png")], source: nil,
            editing: true, refs: [], refLimit: 3)
        XCTAssertEqual(placed.source, u("a.png"))
        XCTAssertEqual(placed.refs, [u("b.png"), u("c.png")])
    }

    /// A full reference list leaves the source ALONE — replacing an image the
    /// user chose because their reference list happened to be full is the
    /// surprise.
    func testAFullReferenceListNeverReplacesTheSource() {
        let refs = [u("r1.png"), u("r2.png"), u("r3.png")]
        let placed = ImageDropPlacement.place(
            [u("a.png")], source: u("source.png"), editing: true, refs: refs, refLimit: 3)
        XCTAssertEqual(placed.source, u("source.png"))
        XCTAssertEqual(placed.refs, refs)
    }

    /// …which is exactly why that pane must report NO room: the old limit
    /// (`1 + refLimit - refs.count`) said 1 with the source set and the
    /// references full, so the file animated in and landed nowhere.
    func testAPaneWithNothingLeftToFillReportsNoRoom() {
        XCTAssertEqual(ImageDropPlacement.room(source: u("source.png"), editing: true,
                                               refs: 3, refLimit: 3), 0)
        XCTAssertEqual(ImageDropPlacement.room(source: u("source.png"), editing: true,
                                               refs: 1, refLimit: 3), 2)
        XCTAssertEqual(ImageDropPlacement.room(source: nil, editing: true,
                                               refs: 0, refLimit: 3), 4)
    }

    /// Variation mode always has room for exactly one file, whether the slot
    /// is empty or being replaced — the references it cannot use are not part
    /// of the budget.
    func testVariationModeAlwaysHasRoomForExactlyOneFile() {
        XCTAssertEqual(ImageDropPlacement.room(source: nil, editing: false,
                                               refs: 0, refLimit: 3), 1)
        XCTAssertEqual(ImageDropPlacement.room(source: u("source.png"), editing: false,
                                               refs: 0, refLimit: 3), 1)
    }

    // MARK: - The H3 references section

    /// One target, three lists: the file's own type picks its list, under both
    /// that type's cap and the combined budget the Add buttons respect.
    func testAMixedDropRoutesEachFileToItsOwnListUnderBothCaps() {
        let routed = H3RefDrop.route([u("a.png"), u("b.mov"), u("c.wav")],
                                     images: [], videos: [], audios: [])
        XCTAssertEqual(routed.images, [u("a.png")])
        XCTAssertEqual(routed.videos, [u("b.mov")])
        XCTAssertEqual(routed.audios, [u("c.wav")])
    }

    /// Per-type cap and combined cap both bind, and a file no pane wants is
    /// skipped without spending either.
    func testRoutingRespectsThePerTypeCapTheCombinedCapAndSkipsUnknownFiles() {
        let images = (1...8).map { u("i\($0).png") }        // 8 of 9
        let videos = (1...3).map { u("v\($0).mov") }        // 3 of 3, and 11 of 12 total
        let routed = H3RefDrop.route([u("new.png"), u("also.png"), u("more.mov"),
                                      u("notes.txt"), u("tune.wav")],
                                     images: images, videos: videos, audios: [])
        // The first image takes the last of the combined budget; everything
        // after it is refused by one cap or the other.
        XCTAssertEqual(routed.images, images + [u("new.png")])
        XCTAssertEqual(routed.videos, videos)
        XCTAssertEqual(routed.audios, [])
    }
}
