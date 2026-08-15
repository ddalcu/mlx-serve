import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Drag-and-drop file input for the Create panes, in ONE place.
///
/// Every gen pane already had a picker button; a drop target is the same
/// question asked with the mouse, and the two must never disagree about what
/// they accept. So the allow-lists live here rather than beside each picker,
/// and the panes hand this modifier the slot's remaining ROOM instead of each
/// re-deriving a cap. What a picker never had to answer — several files at
/// once, arriving out of order — is `MediaDrop`/`H3RefDrop` below.
enum MediaDropKind {
    case image, video, audio

    /// Mirrors the matching picker's `allowedContentTypes`, widened to the
    /// spellings the same decoder opens anyway (an `NSImage` reads webp/tiff;
    /// AVFoundation reads m4a/aiff). Kept as extensions, not UTTypes, because
    /// a drop hands us a plain file URL and this is the one thing we can read
    /// from it without touching the disk.
    var extensions: Set<String> {
        switch self {
        case .image: ["png", "jpg", "jpeg", "heic", "heif", "webp", "tiff", "tif", "gif", "bmp"]
        case .video: ["mov", "mp4", "m4v"]
        case .audio: ["wav", "mp3", "m4a", "aac", "aiff", "aif", "caf", "flac"]
        }
    }

    func accepts(_ url: URL) -> Bool {
        extensions.contains(url.pathExtension.lowercased())
    }

    /// Which list a mixed drop belongs to, or nil for a file no pane wants.
    static func of(_ url: URL) -> MediaDropKind? {
        [.image, .video, .audio].first { $0.accepts(url) }
    }
}

/// The verdict a drag needs while it is still in the air.
///
/// The type check can't wait for the providers: they resolve after SwiftUI has
/// accepted the drop and animated the file in, so a dropped `.txt` lit the
/// dashed border, flew home and did nothing. Naming the kind's UTTypes in the
/// target's `of:` is NOT the fix — a Finder drag carries `public.file-url` and
/// nothing else (the file's own content type is not on the pasteboard), so a
/// target registered for `.image`/`.movie`/`.audio` is never offered the drag
/// at all and every pane silently stopped accepting files. So the target
/// registers `.fileURL` — which is what a file drag actually is — and the
/// refusal happens in `validateDrop`, reading the drag pasteboard, which is
/// the one synchronous view of what is being dragged.
enum MediaDropValidation {
    /// Whether this drop is worth lighting the border for: something in it
    /// fits the slot, and the slot has room. A mixed target (`kind == nil`)
    /// takes anything one of the three lists would.
    ///
    /// A drag we could read NOTHING off (`urls` empty while it still claims to
    /// carry files) is no information, not a refusal — it falls back to the
    /// old behaviour, accepting and letting the post-resolve filter drop what
    /// the pane can't use. Bouncing everything is the worse failure of the
    /// two: it is the one that makes the pane look broken.
    static func accepts(_ urls: [URL], carriesFileURLs: Bool,
                        kind: MediaDropKind?, limit: Int) -> Bool {
        guard limit > 0 else { return false }
        guard !urls.isEmpty else { return carriesFileURLs }
        guard let kind else { return urls.contains { MediaDropKind.of($0) != nil } }
        return urls.contains { kind.accepts($0) }
    }

    /// The files a live drag is carrying. `.drag` is the system-wide dragging
    /// pasteboard — the same one the providers are bridged from — so this is
    /// the drop's own contents, not a guess from a type identifier.
    static func draggedFileURLs(
        _ pasteboard: NSPasteboard = NSPasteboard(name: .drag)
    ) -> [URL] {
        let options: [NSPasteboard.ReadingOptionKey: Any] = [.urlReadingFileURLsOnly: true]
        return pasteboard.readObjects(forClasses: [NSURL.self], options: options) as? [URL] ?? []
    }
}

enum MediaDrop {
    /// Turns the raw per-provider results into the files a slot will take.
    ///
    /// `slots` is positional — one entry per provider, in the order the user
    /// dropped them, `nil` where a provider failed to resolve. That is the
    /// whole point: providers resolve asynchronously and independently, so
    /// appending as each finishes puts them in a RACE order, and the
    /// reference lists are numbered (`<Picture 2>` is a contract with the
    /// model, per type, in list order). A hole is skipped without shifting
    /// its neighbours; over the cap the earliest files win.
    static func accepted(_ slots: [URL?], as kind: MediaDropKind, limit: Int) -> [URL] {
        guard limit > 0 else { return [] }
        return Array(slots.compactMap { $0 }.filter { kind.accepts($0) }.prefix(limit))
    }
}

/// Where a dropped image lands on the Image pane, which is the one pane with
/// two destinations. Kept pure and out of the view so the routing can be read
/// (and tested) on its own.
enum ImageDropPlacement {
    /// How many files this pane can actually take, which is what the target
    /// advertises to SwiftUI — the drop animates in on the strength of it, so
    /// a number bigger than `place` can use is a file swallowed with the
    /// accept animation playing. Zero bounces, which is the honest answer for
    /// a pane with a source AND a full reference list: the user has to remove
    /// one first, same as the Add button disappearing at the cap.
    static func room(source: URL?, editing: Bool, refs: Int, refLimit: Int) -> Int {
        // Variation is ONE slot whether it is empty or being replaced. The
        // reference list is not part of that budget — it isn't even shown.
        guard editing else { return 1 }
        return (source == nil ? 1 : 0) + max(0, refLimit - refs)
    }

    /// Priority: fill the empty source first, then references while editing;
    /// otherwise the drop REPLACES the source, since variation mode has a
    /// single image slot and silently discarding the file would look like the
    /// drop missed. A full reference list drops the extra file and leaves the
    /// source alone — replacing a source the user chose because their
    /// reference list happened to be full is the surprise.
    ///
    /// `editing` means `effectiveEditMode`, which already includes the
    /// backend's `supportsReferenceEdit`.
    static func place(_ urls: [URL], source: URL?, editing: Bool,
                      refs: [URL], refLimit: Int) -> (source: URL?, refs: [URL]) {
        guard let first = urls.first else { return (source, refs) }
        // One slot, so the FIRST file wins and the rest have nowhere to go —
        // replacing the source once per file kept the LAST of a multi-file
        // drop, which reads as the pane choosing a file at random. `room`
        // caps the drop at one anyway; this keeps the two from disagreeing.
        guard editing else { return (first, refs) }
        var source = source
        var refs = refs
        for url in urls {
            if source == nil {
                source = url
            } else if refs.count < refLimit {
                refs.append(url)
            }
        }
        return (source, refs)
    }
}

/// Splits one drop across the H3 reference lists. The section is a single
/// target — three separate ones would make attaching a clip a game of hitting
/// the right 20pt row — so the file's own type picks its list, under both that
/// type's cap and the combined budget the Add buttons already respect.
enum H3RefDrop {
    static func route(_ urls: [URL], images: [URL], videos: [URL],
                      audios: [URL]) -> (images: [URL], videos: [URL], audios: [URL]) {
        var images = images, videos = videos, audios = audios
        for url in urls {
            guard let kind = MediaDropKind.of(url) else { continue }
            let attached = images.count + videos.count + audios.count
            switch kind {
            case .image:
                if H3RefLimits.remaining(perType: H3RefLimits.images, current: images.count,
                                         totalAttached: attached) > 0 { images.append(url) }
            case .video:
                if H3RefLimits.remaining(perType: H3RefLimits.videos, current: videos.count,
                                         totalAttached: attached) > 0 { videos.append(url) }
            case .audio:
                if H3RefLimits.remaining(perType: H3RefLimits.audios, current: audios.count,
                                         totalAttached: attached) > 0 { audios.append(url) }
            }
        }
        return (images, videos, audios)
    }
}

// MARK: - The view side

/// Collects a drop's providers and delivers them ONCE, in drop order.
private enum MediaDropLoader {
    /// Boxed so the escaping per-provider callbacks can fill their own slot
    /// without capturing a `var` across threads.
    private final class Slots: @unchecked Sendable {
        private let lock = NSLock()
        private var urls: [URL?]
        init(count: Int) { urls = .init(repeating: nil, count: count) }
        func set(_ url: URL?, at i: Int) { lock.lock(); urls[i] = url; lock.unlock() }
        var all: [URL?] { lock.lock(); defer { lock.unlock() }; return urls }
    }

    /// Runs after the drop is accepted, so this is the SECOND filter: the
    /// verdict in front of it works off the pasteboard, and this one off what
    /// the providers actually resolved to. Returns false when there was
    /// nothing to resolve at all.
    static func load(_ providers: [NSItemProvider], kind: MediaDropKind?, limit: Int,
                     completion: @escaping ([URL]) -> Void) -> Bool {
        let files = providers.filter {
            $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier)
        }
        guard !files.isEmpty, limit > 0 else { return false }
        let slots = Slots(count: files.count)
        let group = DispatchGroup()
        for (i, provider) in files.enumerated() {
            group.enter()
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                slots.set(url, at: i)
                group.leave()
            }
        }
        group.notify(queue: .main) {
            // A nil kind means the caller sorts the types itself (the H3
            // references section); it still wants the drop order and the cap.
            let resolved = slots.all.compactMap { $0 }
            let urls = kind.map { MediaDrop.accepted(slots.all, as: $0, limit: limit) }
                ?? Array(resolved.prefix(limit))
            guard !urls.isEmpty else { return }
            completion(urls)
        }
        return true
    }
}

/// Refuses the drag before it is ever highlighted, then hands the accepted
/// files over. A `DropDelegate` rather than `onDrop(of:isTargeted:perform:)`
/// purely for `validateDrop`: it is the only hook that answers WHILE the drag
/// is in the air, and SwiftUI calls `dropEntered` (where the border lights) on
/// validated drops only, so one verdict does both jobs.
private struct MediaDropDelegate: DropDelegate {
    let kind: MediaDropKind?
    let limit: Int
    @Binding var isTargeted: Bool
    let onURLs: ([URL]) -> Void

    func validateDrop(info: DropInfo) -> Bool {
        MediaDropValidation.accepts(MediaDropValidation.draggedFileURLs(),
                                    carriesFileURLs: info.hasItemsConforming(to: [.fileURL]),
                                    kind: kind, limit: limit)
    }

    func dropEntered(info: DropInfo) { isTargeted = true }
    func dropExited(info: DropInfo) { isTargeted = false }

    func performDrop(info: DropInfo) -> Bool {
        isTargeted = false
        return MediaDropLoader.load(info.itemProviders(for: [.fileURL]),
                                    kind: kind, limit: limit, completion: onURLs)
    }
}

private struct MediaDropModifier: ViewModifier {
    let kind: MediaDropKind?
    let limit: Int
    @Binding var isTargeted: Bool
    let onURLs: ([URL]) -> Void

    private static let cornerRadius: CGFloat = 8

    func body(content: Content) -> some View {
        content
            .padding(6)
            // Without contentShape the gaps between rows aren't hit-testable
            // and a drop there falls through to the ScrollView behind — which
            // is most of a section made of small rows.
            .contentShape(RoundedRectangle(cornerRadius: Self.cornerRadius))
            // `.fileURL` is what a file drag IS — see `MediaDropValidation`
            // for why the kind's own UTTypes belong in the verdict rather than
            // here.
            .onDrop(of: [.fileURL],
                    delegate: MediaDropDelegate(kind: kind, limit: limit,
                                                isTargeted: $isTargeted, onURLs: onURLs))
            .overlay {
                RoundedRectangle(cornerRadius: Self.cornerRadius)
                    .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 2, dash: [6]))
                    .opacity(isTargeted ? 1 : 0)
                    .allowsHitTesting(false)
            }
    }
}

extension View {
    /// Makes this section a drop target for `kind`, highlighted while a drag
    /// is over it. `limit` is the slot's REMAINING room (1 for a single-image
    /// slot, since a drop there replaces).
    func mediaDrop(_ kind: MediaDropKind, limit: Int = 1, isTargeted: Binding<Bool>,
                   onURLs: @escaping ([URL]) -> Void) -> some View {
        modifier(MediaDropModifier(kind: kind, limit: limit, isTargeted: isTargeted, onURLs: onURLs))
    }

    /// Mixed-type variant: every file is handed over and the caller routes by
    /// type (`H3RefDrop`).
    func mediaDropAnyKind(limit: Int, isTargeted: Binding<Bool>,
                          onURLs: @escaping ([URL]) -> Void) -> some View {
        modifier(MediaDropModifier(kind: nil, limit: limit, isTargeted: isTargeted, onURLs: onURLs))
    }
}

/// The empty state's target: something to see and aim at, rather than a bare
/// button inside an invisible drop region.
struct MediaDropWell: View {
    let title: String
    let systemImage: String
    let isTargeted: Bool
    let action: () -> Void

    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: systemImage)
                .font(.title2)
                .foregroundStyle(.secondary)
            Button(title, action: action)
                .buttonStyle(.link)
                .font(.caption)
            Text("or drag one here")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, minHeight: 84)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.secondary.opacity(isTargeted ? 0.12 : 0.06))
        )
    }
}
