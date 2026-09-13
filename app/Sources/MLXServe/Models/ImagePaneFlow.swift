import Foundation
import AppKit
import ImageIO

/// Pure flow logic for the Image pane — the decisions that have to hold
/// whether or not a view is on screen, kept out of the view so they can be
/// tested without one.

// MARK: - Handing a finished picture to the next run

/// Turning a result the pane just produced into the source image for the next
/// one. The button lives on the preview ("Enlarge"), so its input is a path
/// the pane is currently DRAWING — which is not the same as a path that is
/// still on disk, and not necessarily a moment when swapping the source is
/// safe.
enum ImageSourceHandoff {

    enum Outcome: Equatable {
        /// Attach this file as the source image.
        case accepted(URL)
        /// The file is gone (deleted, or on a volume that went away). Carries
        /// the last path component, which is what an error sentence can name.
        case missing(String)
        /// A run is in flight. Its own source is what produced what is on
        /// screen, so it stands until the run ends.
        case busy
    }

    static func resolve(path: String,
                        isRunning: Bool,
                        exists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }) -> Outcome {
        if isRunning { return .busy }
        guard exists(path) else { return .missing((path as NSString).lastPathComponent) }
        return .accepted(URL(fileURLWithPath: path))
    }
}

// MARK: - What a source image is FOR

/// The three things the Image pane can do with an attached picture. This
/// replaces the old top-level `Create | Upscale` switch: "Upscale" was never a
/// sibling of "Create" — Create is a place you stay in and write prompts,
/// while enlarging is one thing you do to one picture and then walk back from.
/// Modelling it as a verb ON a source puts it beside the two verbs that were
/// already there, and asks the pane's real question once: *I have a picture —
/// what do I want done to it?*
///
/// `edit` and `variation` are capabilities of the IMAGE model. `enlarge` is
/// not: it runs SeedVR2, a different model family entirely, so it is available
/// on every preset and cannot be taken away by a model switch.
enum ImageSourceVerb: String, CaseIterable, Identifiable, Codable {
    case edit
    case variation
    case enlarge

    var id: String { rawValue }

    var label: String {
        switch self {
        case .edit: return "Edit"
        case .variation: return "Variation"
        case .enlarge: return "Enlarge"
        }
    }
}

extension ImageSourceVerb {

    /// What this model's backend can actually be asked to do with a source
    /// image, in picker order. Never empty — `enlarge` always applies.
    ///
    /// A txt2img-only preset (Mage-Flow Turbo: no in-context editing, no VAE
    /// encoder) returns `[.enlarge]` alone, which also closes a dead state
    /// that shipped before this existed: a source image attached there drew a
    /// thumbnail, offered no mode, and made Generate send `image` without
    /// `mode:"edit"` — a named 400.
    static func available(for preset: ImageModelPreset) -> [ImageSourceVerb] {
        var verbs: [ImageSourceVerb] = []
        if preset.supportsReferenceEdit { verbs.append(.edit) }
        if preset.supportsImg2Img { verbs.append(.variation) }
        verbs.append(.enlarge)
        return verbs
    }

    /// Keep a selection meaningful across a model switch. A verb the new model
    /// cannot serve falls back to the first one it can — the old
    /// `effectiveEditMode` rule ("where editing is the only thing a source can
    /// do, a source MEANS edit"), now stated once for all three verbs.
    static func resolve(_ wanted: ImageSourceVerb, for preset: ImageModelPreset) -> ImageSourceVerb {
        let ok = available(for: preset)
        return ok.contains(wanted) ? wanted : (ok.first ?? .enlarge)
    }
}

// MARK: - One preview, two services

/// Which run the pane's single preview is showing.
///
/// Generation and enlargement are separate services with separate phases, and
/// the preview used to belong to whichever PANE was mounted — so setting up an
/// enlarge threw away the generated image you were looking at. Deciding this
/// from the two phases plus the selected strip row instead of from the current
/// verb is what makes that impossible: `resolve` deliberately takes no verb,
/// because what is on screen is a property of what has been MADE and what the
/// user picked out of it, not of what the controls are set to.
enum ImagePanePreview {

    /// Which service produced what is being shown. Kept through to the view
    /// because the two fail for different reasons and offer different
    /// remedies — a prompt to change, or a scale to lower.
    enum Origin: Equatable { case generated, enlarged }

    /// One service's phase, flattened to the four states the preview cares
    /// about. The services' own phases carry more (step counts, logs); mapping
    /// down here keeps this resolver free of their actor isolation.
    enum Run: Equatable {
        case idle
        case running(String)
        case done(String)
        case failed(String)
    }

    enum State: Equatable {
        case empty
        case running(Origin, String)
        case result(Origin, String)
        case failed(Origin, String)
    }

    /// - Parameter selected: the strip row the user is looking at. Set on
    ///   every completion too, so a finished run selects itself.
    static func resolve(generate: Run, enlarge: Run, selected: MediaSessionItem?) -> State {
        // A run in flight always wins: it is the only thing on screen that is
        // still changing. With both somehow in flight the selection breaks the
        // tie, so the answer is stable rather than order-dependent.
        let running: [(Origin, Run)] = [(.generated, generate), (.enlarged, enlarge)]
            .filter { if case .running = $0.1 { return true } else { return false } }
        if running.count == 2, let origin = selected?.origin,
           let pick = running.first(where: { $0.0 == origin }),
           case .running(let msg) = pick.1 {
            return .running(pick.0, msg)
        }
        if let (origin, run) = running.first, case .running(let msg) = run {
            return .running(origin, msg)
        }

        // Then whatever is selected — which is how picking an older picture out
        // of the strip brings it back, and how a just-finished run takes over.
        if let selected { return .result(selected.origin, selected.path) }

        // A failure has no row to select and is news, so it takes the preview.
        for (origin, run) in [(Origin.generated, generate), (.enlarged, enlarge)] {
            if case .failed(let msg) = run { return .failed(origin, msg) }
        }
        // Last resort: the pane can be remounted while a service still holds a
        // finished run from an earlier visit, before anything has selected it.
        for (origin, run) in [(Origin.generated, generate), (.enlarged, enlarge)] {
            if case .done(let path) = run { return .result(origin, path) }
        }
        return .empty
    }
}

// MARK: - The session strip

/// One picture this pane has made.
struct MediaSessionItem: Identifiable, Equatable {
    let path: String
    let origin: ImagePanePreview.Origin

    var id: String { path }
    var filename: String { (path as NSString).lastPathComponent }
}

/// Everything the Image pane has produced, as one timeline.
///
/// Generation and enlargement write to different folders and are tracked by
/// different services, but "what have I made" is one question — so the strip
/// interleaves them. Grouping by which service produced a picture is a
/// distinction nobody is looking for.
enum MediaSessionStrip {

    /// How many rows the strip holds. Each service already caps its own
    /// `recent` at 60; this is the cap on the merged view of them.
    static let limit = 60

    /// The `yyyy-MM-dd_HH-mm-ss` stamp both writers put at the head of every
    /// output filename, or nil for a file that arrived some other way (the
    /// output folders are ordinary folders and `recent` scans whatever is in
    /// them). Reading the name is what lets the merge order 120 files without
    /// stat-ing any of them on every redraw.
    static func timestampKey(_ path: String) -> String? {
        let name = (path as NSString).lastPathComponent
        guard name.count >= 19 else { return nil }
        let stamp = String(name.prefix(19))
        // yyyy-MM-dd_HH-mm-ss — check the shape, not just the length, so a
        // file called "2026 summer holiday.png" isn't read as a date.
        let separators: [(Int, Character)] = [(4, "-"), (7, "-"), (10, "_"), (13, "-"), (16, "-")]
        let chars = Array(stamp)
        for (i, c) in separators where chars[i] != c { return nil }
        for (i, c) in chars.enumerated() where !separators.contains(where: { $0.0 == i }) {
            guard c.isNumber else { return nil }
        }
        return stamp
    }

    /// Newest first. A file with no readable stamp keeps its place in the list
    /// but sorts last — dropping it would make the strip disagree with the
    /// folder the "Open output folder" link opens.
    static func items(generated: [String], enlarged: [String], limit: Int = limit) -> [MediaSessionItem] {
        let all = generated.map { MediaSessionItem(path: $0, origin: .generated) }
            + enlarged.map { MediaSessionItem(path: $0, origin: .enlarged) }
        return all
            .enumerated()
            .sorted { a, b in
                let ka = timestampKey(a.element.path) ?? ""
                let kb = timestampKey(b.element.path) ?? ""
                // Descending by stamp; the original index breaks ties so the
                // order is stable rather than dependent on the sort's internals.
                if ka != kb { return ka > kb }
                return a.offset < b.offset
            }
            .prefix(limit)
            .map(\.element)
    }

    /// What stays selected once `path` is gone.
    ///
    /// Clearing out a run of bad results is the reason delete exists, and a
    /// strip that blanks the preview after every one makes the app feel like it
    /// lost its place. The next picture is right there, so it takes over; at
    /// the end of the list the previous one does. Only an emptied strip
    /// legitimately clears the selection.
    static func selectionAfterDelete(_ items: [MediaSessionItem],
                                     removing path: String,
                                     selected: String?) -> String? {
        guard selected == path, let i = items.firstIndex(where: { $0.path == path }) else {
            return selected
        }
        if i + 1 < items.count { return items[i + 1].path }
        if i > 0 { return items[i - 1].path }
        return nil
    }
}
/// Strip thumbnails.
///
/// A strip row is 56pt and the pane's own output runs to 4096 square, with 60
/// rows in the list — `NSImage(contentsOfFile:)` on every SwiftUI body pass
/// decodes all of that at full size. `CGImageSource` reads a REDUCED image
/// straight from the file instead (using an embedded thumbnail where the
/// encoder wrote one), and the results are cached by path so a redraw costs
/// nothing.
enum MediaThumbnails {

    /// Longest edge, in pixels, of a cached thumbnail. 2x the 56pt tile, so it
    /// stays sharp on Retina without holding full frames in memory.
    static let maxPixel = 128

    private static let cache: NSCache<NSString, NSImage> = {
        let c = NSCache<NSString, NSImage>()
        c.countLimit = 200
        return c
    }()

    /// nil when the file is unreadable — the strip keeps the row anyway, so it
    /// can still be selected and moved to the Trash.
    static func load(path: String, maxPixel: Int = maxPixel) -> NSImage? {
        let url = URL(fileURLWithPath: path)
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixel,
        ]
        guard let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, options as CFDictionary) else {
            return nil
        }
        return NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
    }

    /// Cached by path. Deleting a picture drops its entry, so a path reused by
    /// a later run never draws the old picture.
    static func cached(path: String) -> NSImage? {
        if let hit = cache.object(forKey: path as NSString) { return hit }
        guard let img = load(path: path) else { return nil }
        cache.setObject(img, forKey: path as NSString)
        return img
    }

    static func forget(path: String) {
        cache.removeObject(forKey: path as NSString)
    }
}
