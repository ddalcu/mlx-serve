import Foundation

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
/// from the two phases plus a focus (set when a run finishes) instead of from
/// the current verb is what makes that impossible: `resolve` deliberately
/// takes no verb, because what is on screen is a property of what has
/// FINISHED, not of what the controls are set to.
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

    static func resolve(generate: Run, enlarge: Run, focus: Origin?) -> State {
        // A run in flight always wins: it is the only thing on screen that is
        // still changing. With both somehow in flight the focus breaks the
        // tie, so the answer is stable rather than order-dependent.
        let running: [(Origin, Run)] = [(.generated, generate), (.enlarged, enlarge)]
            .filter { if case .running = $0.1 { return true } else { return false } }
        if running.count == 2, let focus,
           let pick = running.first(where: { $0.0 == focus }),
           case .running(let msg) = pick.1 {
            return .running(pick.0, msg)
        }
        if let (origin, run) = running.first, case .running(let msg) = run {
            return .running(origin, msg)
        }

        // Otherwise the focus names which finished run to show, and a focus
        // whose side has nothing to say falls through to the other rather than
        // blanking a picture that is still perfectly good.
        let order: [(Origin, Run)] = focus == .enlarged
            ? [(.enlarged, enlarge), (.generated, generate)]
            : [(.generated, generate), (.enlarged, enlarge)]
        for (origin, run) in order {
            if case .done(let path) = run { return .result(origin, path) }
        }
        for (origin, run) in order {
            if case .failed(let msg) = run { return .failed(origin, msg) }
        }
        return .empty
    }
}
