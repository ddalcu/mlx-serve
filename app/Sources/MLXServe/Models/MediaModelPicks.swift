import Foundation

/// Which media models a Create pane leads with.
enum MediaModelPicks {

    /// A featured model, and why it is being offered.
    struct Pick<P: MediaModelSizing>: Identifiable {
        let preset: P
        /// The job this model is here to do — rendered as the reason under it.
        let capability: String
        /// Whether this Mac's RAM covers it. False picks are still shown, with
        /// the shortfall named: warn, don't block (`meetsSystemRequirements`,
        /// `RecommendedModelPick`, the oversized-model alert — same policy).
        let fits: Bool

        var id: String { preset.id }
    }

    /// The best model per capability: the largest that fits this Mac, or — when
    /// nothing in that capability fits — the smallest one, flagged. Capabilities
    /// come out in the order they first appear in the catalogue, so the pane's
    /// order is the catalogue's order and neither can surprise the other.
    static func featured<P: MediaModelSizing>(
        _ presets: [P],
        physicalMemoryBytes: UInt64,
        capabilityOf: (P) -> String
    ) -> [Pick<P>] {
        var order: [String] = []
        var grouped: [String: [P]] = [:]
        for preset in presets {
            let capability = capabilityOf(preset)
            if grouped[capability] == nil { order.append(capability) }
            grouped[capability, default: []].append(preset)
        }

        return order.compactMap { capability in
            guard let candidates = grouped[capability], !candidates.isEmpty else { return nil }
            let fitting = candidates.filter { $0.meetsSystemRequirements(physicalMemoryBytes: physicalMemoryBytes) }
            if let best = fitting.max(by: { $0.approxRAMGB < $1.approxRAMGB }) {
                return Pick(preset: best, capability: capability, fits: true)
            }
            // Nothing fits: offer the cheapest rather than an empty picker,
            // which reads as "this Mac can't do this at all".
            guard let smallest = candidates.min(by: { $0.approxRAMGB < $1.approxRAMGB }) else { return nil }
            return Pick(preset: smallest, capability: capability, fits: false)
        }
    }

    /// Everything the featured list didn't take, in catalogue order. A featured
    /// model never appears here too — one entry, one place.
    static func others<P: MediaModelSizing>(_ presets: [P], featured: [Pick<P>]) -> [P] {
        let taken = Set(featured.map(\.preset.id))
        return presets.filter { !taken.contains($0.id) }
    }
}

// MARK: - What each catalogue's models are FOR

/// The labels are the reason line under a featured model, so they name the JOB,
/// not the architecture. They're also the grouping key, so two models sharing a
/// label are treated as a ranking and only the better one is featured.
extension ImageModelPreset {
    var capabilityLabel: String {
        supportsReferenceEdit ? "Best for editing photos" : "Best for creating images"
    }
}

extension VideoModelPreset {
    var capabilityLabel: String {
        generatesAudio ? "Best for video with sound" : "Best for video"
    }
}

extension AudioModelPreset {
    /// One job: speech (and every entry in `.all` can clone a voice — the
    /// voice-only backends live in `allIncludingVoiceOnly` and never reach a
    /// pane's picker).
    var capabilityLabel: String { "Best for speech and voice cloning" }
}

extension MusicModelPreset {
    var capabilityLabel: String {
        family == .acestep ? "Fastest music (8-step Turbo)" : "Best for full songs with vocals"
    }
}

extension Model3DModelPreset {
    var capabilityLabel: String { "Best for turning a photo into a 3D model" }
}
