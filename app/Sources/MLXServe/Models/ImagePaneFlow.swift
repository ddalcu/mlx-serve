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
