import Foundation

/// What a My Models row lets you DO, as pure functions the row reads.
///
/// The row shipped with two controls: "Use", and — depending on where the file
/// lived — either a trash or `externaldrive.badge.icloud`, a glyph that reads
/// as "external drive" or "cloud" and meant neither. It was an `Image`, not a
/// `Button`, so clicking it did nothing. For every model outside
/// `~/.mlx-serve/models` that glyph WAS the row's entire answer to "where is
/// this file and how do I get rid of it".
///
/// Three answers now, and the logic lives here rather than inside the view so
/// it can be tested:
///
/// * **Reveal** — every row, no exceptions. Six other panes already had it;
///   this one, the only pane that is ABOUT files on disk, did not.
/// * **Lock, then trash** — a foreign tree still starts locked, because
///   deleting another app's working model by accident is worth one click of
///   friction. But the lock is now the click that removes the friction, not a
///   wall: these are all apps the user runs, on the user's disk.
/// * **Right-click** — the same actions, where people look for them.
enum ModelRowActions {

    /// True when the trash is offered right now. A broken folder never needs
    /// unlocking: it is junk, not somebody's model (see `ModelDefect`).
    static func showsTrash(_ model: LocalModel, unlocked: Bool) -> Bool {
        model.isDeletable || unlocked
    }

    /// True when the row shows a lock instead. Mutually exclusive with the
    /// trash — they occupy the same slot, and a row showing both is a row where
    /// one of them is lying.
    static func showsLock(_ model: LocalModel, unlocked: Bool) -> Bool {
        !showsTrash(model, unlocked: unlocked)
    }

    /// Tooltip for the lock. Names the owning app AND says what clicking does —
    /// a lock that cannot explain itself is the mystery glyph in a nicer shape.
    static func lockHelp(_ model: LocalModel) -> String {
        // Phrased per source rather than by slotting a name into one template:
        // "a folder you added" is not a possessive, and the first version of
        // this read "a folder you added's folder".
        let where_: String
        switch model.source {
        case .mlxServe: where_ = "in MLX Core\u{2019}s own models folder"
        case .lmStudio: where_ = "in LM Studio\u{2019}s models folder"
        case .huggingFace: where_ = "in the Hugging Face cache, where models share files"
        case .mtplx: where_ = "in MTPLX\u{2019}s models folder"
        case .osaurus: where_ = "in Osaurus\u{2019}s models folder"
        case .custom: where_ = "in a custom folder you added"
        }
        return "This model is \(where_). Click to unlock and delete it."
    }

    /// Confirmation text. A foreign row names the PATH, because "Delete org/m?"
    /// does not say which of four tools' copies is about to go.
    static func deleteMessage(_ model: LocalModel) -> String {
        if let defect = model.defect {
            return "\(defect.explanation)\n\nDelete \(model.path)?"
        }
        if model.quantFile != nil {
            return "Delete \(model.displayLabel)? Other quants of this model stay on disk."
        }
        switch model.source {
        case .mlxServe:
            return "Delete \(model.name)? This will remove all downloaded files."
        case .huggingFace:
            // The one tree where deleting damages models we are NOT deleting:
            // snapshots hard-link shared blobs. Allowed — it is the user's disk
            // — but never silently.
            return """
            This is in the Hugging Face cache, where models share downloaded files. \
            Deleting it can break other models that share the same files.

            Delete \(model.path)?
            """
        default:
            return "This is in another app\u{2019}s models folder.\n\nDelete \(model.path)?"
        }
    }

    /// Tooltip for Reveal. Offered on every row including broken folders —
    /// especially those, since opening the folder is how you confirm it is junk.
    static func revealHelp(_ model: LocalModel) -> String {
        model.quantFile != nil ? "Show this quant in Finder" : "Show in Finder"
    }
}
