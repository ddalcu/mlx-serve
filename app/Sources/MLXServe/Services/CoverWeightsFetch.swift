import Foundation

/// Whether ACE-Step **Cover** can run against the pack on this Mac, and what
/// the Music tab should say when it cannot.
///
/// Cover re-sings a source track through the FSQ audio tokenizer, which ships
/// as `fsq.safetensors` beside `model.safetensors`. The file was added to the
/// mirror AFTER cover mode landed, so a pack downloaded before then is
/// complete in every other way and simply has no tokenizer — the server's only
/// signal was a named 400, earned minutes into a generation the user had
/// already set up (#269).
///
/// The check is purely LOCAL: the pack is on disk and `ServerManager
/// .resolveModelDir` already resolves it across every SERVED root
/// (`ModelRoots.readRoots`), so an LM Studio or custom folder is covered too
/// and no protocol change is needed to learn this.
///
/// `model.safetensors` is byte-identical across the two pack revisions, so the
/// repair is the ONE file — `DownloadManager.startPackFile`, the same
/// single-file-into-an-existing-pack contract `startTurboLora` uses.
enum CoverWeightsFetch {
    /// The name the SERVER resolves (`acestep.FSQ_FILE`) and the ACE-Step
    /// bundle allowlists. One constant so a rename cannot land the file where
    /// nothing reads it.
    static let fileName = "fsq.safetensors"

    /// Roughly what it costs, for the sentence shown before it starts. The
    /// file is 420,148,465 bytes; this is only ever prose.
    static let approxMB = 420

    enum Decision: Equatable {
        /// The tokenizer is on disk — Cover works.
        case ready
        /// Missing, and the pack's folder is ours to write: offer the ONE file.
        case fetch
        /// That fetch is in flight.
        case downloading
        /// Missing, and we cannot write the pack (another tool's tree, or a
        /// folder outside our sandbox grants). Say where it goes; never offer
        /// a button that would fail.
        case missingUnwritable(dir: String)
        /// The model is on another Mac. Its pack is not ours to complete; that
        /// server answers its own named 400.
        case unavailableRemotely
        /// Not a cover request, or the pack is not installed at all (the model
        /// row's own Download button owns that case).
        case notApplicable
    }

    /// `packDir` is the resolved pack directory (nil = not installed here);
    /// `fetching` is a single-file fetch already in flight for this repo.
    static func decide(task: MusicTask, modelSupportsSourceAudio: Bool, isRemote: Bool,
                       packDir: String?, fetching: Bool) -> Decision {
        guard task == .cover, modelSupportsSourceAudio else { return .notApplicable }
        if isRemote { return .unavailableRemotely }
        guard let packDir else { return .notApplicable }
        if fetching { return .downloading }
        let path = (packDir as NSString).appendingPathComponent(fileName)
        if FileManager.default.fileExists(atPath: path) { return .ready }
        // The honest check available to us. A sandboxed build without a grant
        // for another tool's tree reads as not-writable here, which is exactly
        // the case that must not be offered a button.
        guard FileManager.default.isWritableFile(atPath: packDir) else {
            return .missingUnwritable(dir: packDir)
        }
        return .fetch
    }

    /// The mode picker's own label. Requirement of #269: the reason Cover does
    /// not work is readable BEFORE the mode is selected, so it lives in the
    /// label rather than in a disabled control that explains nothing.
    static func modeLabel(_ task: MusicTask, decision: Decision) -> String {
        guard task == .cover else { return task.label }
        switch decision {
        case .ready, .notApplicable: return task.label
        case .fetch, .downloading, .missingUnwritable, .unavailableRemotely:
            return "\(task.label) — \(fileName) missing"
        }
    }

    /// The sentence under the mode control. nil when there is nothing wrong.
    static func notice(_ decision: Decision) -> String? {
        switch decision {
        case .ready, .notApplicable:
            return nil
        case .fetch:
            return "Cover needs \(fileName), a \(approxMB) MB tokenizer this pack predates. "
                + "It downloads into the pack you already have — not the whole model again."
        case .downloading:
            return "Fetching \(fileName) (\(approxMB) MB) into this pack. Cover works as soon as it lands."
        case .missingUnwritable(let dir):
            return "Cover needs \(fileName) (\(approxMB) MB), and MLX Core cannot write to this pack's "
                + "folder. Download it from the model's Hugging Face page and put it in:\n\(dir)"
        case .unavailableRemotely:
            return "Cover runs on the Mac hosting this model; it needs \(fileName) in ITS copy of the pack."
        }
    }
}
