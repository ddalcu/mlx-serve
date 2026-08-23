import Foundation

/// The canonical model folders of OTHER local-inference tools on this Mac.
///
/// LM Studio's folder was detected from the first version and nobody else's
/// was, which made "MLX Core cannot see my models" a configuration answer
/// wearing an engine costume. Both folders added here hold layouts the Zig
/// discovery has always understood — MTPLX writes flat `Org--Name` dirs and
/// Osaurus writes plain `org/repo`, and `discoverModelsInDir` reads both — so
/// the models were loadable the whole time and merely unreachable.
///
/// Rule for adding one: it must be the CANONICAL, tool-owned location of a
/// local-inference runtime, not a folder a person happens to like. `~/models`
/// is the case this rule excludes — a common convention, but claiming a
/// generic home folder for every install is a surprise, and the "Custom
/// folder" setting exists precisely for it.
///
/// Everything here is read-only. These are other tools' trees: the app serves
/// out of them and never writes, moves, or deletes inside them (`ownedRoots`
/// stays the download destination plus the built-in root).
struct ToolModelRoots: Equatable {

    var lmStudio: String?
    var mtplx: String?
    var osaurus: String?

    init(lmStudio: String? = nil, mtplx: String? = nil, osaurus: String? = nil) {
        self.lmStudio = lmStudio
        self.mtplx = mtplx
        self.osaurus = osaurus
    }

    /// Scan order paired with the `LocalModelSource` each root's models are
    /// listed under, absent tools dropped. The pairing lives here so a root
    /// and its heading cannot drift apart: the picker enumerates from this
    /// list, and a path listed under a foreign tool's heading sends you to the
    /// wrong app to manage the file.
    ///
    /// LM Studio stays first because it was there before the others and a
    /// reordering would change which copy of a duplicated repo id wins the
    /// first-wins merge.
    var orderedWithSource: [(path: String, source: LocalModelSource)] {
        var out: [(path: String, source: LocalModelSource)] = []
        if let p = lmStudio { out.append((p, .lmStudio)) }
        if let p = mtplx { out.append((p, .mtplx)) }
        if let p = osaurus { out.append((p, .osaurus)) }
        return out
    }

    /// Just the paths, for `--model-dir` flags, which have no notion of source.
    var ordered: [String] { orderedWithSource.map(\.path) }

    /// Resolve every tool folder against `home`. Each one is existence-gated
    /// by `ModelRoots.existingDirectory`, so an uninstalled tool contributes
    /// nothing and a FILE sitting at the canonical path is not mistaken for a
    /// root — the server exits on a `--model-dir` it cannot open.
    ///
    /// `home` is a parameter so this is testable without depending on what the
    /// machine running the tests happens to have installed.
    static func detected(home: String = NSHomeDirectory(),
                         lmStudioRoot: String? = nil) -> ToolModelRoots {
        func under(_ rel: String) -> String? {
            ModelRoots.existingDirectory((home as NSString).appendingPathComponent(rel))
        }
        return ToolModelRoots(
            // LM Studio reads its own settings.json for a moved folder, so it
            // resolves itself rather than being guessed from `home` — but the
            // resolver takes `home` too, so an injected one stays injected.
            lmStudio: (lmStudioRoot ?? DownloadManager.lmStudioRootPath(home: home))
                .flatMap { ModelRoots.existingDirectory($0) },
            // MTPLX: `~/.mtplx/models/<Org>--<Name>/`, one flat dir per model.
            mtplx: under(".mtplx/models"),
            // Osaurus: `~/MLXModels/<org>/<repo>/`, the HF-style layout.
            osaurus: under("MLXModels"))
    }
}
