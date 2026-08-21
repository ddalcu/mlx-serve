import Foundation

/// Every folder models can live in, and which of them the server is told to
/// scan.
///
/// Two settings existed before this and neither did what people assumed. The
/// download destination was a hardcoded `~/.mlx-serve/models` (a `let` in
/// `DownloadManager`, plus a second copy in `ServerManager`), and "Custom
/// folder" fed only the app's OWN scan — the server takes `--model-dir`, so a
/// model in the custom folder was listed by the picker and absent from
/// `/v1/models`, and switching to one cost a full server restart instead of a
/// hot swap.
///
/// This is now the one place that answers both questions, and `--model-dir` is
/// repeatable server-side so every folder here is actually served.
struct ModelRoots {

    /// Where downloads have always gone. Not a fallback so much as the answer
    /// for every install that never touches the setting.
    static let builtInRoot = NSString(string: "~/.mlx-serve/models").expandingTildeInPath

    /// `main.zig` accepts this many `--model-dir` folders and EXITS on one
    /// more, so the client's list is bounded by the server's own cap rather
    /// than by a number that could drift away from it.
    static let serverRootLimit = 8

    private static let downloadRootKey = "defaultModelDownloadPath"
    private static let customRootKey = "customModelPath"

    private let defaults: UserDefaults
    private let allowCustomFolders: Bool

    init(defaults: UserDefaults = .standard,
         allowCustomFolders: Bool = BuildFeatures.current.customModelFolders) {
        self.defaults = defaults
        self.allowCustomFolders = allowCustomFolders
    }

    // MARK: - Stored settings

    /// The user's chosen download folder, VERBATIM — kept even when it does not
    /// resolve, so Settings can show the path that stopped working instead of
    /// quietly forgetting it (the same rule `customRoot` already followed).
    var configuredDownloadRoot: String? {
        get { Self.nonEmpty(defaults.string(forKey: Self.downloadRootKey)) }
        nonmutating set { Self.write(newValue, key: Self.downloadRootKey, to: defaults) }
    }

    /// The extra folder to SCAN. Downloads never go here — that is what
    /// `configuredDownloadRoot` is for, and conflating them is how "Custom
    /// folder" came to mean something different from what it says.
    var customRoot: String? {
        get { Self.nonEmpty(defaults.string(forKey: Self.customRootKey)) }
        nonmutating set { Self.write(newValue, key: Self.customRootKey, to: defaults) }
    }

    // MARK: - Answers

    /// Where a download lands. The configured folder when it is allowed and
    /// actually there, otherwise the built-in one.
    var downloadRoot: String {
        resolvedDownloadRoot ?? Self.builtInRoot
    }

    /// True when a folder IS configured and cannot be reached — an external
    /// drive that is not mounted. Downloads fall back rather than creating a
    /// directory inside a mountpoint stub and writing gigabytes into it.
    var downloadRootIsUnavailable: Bool {
        guard allowCustomFolders, configuredDownloadRoot != nil else { return false }
        return resolvedDownloadRoot == nil
    }

    private var resolvedDownloadRoot: String? {
        guard allowCustomFolders, let raw = configuredDownloadRoot else { return nil }
        return Self.existingDirectory(raw)
    }

    /// The folders the app ITSELF downloads into — destination first, then the
    /// built-in root every install started with. `scanRoots` is what the SERVER
    /// scans; this is what the app's own reads ("is this repo on disk?",
    /// discovery, delete) must check. Reading only the destination is how
    /// moving it made the pre-move library vanish from the picker while
    /// `/v1/models`, which scans both, kept serving it. Not existence-filtered
    /// (a read against a missing folder just finds nothing), and never LM
    /// Studio / custom folders — other tools' trees the app must not delete
    /// into.
    var ownedRoots: [String] {
        let dest = downloadRoot
        return dest == Self.builtInRoot ? [dest] : [dest, Self.builtInRoot]
    }

    /// The folders the app READS when asking "is this repo on disk?" — the
    /// same list the server scans, in the same first-wins order. A pack in ANY
    /// served folder must not be offered as a download (live 2026-08-09: a
    /// Mage-Flow pack in the custom scan folder showed a Download bar, and
    /// Generate could not resolve its dir). Only WRITES, cancel cleanup and
    /// deletes stay on `ownedRoots` — the extra folders are the user's or
    /// another tool's trees the app must not delete into.
    func readRoots(toolRoots: ToolModelRoots) -> [String] {
        scanRoots(toolRoots: toolRoots)
    }

    /// Static convenience for read sites without a DownloadManager instance
    /// (ServerManager's resolver, the voice-clone disk check).
    static func readRoots() -> [String] {
        ModelRoots().readRoots(toolRoots: ToolModelRoots.detected())
    }

    /// Every folder to scan, in the order the server should take them:
    /// download destination first, because `--model-dir` is first-wins on a
    /// repeated model id and the folder we write into holds the live copy.
    ///
    /// The built-in root stays in the list after the destination moves —
    /// everything downloaded before the change is there, and dropping it would
    /// make a library disappear from the picker.
    /// Other tools' folders come after ours and before the custom slot: a repo
    /// id we hold a copy of keeps resolving to OUR copy, and the custom folder
    /// stays last because it is the one the person chose explicitly and the
    /// server's root cap could otherwise spend every slot on tools they did
    /// not ask about.
    func scanRoots(toolRoots: ToolModelRoots) -> [String] {
        var out: [String] = []
        var seen = Set<String>()
        func add(_ path: String?) {
            guard out.count < Self.serverRootLimit,
                  let raw = path,
                  let resolved = Self.existingDirectory(raw),
                  seen.insert(resolved).inserted
            else { return }
            out.append(resolved)
        }
        add(resolvedDownloadRoot)
        add(Self.builtInRoot)
        for tool in toolRoots.ordered { add(tool) }
        if allowCustomFolders { add(customRoot) }
        return out
    }

    // MARK: - Helpers

    private static func nonEmpty(_ s: String?) -> String? {
        guard let t = s?.trimmingCharacters(in: .whitespacesAndNewlines), !t.isEmpty else { return nil }
        return t
    }

    private static func write(_ value: String?, key: String, to defaults: UserDefaults) {
        if let t = nonEmpty(value) {
            defaults.set(t, forKey: key)
        } else {
            defaults.removeObject(forKey: key)
        }
    }

    /// Expand, standardize, and require an existing DIRECTORY. Standardizing is
    /// what makes the de-dup work: `<p>` and `<p>/` are one folder, and the
    /// same path reached as the destination and as LM Studio's root must not
    /// become two `--model-dir` flags.
    static func existingDirectory(_ raw: String) -> String? {
        guard let t = nonEmpty(raw) else { return nil }
        let path = URL(fileURLWithPath: (t as NSString).expandingTildeInPath).standardizedFileURL.path
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else { return nil }
        return path
    }
}
