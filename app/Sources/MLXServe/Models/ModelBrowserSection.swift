import Foundation

/// The Model Browser's sidebar destinations.
enum ModelBrowserSection: String, CaseIterable, Identifiable, Hashable {
    /// Every curated Gemma 4 / Qwen 3.5-3.6 checkpoint, grouped by family and
    /// explained in plain English — the friendly front door for someone who
    /// has never picked a local model before. Lands first; Discover is the
    /// power-user HuggingFace search.
    case recommended
    /// Search HuggingFace. On-disk models stay listed here, marked, never hidden.
    case discover
    /// Everything on this Mac that the tray picker can offer, grouped by source.
    case myModels
    /// The transfer queue: in-flight and failed downloads.
    case downloads
    /// Media-gen model catalog (image/audio/video/music), grouped by modality.
    case media

    var id: String { rawValue }

    var title: String {
        switch self {
        case .recommended: return "Recommended"
        case .discover:     return "Discover"
        case .myModels:     return "My Models"
        case .downloads:    return "Downloads"
        case .media:        return "Media"
        }
    }

    var systemImage: String {
        switch self {
        case .recommended: return "star"
        case .discover:     return "magnifyingglass"
        case .myModels:     return "internaldrive"
        case .downloads:    return "arrow.down.circle"
        case .media:        return "photo.on.rectangle.angled"
        }
    }

    /// The HuggingFace search field, the weight-format picker, and the Search
    /// button. They belong to Discover alone — rendering them above a
    /// filesystem list is what made the old toggle read as a filter.
    var showsSearchControls: Bool { self == .discover }

    /// Panes that render on-disk state, and therefore need the periodic
    /// `refreshModels()` rescan while bytes are landing.
    private var showsDiskState: Bool { self == .myModels || self == .downloads }

    /// Whether to run the 1 Hz disk rescan: only on a pane that shows on-disk
    /// state, and only while a download is actually in flight, so it
    /// self-terminates. Discover doesn't need it — its rows re-evaluate off
    /// `DownloadManager`'s published state.
    static func shouldLivePoll(section: ModelBrowserSection, hasActiveDownloads: Bool) -> Bool {
        hasActiveDownloads && section.showsDiskState
    }
}

/// Sidebar badge counts. A zero count shows no badge at all rather than a "0".
struct ModelBrowserBadgeCounts: Equatable {
    let myModels: Int
    /// Downloading *or* failed — both want the user's attention, and both are
    /// what the Downloads pane lists.
    let activeDownloads: Int
    /// Media (image/audio/video/music) bundles fully on disk.
    let mediaReady: Int

    /// The live counts, from the two objects that hold them. ONE place, because
    /// the sidebar renders the badges and the panes render what they count — a
    /// second copy of this arithmetic is how a badge starts disagreeing with the
    /// list it sits next to.
    static func live(localModelCount: Int,
                     activeDownloadCount: Int,
                     mediaReadyCount: Int) -> ModelBrowserBadgeCounts {
        ModelBrowserBadgeCounts(myModels: localModelCount,
                                activeDownloads: activeDownloadCount,
                                mediaReady: mediaReadyCount)
    }

    func badge(for section: ModelBrowserSection) -> String? {
        let n: Int
        switch section {
        case .recommended: return nil
        case .discover:     return nil
        case .myModels:     n = myModels
        case .downloads:    n = activeDownloads
        case .media:        n = mediaReady
        }
        return n > 0 ? "\(n)" : nil
    }
}

/// What a browser row's action cell should offer. Pure so the branch ladder is
/// testable; the view maps each case onto a control (a GGUF repo renders
/// `.notDownloaded` as a quant menu, everything else as a button).
enum ModelRowAction: Equatable {
    /// Architecture we can't serve. No action.
    case unsupported
    /// Present on this Mac: offer Use + Delete instead of removing the row.
    case onDisk
    case downloading(progress: Double)
    case failed(resumable: Bool)
    case notDownloaded(resumable: Bool)

    /// Resolution order mirrors the original view's `if` ladder, so behaviour is
    /// unchanged apart from `.onDisk` rows staying visible in Discover.
    static func resolve(
        isCompatible: Bool,
        isReady: Bool,
        status: DownloadManager.DownloadState.Status?,
        hasPartial: Bool,
        progress: Double = 0
    ) -> ModelRowAction {
        if !isCompatible { return .unsupported }
        if isReady { return .onDisk }
        guard let status else { return .notDownloaded(resumable: hasPartial) }
        switch status {
        case .completed:   return .onDisk
        case .downloading: return .downloading(progress: progress)
        case .failed:      return .failed(resumable: hasPartial)
        case .idle:        return .notDownloaded(resumable: hasPartial)
        }
    }
}

/// Feedback for the model the user picked with "Use".
enum ModelUseState: Equatable {
    /// Not the selected model — offer the Use button.
    case idle
    /// Selected, and the server is up with it.
    case inUse
    /// Selected, and the server is coming up.
    case loading
    /// Selected, but the server is stopped — it'll load on next start.
    case selected

    /// `selected` is `appState.selectedModelPath == model.path`.
    static func resolve(selected: Bool, serverStatus: ServerStatus) -> ModelUseState {
        guard selected else { return .idle }
        switch serverStatus {
        case .running:  return .inUse
        case .starting: return .loading
        case .stopped, .error: return .selected
        }
    }

    var label: String {
        switch self {
        case .idle:     return ""
        case .inUse:    return "In use"
        case .loading:  return "Loading…"
        case .selected: return "Selected"
        }
    }

    /// Tooltip. Says what the state *means* for the server, not what the badge says.
    var help: String {
        switch self {
        case .idle:     return ""
        case .inUse:    return "This model is loaded and serving requests."
        case .loading:  return "The server is loading this model…"
        case .selected: return "This model will load when you start the server."
        }
    }
}

/// One source-grouped bucket of `My Models`.
struct LocalModelGroup: Identifiable {
    let source: LocalModelSource
    let models: [LocalModel]
    /// Broken folders are grouped by their DEFECT, not by which tool's folder
    /// they happen to sit in — they are not that tool's models, and padding a
    /// tool's section with junk misreports what you have.
    var isDefectGroup: Bool = false
    var id: String { isDefectGroup ? "defect" : source.rawValue }
    var title: String {
        isDefectGroup ? ModelBrowserUse.defectGroupTitle : ModelBrowserUse.groupTitle(source)
    }
}

/// Pure helpers for the "what do I already have, and can I load it?" side of
/// the browser.
enum ModelBrowserUse {

    /// The local model at `path`, if it's something the server can load as its
    /// chat model. Drafters, encoders, and media checkpoints resolve to nil —
    /// they're real files worth listing and deleting, but "Use" would load a
    /// checkpoint that can't serve a completion.
    static func pickableModel(atPath path: String?, in models: [LocalModel]) -> LocalModel? {
        guard let path, !path.isEmpty else { return nil }
        let wanted = normalize(path)
        return models.first { normalize($0.path) == wanted && $0.isChatPickable }
    }

    /// The media model at `path` and the pane it belongs to, if any.
    ///
    /// The sibling of `pickableModel` for the other half of the catalogue. Both
    /// return nil for the other's kind, so a row asks both and gets at most one
    /// answer — a media checkpoint is never chat-pickable and a chat model has
    /// no modality. Every on-disk row can then offer a Use, instead of Discover
    /// and the Media pane quietly ending at Delete the way My Models did (#228).
    static func mediaModel(atPath path: String?, in models: [LocalModel]) -> (model: LocalModel, modality: MediaModality)? {
        guard let path, !path.isEmpty else { return nil }
        let wanted = normalize(path)
        guard let m = models.first(where: { normalize($0.path) == wanted }),
              let modality = MediaModality(modelType: m.modelType) else { return nil }
        return (m, modality)
    }

    private static func normalize(_ path: String) -> String {
        var p = (path as NSString).standardizingPath
        while p.count > 1, p.hasSuffix("/") { p.removeLast() }
        return p
    }

    /// Display order for the sidebar's My Models list. Mirrors the tray picker's
    /// sections (`StatusMenuView`), which is the point — the old "Downloaded"
    /// tab filtered to `.mlxServe` only and so never matched what you could
    /// actually select.
    static let sourceOrder: [LocalModelSource] = [.mlxServe, .lmStudio, .huggingFace, .mtplx, .osaurus, .custom]

    /// Heading for the broken-folder group. Names the two things it holds —
    /// folders that cannot load, and downloads that never finished.
    static let defectGroupTitle = "Incomplete & Orphaned"

    static func groupTitle(_ source: LocalModelSource) -> String {
        switch source {
        case .mlxServe:    return "Downloaded by MLX Core"
        case .lmStudio:    return "LM Studio Models"
        case .huggingFace: return "Hugging Face Cache"
        case .mtplx:       return LocalModelSource.mtplx.sectionTitle
        case .osaurus:     return LocalModelSource.osaurus.sectionTitle
        case .custom:      return "Custom Folder"
        }
    }

    /// Group by source in `sourceOrder`, applying the name filter first and
    /// dropping any bucket left empty.
    static func groupedBySource(_ models: [LocalModel], filter: String) -> [LocalModelGroup] {
        let needle = filter.trimmingCharacters(in: .whitespaces)
        let matching = needle.isEmpty
            ? models
            : models.filter { $0.name.localizedCaseInsensitiveContains(needle) }

        var out = sourceOrder.compactMap { source -> LocalModelGroup? in
            let bucket = matching.filter { $0.source == source && $0.defect == nil }
            return bucket.isEmpty ? nil : LocalModelGroup(source: source, models: bucket)
        }
        // Last: these are not what you came to pick, but they ARE the reason a
        // folder you do not recognise is sitting in your library.
        let broken = matching.filter { $0.defect != nil }
        if let first = broken.first {
            out.append(LocalModelGroup(source: first.source, models: broken, isDefectGroup: true))
        }
        return out
    }
}
