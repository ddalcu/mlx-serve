import SwiftUI

/// Custom (user-added) media models in the panes' pickers.
///
/// The preset catalogs are deliberately hand-written — every entry DECLARES
/// what its backend can do, and the panes gate controls and request fields on
/// those declarations. A checkpoint the user drops into a model root can still
/// be offered without breaking that discipline: the server already discovers
/// it, classifies it by `model_type`, and advertises it in `/v1/models` with
/// its modality capability, so we synthesize a preset from the matching FAMILY
/// (same knobs, same capability flags) with the discovered id as both the
/// picker id and the on-disk repo. An arch the engine can't serve never gets a
/// row, because discovery never lists it.
///
/// Source of truth is `ServerManager.allModels` — the list the LAN rows
/// already read — so custom rows share its one limitation: nothing renders
/// while the server is down.
enum CustomMediaModels {

    /// Everything a synthesized preset needs from a registry entry.
    private static func entry(for id: String, in models: [ModelInfo]) -> ModelInfo? {
        models.first {
            ($0.lanPeer == nil && $0.name == id) ||
            ($0.lanPeer != nil && LanPick.base(of: $0.name) == id)
        }
    }

    /// Mirrors the server's `mage_flow.dirIsEdit`: the Turbo and Edit
    /// checkpoints are byte-identical, so the directory name is the only
    /// signal either side has.
    private static func isEditDir(_ id: String) -> Bool {
        id.range(of: "mage-flow-edit", options: .caseInsensitive) != nil ||
        id.range(of: "mageflow-edit", options: .caseInsensitive) != nil
    }

    // MARK: - Arch → family templates

    private static func imageFamily(arch: String, id: String) -> ImageModelPreset? {
        if arch.hasPrefix("flux2") {
            return arch.contains("9b") ? .flux2Klein9B_Q4 : .flux2Klein4B_Q4
        }
        if arch.hasPrefix("krea") { return .krea2Turbo }
        if arch.hasPrefix("mage_flow") || arch == "mageflow" {
            return isEditDir(id) ? .mageFlowEditTurbo : .mageFlowTurbo
        }
        return nil
    }

    /// A custom H3 maps to the FL2VA family: `tasks` (the ref2va
    /// discriminator) is a config fact `/v1/models` doesn't carry, and
    /// reference fields sent to an fl2va DiT are the 400 class.
    private static func videoFamily(arch: String) -> VideoModelPreset? {
        if arch == "AudioVideo" { return .ltx23Q4 }
        if arch == "minimax_h3" { return .minimaxH3 }
        return nil
    }

    /// Cloning-capable TTS only — Kokoro is voice-mode-only by the catalog
    /// rule (`ref_audio` there is a named 400), and ACE-Step advertises
    /// "audio" too but answers the music endpoint.
    private static func audioFamily(arch: String) -> AudioModelPreset? {
        arch == "qwen3_tts" ? .qwen3TTS06B8bit : nil
    }

    private static func musicFamily(arch: String) -> MusicModelPreset? {
        arch == "acestep" ? .acestepXLTurbo8bit : nil
    }

    private static func meshFamily(arch: String) -> Model3DModelPreset? {
        arch.hasPrefix("hunyuan3d") ? .hunyuan3d21_8bit : nil
    }

    /// The family download bundle for a declared media arch — the exact
    /// allowlists + ready markers the catalog packs use, with the repo
    /// swapped in. This is what the Model Browser downloads a community pack
    /// with, and what its tree is verified against beforehand. nil for archs
    /// no pane serves.
    static func bundle(arch: String, repoId: String) -> MediaBundle? {
        if let p = imageFamily(arch: arch, id: repoId) { return p.asCustom(id: repoId).bundle }
        if let p = videoFamily(arch: arch) { return p.asCustom(id: repoId).bundle }
        if let p = audioFamily(arch: arch) { return p.asCustom(id: repoId).bundle }
        if let p = musicFamily(arch: arch) { return p.asCustom(id: repoId).bundle }
        if let p = meshFamily(arch: arch) { return p.asCustom(id: repoId).bundle }
        return nil
    }

    // MARK: - By-id resolution (persisted picks, LAN base ids)

    static func imagePreset(for id: String, from models: [ModelInfo]) -> ImageModelPreset? {
        entry(for: id, in: models).flatMap { imageFamily(arch: $0.architecture, id: id) }?.asCustom(id: id)
    }

    static func videoPreset(for id: String, from models: [ModelInfo]) -> VideoModelPreset? {
        entry(for: id, in: models).flatMap { videoFamily(arch: $0.architecture) }?.asCustom(id: id)
    }

    static func audioPreset(for id: String, from models: [ModelInfo]) -> AudioModelPreset? {
        entry(for: id, in: models).flatMap { audioFamily(arch: $0.architecture) }?.asCustom(id: id)
    }

    static func musicPreset(for id: String, from models: [ModelInfo]) -> MusicModelPreset? {
        entry(for: id, in: models).flatMap { musicFamily(arch: $0.architecture) }?.asCustom(id: id)
    }

    static func meshPreset(for id: String, from models: [ModelInfo]) -> Model3DModelPreset? {
        entry(for: id, in: models).flatMap { meshFamily(arch: $0.architecture) }?.asCustom(id: id)
    }

    // MARK: - "On This Mac" rows (local, non-catalog, sorted)

    private static func localIds(in models: [ModelInfo], excluding repos: Set<String>) -> [String] {
        models.filter { $0.lanPeer == nil && !repos.contains($0.name) }
            .map(\.name).sorted()
    }

    static func imagePresets(from models: [ModelInfo]) -> [ImageModelPreset] {
        let known = Set(ImageModelPreset.all.map(\.repo))
        return localIds(in: models, excluding: known).compactMap { imagePreset(for: $0, from: models) }
    }

    static func videoPresets(from models: [ModelInfo]) -> [VideoModelPreset] {
        let known = Set(VideoModelPreset.all.map(\.repo))
        return localIds(in: models, excluding: known).compactMap { videoPreset(for: $0, from: models) }
    }

    static func audioPresets(from models: [ModelInfo]) -> [AudioModelPreset] {
        let known = Set(AudioModelPreset.allIncludingVoiceOnly.map(\.repo))
        return localIds(in: models, excluding: known).compactMap { audioPreset(for: $0, from: models) }
    }

    static func musicPresets(from models: [ModelInfo]) -> [MusicModelPreset] {
        let known = Set(MusicModelPreset.all.map(\.repo))
        return localIds(in: models, excluding: known).compactMap { musicPreset(for: $0, from: models) }
    }

    static func meshPresets(from models: [ModelInfo]) -> [Model3DModelPreset] {
        let known = Set(Model3DModelPreset.all.map(\.repo))
        return localIds(in: models, excluding: known).compactMap { meshPreset(for: $0, from: models) }
    }
}

/// A preset that can stand in for a user-added checkpoint of its family:
/// same knobs and capability declarations, its own registry id — which is
/// also the repo, since a discovered id IS the on-disk `<org>/<name>` dir
/// every resolver reads.
protocol CustomizableMediaPreset {
    var id: String { get set }
    var name: String { get set }
    var repo: String { get set }
}

extension CustomizableMediaPreset {
    func asCustom(id newId: String) -> Self {
        var c = self
        c.id = newId
        c.name = newId
        c.repo = newId
        return c
    }
}

extension ImageModelPreset: CustomizableMediaPreset {}
extension VideoModelPreset: CustomizableMediaPreset {}
extension AudioModelPreset: CustomizableMediaPreset {}
extension MusicModelPreset: CustomizableMediaPreset {}
extension Model3DModelPreset: CustomizableMediaPreset {}

/// "On This Mac" rows for a media pane's model Picker — the custom-model twin
/// of `LanModelPickerRows`. Renders nothing when there are no custom models
/// (or the server is down, since the list comes from `/v1/models`).
struct CustomModelPickerRows<P: CustomizableMediaPreset & Identifiable>: View where P.ID == String {
    let presets: [P]

    var body: some View {
        if !presets.isEmpty {
            Section("On This Mac") {
                ForEach(presets) { Text($0.name).tag($0.id) }
            }
        }
    }
}
