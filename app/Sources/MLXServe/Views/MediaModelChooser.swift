import SwiftUI

/// The Model control at the top of every Create pane.
struct MediaModelChooser<P: MediaModelSizing>: View {
    /// Featured picks — one per capability, best that fits.
    let featured: [MediaModelPicks.Pick<P>]
    /// Everything else in the CATALOGUE, behind the "Other Models" menu.
    let others: [P]
    /// Checkpoints discovered on disk that the catalogue doesn't know about
    /// (`CustomMediaModels`) — your own conversions, community packs. Listed
    /// under their own heading rather than mixed into `others`: "Other Models"
    /// means "the rest of what we ship", and a model the user put there
    /// themselves is a different kind of thing. Default empty so a pane that
    /// has no notion of custom checkpoints doesn't have to say so.
    var onThisMac: [P] = []
    /// Currently selected preset id (local pick).
    let selectedId: String
    /// nil for a local pick, otherwise the LAN routing id it runs on.
    let lanModel: String?
    /// Capability line for a model that isn't featured (a pick from "Other").
    let capabilityOf: (P) -> String
    /// Bundle state for the row's trailing control.
    let isDownloaded: (P) -> Bool
    let downloadLabel: (P) -> String
    let onSelect: (P) -> Void
    let onDownload: (P) -> Void
    /// The LAN rows, if this pane has any peers offering the modality.
    let lanCapability: String

    @EnvironmentObject var server: ServerManager

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Model").font(.subheadline.weight(.semibold))

            VStack(spacing: 0) {
                ForEach(featured) { pick in
                    row(pick.preset, capability: pick.capability, fits: pick.fits)
                    if pick.id != featured.last?.id || showsSelectedExtraRow {
                        Divider().padding(.leading, 26)
                    }
                }
                // A pick from "Other Models" gets the same row — so its download
                // control is in the same place as everything else's, rather than
                // the user having to hunt for where the button went.
                if let extra = selectedNonFeatured {
                    row(extra, capability: capabilityOf(extra), fits: true, isOtherPick: true)
                }
            }
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color.primary.opacity(0.04))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
            )

            HStack(spacing: 8) {
                if !others.isEmpty || !onThisMac.isEmpty || !lanModels.isEmpty {
                    Menu("Other Models") {
                        ForEach(others) { preset in
                            Button {
                                onSelect(preset)
                            } label: {
                                Text(isDownloaded(preset)
                                     ? preset.name
                                     : "\(preset.name) — not downloaded")
                            }
                        }
                        // Yours, found on disk. No "not downloaded" suffix is
                        // possible here — discovery IS the evidence it's there.
                        if !onThisMac.isEmpty {
                            Divider()
                            Section("On This Mac") {
                                ForEach(onThisMac) { preset in
                                    Button(preset.name) { onSelect(preset) }
                                }
                            }
                        }
                        if !lanModels.isEmpty {
                            Divider()
                            Section("On Your Network") {
                                ForEach(lanModels, id: \.name) { m in
                                    Button(m.lanDisplayName) { onSelectLan(m.name) }
                                }
                            }
                        }
                    }
                    .menuStyle(.button)
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .fixedSize()
                }
                if let lanModel {
                    Text("Running on \(LanPick.peer(of: lanModel)) over your network — nothing to download.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }
        }
    }

    /// LAN selection is a plain callback rather than a second binding: the pane
    /// owns both halves of the choice (`LanPick.selection`) and this control
    /// only reports what was clicked.
    var onSelectLan: (String) -> Void = { _ in }

    private var lanModels: [ModelInfo] { server.lanModels(capability: lanCapability) }

    /// The selected preset when it came from "Other Models" — nil when the
    /// selection is one of the featured rows (or a LAN model).
    /// Searches BOTH menu groups: a discovered checkpoint you selected needs
    /// the same row — with the same download control in the same place — as
    /// one picked out of the catalogue, or choosing your own model makes the
    /// row disappear.
    private var selectedNonFeatured: P? {
        guard lanModel == nil,
              !featured.contains(where: { $0.preset.id == selectedId }) else { return nil }
        return others.first { $0.id == selectedId } ?? onThisMac.first { $0.id == selectedId }
    }

    private var showsSelectedExtraRow: Bool { selectedNonFeatured != nil }

    /// Two REAL side-by-side buttons — select and download — never a tap
    /// gesture wrapped around the row. A parent `onTapGesture` + `contentShape`
    /// swallows plain child buttons on macOS: the download click would die
    /// silently, with no error anywhere (see the app-side rule; it has shipped
    /// twice).
    @ViewBuilder
    private func row(_ preset: P, capability: String, fits: Bool,
                     isOtherPick: Bool = false) -> some View {
        let selected = lanModel == nil && preset.id == selectedId
        HStack(alignment: .top, spacing: 8) {
            Button {
                onSelect(preset)
            } label: {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: selected ? "largecircle.fill.circle" : "circle")
                        .font(.system(size: 13))
                        .foregroundStyle(selected ? Color.accentColor : Color.secondary)
                        .padding(.top, 1)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(preset.name)
                            .font(.subheadline.weight(selected ? .semibold : .regular))
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                        HStack(spacing: 4) {
                            // The REASON this model is on screen. Without it the
                            // list is just names again — the state this replaced.
                            Text(capability)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text("· ~\(preset.approxRAMGB) GB RAM")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }
                        if !fits {
                            // Warn, don't block: it may still run, slowly.
                            Label("Needs more memory than this Mac has",
                                  systemImage: "exclamationmark.triangle.fill")
                                .font(.caption2)
                                .foregroundStyle(.orange)
                        }
                    }
                    Spacer(minLength: 8)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            // Downloading is part of CHOOSING — so it lives here, on the model,
            // and never beside Generate.
            if isDownloaded(preset) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .help("Downloaded and ready to use")
                    .padding(.top, 1)
            } else {
                Button {
                    onDownload(preset)
                } label: {
                    Text(downloadLabel(preset))
                        .font(.caption.weight(.medium))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .help("Download this model — you can keep using the app while it transfers")
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
    }
}

extension MediaModelChooser {
    /// The Create panes' whole Model control.
    ///
    /// All four panes differ only in their catalogue, their discovered
    /// checkpoints and their capability tag — the featured/others split, the
    /// download wiring and the LAN handoff were copied four times, so a fix to
    /// one copy left three behind.
    static func pane(all: [P], onThisMac: [P], capability: String,
                     selected: Binding<P>, lanModel: Binding<String?>,
                     capabilityOf: @escaping (P) -> String,
                     bundleOf: @escaping (P) -> MediaBundle,
                     downloads: DownloadManager,
                     onDownloadFinished: @escaping () -> Void,
                     persist: @escaping () -> Void) -> MediaModelChooser<P> {
        let featured = MediaModelPicks.featured(
            all,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            capabilityOf: capabilityOf)
        return MediaModelChooser(
            featured: featured,
            others: MediaModelPicks.others(all, featured: featured),
            onThisMac: onThisMac,
            selectedId: selected.wrappedValue.id,
            lanModel: lanModel.wrappedValue,
            capabilityOf: capabilityOf,
            isDownloaded: { downloads.bundleReady(bundleOf($0)) },
            downloadLabel: { "Download \(bundleOf($0).approxSizeLabel)" },
            onSelect: { preset in
                lanModel.wrappedValue = nil
                selected.wrappedValue = preset
                // Persist HERE, like onSelectLan: the panes persist on
                // `.onChange(of: model)`, which never fires when the local
                // pick already IS this preset — deselecting a LAN model back
                // to the same local one would survive only until relaunch.
                persist()
            },
            onDownload: { preset in
                // Downloading also selects — you fetch the one you mean to use.
                lanModel.wrappedValue = nil
                selected.wrappedValue = preset
                persist()
                downloads.startBundle(bundleOf(preset)) { onDownloadFinished() }
            },
            lanCapability: capability,
            onSelectLan: { id in
                lanModel.wrappedValue = id
                if let base = all.first(where: { $0.id == LanPick.base(of: id) }) {
                    selected.wrappedValue = base
                }
                persist()
            })
    }
}
