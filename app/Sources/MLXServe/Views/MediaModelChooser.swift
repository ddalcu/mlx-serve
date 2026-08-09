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

            // ONE row: the model that will run. A stack of radio rows read as
            // a multi-select and grew with the catalogue; what the pane has to
            // SAY is the current pick — changing it is the menu's job.
            currentRow
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color.primary.opacity(0.04))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
                )

            switcherMenu
        }
    }

    /// LAN selection is a plain callback rather than a second binding: the pane
    /// owns both halves of the choice (`LanPick.selection`) and this control
    /// only reports what was clicked.
    var onSelectLan: (String) -> Void = { _ in }

    private var lanModels: [ModelInfo] { server.lanModels(capability: lanCapability) }

    /// What the one row shows. A LAN pick wins (the local preset underneath is
    /// only the family the request is shaped by); a local id resolves through
    /// featured, then the rest of the catalogue, then discovered checkpoints —
    /// a persisted id nothing resolves any more (a deleted custom model) still
    /// renders by name rather than drawing an empty box.
    private var current: (name: String, detail: String, ram: String?, fits: Bool, preset: P?) {
        if let lanModel {
            let display = lanModels.first { $0.name == lanModel }?.lanDisplayName
                ?? lanModel.replacingOccurrences(of: "@", with: " · ")
            return (display,
                    "Running on \(LanPick.peer(of: lanModel)) over your network — nothing to download.",
                    nil, true, nil)
        }
        if let pick = featured.first(where: { $0.preset.id == selectedId }) {
            return (pick.preset.name, pick.capability,
                    "· ~\(pick.preset.approxRAMGB) GB RAM", pick.fits, pick.preset)
        }
        if let p = others.first(where: { $0.id == selectedId })
            ?? onThisMac.first(where: { $0.id == selectedId }) {
            // Same warn-don't-block check the featured picks carry — the
            // models most likely to overrun this Mac are exactly the big ones
            // that DIDN'T make the featured list.
            let fits = p.meetsSystemRequirements(
                physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
            return (p.name, capabilityOf(p), "· ~\(p.approxRAMGB) GB RAM", fits, p)
        }
        return (selectedId, "", nil, true, nil)
    }

    /// The current pick's row: name, why it's here, and its download control —
    /// a real sibling Button, never a tap gesture around the row (the
    /// swallowed-click class).
    private var currentRow: some View {
        let cur = current
        return HStack(alignment: .top, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(cur.name)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                if !cur.detail.isEmpty || cur.ram != nil {
                    HStack(spacing: 4) {
                        if !cur.detail.isEmpty {
                            Text(cur.detail)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        if let ram = cur.ram {
                            Text(ram)
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }
                    }
                }
                if !cur.fits {
                    // Warn, don't block: it may still run, slowly.
                    Label("Needs more memory than this Mac has",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }
            }
            Spacer(minLength: 8)
            // Downloading is part of CHOOSING — so it lives here, on the model,
            // and never beside Generate. Absent for a LAN pick: the pack is on
            // someone else's Mac.
            if let preset = cur.preset {
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
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
    }

    /// The one switcher: everything pickable, grouped, current pick ticked.
    /// Checkmarks are our own per-row Labels keyed by ID — never a Picker,
    /// whose menu ticks by row TITLE and double-ticks duplicate names.
    private var switcherMenu: some View {
        Menu("Change Model") {
            if !featured.isEmpty {
                Section("Recommended for This Mac") {
                    ForEach(featured) { pick in menuRow(pick.preset) }
                }
            }
            if !others.isEmpty {
                Section("More Models") {
                    ForEach(others) { menuRow($0) }
                }
            }
            // Yours, found on disk. No "not downloaded" suffix is possible
            // here — discovery IS the evidence it's there.
            if !onThisMac.isEmpty {
                Section("On This Mac") {
                    ForEach(onThisMac) { menuRow($0, showDownloadState: false) }
                }
            }
            if !lanModels.isEmpty {
                Section("On Your Network") {
                    ForEach(lanModels, id: \.name) { m in
                        Button {
                            onSelectLan(m.name)
                        } label: {
                            if lanModel == m.name {
                                Label(m.lanDisplayName, systemImage: "checkmark")
                            } else {
                                Text(m.lanDisplayName)
                            }
                        }
                    }
                }
            }
        }
        .menuStyle(.button)
        .buttonStyle(.bordered)
        .controlSize(.small)
        .fixedSize()
    }

    @ViewBuilder
    private func menuRow(_ preset: P, showDownloadState: Bool = true) -> some View {
        let title = showDownloadState && !isDownloaded(preset)
            ? "\(preset.name) — not downloaded"
            : preset.name
        Button {
            onSelect(preset)
        } label: {
            if lanModel == nil && preset.id == selectedId {
                Label(title, systemImage: "checkmark")
            } else {
                Text(title)
            }
        }
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
                     resolveCustom: @escaping (String) -> P?,
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
                // Adopt the preset the request will be SHAPED by: catalogue
                // first, then the pane's custom-family resolver — a peer's own
                // conversion has no catalogue row, and leaving `model` on the
                // previous pick sends another family's canvas and frame
                // ladders (the H3-sent-LTX-shapes class).
                let base = LanPick.base(of: id)
                if let preset = all.first(where: { $0.id == base }) ?? resolveCustom(base) {
                    selected.wrappedValue = preset
                }
                persist()
            })
    }
}
