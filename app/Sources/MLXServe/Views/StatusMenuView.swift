import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Claude logo icon from the official Claude AI symbol SVG.
struct ClaudeIcon: View {
    var size: CGFloat = 14

    var body: some View {
        ClaudeShape()
            .frame(width: size, height: size)
    }
}

private struct ClaudeShape: Shape {
    // SVG path coordinates from the official Claude AI symbol (1200x1200 viewBox).
    // One M, many L, one C, one Z — stored as (x,y) pairs for compact code.
    private static let points: [(CGFloat, CGFloat)] = [
        (233.96, 800.21), (468.64, 668.54), (472.59, 657.10), (468.64, 650.74),
        (457.21, 650.74), (417.99, 648.32), (283.89, 644.70), (167.60, 639.87),
        (54.93, 633.83), (26.58, 627.79), (0, 592.75), (2.74, 575.28),
        (26.58, 559.25), (60.72, 562.23), (136.19, 567.38), (249.42, 575.19),
        (331.57, 580.03), (453.26, 592.67), (472.59, 592.67), (475.33, 584.86),
        (468.72, 580.03), (463.57, 575.19), (346.39, 495.79), (219.54, 411.87),
        (153.10, 363.54), (117.18, 339.06), (99.06, 316.11), (91.25, 266.01),
        (123.87, 230.09), (167.68, 233.07), (178.87, 236.05), (223.25, 270.20),
        (318.04, 343.57), (441.83, 434.74), (459.95, 449.80), (467.19, 444.64),
        (468.08, 441.02), (459.95, 427.41), (392.62, 305.72), (320.78, 181.93),
        (288.81, 130.63), (280.35, 99.87),
    ]
    // C 277.37 87.22 275.19 76.59 275.19 63.62 (cubic bezier)
    private static let curveEnd: (c1: (CGFloat, CGFloat), c2: (CGFloat, CGFloat), to: (CGFloat, CGFloat)) =
        ((277.37, 87.22), (275.19, 76.59), (275.19, 63.62))
    private static let points2: [(CGFloat, CGFloat)] = [
        (312.32, 13.21), (332.86, 6.60), (382.39, 13.21), (403.25, 31.33),
        (434.01, 101.72), (483.87, 212.54), (561.18, 363.22), (583.81, 407.92),
        (595.89, 449.32), (600.40, 461.96), (608.21, 461.96), (608.21, 454.71),
        (614.58, 369.83), (626.34, 265.61), (637.77, 131.52), (641.72, 93.75),
        (660.40, 48.48), (697.53, 24.00), (726.52, 37.85), (750.36, 72.00),
        (747.06, 94.07), (732.89, 186.20), (705.10, 330.52), (686.98, 427.17),
        (697.53, 427.17), (709.61, 415.09), (758.50, 350.17), (840.64, 247.49),
        (876.89, 206.74), (919.17, 161.72), (946.31, 140.30), (997.61, 140.30),
        (1035.38, 196.43), (1018.47, 254.42), (965.64, 321.42), (921.83, 378.20),
        (859.01, 462.77), (819.79, 530.42), (823.41, 535.81), (832.75, 534.93),
        (974.66, 504.72), (1051.33, 490.87), (1142.82, 475.17), (1184.21, 494.50),
        (1188.72, 514.15), (1172.46, 554.34), (1074.60, 578.50), (959.84, 601.45),
        (788.94, 641.88), (786.85, 643.41), (789.26, 646.39), (866.26, 653.64),
        (899.19, 655.41), (979.81, 655.41), (1129.93, 666.60), (1169.15, 692.54),
        (1192.67, 724.27), (1188.72, 748.43), (1128.32, 779.19), (1046.82, 759.87),
        (856.59, 714.60), (791.36, 698.34), (782.34, 698.34), (782.34, 703.73),
        (836.70, 756.89), (936.32, 846.85), (1061.07, 962.82), (1067.44, 991.49),
        (1051.41, 1014.12), (1034.50, 1011.70), (924.89, 929.23), (882.60, 892.11),
        (786.85, 811.49), (780.48, 811.49), (780.48, 819.95), (802.55, 852.24),
        (919.09, 1027.41), (925.13, 1081.13), (916.67, 1098.60), (886.47, 1109.15),
        (853.29, 1103.11), (785.07, 1007.36), (714.68, 899.52), (657.91, 802.87),
        (650.98, 806.82), (617.48, 1167.70), (601.77, 1186.15), (565.53, 1200.00),
        (535.33, 1177.05), (519.30, 1139.92), (535.33, 1066.55), (554.66, 970.79),
        (570.36, 894.68), (584.54, 800.13), (593.00, 768.72), (592.43, 766.63),
        (585.50, 767.52), (514.23, 865.37), (405.83, 1011.87), (320.05, 1103.68),
        (299.52, 1111.81), (263.92, 1093.37), (267.22, 1060.43), (287.11, 1031.11),
        (405.83, 880.11), (477.42, 786.52), (523.65, 732.48), (523.33, 724.67),
        (520.59, 724.67), (205.29, 929.40), (149.15, 936.64), (125.00, 914.01),
        (127.97, 876.89), (139.41, 864.81), (234.20, 799.57),
    ]

    func path(in rect: CGRect) -> Path {
        let sx = rect.width / 1200
        let sy = rect.height / 1200

        var p = Path()
        let first = Self.points[0]
        p.move(to: CGPoint(x: first.0 * sx, y: first.1 * sy))

        for pt in Self.points.dropFirst() {
            p.addLine(to: CGPoint(x: pt.0 * sx, y: pt.1 * sy))
        }

        let c = Self.curveEnd
        p.addCurve(
            to: CGPoint(x: c.to.0 * sx, y: c.to.1 * sy),
            control1: CGPoint(x: c.c1.0 * sx, y: c.c1.1 * sy),
            control2: CGPoint(x: c.c2.0 * sx, y: c.c2.1 * sy)
        )

        for pt in Self.points2 {
            p.addLine(to: CGPoint(x: pt.0 * sx, y: pt.1 * sy))
        }

        p.closeSubpath()
        return p
    }
}

/// The native media-generation tools (image / video / audio) shown in the
/// "Experiments" section of the menu popover. Pure data (no SwiftUI) so the
/// section's membership, ordering, and help text stay unit-testable. Kept in
/// sync with `GenExperimentTests`.
enum GenExperiment: String, CaseIterable, Identifiable {
    case image, video, audio

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .image: "photo.on.rectangle.angled"
        case .video: "film.stack"
        case .audio: "waveform"
        }
    }

    var title: String {
        switch self {
        case .image: "ImageGen"
        case .video: "VideoGen"
        case .audio: "AudioGen"
        }
    }

    /// Tooltip for the tile.
    var help: String {
        switch self {
        case .image: "Image Generation (FLUX.2 / Krea-2)"
        case .video: "Video Generation (LTX-Video 2.3)"
        case .audio: "Audio Generation — neural TTS & voice cloning"
        }
    }
}

struct StatusMenuView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager
    @State private var showDownloads = false
    /// Slot ids with an unload in flight (the eject button becomes a spinner —
    /// an unload can take seconds while the server drains a running request).
    @State private var unloadingIds: Set<String> = []
    let openChat: () -> Void
    let openModelBrowser: () -> Void
    let openImageGen: () -> Void
    let openVideoGen: () -> Void
    let openAudioGen: () -> Void
    let openSettings: () -> Void
    let openServerLog: () -> Void
    let openTasks: () -> Void
    var openSandboxTerminal: () -> Void = {}

    /// Observes the shared sandbox so the tray badge appears/updates live when
    /// the Agent Sandbox is turned on and when its guest boots. Safe to observe
    /// from the tray: the per-command transcript lives in the separate
    /// `AgentSandbox.transcriptStore` (observed only by the Sandbox Terminal),
    /// so command churn never re-renders this menu.
    @ObservedObject private var sandbox = AgentSandbox.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Text("MLX Core")
                    .font(.headline)
                Text(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                Spacer()
                StatusDot(status: server.status)
            }
            .padding(.horizontal, 16)
            .padding(.top, 14)
            .padding(.bottom, 10)

            Divider().padding(.horizontal, 12)

            // Server Control
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Server")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(server.status.label)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if appState.localModels.isEmpty {
                    Text("No models found. Download one below.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                } else {
                    // Model picker + Settings shortcut on the same row so the
                    // gear lives where the user picks what to load. Tuning
                    // anything else lives in the Settings window the gear opens.
                    HStack(spacing: 6) {
                        Picker("Model", selection: $appState.selectedModelPath) {
                            // Hide drafter checkpoints — they pair with a base
                            // model via the Drafter toggle in Settings, not
                            // loadable as a target on their own.
                            let pickable = appState.localModels.filter { $0.kind == .base }
                            // macOS .menu Pickers key the checkmark by item
                            // TITLE — two same-named rows (one GGUF, one MLX)
                            // both rendered selected. Suffix duplicated names
                            // with the engine tag so titles stay unique.
                            let dupNames = LocalModel.duplicateNames(in: pickable)
                            let mlxServe = pickable.filter { $0.source == .mlxServe }
                            let lmStudio = pickable.filter { $0.source == .lmStudio }
                            let custom = pickable.filter { $0.source == .custom }
                            if !mlxServe.isEmpty {
                                Section("MLX-Serve Models") {
                                    ForEach(mlxServe) { model in
                                        Text(modelPickerLabel(model, dupNames: dupNames)).tag(model.path)
                                    }
                                }
                            }
                            if !lmStudio.isEmpty {
                                Section("Other Discovered Models") {
                                    ForEach(lmStudio) { model in
                                        Text(modelPickerLabel(model, dupNames: dupNames)).tag(model.path)
                                    }
                                }
                            }
                            if !custom.isEmpty {
                                Section("Custom Folder") {
                                    ForEach(custom) { model in
                                        Text(modelPickerLabel(model, dupNames: dupNames)).tag(model.path)
                                    }
                                }
                            }
                        }
                        .labelsHidden()
                        .pickerStyle(.menu)

                        Button { openSettings() } label: {
                            Image(systemName: "gear")
                        }
                        .buttonStyle(.bordered)
                        .help("Settings")
                    }

                    HStack(spacing: 6) {
                        let control = ServerControlButtonPresentation(status: server.status)
                        Button {
                            server.toggle(modelPath: appState.selectedModelPath, options: appState.serverOptions)
                        } label: {
                            HStack(spacing: 8) {
                                if control.showsProgress {
                                    ProgressView()
                                        .controlSize(.small)
                                } else if let systemImageName = control.systemImageName {
                                    Image(systemName: systemImageName)
                                }
                                Text(control.title)
                            }
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(control.tint.color)
                        .disabled(appState.selectedModelPath.isEmpty)
                        .controlSize(.regular)
                        .help(control.help)

                        Button {
                            openServerLog()
                        } label: {
                            Image(systemName: "macwindow.on.rectangle")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.regular)
                        .help("Open Server Log in a separate window (easier copy/paste)")
                    }

                    HStack {
                        Toggle("Auto-start on launch", isOn: $appState.autoStartServer)
                            .toggleStyle(.switch)
                            .controlSize(.mini)
                        Spacer()
                        // Which embedded engine the selected model routes to
                        // (MLX safetensors, llama.cpp GGUF, or ds4 GGUF).
                        if let engine = appState.localModels
                            .first(where: { $0.path == appState.selectedModelPath })?.engine
                        {
                            Text(engine.displayName)
                                .help("Engine the selected model runs on")
                        }
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)

                    // Show error details
                    if case .error = server.status, !server.lastError.isEmpty {
                        Text(server.lastError)
                            .font(.caption2)
                            .foregroundStyle(.red)
                            .lineLimit(4)
                            .textSelection(.enabled)
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            // Model Info & Memory (when running)
            if case .running = server.status {
                Divider().padding(.horizontal, 12)

                VStack(alignment: .leading, spacing: 6) {
                    // Model slots — one row per RESIDENT registry entry (chat,
                    // image/video/audio gen, embeddings), each with an eject
                    // button that frees its memory. Unloaded stubs are hidden.
                    let loadedModels = server.allModels.filter(\.loaded)
                    HStack {
                        Text(loadedModels.count > 1 ? "Models" : "Model")
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                    if loadedModels.isEmpty {
                        Text("None loaded — models load on demand")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                    ForEach(loadedModels, id: \.name) { info in
                        modelSlotRow(info)
                    }

                    if let mem = server.memoryInfo {
                        // Both bars share total physical RAM as the denominator,
                        // so they're directly comparable and never stuck (the old
                        // GPU bar used peak×2 → pinned at 50% once active≈peak).
                        let totalRAM = Int64(ProcessInfo.processInfo.physicalMemory)

                        HStack {
                            Text("GPU Memory")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                            Text(mem.activeFormatted)
                                .font(.caption.monospaced())
                        }
                        ProgressView(value: mem.gpuFraction(ofTotal: totalRAM))
                            .tint(.blue)

                        // Available RAM — same calc the model-load pre-flight
                        // uses (total − wired − compressor), so this can't drift
                        // from what gates a load. It's reclaimable-available
                        // (includes evictable file cache), not unused memory.
                        // Hidden when 0 (older server build that omits the field).
                        if mem.availableBytes > 0 {
                            HStack {
                                Text("Available RAM")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Text(mem.availableFormatted)
                                    .font(.caption.monospaced())
                            }
                            ProgressView(value: mem.availableFraction(ofTotal: totalRAM))
                                .tint(.green)
                        }

                        if totalRAM > 0 {
                            Text("\(MemoryInfo.format(totalRAM)) total")
                                .font(.caption2)
                                .foregroundStyle(.tertiary)
                        }
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)

                // Endpoints
                Divider().padding(.horizontal, 12)
                EndpointsSection(baseURL: server.baseURL)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)

            }

            Divider().padding(.horizontal, 12)

            // Downloads
            VStack(alignment: .leading, spacing: 6) {
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        showDownloads.toggle()
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.right")
                            .font(.caption2)
                            .rotationEffect(.degrees(showDownloads ? 90 : 0))
                        Text("Download Models")
                            .font(.subheadline.weight(.medium))
                        Spacer()
                    }
                    .foregroundStyle(.secondary)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)

                if showDownloads {
                    ModelDownloadView()
                        .environmentObject(downloads)
                        .environmentObject(appState)

                    Button {
                        openModelBrowser()
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "magnifyingglass")
                            Text("Browse More Models")
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .padding(.top, 4)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)

            Divider().padding(.horizontal, 12)

            // Experiments — native media-generation tools (image / audio /
            // video). Shown inline now that they're built in (no Python, no
            // accordion); the tiles open their windows directly.
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 4) {
                    Text("Media Generation")
                        .font(.subheadline.weight(.medium))
                    Spacer()
                }
                .foregroundStyle(.secondary)

                HStack(spacing: 6) {
                    ForEach(GenExperiment.allCases) { exp in
                        genFeatureButton(
                            icon: exp.icon,
                            title: exp.title,
                            help: exp.help,
                            action: { open(exp) })
                    }
                }
                .padding(.top, 4)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)

            Divider().padding(.horizontal, 12)

            // Persistent, window-independent voice assistant — its own row, not
            // a button. Toggle it on and talk hands-free with no chat window.
            VoiceTrayPanel(voice: appState.voice)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)

            Divider().padding(.horizontal, 12)

            // Quick launcher — Spotlight-style ⌃Space prompt panel, summonable
            // from any app while MLX Core runs in the tray.
            QuickLauncherTrayRow()
                .padding(.horizontal, 16)
                .padding(.vertical, 8)

            Divider().padding(.horizontal, 12)

            // Agent-sandbox badge — visible only while the sandbox is enabled.
            // Green box = a guest is live; click to open the Sandbox Terminal and
            // run commands / watch the agent in the isolated Linux VM.
            if sandbox.isEnabled {
                Button {
                    openSandboxTerminal()
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: "shippingbox.fill")
                            .foregroundStyle(sandbox.guestRunning ? Color.green : Color.orange)
                        VStack(alignment: .leading, spacing: 1) {
                            Text("Agent Sandbox is on").font(.callout.weight(.medium))
                            // guestMemoryText is quantized + published only on
                            // change, so this row doesn't re-render per second.
                            Text(sandbox.guestRunning
                                 ? "Guest running" + (sandbox.guestMemoryText.map { " · \($0)" } ?? "")
                                 : "Idle — boots on the first command")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Image(systemName: "terminal").foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 16).padding(.vertical, 8)
                .help("Open the Sandbox Terminal")

                Divider().padding(.horizontal, 12)
            }

            // Chat, Tasks, Claude Code & Quit
            HStack(spacing: 8) {
                Button {
                    openChat()
                } label: {
                    HStack {
                        Image(systemName: "bubble.left.and.bubble.right")
                        Text("Chat")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(server.status != .running)

                Button {
                    openTasks()
                } label: {
                    HStack {
                        Image(systemName: "clock.badge.checkmark")
                        Text("Tasks")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .help("Scheduled Tasks")

                CLILauncherButton(
                    baseURL: server.baseURL,
                    servedModelId: server.modelInfo?.name ?? "mlx-serve",
                    isEnabled: server.status == .running
                )

                Button {
                    server.stop()
                    NSApplication.shared.terminate(nil)
                } label: {
                    Image(systemName: "power")
                }
                .buttonStyle(.bordered)
            }
            .padding(.horizontal, 16)
            .padding(.top, 6)
            .padding(.bottom, 14)
        }
        .frame(width: 320)
        // Drive ServerManager's /props live-polling from the popover's
        // visibility: poll on open, idle on close. SwiftUI's MenuBarExtra
        // (.window style) fires onAppear when the popover shows and
        // onDisappear when it dismisses — perfect hook for "user is or isn't
        // looking at the GPU-memory bar".
        .onAppear { server.setMenuVisible(true) }
        .onDisappear { server.setMenuVisible(false) }
    }

    /// One button in the secondary "optional Python features" row. Three of
    /// these share a 320pt-wide popover, so the label is compact: caption font
    /// (which also shrinks the SF Symbol), tight spacing, small control size,
    /// and `minimumScaleFactor` as a safety net so a long title scales down
    /// instead of truncating to "AudioG…".
    /// Route an experiment tile to its window opener.
    private func open(_ exp: GenExperiment) {
        switch exp {
        case .image: openImageGen()
        case .video: openVideoGen()
        case .audio: openAudioGen()
        }
    }

    private func genFeatureButton(
        icon: String, title: String, help: String, action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                Text(title)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
            }
            .font(.caption)
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .help(help)
    }

    /// Append a "+ assist" suffix to every model row that *could* use the
    /// assistant drafter — i.e. drafter is currently enabled overall AND a
    /// matching `gemma-4-*-it-assistant-bf16` checkpoint is on disk for this
    /// row. Lets the user see at a glance which models keep the speedup if
    /// they switch (auto-sync swaps `drafterPath` to the matching one on
    /// model change). When drafter is off, no badges anywhere.
    private func modelPickerLabel(_ model: LocalModel, dupNames: Set<String>) -> String {
        var label = model.name
        if dupNames.contains(model.name) {
            label += " · \(model.engine.shortLabel)"
        }
        guard !appState.serverOptions.drafterPath.isEmpty,
              downloads.recommendedDrafterFromPath(model.path) != nil else {
            return label
        }
        return "\(label) + assist"
    }

    /// One resident-model slot: modality icon, name, badges (chat quant /
    /// spec-decode), resident size, and an eject button that unloads it.
    @ViewBuilder
    private func modelSlotRow(_ info: ModelInfo) -> some View {
        HStack(spacing: 6) {
            Image(systemName: info.slotKind.icon)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 16)
                .help(info.slotKind.label)
            Text(info.name)
                .font(.caption.monospaced())
                .lineLimit(1)
                .truncationMode(.middle)
                .help(info.slotKind == .chat
                      ? "\(info.slotKind.label) — \(info.layers) layers, \(info.hiddenSize)-dim"
                      : info.slotKind.label)
            if info.quantBits > 0 {
                Text("\(info.quantBits)-bit")
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(.quaternary)
                    .clipShape(Capsule())
            }
            // Speculative-decoding speedup badge (MTP / drafter).
            if let badge = info.specDecodeBadge {
                Text(badge)
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.green)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(Color.green.opacity(0.15))
                    .clipShape(Capsule())
                    .help(info.mtpLoaded
                          ? "Native multi-token-prediction head loaded — faster decode via speculative decoding"
                          : "Assistant drafter loaded — faster decode via speculative decoding")
            }
            Spacer()
            if info.bytesResident > 0 {
                Text(MemoryInfo.format(Int64(clamping: info.bytesResident)))
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
            }
            if unloadingIds.contains(info.name) {
                ProgressView()
                    .controlSize(.mini)
                    .help("Unloading — waits for any running request to finish")
            } else {
                Button {
                    unloadSlot(info.name)
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.caption)
                }
                .buttonStyle(.borderless)
                .foregroundStyle(.secondary)
                .help("Unload from memory (the model reloads on its next use)")
            }
        }
    }

    /// Eject a resident model. The server marks the entry evicting, waits for
    /// in-flight requests to drain, frees it on the inference thread, and the
    /// registry keeps the stub — the next request that targets it reloads.
    private func unloadSlot(_ id: String) {
        unloadingIds.insert(id)
        Task {
            try? await server.unloadModel(id: id)
            unloadingIds.remove(id)
        }
    }
}

struct ServerControlButtonPresentation: Equatable {
    enum Tint: Equatable {
        case accent
        case loading
        case red

        var color: Color {
            switch self {
            case .accent: .accentColor
            case .loading: Color(red: 0.78, green: 0.32, blue: 0.0)
            case .red: .red
            }
        }
    }

    let title: String
    let systemImageName: String?
    let showsProgress: Bool
    let tint: Tint
    let help: String

    init(status: ServerStatus) {
        switch status {
        case .starting:
            title = "Loading Model..."
            systemImageName = nil
            showsProgress = true
            tint = .loading
            help = "Loading model. Click to stop."
        case .running:
            title = "Stop Server"
            systemImageName = "stop.fill"
            showsProgress = false
            tint = .red
            help = "Stop the running server."
        case .stopped, .error:
            title = "Start Server"
            systemImageName = "play.fill"
            showsProgress = false
            tint = .accent
            help = "Start the selected model."
        }
    }
}

struct StatusDot: View {
    let status: ServerStatus

    var body: some View {
        Circle()
            .fill(dotColor)
            .frame(width: 8, height: 8)
            .overlay {
                if case .starting = status {
                    Circle()
                        .stroke(dotColor.opacity(0.5), lineWidth: 2)
                        .frame(width: 14, height: 14)
                        .opacity(0.6)
                }
            }
    }

    var dotColor: Color {
        switch status {
        case .running: .green
        case .starting: .orange
        case .stopped: .red
        case .error: .red
        }
    }
}

struct EndpointsSection: View {
    let baseURL: String
    @State private var copiedEndpoint: String?
    @State private var isExpanded = false

    private let endpoints: [(method: String, path: String)] = [
        ("GET", "/health"),
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/responses"),
        ("POST", "/v1/messages"),
        ("POST", "/v1/embeddings"),
    ]

    /// The server root (`/`) — the human-friendly status page. Pure so the
    /// "open in browser" wiring is testable without rendering the view, and so
    /// it normalizes to exactly one trailing slash regardless of how `baseURL`
    /// is formatted.
    static func rootURL(_ baseURL: String) -> URL? {
        let trimmed = baseURL.hasSuffix("/") ? String(baseURL.dropLast()) : baseURL
        return URL(string: trimmed + "/")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "chevron.right")
                        .font(.caption2)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                    Text("Endpoints")
                        .font(.subheadline.weight(.medium))
                    Spacer()
                }
                .foregroundStyle(.secondary)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if isExpanded {
                // Root status page — click anywhere on the row to open it in the
                // default browser (not a copy target like the API endpoints below).
                Button {
                    if let url = Self.rootURL(baseURL) {
                        NSWorkspace.shared.open(url)
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text("GET")
                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                            .foregroundStyle(.green)
                            .frame(width: 30, alignment: .leading)
                        Text(baseURL + "/")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.blue)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Image(systemName: "arrow.up.right.square")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Open the server status page in your browser")
                .padding(.vertical, 1)

                ForEach(endpoints, id: \.path) { ep in
                    HStack(spacing: 4) {
                        Text(ep.method)
                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                            .foregroundStyle(ep.method == "GET" ? .green : .blue)
                            .frame(width: 30, alignment: .leading)
                        Text(baseURL + ep.path)
                            .font(.system(size: 10, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(baseURL + ep.path, forType: .string)
                            copiedEndpoint = ep.path
                            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                if copiedEndpoint == ep.path { copiedEndpoint = nil }
                            }
                        } label: {
                            Image(systemName: copiedEndpoint == ep.path ? "checkmark" : "doc.on.doc")
                                .font(.caption2)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(copiedEndpoint == ep.path ? .green : .secondary)
                    }
                    .padding(.vertical, 1)
                }
            }
        }
    }
}

/// Show folder picker and launch Claude Code in the selected directory.
func launchClaudeCodeWithPicker(baseURL: String) {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.allowsMultipleSelection = false
    panel.canCreateDirectories = true
    panel.prompt = "Open"
    panel.message = "Select or create a working directory"
    let defaultWS = NSString(string: "~/.mlx-serve/workspace").expandingTildeInPath
    try? FileManager.default.createDirectory(atPath: defaultWS, withIntermediateDirectories: true)
    panel.directoryURL = URL(fileURLWithPath: defaultWS)
    guard panel.runModal() == .OK, let url = panel.url else { return }
    launchClaudeCode(baseURL: baseURL, workingDirectory: url.path)
}

/// Launch Claude Code CLI configured to use the local mlx-serve server.
func launchClaudeCode(baseURL: String, workingDirectory: String? = nil) {
    let model = "mlx-serve"
    let cdLine = workingDirectory.map { "cd '\($0)'" } ?? ""
    let scriptContent = """
    #!/bin/zsh -l
    export ANTHROPIC_BASE_URL='\(baseURL)'
    export ANTHROPIC_API_KEY=
    export ANTHROPIC_AUTH_TOKEN=mlx-serve
    export CLAUDE_CODE_ATTRIBUTION_HEADER=0
    export ANTHROPIC_DEFAULT_OPUS_MODEL=\(model)
    export ANTHROPIC_DEFAULT_SONNET_MODEL=\(model)
    export ANTHROPIC_DEFAULT_HAIKU_MODEL=\(model)
    export CLAUDE_CODE_SUBAGENT_MODEL=\(model)
    \(cdLine)
    claude --model \(model)
    """

    let path = NSTemporaryDirectory() + "mlx-claude-code.command"
    try? scriptContent.write(toFile: path, atomically: true, encoding: .utf8)
    try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: path)
    NSWorkspace.shared.open(URL(fileURLWithPath: path))
}

/// Full-window terminal-style view of the live server stderr buffer.
///
/// Crucially, the view *pulls* — it owns a `LogPoller` that ticks at 2 Hz
/// while the window is open and reads `server.currentServerLogSnapshot()`.
/// `ServerManager` itself does NOT publish the log (see the `logBuffer`
/// comment there). Result: stderr volume can't slow ChatView's SSE token
/// loop — those views never re-render due to log activity, regardless of
/// whether this window is open.
///
/// Auto-scroll defaults on; the toggle pins to the user's last interaction
/// so selecting a region above doesn't get yanked away by new output.
struct ServerLogWindowView: View {
    @EnvironmentObject var server: ServerManager
    @State private var autoScroll = true
    @State private var copied = false
    @StateObject private var poller: LogPoller

    init() {
        // Closure captures nothing at init — it'll be rebound to `server`
        // on first appear via the environment-object lookup pattern below.
        // We can't reach `@EnvironmentObject` in init(), so the poller
        // starts with a placeholder snapshot that returns "" until
        // `.onAppear` rebinds it through `start(server:)`.
        _poller = StateObject(wrappedValue: LogPoller(interval: 0.5) { "" })
    }

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            logBody
        }
        .frame(minWidth: 600, minHeight: 360)
        .onAppear { startPolling() }
        .onDisappear { poller.stop() }
    }

    /// Bind the poller's snapshot closure to the environment server, then
    /// start ticking. Has to happen on appear because @StateObject's init
    /// runs before SwiftUI injects the environment object.
    private func startPolling() {
        poller.bind { [weak server] in
            server?.currentServerLogSnapshot() ?? ""
        }
        poller.start()
    }

    private var toolbar: some View {
        HStack(spacing: 10) {
            StatusDot(status: server.status)
            Text(statusLabel)
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
            Text("·")
                .foregroundStyle(.tertiary)
            // Byte counter from the poller mirror — same data the body
            // shows, so the number matches what's on screen rather than
            // racing with the live buffer.
            Text("\(poller.text.count) bytes")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)

            Spacer()

            Toggle(isOn: $autoScroll) {
                Label("Auto-scroll", systemImage: "arrow.down.to.line")
            }
            .toggleStyle(.button)
            .controlSize(.small)
            .help(autoScroll
                  ? "Auto-scroll is ON — new lines pin the view to the bottom"
                  : "Auto-scroll is OFF — the view stays where you left it")

            Button {
                copyLog()
            } label: {
                Label(copied ? "Copied" : "Copy",
                      systemImage: copied ? "checkmark" : "doc.on.doc")
            }
            .controlSize(.small)
            .help("Copy the entire log to the clipboard")

            Button {
                saveLog()
            } label: {
                Label("Save…", systemImage: "square.and.arrow.down")
            }
            .controlSize(.small)
            .help("Save the log to a .log file")

            Button(role: .destructive) {
                server.clearServerLog()
            } label: {
                Label("Clear", systemImage: "trash")
            }
            .controlSize(.small)
            .help("Clear the in-memory log buffer (does not affect the running server)")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.bar)
    }

    private var logBody: some View {
        ZStack {
            // NSTextView, not SwiftUI Text — incremental textStorage.append
            // for new bytes (a few ms regardless of buffer size), preserves
            // user selection, native scroll. Driven by the poller's mirror;
            // updates at the poller's rate (~2 Hz) not at stderr arrival
            // rate.
            TerminalLogTextView(text: poller.text, autoScroll: autoScroll)

            // Empty-state placeholder. Pure poller.text check — re-renders
            // only when the log transitions to/from empty.
            if poller.text.isEmpty {
                Text("(server has produced no output yet)")
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .padding(12)
                    .frame(maxWidth: .infinity, maxHeight: .infinity,
                           alignment: .topLeading)
                    .allowsHitTesting(false)
            }
        }
        .background(Color.black)
    }

    private var statusLabel: String {
        switch server.status {
        case .running:  return "Running"
        case .starting: return "Starting…"
        case .stopped:  return "Stopped"
        case .error:    return "Error"
        }
    }

    private func copyLog() {
        // Pull from the live raw buffer so a copy taken right after a fresh
        // log line is never up-to-100-ms-stale relative to the displayed text.
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(server.currentServerLogSnapshot(), forType: .string)
        copied = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            copied = false
        }
    }

    private func saveLog() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText]
        panel.canCreateDirectories = true
        // ISO timestamp with `:` replaced — POSIX-safe filename on macOS.
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        panel.nameFieldStringValue = "mlx-serve-\(stamp).log"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        try? server.currentServerLogSnapshot()
            .write(to: url, atomically: true, encoding: .utf8)
    }
}

/// AppKit-backed terminal-style log view. Wraps `NSTextView` in an
/// `NSScrollView` and exposes a SwiftUI-friendly `text` + `autoScroll`
/// surface.
///
/// Why not SwiftUI `Text` in a `ScrollView`? Because SwiftUI `Text`
/// re-lays-out its entire string on every `@Published` change. At 10 Hz
/// with a 64 KB monospace block that's enough main-thread work to starve
/// ChatView's SSE loop when this window is open. `NSTextView` does
/// incremental `textStorage.append` for the common case (new bytes at the
/// end of an existing prefix) — a few ms regardless of buffer size — and
/// only falls back to a full replacement when the in-memory buffer is
/// trimmed from the head (i.e. the 64 KB cap kicks in).
struct TerminalLogTextView: NSViewRepresentable {
    let text: String
    let autoScroll: Bool

    private static let textFont = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)
    private static let textColor = NSColor(red: 0.86, green: 0.95, blue: 0.88, alpha: 1.0)
    private static let bg = NSColor.black

    func makeCoordinator() -> Coordinator { Coordinator() }

    /// Carries cross-update state SwiftUI itself doesn't preserve — used
    /// here to detect the autoScroll toggle so flipping it back on
    /// scrolls to bottom even if `text` hasn't changed since.
    final class Coordinator {
        var lastAutoScroll: Bool = true
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        scroll.borderType = .noBorder
        scroll.hasVerticalScroller = true
        scroll.hasHorizontalScroller = false
        scroll.drawsBackground = true
        scroll.backgroundColor = Self.bg

        guard let textView = scroll.documentView as? NSTextView else { return scroll }
        textView.isEditable = false
        textView.isSelectable = true
        textView.isRichText = false
        textView.allowsUndo = false
        textView.usesFindBar = true
        textView.font = Self.textFont
        textView.textColor = Self.textColor
        textView.backgroundColor = Self.bg
        textView.drawsBackground = true
        textView.textContainerInset = NSSize(width: 8, height: 8)
        // Wrap lines to the visible width instead of horizontal-scrolling.
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.containerSize = NSSize(
            width: scroll.contentSize.width,
            height: .greatestFiniteMagnitude
        )
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        guard let textView = scroll.documentView as? NSTextView,
              let storage = textView.textStorage else { return }

        let attrs: [NSAttributedString.Key: Any] = [
            .font: Self.textFont,
            .foregroundColor: Self.textColor,
        ]

        let current = storage.string
        let textChanged = (text != current)
        if textChanged {
            if !current.isEmpty && text.hasPrefix(current) {
                // Cheap incremental append — preserves user selection and
                // scroll position. This is the hot path while the server
                // streams stderr below the 64 KB cap.
                let suffix = String(text.dropFirst(current.count))
                storage.append(NSAttributedString(string: suffix, attributes: attrs))
            } else {
                // Buffer was trimmed from the head (cap kicked in) or
                // cleared via `clearServerLog()`. Full replacement is
                // unavoidable; selection is lost but the user explicitly
                // accepted that by streaming past the cap.
                storage.beginEditing()
                storage.setAttributedString(NSAttributedString(string: text, attributes: attrs))
                storage.endEditing()
            }
        }

        // Auto-scroll: every text change while on, plus the off→on toggle
        // (so flipping the switch back to ON catches up immediately).
        let toggledOn = autoScroll && !context.coordinator.lastAutoScroll
        if autoScroll && (textChanged || toggledOn) {
            textView.scrollToEndOfDocument(nil)
        }
        context.coordinator.lastAutoScroll = autoScroll
    }
}

