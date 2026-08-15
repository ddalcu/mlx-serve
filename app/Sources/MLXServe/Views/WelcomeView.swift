import SwiftUI

struct WelcomeView: View {
    let onDismiss: () -> Void
    /// False when no downloaded model can serve chat — the Run-models panel
    /// shows the recommended-download card instead of an "installed" note. A
    /// one-time snapshot taken when the window is shown (this window isn't
    /// live-updating), not a reactive binding.
    let hasChatModels: Bool
    /// Calls `AppState.showModels()`.
    let onOpenModelBrowser: () -> Void
    /// Bumps `AppState.pendingChatOpenTick`, same bridge. Fired on dismiss and
    /// once the starter download has the server up — the window's job is to end
    /// in a chat, not in a closed window.
    let onOpenChat: () -> Void

    /// Injected AT THE SHEET (see the chat scene) — a sheet does not inherit
    /// the environment of the view it hangs on.
    @EnvironmentObject var appState: AppState
    /// For the live memory meter (GPU bar shows when a model is loaded).
    @EnvironmentObject var server: ServerManager

    /// Which left-column bullet is selected — drives the right column.
    @State private var selected: WelcomeFeature = .default
    @State private var appeared = false
    /// UserDefaults-backed: when set, the next launch skips this window and
    /// opens Chat directly (`LaunchDecision.resolve`).
    @AppStorage(LaunchDecision.suppressDefaultsKey) private var suppressWelcome = false

    // CLI install row state. nil probe = still checking (the probe spawns the
    // user's login shell to read the real PATH, so it runs off-main).
    @State private var cliProbe: CLIInstaller.Probe?
    @State private var cliInstalling = false
    @State private var cliError: String?

    /// The white monochrome mark (not the colored app icon) — reads cleanly on
    /// the dark welcome surface.
    private static let logoImage: NSImage? = BundledAsset.image("mlx-white.png")

    /// Derived from `UpdateChecker.repo` (the app's single source of truth
    /// for the GitHub repo) so the star link can never drift from it.
    static let gitHubStarURL = URL(string: "https://github.com/\(UpdateChecker.repo)")!

    // MARK: - Layout constants

    // Both dimensions are pinned so the sheet has a deterministic size; the
    // interior is free to use flexible/centered layout inside it.
    private static let windowWidth: CGFloat = 850
    private static let windowHeight: CGFloat = 550
    private static let leftColumnWidth: CGFloat = 292
    /// Width of the drawn connector between the selected card and the panel.
    private static let connectorWidth: CGFloat = 28
    private static let edgePadding: CGFloat = 32

    /// Coordinate space the selected card reports its centre in, so the
    /// connector can meet it at exactly its middle whatever the copy length.
    static let bodySpace = "welcomeBody"

    /// Vertical centre of the SELECTED card, in `bodySpace`. Published by the
    /// card itself — the three have different heights (their descriptions run
    /// one to three lines), so this can't be arithmetic.
    @State private var connectorY: CGFloat?

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(.horizontal, Self.edgePadding)
                .padding(.top, 26)
                .padding(.bottom, 22)

            // Body: the feature list, a connector, and the content panel it
            // points at. The panel does NOT repeat the selected card's title
            // and description — the card is right there, joined to it.
            HStack(alignment: .top, spacing: 0) {
                featureCards
                connector
                rightPanel(for: selected)
            }
            .coordinateSpace(name: Self.bodySpace)
            .onPreferenceChange(WelcomeCardAnchorKey.self) { connectorY = $0 }
            .padding(.horizontal, Self.edgePadding)
            .frame(maxHeight: .infinity)

            footer
                .padding(.horizontal, Self.edgePadding)
                .padding(.top, 20)
                .padding(.bottom, 24)
        }
        .frame(width: Self.windowWidth, height: Self.windowHeight)
        .background(.ultraThinMaterial)
        .onAppear {
            withAnimation(.easeOut(duration: 0.4)) { appeared = true }
        }
        .opacity(appeared ? 1 : 0)
        .task {
            // The App Store build can't install a CLI symlink, so don't probe.
            guard BuildFeatures.current.cliInstaller else { return }
            let probe = await Task.detached { CLIInstaller.probe() }.value
            cliProbe = probe
        }
    }

    // MARK: - Leaving

    /// The ONE way out of this window. Chat is opened first so that when Browse
    /// also opens the Model Browser, the browser lands in front of a chat
    /// window that's already there — closing it drops the user on a composer
    /// instead of an empty desktop (`WelcomeExit`, live dead end 2026-08-08 —
    /// unbuildable now that this is a sheet ON the chat window).
    private func leave(_ exit: WelcomeExit) {
        if exit.opensChat { onOpenChat() }
        if exit.opensModelBrowser { onOpenModelBrowser() }
        if exit.closesWelcome { onDismiss() }
    }

    // MARK: - Header (logo + title, star link)

    private var header: some View {
        HStack(alignment: .center, spacing: 14) {
            logoTile
            VStack(alignment: .leading, spacing: 2) {
                Text("MLX Core")
                    .font(.system(size: 22, weight: .semibold))
                Text("Local AI on Apple Silicon")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 12)

            // A real (if quiet) control rather than underlined link text: this
            // is an ordinary button that opens a web page, and macOS spells
            // that as a bordered control.
            Button {
                NSWorkspace.shared.open(Self.gitHubStarURL)
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: "star.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(.yellow)
                    Text("Star on GitHub")
                }
                .font(.subheadline)
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
            .help("Open the mlx-serve repository on GitHub")
        }
    }

    private var logoTile: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.white.opacity(0.06))
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Color.white.opacity(0.12), lineWidth: 1)
            if let logo = Self.logoImage {
                Image(nsImage: logo)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 30, height: 30)
            }
        }
        .frame(width: 46, height: 46)
    }

    // MARK: - Feature list (the selector)

    private var featureCards: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("GET STARTED")
                .font(.caption2.weight(.semibold))
                .tracking(0.7)
                .foregroundStyle(.tertiary)
                .padding(.leading, 2)
                .padding(.bottom, 2)

            ForEach(WelcomeFeature.ordered) { feature in
                WelcomeFeatureCard(feature: feature, isSelected: selected == feature) {
                    withAnimation(.easeInOut(duration: 0.18)) { selected = feature }
                }
            }
            Spacer(minLength: 0)
        }
        .frame(width: Self.leftColumnWidth, alignment: .top)
    }

    // MARK: - Connector

    /// The line that joins the selected card to the panel. It replaces the
    /// panel's old duplicated header: with the two visibly connected, the copy
    /// only has to exist once. Gradient accent → neutral so it leaves the
    /// selection and arrives as the panel's own hairline.
    private var connector: some View {
        GeometryReader { geo in
            LinearGradient(
                colors: [Color.accentColor.opacity(0.8), Color.primary.opacity(0.14)],
                startPoint: .leading, endPoint: .trailing
            )
            .frame(height: 1.5)
            .position(x: geo.size.width / 2,
                      // Before the first card reports in, sit at the panel's
                      // top rather than jumping in from the middle.
                      y: connectorY ?? 40)
            .animation(.easeInOut(duration: 0.18), value: connectorY)
        }
        .frame(width: Self.connectorWidth)
        .allowsHitTesting(false)
    }

    // MARK: - Right panel (content only — the card carries the copy)

    @ViewBuilder
    private func rightPanel(for feature: WelcomeFeature) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            panelContent(for: feature.rightPanel)
            Spacer(minLength: 0)
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.primary.opacity(0.05))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        )
    }

    @ViewBuilder
    private func panelContent(for panel: WelcomeRightPanel) -> some View {
        switch panel {
        case .modelDownload:
            runModelsPanel
        case .surfaces:
            surfacesPanel
        case .toolsDemo:
            toolsDemoPanel
        }
    }

    // MARK: - Surfaces panel (App / Menu bar / Terminal)

    /// Where you can drive this app from. Two of the three ship inside the
    /// bundle, so they state "Installed" rather than offering a control that
    /// could do nothing; only the Terminal command is something to add, which
    /// is what makes it read as the one action on the panel instead of a lone
    /// install row with no context.
    private var surfacesPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(WelcomeSurface.ordered) { surface in
                surfaceRow(surface)
            }
            Spacer(minLength: 0)
        }
    }

    private func surfaceRow(_ surface: WelcomeSurface) -> some View {
        HStack(alignment: .center, spacing: 10) {
            Image(systemName: surface.icon)
                .font(.system(size: 18))
                .foregroundColor(.accentColor)
                .frame(width: 26, alignment: .center)
            VStack(alignment: .leading, spacing: 2) {
                Text(surface.title)
                    .font(.headline)
                Text(caption(for: surface))
                    .font(.callout)
                    .foregroundStyle(captionStyle(for: surface))
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                    .help(caption(for: surface))
            }
            Spacer(minLength: 0)
            if surface.shipsWithTheApp {
                installedBadge
            } else {
                cliTrailingControl
            }
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.primary.opacity(0.05)))
    }

    /// Terminal has no constant caption — its line is live (`cliCaption`).
    private func caption(for surface: WelcomeSurface) -> String {
        surface.caption ?? cliCaption
    }

    private func captionStyle(for surface: WelcomeSurface) -> AnyShapeStyle {
        surface.caption == nil && cliError != nil
            ? AnyShapeStyle(Color.red) : AnyShapeStyle(.secondary)
    }

    /// Stated, not offered: these two are the app itself.
    private var installedBadge: some View {
        Label("Installed", systemImage: "checkmark.circle.fill")
            .font(.callout.weight(.semibold))
            .foregroundStyle(.green)
            .labelStyle(.titleAndIcon)
    }

    // MARK: - Tools demo panel

    /// A silent, looping screen recording of the agent using its tools. A demo
    /// answers "what does this actually do" in a way the sentence on the card
    /// beside it cannot — and it is muted and chrome-free because it is
    /// illustration, not media the user came here to operate.
    @ViewBuilder
    private var toolsDemoPanel: some View {
        if let url = BundledAsset.url(Self.toolsDemoFileName) {
            LoopingVideoView(url: url)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        } else {
            // A build that shipped without the asset says so quietly rather
            // than leaving a black rectangle that reads as a broken player.
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.gray.opacity(0.22))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .overlay(
                    Text("Demo unavailable in this build.")
                        .font(.callout)
                        .foregroundStyle(.secondary))
        }
    }

    static let toolsDemoFileName = "tools.mov"

    /// The Run-models panel: the live memory meter, then the best model of each
    /// type that fits this Mac (`WelcomeModelPicks`), each with a one-line
    /// strength and a Get/Use control. The overall recommended pick (the
    /// app-wide `starterPick`) is marked.
    private var runModelsPanel: some View {
        let picks = WelcomeModelPicks.forMemory(SystemMemoryInfo.current())
        return VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 8) {
                panelLabel("THIS MAC")
                // The same GPU + Available RAM meter the menu bar shows.
                MemoryMeter.live(server: server.memoryInfo)
            }

            VStack(alignment: .leading, spacing: 8) {
                panelLabel("BEST MODELS FOR YOUR MAC")

                // The first entry (General) is the everyday default — marked as
                // the suggested starting point.
                ForEach(Array(picks.enumerated()), id: \.element.id) { index, entry in
                    WelcomeModelRow(
                        entry: entry,
                        isRecommended: index == 0,
                        onOpenChat: { leave(.useModel) }
                    )
                }

                Button {
                    leave(.browseModels)
                } label: {
                    HStack(spacing: 4) {
                        Text("Browse all models")
                        Image(systemName: "arrow.right")
                            .font(.caption2.weight(.semibold))
                    }
                    .font(.callout)
                }
                .buttonStyle(.plain)
                .foregroundStyle(Color.accentColor)
                .padding(.top, 2)
            }
        }
    }

    /// One small-caps label inside the panel. Same treatment as the tray's
    /// section headers, so the two surfaces read as one design.
    private func panelLabel(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .tracking(0.6)
            .foregroundStyle(.secondary)
    }

    // MARK: - Footer ("Don't show again" leading, primary action trailing)

    private var footer: some View {
        HStack(spacing: 16) {
            Toggle("Don't show again", isOn: $suppressWelcome)
                .toggleStyle(.checkbox)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Spacer(minLength: 0)

            // Bottom-trailing default button, as every macOS sheet places it —
            // and it says what happens next rather than acknowledging the
            // window. Return activates it.
            Button {
                leave(.startChatting)
            } label: {
                Text(hasChatModels ? "Start Chatting" : "Continue")
                    .font(.headline)
                    .frame(minWidth: 150)
                    .padding(.vertical, 4)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .keyboardShortcut(.defaultAction)
        }
    }

    // MARK: - CLI install state (the Terminal row of the surfaces panel)

    /// The App Store build never probes (it cannot symlink out of the sandbox),
    /// so `cliProbe` stays nil there forever — which as a bare `case nil` was a
    /// spinner that never resolves. The row states the situation instead.
    private var cliInstallable: Bool { BuildFeatures.current.cliInstaller }

    @ViewBuilder private var cliTrailingControl: some View {
        if !cliInstallable {
            EmptyView()
        } else {
            cliProbeControl
        }
    }

    @ViewBuilder private var cliProbeControl: some View {
        switch cliProbe {
        case nil:
            ProgressView().controlSize(.small)
        case .installed:
            Label("Installed", systemImage: "checkmark.circle.fill")
                .font(.callout.weight(.semibold))
                .foregroundStyle(.green)
                .labelStyle(.titleAndIcon)
        case .binaryMissing:
            EmptyView()
        case .available(let target):
            Button {
                installCLI(target: target)
            } label: {
                Text(cliInstalling ? "Installing…" : "Install")
                    .font(.callout.weight(.semibold))
            }
            .controlSize(.large)
            .disabled(cliInstalling)
        }
    }

    private var cliCaption: String {
        if let cliError { return cliError }
        guard cliInstallable else {
            return "Not available in the App Store build — get the CLI from GitHub."
        }
        switch cliProbe {
        case nil:
            return "Run mlx-serve from Terminal."
        case .installed(let link):
            return "Installed at \(abbreviateHome(link))"
        case .binaryMissing:
            return "mlx-serve binary not found in this build."
        case .available(let target):
            return target.requiresAdmin
                ? "Adds a link in /usr/local/bin (asks for your password)."
                : "Adds a link in \(abbreviateHome(target.directory)) — no password needed."
        }
    }

    private func abbreviateHome(_ path: String) -> String {
        (path as NSString).abbreviatingWithTildeInPath
    }

    private func installCLI(target: CLIInstaller.Target) {
        cliInstalling = true
        cliError = nil
        Task.detached {
            let result: Result<String, Error>
            do {
                guard let source = CLIInstaller.resolveBinarySource() else {
                    throw CLIInstaller.InstallError.binaryNotFound
                }
                let link = target.requiresAdmin
                    ? try CLIInstaller.installWithAdmin(binarySource: source)
                    : try CLIInstaller.installIntoHomeBin(directory: target.directory,
                                                          binarySource: source)
                result = .success(link)
            } catch {
                result = .failure(error)
            }
            await MainActor.run {
                cliInstalling = false
                switch result {
                case .success(let link): cliProbe = .installed(linkPath: link)
                case .failure(let error): cliError = error.localizedDescription
                }
            }
        }
    }
}

/// Where the connector should meet the card column: the vertical centre of the
/// SELECTED card, in `WelcomeView.bodySpace`. Only the selected card publishes a
/// value — the reduce keeps whichever child sent a non-nil one.
struct WelcomeCardAnchorKey: PreferenceKey {
    static let defaultValue: CGFloat? = nil
    static func reduce(value: inout CGFloat?, nextValue: () -> CGFloat?) {
        value = nextValue() ?? value
    }
}

/// One clickable feature in the welcome screen's left column. Selecting it
/// drives the right panel.
private struct WelcomeFeatureCard: View {
    let feature: WelcomeFeature
    let isSelected: Bool
    let onTap: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: onTap) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: feature.icon)
                    .font(.system(size: 17))
                    .foregroundColor(isSelected ? .accentColor : .secondary)
                    .frame(width: 24, alignment: .center)
                    .padding(.top, 1)
                VStack(alignment: .leading, spacing: 3) {
                    Text(feature.title)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(.primary)
                    Text(feature.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 13)
            .padding(.vertical, 12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(isSelected
                          ? Color.accentColor.opacity(0.13)
                          : Color.primary.opacity(hovering ? 0.05 : 0))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(isSelected ? Color.accentColor.opacity(0.55) : Color.clear,
                                  lineWidth: 1)
            )
            .contentShape(Rectangle())
            // Publish this card's centre so the connector can meet it. Gated on
            // selection: an unselected card publishing would leave the line
            // pointing at whichever row reported last.
            .background(
                GeometryReader { geo in
                    Color.clear.preference(
                        key: WelcomeCardAnchorKey.self,
                        value: isSelected
                            ? geo.frame(in: .named(WelcomeView.bodySpace)).midY
                            : nil)
                }
            )
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
    }
}

/// One compact row in the welcome screen's "best models for your Mac" list: a
/// type chip, the model name, a one-line strength, and a Get / progress / Use
/// control. Every welcome pick is a safetensors repo (no GGUF quant to name).
private struct WelcomeModelRow: View {
    let entry: WelcomeModelPick
    let isRecommended: Bool
    let onOpenChat: () -> Void

    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState

    private var pick: RecommendedModelPick { entry.pick }
    private var state: DownloadManager.DownloadState? { downloads.downloads[pick.repoId] }
    private var isReady: Bool { downloads.isReady(pick.repoId) }

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(entry.category)
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        .background(Capsule().fill(Color.secondary.opacity(0.15)))
                        .foregroundStyle(.secondary)
                    Text(pick.name)
                        .font(.callout.weight(.medium))
                    if isRecommended {
                        Image(systemName: "sparkles")
                            .font(.caption2)
                            .foregroundStyle(.tint)
                            .help("Recommended for your Mac")
                    }
                }
                Text(entry.strength)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 8)
            control
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(isRecommended ? Color.accentColor.opacity(0.08) : Color.primary.opacity(0.04))
        )
    }

    @ViewBuilder private var control: some View {
        if isReady {
            Button {
                useAndChat()
            } label: {
                Text("Use").font(.caption.weight(.semibold))
            }
            .controlSize(.small)
            .buttonStyle(.borderedProminent)
        } else if let state, state.status == .downloading {
            VStack(spacing: 2) {
                ProgressView(value: state.progress).frame(width: 58)
                Text(state.percentFormatted)
                    .font(.system(size: 9).monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        } else {
            Button {
                startDownload()
            } label: {
                Text(downloads.hasPartialDownload(pick.repoId) ? "Resume" : "Get")
                    .font(.caption.weight(.semibold))
            }
            .controlSize(.small)
            .buttonStyle(.bordered)
        }
    }

    private func useAndChat() {
        Task {
            if let dir = downloads.existingModelDir(for: pick.repoId) {
                _ = await appState.useModelAndAwaitReady(atPath: dir)
            }
            onOpenChat()
        }
    }

    private func startDownload() {
        downloads.start(repoId: pick.repoId) {
            appState.refreshModels()
            // `onFinish` runs on failure and cancel too (DownloadManager's
            // contract) — only a download that actually landed proceeds.
            // Otherwise the row re-offers Get/Resume; dismissing the welcome
            // sheet into a chat with no model is the dead end it exists to
            // prevent.
            guard downloads.isReady(pick.repoId) else { return }
            useAndChat()
        }
    }
}
