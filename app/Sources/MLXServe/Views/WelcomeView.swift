import SwiftUI

struct WelcomeView: View {
    let onDismiss: () -> Void

    @State private var pulseMenu = false
    @State private var appeared = false

    // CLI install row state. nil probe = still checking (the probe spawns the
    // user's login shell to read the real PATH, so it runs off-main).
    @State private var cliProbe: CLIInstaller.Probe?
    @State private var cliInstalling = false
    @State private var cliError: String?

    private static func loadBundledImage(_ name: String) -> NSImage? {
        let candidates: [URL?] = [
            Bundle.main.resourceURL?.appendingPathComponent(name),
            Bundle.main.bundleURL.appendingPathComponent("MLXCore_MLXCore.bundle/Resources/\(name)"),
            // Dev builds: look relative to source
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent(name),
        ]
        for case let url? in candidates {
            if let img = NSImage(contentsOf: url) { return img }
        }
        return nil
    }

    private static let appIcon: NSImage? = loadBundledImage("appiconb.png")

    private static let trayIcon: NSImage? = {
        guard let img = loadBundledImage("tray.png") else { return nil }
        img.isTemplate = true  // adapts to light/dark mode
        return img
    }()

    var body: some View {
        VStack(spacing: 0) {
            Spacer().frame(height: 24)

            // App icon
            if let icon = Self.appIcon {
                Image(nsImage: icon)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 72, height: 72)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                    .padding(.bottom, 10)
            }

            Text("Welcome to MLX Core")
                .font(.system(size: 22, weight: .semibold))
                .padding(.bottom, 3)

            Text("Local AI on Apple Silicon")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .padding(.bottom, 20)

            // Feature cards
            VStack(alignment: .leading, spacing: 12) {
                FeatureRow(
                    icon: "menubar.rectangle",
                    title: "Lives in your menu bar",
                    description: "Click the icon in the top-right of your screen to start a server, download models, and chat."
                )
                FeatureRow(
                    icon: "bolt.fill",
                    title: "Run models locally",
                    description: "No cloud, no API keys. All processing stays on your device."
                )
                FeatureRow(
                    icon: "wrench.and.screwdriver.fill",
                    title: "Agent with tools",
                    description: "Let the model read files, run commands, search the web, and write code."
                )
            }
            .padding(.horizontal, 28)
            .padding(.bottom, 16)

            // Tray hint
            HStack(spacing: 6) {
                Image(systemName: "arrow.up.right")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.accentColor)
                    .offset(y: pulseMenu ? -2 : 2)
                    .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: pulseMenu)
                Text("Look for the")
                    .foregroundStyle(.secondary)
                if let tray = Self.trayIcon {
                    Image(nsImage: tray)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 16, height: 16)
                } else {
                    Image(systemName: "brain.head.profile")
                        .foregroundColor(.accentColor)
                }
                Text("icon in your menu bar")
                    .foregroundStyle(.secondary)
            }
            .font(.caption)
            .padding(.bottom, 14)

            // CLI install row — puts an `mlx-serve` symlink on the PATH so
            // the server runs from Terminal. Rendered in every state (fixed
            // height) so the pre-sized welcome window never clips.
            // The App Store build never probes (see the `.task` below), so the
            // row would sit on `cliProbe == nil` forever — a permanent spinner
            // offering an install that can't happen. Drop it entirely.
            if BuildFeatures.current.cliInstaller {
                cliSection
                    .padding(.horizontal, 28)
                    .padding(.bottom, 14)
            }

            // Dismiss button
            Button {
                onDismiss()
                NSApp.keyWindow?.close()
            } label: {
                Text("Got it")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 6)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .padding(.horizontal, 28)
            .padding(.bottom, 20)
        }
        .frame(width: 420)
        .fixedSize(horizontal: true, vertical: true)
        .background(.ultraThinMaterial)
        .onAppear {
            pulseMenu = true
            withAnimation(.easeOut(duration: 0.5)) { appeared = true }
        }
        .opacity(appeared ? 1 : 0)
        .task {
            // The App Store build can't install a CLI symlink, so don't offer it.
            guard BuildFeatures.current.cliInstaller else { return }
            let probe = await Task.detached { CLIInstaller.probe() }.value
            cliProbe = probe
        }
    }

    // MARK: - CLI install row

    @ViewBuilder private var cliSection: some View {
        HStack(alignment: .center, spacing: 10) {
            Image(systemName: "terminal")
                .font(.system(size: 16))
                .foregroundColor(.accentColor)
                .frame(width: 24, alignment: .center)
            VStack(alignment: .leading, spacing: 2) {
                Text("Terminal command")
                    .font(.subheadline.weight(.semibold))
                Text(cliCaption)
                    .font(.caption)
                    .foregroundStyle(cliError == nil ? AnyShapeStyle(.secondary) : AnyShapeStyle(Color.red))
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(cliCaption)
            }
            Spacer()
            cliTrailingControl
        }
        .frame(minHeight: 34)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.primary.opacity(0.05)))
    }

    @ViewBuilder private var cliTrailingControl: some View {
        switch cliProbe {
        case nil:
            ProgressView().controlSize(.small)
        case .installed:
            Label("Installed", systemImage: "checkmark.circle.fill")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.green)
                .labelStyle(.titleAndIcon)
        case .binaryMissing:
            EmptyView()
        case .available(let target):
            Button {
                installCLI(target: target)
            } label: {
                Text(cliInstalling ? "Installing…" : "Install")
                    .font(.caption.weight(.semibold))
            }
            .controlSize(.small)
            .disabled(cliInstalling)
        }
    }

    private var cliCaption: String {
        if let cliError { return cliError }
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

private struct FeatureRow: View {
    let icon: String
    let title: String
    let description: String

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 16))
                .foregroundColor(.accentColor)
                .frame(width: 24, alignment: .center)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                Text(description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}
