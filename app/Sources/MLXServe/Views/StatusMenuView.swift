import SwiftUI

struct StatusMenuView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager
    @State private var showDownloads = false
    @State private var showLog = false
    let openChat: () -> Void
    let openBrowser: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Text("MLX Claw")
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
                    Picker("Model", selection: $appState.selectedModelPath) {
                        ForEach(appState.localModels) { model in
                            Text(model.name).tag(model.path)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)

                    ContextSizeSection(contextSize: $appState.contextSize, isRunning: server.status == .running || server.status == .starting)

                    HStack(spacing: 6) {
                        Button {
                            server.toggle(modelPath: appState.selectedModelPath, contextSize: appState.contextSize)
                        } label: {
                            HStack {
                                Image(systemName: server.status == .running || server.status == .starting ? "stop.fill" : "play.fill")
                                Text(server.status == .running || server.status == .starting ? "Stop Server" : "Start Server")
                            }
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(server.status == .running ? .red : .accentColor)
                        .disabled(appState.selectedModelPath.isEmpty)
                        .controlSize(.regular)

                        Button {
                            showLog.toggle()
                        } label: {
                            Image(systemName: "text.alignleft")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.regular)
                        .help("Server Log")
                    }

                    Toggle("Auto-start on launch", isOn: $appState.autoStartServer)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .toggleStyle(.switch)
                        .controlSize(.mini)

                    if showLog {
                        ServerLogView(log: server.serverLog)
                    }

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
                    if let info = server.modelInfo {
                        HStack {
                            Text("Model")
                                .font(.subheadline.weight(.medium))
                                .foregroundStyle(.secondary)
                            Spacer()
                        }
                        HStack(spacing: 6) {
                            Text(info.name)
                                .font(.caption.monospaced())
                                .lineLimit(1)
                            if info.quantBits > 0 {
                                Text("\(info.quantBits)-bit")
                                    .font(.caption2.weight(.semibold))
                                    .padding(.horizontal, 5)
                                    .padding(.vertical, 1)
                                    .background(.quaternary)
                                    .clipShape(Capsule())
                            }
                        }
                        Text("\(info.layers) layers, \(info.hiddenSize)-dim")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }

                    if let mem = server.memoryInfo {
                        HStack {
                            Text("GPU Memory")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                            Text(mem.activeFormatted)
                                .font(.caption.monospaced())
                        }
                        ProgressView(value: Double(mem.activeBytes), total: Double(max(mem.peakBytes, mem.activeBytes) * 2))
                            .tint(.blue)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)

                // Max Tokens
                Divider().padding(.horizontal, 12)
                MaxTokensSection(maxTokens: $appState.maxTokens)
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
                    showDownloads.toggle()
                } label: {
                    HStack {
                        Image(systemName: "arrow.down.circle")
                        Text("Download Models")
                            .font(.subheadline)
                        Spacer()
                        Image(systemName: showDownloads ? "chevron.up" : "chevron.down")
                            .font(.caption)
                    }
                }
                .buttonStyle(.plain)

                if showDownloads {
                    ModelDownloadView()
                        .environmentObject(downloads)
                        .environmentObject(appState)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)

            Divider().padding(.horizontal, 12)

            // Chat, Browser, Claude Code & Quit
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
                    openBrowser()
                } label: {
                    HStack {
                        Image(systemName: "globe")
                        Text("Browser")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)

                Button {
                    let path = NSString(string: "~/.mlx-serve").expandingTildeInPath
                    NSWorkspace.shared.open(URL(fileURLWithPath: path))
                } label: {
                    Image(systemName: "folder.badge.gearshape")
                }
                .buttonStyle(.bordered)
                .help("MLX Serve Folder")

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
            .padding(.bottom, 4)

            // Claude Code launcher
            if case .running = server.status {
                Button {
                    launchClaudeCode(baseURL: server.baseURL)
                } label: {
                    HStack {
                        Image(systemName: "terminal")
                        Text("Launch Claude Code")
                        Spacer()
                        Image(systemName: "arrow.up.forward.square")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .tint(.purple)
                .padding(.horizontal, 16)
                .padding(.bottom, 14)
            } else {
                Spacer().frame(height: 10)
            }
        }
        .frame(width: 320)
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

    private let endpoints: [(method: String, path: String)] = [
        ("GET", "/health"),
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/embeddings"),
        ("POST", "/v1/messages"),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Endpoints")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.secondary)

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

struct MaxTokensSection: View {
    @Binding var maxTokens: Int

    private let presets: [(String, Int)] = [
        ("2K", 2048),
        ("4K", 4096),
        ("8K", 8192),
        ("16K", 16384),
        ("32K", 32768),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Max Tokens")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.secondary)
                Spacer()
                Text(formatted)
                    .font(.caption.monospaced())
            }
            HStack(spacing: 4) {
                ForEach(presets, id: \.1) { label, value in
                    Button(label) {
                        maxTokens = value
                    }
                    .buttonStyle(.bordered)
                    .tint(maxTokens == value ? .accentColor : nil)
                    .controlSize(.mini)
                }
            }
        }
    }

    private var formatted: String {
        if maxTokens >= 1024 {
            return "\(maxTokens / 1024)K"
        }
        return "\(maxTokens)"
    }
}

struct ContextSizeSection: View {
    @Binding var contextSize: Int
    var isRunning: Bool

    private let presets: [(String, Int)] = [
        ("16K", 16384),
        ("32K", 32768),
        ("64K", 65536),
        ("128K", 131072),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Context Size")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.secondary)
                Spacer()
                Text(formatted)
                    .font(.caption.monospaced())
            }
            HStack(spacing: 4) {
                ForEach(presets, id: \.1) { label, value in
                    Button(label) {
                        contextSize = value
                    }
                    .buttonStyle(.bordered)
                    .tint(contextSize == value ? .accentColor : nil)
                    .controlSize(.mini)
                    .disabled(isRunning)
                }
            }
            if isRunning {
                Text("Restart server to apply")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private var formatted: String {
        if contextSize >= 1024 {
            return "\(contextSize / 1024)K"
        }
        return "\(contextSize)"
    }
}

/// Launch Claude Code CLI configured to use the local mlx-serve server.
private func launchClaudeCode(baseURL: String) {
    let model = "mlx-serve"
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
    claude --model \(model)
    """

    let path = NSTemporaryDirectory() + "mlx-claude-code.command"
    try? scriptContent.write(toFile: path, atomically: true, encoding: .utf8)
    try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: path)
    NSWorkspace.shared.open(URL(fileURLWithPath: path))
}

struct ServerLogView: View {
    let log: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Server Log")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(log, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                        .font(.caption2)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("Copy Log")
            }

            ScrollView {
                Text(log.isEmpty ? "(no output yet)" : log)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(log.isEmpty ? .tertiary : .primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(height: 180)
            .background(.black.opacity(0.3))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }
}

