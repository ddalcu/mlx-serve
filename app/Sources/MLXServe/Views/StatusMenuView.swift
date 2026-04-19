import SwiftUI

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

struct StatusMenuView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager
    @State private var showDownloads = false
    @State private var showLog = false
    let openChat: () -> Void
    let openBrowser: () -> Void
    let openModelBrowser: () -> Void

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
                    Picker("Model", selection: $appState.selectedModelPath) {
                        ForEach(appState.localModels) { model in
                            Text(model.name).tag(model.path)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)

                    ContextSizeSection(contextSize: $appState.contextSize, isRunning: server.status == .running || server.status == .starting, maxSafeContext: server.memoryInfo?.maxSafeContext ?? 0)

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
                            Text("Browse All MLX Models")
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
                    Image(systemName: "globe")
                }
                .buttonStyle(.bordered)
                .help("Browser")

                Button {
                    let path = NSString(string: "~/.mlx-serve").expandingTildeInPath
                    NSWorkspace.shared.open(URL(fileURLWithPath: path))
                } label: {
                    Image(systemName: "folder.badge.gearshape")
                }
                .buttonStyle(.bordered)
                .help("MLX Serve Folder")

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
        ("POST", "/v1/completions"),
        ("POST", "/v1/embeddings"),
        ("POST", "/v1/messages"),
    ]

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
    var maxSafeContext: Int

    private let presets: [(String, Int)] = [
        ("Auto", 0),
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
            if maxSafeContext > 0 {
                Text("GPU safe max: \(Self.formatTokens(maxSafeContext))")
                    .font(.caption2)
                    .foregroundColor(contextSize > 0 && contextSize > maxSafeContext ? .orange : .secondary)
            }
            if isRunning {
                Text("Restart server to apply")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private var formatted: String {
        if contextSize == 0 {
            return maxSafeContext > 0 ? "Auto (\(Self.formatTokens(maxSafeContext)))" : "Auto"
        }
        return Self.formatTokens(contextSize)
    }

    static func formatTokens(_ tokens: Int) -> String {
        if tokens >= 1024 { return "\(tokens / 1024)K" }
        return "\(tokens)"
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

