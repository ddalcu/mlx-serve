import SwiftUI

struct StatusMenuView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager
    @State private var showDownloads = false
    let openChat: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Group {
                    if let url = Bundle.module.url(forResource: "appiconb", withExtension: "png", subdirectory: "Resources"),
                       let nsImg = NSImage(contentsOf: url) {
                        Image(nsImage: nsImg)
                            .resizable()
                            .frame(width: 20, height: 20)
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                    } else {
                        Image(systemName: "brain.head.profile")
                            .font(.title3)
                            .foregroundStyle(.secondary)
                    }
                }
                Text("MLX Claw")
                    .font(.headline)
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

                    Button {
                        server.toggle(modelPath: appState.selectedModelPath)
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
                        Text("Download Gemma 4 Models")
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

            // Chat & Quit
            HStack(spacing: 10) {
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
                    NSApplication.shared.terminate(nil)
                } label: {
                    HStack {
                        Image(systemName: "power")
                        Text("Quit")
                    }
                    .frame(maxWidth: .infinity)
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

    private let endpoints: [(method: String, path: String)] = [
        ("GET", "/health"),
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/embeddings"),
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
