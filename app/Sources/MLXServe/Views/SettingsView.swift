import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Single-window form for the user-facing mlx-serve tunables. Bindings flow
/// through `appState.serverOptions`; AppState's `didSet` auto-saves to
/// UserDefaults.
///
/// Intentionally narrow surface: only the things end-users actually want to
/// tune. Request-timeout lives in the CLI for power users; per-request
/// spec-decode overrides duplicate what the Speculative Decoding toggles
/// already express; "Enable thinking" lives on the chat toolbar.
struct SettingsView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    var body: some View {
        VStack(spacing: 0) {
            if server.needsRestartFor(appState.serverOptions) {
                RestartBanner()
            }
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    SettingsSection(
                        title: "Model Folders",
                        subtitle: "Always scans ~/.mlx-serve/models and ~/.lmstudio/models. Add one more folder here if your models live elsewhere — no restart needed."
                    ) {
                        ModelFoldersSectionContent()
                    }
                    SettingsSection(
                        title: "Server",
                        subtitle: "Server-launch flags. Restart the server to apply changes."
                    ) {
                        ServerSectionContent()
                    }
                    // Engine-aware sections. Each panel is hidden when its
                    // controls don't apply to the active engine — flipping
                    // `--kv-quant` on a GGUF model silently no-ops, so we'd
                    // rather not show that picker at all than mislead.
                    EngineAwareSections()
                    SettingsSection(
                        title: "Per-Request Defaults",
                        subtitle: "Apply on the next chat request — no restart needed."
                    ) {
                        RequestDefaultsSectionContent()
                    }

                    SettingsSection(
                        title: "Voice",
                        subtitle: "Clone your voice once — hands-free voice mode answers in it via the local TTS model. No clip set: answers use the macOS system voice. Applies to the next spoken sentence — no restart needed."
                    ) {
                        VoiceCloneSectionContent()
                    }

                    SettingsSection(
                        title: "Agent Sandbox",
                        subtitle: "Run the agent's shell commands inside an isolated Linux sandbox instead of directly on this Mac. Off by default; applies to the next command — no restart needed."
                    ) {
                        SandboxSectionContent()
                    }

                    SettingsSection(
                        title: "Messaging — Telegram bot",
                        subtitle: "Message your local model from your phone via a Telegram bot. No public URL or port-forwarding needed — the app long-polls Telegram over your normal internet connection, so it works behind home Wi-Fi."
                    ) {
                        MessagingSectionContent(bridge: appState.telegramBridge)
                    }

                    SettingsSection(
                        title: "Updates",
                        subtitle: "New versions ship on the project's GitHub releases page. Installing downloads the notarized app, swaps it in place, and relaunches — chats, models, and settings are untouched."
                    ) {
                        UpdatesSectionContent(updates: appState.updates)
                    }

                    ResetDefaultsFooter()
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .navigationTitle("Settings")
    }
}

// MARK: - Reset to Defaults footer

/// Single button that restores every field on the Settings screen to the
/// values in `ServerOptions()` (the struct's default initializer). Confirms
/// before discarding the user's tuning because some fields (drafter path,
/// custom temperature, etc.) take meaningful effort to set up.
private struct ResetDefaultsFooter: View {
    @EnvironmentObject var appState: AppState
    @State private var showConfirm = false

    var body: some View {
        HStack {
            Spacer()
            Button(role: .destructive) {
                showConfirm = true
            } label: {
                Label("Reset to Defaults", systemImage: "arrow.uturn.backward.circle")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help("Restores every Server / Speculative Decoding / Performance / Per-Request field to its built-in default. Server-launch fields still need a restart to take effect.")
        }
        .padding(.top, 4)
        .confirmationDialog(
            "Reset all settings to defaults?",
            isPresented: $showConfirm,
            titleVisibility: .visible
        ) {
            Button("Reset", role: .destructive) {
                appState.serverOptions = ServerOptions()
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("This restores every field on this screen — server flags, speculative decoding, performance (continuous batching / KV-quant / prefix cache), and per-request defaults — to the values that ship with the app. The change is local; the running server keeps its current flags until you hit Restart Now.")
        }
    }
}

// MARK: - Restart banner

private struct RestartBanner: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "arrow.clockwise.circle.fill")
                .font(.title2)
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 2) {
                Text("Some changes require a server restart")
                    .font(.subheadline.weight(.semibold))
                Text("Click Restart Now to apply, or Discard to revert the unsaved server-launch fields.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Restart Now") {
                let opts = appState.serverOptions
                let model = appState.selectedModelPath
                server.stop()
                if !model.isEmpty {
                    server.start(modelPath: model, options: opts)
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(appState.selectedModelPath.isEmpty)

            Button("Discard") {
                if let last = server.lastLaunchedOptions {
                    // Revert every server-launch field to the last-launched
                    // snapshot; per-request defaults are preserved. Start
                    // from `last` (which has all server fields right) and
                    // patch the per-request fields back from `current` so
                    // the user's mid-session sampler tweaks survive.
                    let current = appState.serverOptions
                    var reverted = last
                    reverted.defaultMaxTokens       = current.defaultMaxTokens
                    reverted.defaultTemperature     = current.defaultTemperature
                    reverted.defaultTopP            = current.defaultTopP
                    reverted.defaultTopK            = current.defaultTopK
                    reverted.defaultRepeatPenalty   = current.defaultRepeatPenalty
                    reverted.defaultPresencePenalty = current.defaultPresencePenalty
                    reverted.defaultReasoningBudget = current.defaultReasoningBudget
                    reverted.defaultEnableThinking  = current.defaultEnableThinking
                    reverted.perRequestEnablePLD    = current.perRequestEnablePLD
                    reverted.perRequestEnableDrafter = current.perRequestEnableDrafter
                    appState.serverOptions = reverted
                }
            }
            .buttonStyle(.bordered)
            .disabled(server.lastLaunchedOptions == nil)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(Color.orange.opacity(0.10))
        .overlay(Divider(), alignment: .bottom)
    }
}

// MARK: - Engine-aware section composer

/// Renders the engine-specific section set for the active model:
///   - MLX target:   Common Performance + MLX Performance + MLX Spec Decode
///   - GGUF target:  Common Performance + GGUF Performance
///   - DSV4 target:  Common Performance + DeepSeek-V4 (ds4) section
///   - No model yet: All sections shown (so users can pre-tune before
///                   loading); a banner clarifies that some controls only
///                   apply once a matching engine is loaded.
///
/// `ModelInfo.engine` is the source of truth — it's computed from the
/// `architecture` string the server reports in `/v1/models`. Pre-load
/// (`server.modelInfo == nil`) defaults to `.mlx` for display purposes
/// since that's still the most common path, but a banner notes the
/// fallback so power users know what's going on.
private struct EngineAwareSections: View {
    @EnvironmentObject var server: ServerManager

    /// Resolved engine for routing UI decisions. Nil when no model has
    /// loaded yet (server stopped or first start in progress) — that
    /// case shows all sections so users can pre-tune.
    private var engine: ServerEngine? { server.modelInfo?.engine }

    var body: some View {
        // Always-shown universal performance knob (applies to every engine).
        SettingsSection(
            title: "Performance (all engines)",
            subtitle: "Tunables that apply regardless of engine. Server-launch flag — restart to apply."
        ) {
            CommonPerformanceSectionContent()
        }

        // Engine-specific sections. Show all when no model is loaded so
        // the user can pre-tune; otherwise show only the matching set.
        let showMLX = (engine == nil || engine == .mlx)
        let showLlama = (engine == nil || engine == .llama)
        let showDs4 = (engine == nil || engine == .dsv4)

        if showMLX {
            SettingsSection(
                title: "Speculative Decoding (MLX only)",
                subtitle: "Big throughput wins on echo-heavy work; gates auto-disable on novel content. PLD + drafter are MLX-only kernels — they no-op on GGUF / DSV4."
            ) {
                SpecDecodeSectionContent()
            }
            SettingsSection(
                title: "MLX Performance",
                subtitle: "Continuous batching, KV-cache quantization, and the cross-request hot prefix cache. MLX-only — distinct kernels from llama.cpp."
            ) {
                PerformanceSectionContent()
            }
        }

        if showLlama {
            SettingsSection(
                title: "GGUF Performance (llama.cpp)",
                subtitle: "Knobs that apply when an embedded llama.cpp engine is serving a `.gguf` model. Distinct from the MLX Performance section — different kernels, different KV layout."
            ) {
                LlamaPerformanceSectionContent()
            }
        }

        if showDs4 {
            SettingsSection(
                title: "DeepSeek-V4 (ds4 engine)",
                subtitle: "Knobs for the embedded ds4 engine serving DeepSeek-V4-Flash. Ignored by the MLX and llama.cpp engines."
            ) {
                Ds4PerformanceSectionContent()
            }
        }

        if engine == nil {
            HStack(spacing: 8) {
                Image(systemName: "info.circle")
                    .foregroundStyle(.secondary)
                Text("No model loaded yet — every section is shown so you can pre-tune. Once a model is active, sections that don't apply will hide automatically.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, 4)
        }
    }
}

// MARK: - Section frame

private struct SettingsSection<Content: View>: View {
    let title: String
    let subtitle: String
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.title3.weight(.semibold))
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 18) {
                content
            }
            .padding(16)
            .background(Color(NSColor.controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }
}

// MARK: - One row helper

private struct SettingsRow<Control: View>: View {
    let title: String
    let explainer: String
    /// True when this field has been changed since the running server was
    /// last launched — i.e. the user has edited it but not yet hit "Restart
    /// Now". Drives the orange restart icon. False (or always-false for
    /// per-request fields) hides the icon. We deliberately don't show it on
    /// every server-launch row by default — that's noisy when nothing has
    /// actually been changed yet.
    var isDirty: Bool = false
    @ViewBuilder var control: Control

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                HStack(spacing: 6) {
                    Text(title)
                        .font(.body)
                    if isDirty {
                        Image(systemName: "arrow.clockwise.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .help("Restart the server to apply this change")
                    }
                }
                Spacer(minLength: 12)
                control
                    .frame(maxWidth: 280, alignment: .trailing)
            }
            Text(explainer)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// Shared dirty-bit helper. Compares a single `ServerOptions` keypath against
/// the snapshot the server was last launched with. Returns false until the
/// server has been launched at least once (no baseline to compare against).
fileprivate struct ServerLaunchDirty {
    let current: ServerOptions
    let last: ServerOptions?

    func dirty<V: Equatable>(_ keyPath: KeyPath<ServerOptions, V>) -> Bool {
        guard let last else { return false }
        return current[keyPath: keyPath] != last[keyPath: keyPath]
    }
}

// MARK: - Model folders section

/// One row showing the user-configured extra discovery root. The path is
/// rendered verbatim (raw, not standardized) so the user sees exactly what
/// they picked; discovery silently skips it when it doesn't resolve to an
/// existing directory. Picking a folder triggers an immediate refresh so the
/// menu-bar picker updates without a server restart.
private struct ModelFoldersSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var downloads: DownloadManager

    var body: some View {
        let pathText: String = {
            let raw = downloads.customRoot?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return raw.isEmpty ? "(none)" : raw
        }()
        let hasPath = !(downloads.customRoot?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)

        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text("Custom folder")
                    .font(.body)
                Spacer(minLength: 12)
                HStack(spacing: 8) {
                    Text(pathText)
                        .font(.caption.monospaced())
                        .foregroundStyle(hasPath ? .primary : .secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .frame(maxWidth: 220, alignment: .trailing)
                    Button("Choose…") { choose() }
                        .buttonStyle(.bordered)
                    Button("Clear") {
                        downloads.customRoot = nil
                        appState.refreshModels()
                    }
                    .buttonStyle(.bordered)
                    .disabled(!hasPath)
                }
            }
            Text("Accepts both flat layout (<name>/config.json) and 2-level layout (<author>/<name>/config.json).")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        if let existing = downloads.customRoot,
           !existing.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            panel.directoryURL = URL(fileURLWithPath: (existing as NSString).expandingTildeInPath)
        }
        if panel.runModal() == .OK, let url = panel.url {
            downloads.customRoot = url.path
            appState.refreshModels()
        }
    }
}

// MARK: - Server section

private struct ServerSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.lastLaunchedOptions)
    }

    var body: some View {
        if let m = meta["host"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.host)
            ) {
                TextField(
                    "",
                    text: Binding(
                        get: { appState.serverOptions.host },
                        set: { appState.serverOptions.host = $0.trimmingCharacters(in: .whitespaces) }
                    ),
                    prompt: Text("0.0.0.0")
                )
                .textFieldStyle(.roundedBorder)
                .font(.body.monospacedDigit())
                .multilineTextAlignment(.trailing)
                .frame(width: 160)
            }
        }
        PortRow()
        ContextSizeRow()
        if let m = meta["noVision"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.noVision)
            ) {
                Toggle("", isOn: $appState.serverOptions.noVision)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
        if let m = meta["logLevel"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.logLevel)
            ) {
                Picker("", selection: $appState.serverOptions.logLevel) {
                    ForEach(ServerOptions.LogLevel.allCases) { lvl in
                        Text(lvl.label).tag(lvl)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(minWidth: 180)
            }
        }
        if let m = meta["skipMemPreflight"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.skipMemPreflight)
            ) {
                Toggle("", isOn: $appState.serverOptions.skipMemPreflight)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
    }
}

/// Port text field with commit-on-valid semantics. The field edits a local
/// string so the user can clear it or type through invalid intermediate
/// states; only values `ServerOptions.parsePort` accepts are committed to
/// storage. Submitting (or an external change like Reset to Defaults /
/// Discard) snaps the display back to the last committed value.
private struct PortRow: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @State private var text: String = ""

    private var isDirty: Bool {
        guard let last = server.lastLaunchedOptions else { return false }
        return appState.serverOptions.port != last.port
    }

    var body: some View {
        if let m = ServerOptions.serverFlagFields["port"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: isDirty
            ) {
                TextField("", text: $text, prompt: Text("11234"))
                    .textFieldStyle(.roundedBorder)
                    .font(.body.monospacedDigit())
                    .multilineTextAlignment(.trailing)
                    .frame(width: 90)
                    .onAppear { text = "\(appState.serverOptions.port)" }
                    .onChange(of: text) { _, newValue in
                        if let p = ServerOptions.parsePort(newValue) {
                            appState.serverOptions.port = p
                        }
                    }
                    .onChange(of: appState.serverOptions.port) { _, newPort in
                        if ServerOptions.parsePort(text) != newPort {
                            text = "\(newPort)"
                        }
                    }
                    .onSubmit { text = "\(appState.serverOptions.port)" }
            }
        }
    }
}

/// Snapping slider over a fixed list of common context lengths, capped at the
/// model's declared maximum. The slider position 0 is "Auto" (= use model
/// default at load time). A secondary line shows the GPU-safe ceiling for
/// this Mac and warns when the chosen value exceeds it.
private struct ContextSizeRow: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private static let allPresets: [Int] = [
        0, 4_096, 8_192, 16_384, 32_768, 65_536,
        131_072, 262_144, 524_288, 1_048_576,
    ]

    /// Drop any preset larger than the model's `max_position_embeddings` so
    /// the slider can't pick a value the model would refuse. Auto (0) always
    /// stays. We deliberately use `modelMaxTokens` (the architectural cap from
    /// config.json) — NOT `contextLength` (which is the *running* server's
    /// effective context size and would change with this very setting).
    private var presets: [Int] {
        let modelMax = server.modelInfo?.modelMaxTokens ?? 0
        guard modelMax > 0 else { return Self.allPresets }
        return Self.allPresets.filter { $0 == 0 || $0 <= modelMax }
    }

    private var currentIndex: Int {
        let value = appState.serverOptions.ctxSize
        if let i = presets.firstIndex(of: value) { return i }
        // User has a value that doesn't match a preset (legacy data) — snap
        // visually to the closest non-Auto preset without mutating storage.
        guard value > 0 else { return 0 }
        var best = 1
        for i in 1..<presets.count where abs(presets[i] - value) < abs(presets[best] - value) {
            best = i
        }
        return best
    }

    private static func formatTokens(_ n: Int) -> String {
        if n == 0 { return "Auto" }
        if n >= 1_048_576 { return "\(n / 1_048_576)M" }
        if n >= 1024 { return "\(n / 1024)K" }
        return "\(n)"
    }

    private var isDirty: Bool {
        guard let last = server.lastLaunchedOptions else { return false }
        return appState.serverOptions.ctxSize != last.ctxSize
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                HStack(spacing: 6) {
                    Text("Context size")
                        .font(.body)
                    if isDirty {
                        Image(systemName: "arrow.clockwise.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .help("Restart the server to apply this change")
                    }
                }
                Spacer(minLength: 12)
                HStack(spacing: 8) {
                    Slider(
                        value: Binding(
                            get: { Double(currentIndex) },
                            set: { raw in
                                let i = Int(raw.rounded())
                                let clamped = max(0, min(i, presets.count - 1))
                                appState.serverOptions.ctxSize = presets[clamped]
                            }
                        ),
                        in: 0...Double(max(1, presets.count - 1)),
                        step: 1
                    )
                    .frame(width: 200)
                    Text(Self.formatTokens(appState.serverOptions.ctxSize))
                        .font(.body.monospacedDigit())
                        .frame(minWidth: 56, alignment: .trailing)
                }
            }
            Text("Maximum prompt + completion tokens. \"Auto\" uses the model's declared maximum at load time. Higher values use more memory.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            // Cap info: model max + GPU-safe max for this Mac. Visible only
            // when the server has reported them (after first model load).
            HStack(spacing: 12) {
                if let modelMax = server.modelInfo?.modelMaxTokens, modelMax > 0 {
                    capPill(
                        label: "Model max",
                        value: Self.formatTokens(modelMax),
                        warn: false
                    )
                }
                if let safeMax = server.memoryInfo?.maxSafeContext, safeMax > 0 {
                    let chosen = appState.serverOptions.ctxSize
                    let exceeds = chosen > 0 && chosen > safeMax
                    capPill(
                        label: "GPU-safe max",
                        value: Self.formatTokens(safeMax),
                        warn: exceeds
                    )
                }
            }
        }
    }

    @ViewBuilder
    private func capPill(label: String, value: String, warn: Bool) -> some View {
        let labelColor: Color = warn ? .orange : .secondary
        let valueColor: Color = warn ? .orange : .primary
        HStack(spacing: 4) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(labelColor)
            Text(value)
                .font(.caption2.monospacedDigit().weight(.medium))
                .foregroundStyle(valueColor)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background((warn ? Color.orange : Color.secondary).opacity(0.10))
        .clipShape(Capsule())
    }
}

// MARK: - Spec-decode section

private struct SpecDecodeSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.lastLaunchedOptions)
    }

    /// `draftBlockSize` stays CLI-only — `recommendedBlockSize` in drafter.zig
    /// auto-picks per target (E2B=2, E4B=4, 31B=8, 26B-A4B=4); the field is
    /// kept in ServerOptions so power users who set it via CLI keep working.

    var body: some View {
        let opts = $appState.serverOptions
        // Drafter and PLD are mutually exclusive at the request level
        // (`drafter > PLD > regular` in src/server.zig). When drafter is on
        // we lock the PLD toggles down so users can't accidentally enable a
        // setting that would never apply.
        let drafterActive = !appState.serverOptions.drafterPath.isEmpty
        let pldUsable = appState.serverOptions.enablePLD && !drafterActive

        DrafterRow()
        if let m = meta["enablePLD"] {
            let suffix = drafterActive
                ? " Locked off while Drafter is on (Drafter takes priority)."
                : ""
            SettingsRow(
                title: m.title,
                explainer: m.explainer + suffix,
                isDirty: dirty.dirty(\.enablePLD)
            ) {
                Toggle("", isOn: opts.enablePLD)
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .disabled(drafterActive)
            }
        }
        if let m = meta["pldDraftLen"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.pldDraftLen)
            ) {
                Stepper(value: opts.pldDraftLen, in: 1...16) {
                    Text("\(appState.serverOptions.pldDraftLen)")
                        .font(.body.monospacedDigit())
                }
                .disabled(!pldUsable)
            }
        }
        if let m = meta["pldKeyLen"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.pldKeyLen)
            ) {
                Stepper(value: opts.pldKeyLen, in: 1...8) {
                    Text("\(appState.serverOptions.pldKeyLen)")
                        .font(.body.monospacedDigit())
                }
                .disabled(!pldUsable)
            }
        }
    }
}

// MARK: - Performance section

private struct PerformanceSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.lastLaunchedOptions)
    }

    var body: some View {
        let opts = $appState.serverOptions

        if let m = meta["maxConcurrent"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.maxConcurrent)
            ) {
                Stepper(value: opts.maxConcurrent, in: 1...8) {
                    Text("\(appState.serverOptions.maxConcurrent)")
                        .font(.body.monospacedDigit())
                }
            }
        }
        if let m = meta["kvQuant"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.kvQuant)
            ) {
                Picker("", selection: opts.kvQuant) {
                    ForEach(ServerOptions.KVQuant.allCases) { q in
                        Text(q.label).tag(q)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(minWidth: 220)
            }
        }
        if let m = meta["prefixCacheEntries"] {
            // Surface the RAM clamp so a 16 GB Mac user who sets, say, 8 sees
            // that the launcher will actually pass 1 (and why).
            let ram = ProcessInfo.processInfo.physicalMemory
            let set = appState.serverOptions.prefixCacheEntries
            let effective = ServerOptions.ramCappedPrefixCacheEntries(set, physicalMemoryBytes: ram)
            let capNote = effective < set
                ? "  ·  This Mac (\(MemoryInfo.format(Int64(ram)))) launches with \(effective) to keep cache memory bounded."
                : ""
            SettingsRow(
                title: m.title,
                explainer: m.explainer + capNote,
                isDirty: dirty.dirty(\.prefixCacheEntries)
            ) {
                Stepper(value: opts.prefixCacheEntries, in: 0...16) {
                    Text("\(appState.serverOptions.prefixCacheEntries)")
                        .font(.body.monospacedDigit())
                }
            }
        }
        if let m = meta["prefixCacheMem"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.prefixCacheMem)
            ) {
                TextField("", text: opts.prefixCacheMem, prompt: Text("2GB"))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 110)
            }
        }
    }
}

// MARK: - Common-engine performance section

/// Knobs that apply to every backend (MLX / llama.cpp / ds4). Today this
/// is just the chat-template tokenize cache — the warm-path tokenize
/// stripper that brought a 1813-token Gemma 4 repeat from 240 ms to
/// 0.002 ms. Reorg-friendly: anything we add later that crosses engines
/// (e.g. shared HTTP timeout overrides) lands here.
private struct CommonPerformanceSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.lastLaunchedOptions)
    }

    var body: some View {
        let opts = $appState.serverOptions
        if let m = meta["tokenizeCacheEntries"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.tokenizeCacheEntries)
            ) {
                Stepper(value: opts.tokenizeCacheEntries, in: 0...32) {
                    Text("\(appState.serverOptions.tokenizeCacheEntries)")
                        .font(.body.monospacedDigit())
                }
            }
        }
    }
}

// MARK: - GGUF (llama.cpp) performance section

/// Knobs specific to the embedded llama.cpp engine — surfaced only when
/// the active model loaded through that path (or pre-load, when no
/// engine has been chosen yet). MLX's `--kv-quant` and `--prefix-cache-*`
/// don't apply here; llama.cpp has its own KV scheme and its own
/// multi-session LRU.
private struct LlamaPerformanceSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.lastLaunchedOptions)
    }

    var body: some View {
        let opts = $appState.serverOptions
        if let m = meta["llamaKvQuant"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.llamaKvQuant)
            ) {
                Picker("", selection: opts.llamaKvQuant) {
                    ForEach(ServerOptions.LlamaKVQuant.allCases) { q in
                        Text(q.label).tag(q)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(minWidth: 260)
            }
        }
        if let m = meta["llamaCacheEntries"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.llamaCacheEntries)
            ) {
                Stepper(value: opts.llamaCacheEntries, in: 1...8) {
                    Text("\(appState.serverOptions.llamaCacheEntries)")
                        .font(.body.monospacedDigit())
                }
            }
        }
    }
}

// MARK: - ds4 (DeepSeek-V4-Flash) performance section

/// Knobs specific to the embedded ds4 engine — surfaced only when the active
/// model loaded through that path (DeepSeek-V4-Flash GGUF), or pre-load when
/// no engine has been chosen yet. Today this is just SSD weight streaming:
/// the lever that lets a model larger than RAM load by streaming experts off
/// disk instead of OOMing at warmup (issue #39).
private struct Ds4PerformanceSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.lastLaunchedOptions)
    }

    var body: some View {
        if let m = meta["ssdStreaming"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.ssdStreaming)
            ) {
                Toggle("", isOn: $appState.serverOptions.ssdStreaming)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
    }
}

// MARK: - Drafter row

/// Three-state speculative-decoding toggle for the Gemma 4 assistant drafter.
///
/// State is derived from (loaded model architecture, isMoE, drafter on disk):
///   - **Available, dense Gemma 4** → toggle on/off; status pill shows the
///     auto-discovered checkpoint name in green.
///   - **Available, MoE Gemma 4** → toggle stays usable but flipping on shows
///     a yellow caution pill: drafter regresses on MoE at single-stream
///     batch=1 (verify expert-routing penalty), so PLD is the recommended
///     path. Per-request `enable_drafter:true` still works.
///   - **Unavailable** → disabled toggle, with a one-line explainer naming
///     the reason (non-Gemma-4 target, or no matching drafter on disk). When
///     it's a missing checkpoint, a "Browse" button jumps to the Model
///     Browser.
private struct DrafterRow: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager
    @Environment(\.openWindow) private var openWindow

    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.lastLaunchedOptions)
    }

    /// Drafter the loaded model would pair with — nil for non-Gemma-4 or
    /// when no matching checkpoint is on disk.
    private var recommended: LocalDrafter? {
        guard let info = server.modelInfo else { return nil }
        return downloads.recommendedDrafterFor(
            modelPath: appState.selectedModelPath,
            architecture: info.architecture,
            isMoE: info.isMoE
        )
    }

    /// True when the loaded target is a Gemma 4 model (any size). Tells us
    /// whether to surface "drafter not found" (worth fixing) vs "drafter is
    /// Gemma 4 only" (architectural).
    private var targetIsGemma4: Bool {
        let arch = server.modelInfo?.architecture ?? ""
        return arch == "gemma4" || arch == "gemma4_text"
    }

    private var isMoeTarget: Bool { server.modelInfo?.isMoE ?? false }

    private var explainer: String {
        if let r = recommended {
            return "Pairs with the small assistant drafter for +27–40% on code & agents (dense Gemma 4 only). Auto-discovered: \(r.url.lastPathComponent)."
        }
        // Server hasn't reported a model yet — either it's not started or
        // we're mid-handshake. Don't claim the architecture is wrong.
        if server.modelInfo == nil {
            if appState.selectedModelPath.isEmpty {
                return "Select a model to check drafter compatibility."
            }
            return "Start the server to check drafter compatibility."
        }
        // Server reported a model but didn't include `architecture` in its
        // /v1/models meta — that field landed in the same release that
        // unhid this row, so an older bundled binary will leave it empty.
        if (server.modelInfo?.architecture ?? "").isEmpty {
            return "Drafter status unavailable (server build pre-dates this UI). Use --drafter via CLI."
        }
        if !targetIsGemma4 {
            return "Drafter is Gemma 4 only."
        }
        return "Drafter checkpoint not found. Download from the Model Browser."
    }

    private var toggleEnabled: Bool { recommended != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                HStack(spacing: 6) {
                    Text("Enable Assistant MTP Drafter model")
                        .font(.body)
                    if dirty.dirty(\.drafterPath) {
                        Image(systemName: "arrow.clockwise.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .help("Restart the server to apply this change")
                    }
                }
                Spacer(minLength: 12)
                control
                    .frame(maxWidth: 280, alignment: .trailing)
            }
            Text(explainer)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            // Status pill — green for "ready", yellow for the MoE caution.
            if let r = recommended {
                HStack(spacing: 8) {
                    statusPill(
                        text: "✓ \(r.url.lastPathComponent)",
                        warn: false
                    )
                    if isMoeTarget && !appState.serverOptions.drafterPath.isEmpty {
                        statusPill(
                            text: "⚠ Drafter regresses ~30% on MoE — PLD is recommended",
                            warn: true
                        )
                    }
                }
                .padding(.top, 2)
            } else if server.modelInfo != nil && targetIsGemma4 {
                // Server has a Gemma 4 target loaded but the matching drafter
                // isn't on disk. Jump straight to the Model Browser so the
                // user can pick the right `*-it-assistant-bf16` repo.
                Button("Browse") {
                    openWindow(id: "modelBrowser")
                }
                .controlSize(.small)
                .padding(.top, 2)
            }
        }
    }

    @ViewBuilder
    private var control: some View {
        let isOn = Binding<Bool>(
            get: { !appState.serverOptions.drafterPath.isEmpty },
            set: { newValue in
                if newValue {
                    if let r = recommended {
                        appState.serverOptions.drafterPath = r.url.path
                    }
                } else {
                    appState.serverOptions.drafterPath = ""
                }
            }
        )
        Toggle("", isOn: isOn)
            .labelsHidden()
            .toggleStyle(.switch)
            .disabled(!toggleEnabled)
    }

    @ViewBuilder
    private func statusPill(text: String, warn: Bool) -> some View {
        let fg: Color = warn ? .orange : .green
        Text(text)
            .font(.caption2.monospacedDigit())
            .foregroundStyle(fg)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(fg.opacity(0.10))
            .clipShape(Capsule())
    }
}

// MARK: - Per-request defaults section

private struct RequestDefaultsSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.requestDefaultFields }

    /// Snapping presets for Max Tokens. Position 0 is "Auto" (= 0 sentinel):
    /// the request omits max_tokens and the server pegs generation to the
    /// remaining context window — the right cap on a small-RAM / small-context
    /// machine, where a fixed number would over- or under-shoot. The rest are
    /// powers of 2 from 256 up to 256K.
    private static let maxTokensPresets: [Int] = [
        0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
    ]

    /// Snapping presets for Reasoning Budget. Position 0 is the special
    /// "Unlimited" sentinel (-1); the rest are powers of 2 from 256 up to 32K.
    private static let reasoningPresets: [Int] = [
        -1, 0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    ]

    private static func formatTokens(_ n: Int) -> String {
        if n >= 1_048_576 { return "\(n / 1_048_576)M" }
        if n >= 1024 { return "\(n / 1024)K" }
        return "\(n)"
    }

    var body: some View {
        let opts = $appState.serverOptions

        // Max Tokens — snapping slider
        if let m = meta["defaultMaxTokens"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                snappingSlider(
                    presets: Self.maxTokensPresets,
                    current: appState.serverOptions.defaultMaxTokens,
                    set: { appState.serverOptions.defaultMaxTokens = $0 },
                    label: appState.serverOptions.defaultMaxTokens <= 0
                        ? "Auto"
                        : Self.formatTokens(appState.serverOptions.defaultMaxTokens)
                )
            }
        }
        if let m = meta["defaultTemperature"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                VStack(alignment: .trailing, spacing: 4) {
                    HStack(spacing: 8) {
                        Slider(value: opts.defaultTemperature, in: 0...2, step: 0.05)
                        Text(String(format: "%.2f", appState.serverOptions.defaultTemperature))
                            .font(.body.monospacedDigit())
                            .frame(minWidth: 36, alignment: .trailing)
                    }
                    recPill(server.modelInfo?.recTemperature.map { String(format: "%.2f", $0) })
                }
            }
        }
        if let m = meta["defaultTopP"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                VStack(alignment: .trailing, spacing: 4) {
                    HStack(spacing: 8) {
                        Slider(value: opts.defaultTopP, in: 0.1...1.0, step: 0.01)
                        Text(String(format: "%.2f", appState.serverOptions.defaultTopP))
                            .font(.body.monospacedDigit())
                            .frame(minWidth: 36, alignment: .trailing)
                    }
                    recPill(server.modelInfo?.recTopP.map { String(format: "%.2f", $0) })
                }
            }
        }
        if let m = meta["defaultTopK"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                VStack(alignment: .trailing, spacing: 4) {
                    Stepper(value: opts.defaultTopK, in: 0...1000) {
                        Text(appState.serverOptions.defaultTopK == 0
                             ? "Disabled"
                             : "\(appState.serverOptions.defaultTopK)")
                            .font(.body.monospacedDigit())
                    }
                    // Top-k is the one sampling field that actually falls
                    // through to the model's recommendation: when the slider
                    // reads "Disabled" (0) no `--top-k` flag is sent, so the
                    // model's own value takes effect. Say so when it's live.
                    recPill(
                        server.modelInfo?.recTopK.map { "\($0)" },
                        active: server.modelInfo?.recTopK != nil
                            && appState.serverOptions.defaultTopK == 0
                    )
                }
            }
        }
        if let m = meta["defaultRepeatPenalty"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                HStack(spacing: 8) {
                    Slider(value: opts.defaultRepeatPenalty, in: 1.0...2.0, step: 0.01)
                    Text(String(format: "%.2f", appState.serverOptions.defaultRepeatPenalty))
                        .font(.body.monospacedDigit())
                        .frame(minWidth: 40, alignment: .trailing)
                }
            }
        }
        if let m = meta["defaultPresencePenalty"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                HStack(spacing: 8) {
                    Slider(value: opts.defaultPresencePenalty, in: 0.0...2.0, step: 0.01)
                    Text(String(format: "%.2f", appState.serverOptions.defaultPresencePenalty))
                        .font(.body.monospacedDigit())
                        .frame(minWidth: 40, alignment: .trailing)
                }
            }
        }
        // Reasoning Budget — snapping slider; position 0 is the "Unlimited"
        // sentinel (-1).
        if let m = meta["defaultReasoningBudget"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                snappingSlider(
                    presets: Self.reasoningPresets,
                    current: appState.serverOptions.defaultReasoningBudget,
                    set: { appState.serverOptions.defaultReasoningBudget = $0 },
                    label: appState.serverOptions.defaultReasoningBudget < 0
                        ? "Unlimited"
                        : Self.formatTokens(appState.serverOptions.defaultReasoningBudget)
                )
            }
        }
    }

    /// Small "model recommends" hint pill shown under a sampling slider. The
    /// value comes from the loaded model's `generation_config.json` (surfaced
    /// over `/v1/models`); nil → nothing rendered (no model loaded, or the
    /// model ships no recommendation). `active=true` switches the styling to
    /// green + "(in effect)" for the top-k case, where a Disabled slider
    /// actually lets the model's value win.
    @ViewBuilder
    private func recPill(_ value: String?, active: Bool = false) -> some View {
        if let value {
            let color: Color = active ? .green : .secondary
            HStack(spacing: 4) {
                Text(active ? "Model default (in effect):" : "Model recommends:")
                    .font(.caption2)
                Text(value)
                    .font(.caption2.monospacedDigit().weight(.medium))
            }
            .foregroundStyle(color)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.12))
            .clipShape(Capsule())
        }
    }

    /// Build a snapping slider over a discrete preset list. The slider's float
    /// value is the index into `presets`; rounding pins to the nearest entry.
    /// `label` is the textual readout shown next to the slider.
    @ViewBuilder
    private func snappingSlider(
        presets: [Int],
        current: Int,
        set: @escaping (Int) -> Void,
        label: String
    ) -> some View {
        let safePresets = presets.isEmpty ? [0] : presets
        let currentIdx = Self.closestIndex(in: safePresets, to: current)
        HStack(spacing: 8) {
            Slider(
                value: Binding(
                    get: { Double(currentIdx) },
                    set: { raw in
                        let i = Int(raw.rounded())
                        let clamped = max(0, min(i, safePresets.count - 1))
                        set(safePresets[clamped])
                    }
                ),
                in: 0...Double(max(1, safePresets.count - 1)),
                step: 1
            )
            .frame(width: 200)
            Text(label)
                .font(.body.monospacedDigit())
                .frame(minWidth: 70, alignment: .trailing)
        }
    }

    /// Find the index of the preset closest to `value`, so a stored value not
    /// on the snap grid still positions the slider sensibly.
    private static func closestIndex(in presets: [Int], to value: Int) -> Int {
        if let exact = presets.firstIndex(of: value) { return exact }
        var best = 0
        for i in 1..<presets.count where abs(presets[i] - value) < abs(presets[best] - value) {
            best = i
        }
        return best
    }
}

// MARK: - Voice (clone clip) section

/// The global voice-clone clip: pick an audio file or record a few seconds,
/// normalized via `AudioReference` (24 kHz mono WAV — what Qwen3-TTS
/// `ref_audio` expects) and copied to `~/.mlx-serve/voice-clips/` so it
/// survives relaunch. An app-side setting like the sandbox — binds straight
/// through `appState.serverOptions.voiceClonePath`, no restart banner, no CLI
/// flag. Voice mode's `ClonedVoiceSynthesizer` re-reads the path per sentence.
private struct VoiceCloneSectionContent: View {
    @EnvironmentObject var appState: AppState
    @StateObject private var recorder = AudioRecorder()
    @State private var voiceError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Voice clone clip").font(.subheadline.weight(.semibold))
            HStack(spacing: 8) {
                if !appState.serverOptions.voiceClonePath.isEmpty {
                    Image(systemName: "waveform").foregroundStyle(.secondary)
                    // Prefer the display label — the stored file is always the
                    // normalized "voice-clone.wav", which says nothing.
                    Text(appState.serverOptions.voiceCloneLabel.isEmpty
                         ? (appState.serverOptions.voiceClonePath as NSString).lastPathComponent
                         : appState.serverOptions.voiceCloneLabel)
                        .font(.caption).lineLimit(1).truncationMode(.middle)
                    Button { clearVoice() } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.borderless).foregroundStyle(.secondary)
                        .help("Remove the clip — voice mode falls back to the system voice")
                } else {
                    Text("None — voice mode uses the system voice.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            HStack(spacing: 8) {
                Button { chooseVoiceFile() } label: { Label("Choose file…", systemImage: "folder") }
                if recorder.isRecording {
                    Button(role: .destructive) { stopRecording() } label: {
                        Label(String(format: "Stop (%.1fs)", recorder.duration), systemImage: "stop.circle")
                    }
                } else {
                    Button { startRecording() } label: { Label("Record", systemImage: "mic") }
                }
            }
            .font(.caption)
            Text("A few seconds of clean speech works best. Answers are synthesized locally by the Audio pane's TTS model (downloaded on first use).")
                .font(.caption2).foregroundStyle(.secondary)
            if let voiceError {
                Text(voiceError).font(.caption).foregroundStyle(.red)
            }
        }
    }

    private func chooseVoiceFile() {
        voiceError = nil
        do {
            guard let picked = try VoiceCloneMenuModel.pickAndPersistClip() else { return }
            appState.serverOptions.voiceClonePath = picked.path
            appState.serverOptions.voiceCloneLabel = picked.label
            appState.serverOptions.voiceCloneEnabled = true
        } catch {
            voiceError = error.localizedDescription
        }
    }

    private func startRecording() {
        voiceError = nil
        Task {
            guard await AudioRecorder.requestPermission() else {
                voiceError = "Microphone access denied. Enable it in System Settings ▸ Privacy ▸ Microphone."
                return
            }
            do { try recorder.start() }
            catch { voiceError = error.localizedDescription }
        }
    }

    private func stopRecording() {
        guard let data = recorder.stop() else { voiceError = "Nothing was recorded."; return }
        do {
            let normalized = try AudioReference.normalizedReferenceWav(fromRecordedPCM: data)
            appState.serverOptions.voiceClonePath = VoiceCloneClipStore.persist(normalized)
            appState.serverOptions.voiceCloneLabel = "Recorded clip"
            appState.serverOptions.voiceCloneEnabled = true
        } catch {
            voiceError = error.localizedDescription
        }
    }

    private func clearVoice() {
        appState.serverOptions.voiceClonePath = ""
        appState.serverOptions.voiceCloneLabel = ""
    }
}

// MARK: - Agent sandbox section

/// Toggle + base-image field for the agent execution sandbox. This is an
/// app-side agent-behavior setting (the tool executor reads it), NOT a
/// server-launch flag — so it binds straight through
/// `appState.serverOptions.sandbox` with no restart banner and no CLI flag.
private struct SandboxSectionContent: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        SettingsRow(
            title: "Sandbox agent commands",
            explainer: "OFF = the agent runs shell commands directly on macOS (fast, full access to your files). ON = commands run inside an isolated Linux sandbox that can only touch the current working folder, so a bad command can't harm the rest of your Mac. Costs a bit more memory while active because it spins up a lightweight virtual machine for the session."
        ) {
            Toggle("", isOn: $appState.serverOptions.sandbox.enabled)
                .labelsHidden()
                .toggleStyle(.switch)
        }

        SettingsRow(
            title: "Network + port mapping",
            explainer: "ON = the sandbox has outbound internet (NAT), and any server the agent starts inside it is automatically reachable on this Mac at localhost with the same port — e.g. a dev server on 8080 appears at localhost:8080 (bound to localhost only, never your LAN). OFF = the sandbox gets no network device at all: fully isolated, but installs and downloads inside it will fail. Applies to the next sandbox session."
        ) {
            Toggle("", isOn: $appState.serverOptions.sandbox.network)
                .labelsHidden()
                .toggleStyle(.switch)
        }

        SettingsRow(
            title: "Base image",
            explainer: "The Docker/OCI image the sandbox boots from. Must have an arm64 build (Apple Silicon) — an amd64-only image won't boot. The default ddalcu/agent-shell is a purpose-built agentic shell (Node.js + Python3/pip + git/curl + apt). Pulled once on first use, then cached; a heavier image uses more disk and takes longer the first time."
        ) {
            TextField("", text: $appState.serverOptions.sandbox.baseImage,
                      prompt: Text("ddalcu/agent-shell"))
                .textFieldStyle(.roundedBorder)
                .font(.body.monospaced())
                .frame(width: 260)
        }
    }
}

// MARK: - Messaging (Telegram bot) section

/// Settings for the Telegram bot bridge. The whole thing is two steps for the
/// user: create a bot in @BotFather, paste the token, flip the switch — then
/// message the bot once to lock it to your chat (trust-on-first-use). `@Observed`
/// on the live bridge so the status pill updates as it connects.
private struct MessagingSectionContent: View {
    @EnvironmentObject var appState: AppState
    @ObservedObject var bridge: TelegramBridge

    private var telegram: ServerOptions.TelegramConfig { appState.serverOptions.telegram }

    var body: some View {
        // Live status pill (only meaningful once enabled).
        if telegram.enabled {
            HStack(spacing: 8) {
                Text("Status")
                    .font(.body)
                Spacer(minLength: 12)
                statusPill
            }
        }

        SettingsRow(
            title: "Enable Telegram bot",
            explainer: "Long-polls Telegram for messages and relays them to your local model. Needs a bot token (below) and a running model."
        ) {
            Toggle("", isOn: $appState.serverOptions.telegram.enabled)
                .labelsHidden()
                .toggleStyle(.switch)
        }

        SettingsRow(
            title: "Bot token",
            explainer: "Paste the token @BotFather gives you after /newbot. Stored locally on this Mac and sent only to Telegram's API."
        ) {
            TextField("", text: $appState.serverOptions.telegram.botToken,
                      prompt: Text("123456:ABC-DEF…"))
                .textFieldStyle(.roundedBorder)
                .font(.body.monospaced())
                .frame(width: 260)
        }

        SettingsRow(
            title: "Agent mode (tools)",
            explainer: "OFF = plain chat (safe). ON = the bot can run shell commands and read/write files on this Mac, triggered from your phone. Confined to ~/.mlx-serve/telegram-workspace. Only enable if you understand the risk — anyone who can message the locked chat gets this power."
        ) {
            Toggle("", isOn: $appState.serverOptions.telegram.agentMode)
                .labelsHidden()
                .toggleStyle(.switch)
        }

        SettingsRow(
            title: "MCP tools",
            explainer: "Expose your enabled MCP servers (configured in the MCP marketplace) to the bot and to the tasks it creates. Works with or without Agent mode. Servers start on first use."
        ) {
            Toggle("", isOn: $appState.serverOptions.telegram.useMCP)
                .labelsHidden()
                .toggleStyle(.switch)
        }

        SettingsRow(
            title: "Enable thinking",
            explainer: "Send reasoning-enabled requests for models that support it. The bot replies with the final answer only (no thinking trace)."
        ) {
            Toggle("", isOn: $appState.serverOptions.telegram.enableThinking)
                .labelsHidden()
                .toggleStyle(.switch)
        }

        // Allow-list / lock control.
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text("Locked to")
                    .font(.body)
                Spacer(minLength: 12)
                HStack(spacing: 8) {
                    Text(lockLabel)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(telegram.allowedChatIds.isEmpty ? .secondary : .primary)
                    Button("Reset lock") {
                        appState.serverOptions.telegram.allowedChatIds = []
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(telegram.allowedChatIds.isEmpty)
                }
            }
            Text("The first chat that messages the bot is adopted as the owner; everyone else is refused. Reset to hand the bot to a different chat.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }

        Divider()

        // Setup steps.
        VStack(alignment: .leading, spacing: 6) {
            Text("Setup")
                .font(.caption.weight(.semibold))
            Text("1. In Telegram, open @BotFather and send /newbot.")
                .font(.caption2).foregroundStyle(.secondary)
            Text("2. Copy the token it gives you and paste it above.")
                .font(.caption2).foregroundStyle(.secondary)
            Text("3. Turn on “Enable Telegram bot”, then message your bot once to lock it to your chat.")
                .font(.caption2).foregroundStyle(.secondary)
            Link("Open @BotFather ↗", destination: URL(string: "https://t.me/botfather")!)
                .font(.caption2)
        }
        .fixedSize(horizontal: false, vertical: true)
        .padding(.top, 2)
    }

    private var lockLabel: String {
        let ids = telegram.allowedChatIds
        switch ids.count {
        case 0: return "no chat yet (first to message wins)"
        case 1: return "chat \(ids[0])"
        default: return "\(ids.count) chats"
        }
    }

    @ViewBuilder
    private var statusPill: some View {
        let (text, color): (String, Color) = {
            switch bridge.status {
            case .off:               return ("Off", .secondary)
            case .connecting:        return ("Connecting…", .orange)
            case .listening(let u):  return (u.map { "Listening as @\($0)" } ?? "Listening", .green)
            case .error(let m):      return (m, .red)
            }
        }()
        Text(text)
            .font(.caption2.monospaced())
            .foregroundStyle(color)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.12))
            .clipShape(Capsule())
            .lineLimit(2)
            .frame(maxWidth: 280, alignment: .trailing)
    }
}

// MARK: - Updates section

/// Auto-update controls: the daily-check toggle, a manual check button with
/// inline status, and — once a newer release is known — an install row that
/// mirrors the tray banner's one-click update.
private struct UpdatesSectionContent: View {
    @ObservedObject var updates: UpdateChecker

    var body: some View {
        SettingsRow(
            title: "Check for updates automatically",
            explainer: "Checks the GitHub releases page once a day and shows an update banner in the menu bar tray when a newer version ships. No data beyond the version request leaves this Mac."
        ) {
            Toggle("", isOn: Binding(
                get: { updates.autoCheckEnabled },
                set: { updates.autoCheckEnabled = $0 }))
                .toggleStyle(.switch)
                .labelsHidden()
        }

        SettingsRow(
            title: "Installed version — v\(updates.currentVersion)",
            explainer: statusText
        ) {
            Button {
                Task { await updates.check(userInitiated: true) }
            } label: {
                if case .checking = updates.phase {
                    ProgressView().controlSize(.small)
                } else {
                    Text("Check Now")
                }
            }
            .disabled(busy)
        }

        if let update = updates.available {
            SettingsRow(
                title: "MLX Core v\(update.version) is available",
                explainer: "Downloads MLXCore.dmg from the release, replaces the app, and relaunches."
            ) {
                switch updates.phase {
                case .downloading(let fraction):
                    ProgressView(value: max(0, min(1, fraction)))
                        .progressViewStyle(.linear)
                        .frame(width: 160)
                case .installing:
                    ProgressView().controlSize(.small)
                default:
                    Button("Download & Install") {
                        Task { await updates.downloadAndInstall() }
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        }
    }

    private var busy: Bool {
        switch updates.phase {
        case .checking, .downloading, .installing: return true
        default: return false
        }
    }

    private var statusText: String {
        switch updates.phase {
        case .upToDate:
            return "You're on the latest release."
        case .failed(let message):
            return "Update failed: \(message)"
        case .installing:
            return "Installing — the app will relaunch."
        case .downloading:
            return "Downloading the update…"
        default:
            return "Releases are published at github.com/\(UpdateChecker.repo)/releases."
        }
    }
}
