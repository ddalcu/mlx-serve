import SwiftUI
import AppKit
import AVKit

/// Video generation window — LTX-Video 2 via the shared Python venv.
/// Uses the same Quality / Resolution preset shape as ImageGen, plus a
/// Frames dropdown clamped to LTX's `8N+1` ladder and the user's RAM
/// budget.
struct VideoGenView: View {
    @EnvironmentObject var python: PythonManager
    @EnvironmentObject var service: VideoGenService
    @EnvironmentObject var server: ServerManager

    @State private var prompt: String = ""
    @State private var showAdvanced: Bool = false
    @State private var model: VideoModelPreset = .ltx23Q4
    @State private var quality: QualityPreset = .good
    @State private var resolution: ResolutionOption = VideoModelPreset.ltx23Q4.defaultResolution
    @State private var numFrames: Int = 97
    @State private var fps: Int = 24
    @State private var mode: VideoPipelineMode = .oneStage
    @State private var steps: Int = 12
    @State private var cfgScale: Double = 1.0
    @State private var seed: Int = 42
    @State private var showRAMWarning: Bool = false
    @State private var ramWarningMessage: String = ""
    @State private var pendingRequest: VideoGenRequest? = nil
    @State private var player: AVPlayer?

    var body: some View {
        Group {
            switch python.status {
            case .unknown:
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            case .missingPython, .needsVenv, .needsPackages, .needsFFmpeg:
                GenInstallPane(feature: "Video Generation (LTX-Video 2.3)")
            case .ready:
                readyView
            }
        }
        .frame(minWidth: 880, minHeight: 660)
        .onAppear { applyModelDefaults() }
        .onChange(of: service.phase) { _, phase in
            if case .completed(let path) = phase {
                player = AVPlayer(url: URL(fileURLWithPath: path))
                player?.play()
            }
        }
    }

    private var readyView: some View {
        HSplitView {
            VStack(alignment: .leading, spacing: 14) {
                promptSection
                modelSection
                qualitySection
                resolutionSection
                framesSection
                if showAdvanced { advancedSection } else { advancedToggle }
                Spacer()
                actionRow
            }
            .padding(16)
            .frame(minWidth: 340, idealWidth: 380)

            VStack(spacing: 12) {
                previewArea
                historyShelf
            }
            .padding(16)
            .frame(minWidth: 460)
        }
        .alert("Model exceeds your Mac's RAM", isPresented: $showRAMWarning) {
            Button("Cancel", role: .cancel) { pendingRequest = nil }
            Button("Generate Anyway", role: .destructive) {
                if let req = pendingRequest { service.generate(req) }
                pendingRequest = nil
            }
        } message: {
            Text(ramWarningMessage)
        }
    }

    // MARK: - Sections

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Prompt").font(.subheadline.weight(.semibold))
            TextEditor(text: $prompt)
                .font(.body)
                .frame(height: 90)
                .overlay(
                    RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5)
                )
        }
    }

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Model").font(.subheadline.weight(.semibold))
            Picker("", selection: $model) {
                ForEach(VideoModelPreset.all) { preset in
                    Text(preset.name).tag(preset)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .onChange(of: model) { _, _ in applyModelDefaults() }
            Text("~\(model.approxRAMGB) GB RAM • Includes audio")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var qualitySection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Quality").font(.subheadline.weight(.semibold))
            Picker("", selection: $quality) {
                ForEach(QualityPreset.allCases) { q in
                    Text(q.label).tag(q)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .onChange(of: quality) { _, _ in applyQualityDefaults() }
            Text(qualityHint)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var qualityHint: String {
        let s = model.settings(quality)
        let durationSec = Double(s.numFrames) / Double(model.fps)
        return "\(modeLabel(s.mode)), \(s.steps) steps, \(s.numFrames) frames (~\(String(format: "%.1f", durationSec))s)"
    }

    private func modeLabel(_ m: VideoPipelineMode) -> String {
        switch m {
        case .oneStage:   return "1-stage"
        case .twoStage:   return "2-stage"
        case .twoStageHQ: return "2-stage HQ"
        }
    }

    private var resolutionSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Resolution").font(.subheadline.weight(.semibold))
            Picker("", selection: $resolution) {
                ForEach(model.resolutions) { r in
                    Text(r.label).tag(r)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .onChange(of: resolution) { _, _ in clampFramesToRAM() }
        }
    }

    private var framesSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Frames").font(.subheadline.weight(.semibold))
                Spacer()
                Text("~\(String(format: "%.1f", Double(numFrames) / Double(fps)))s")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Picker("", selection: $numFrames) {
                ForEach(availableFrameOptions, id: \.self) { n in
                    Text("\(n) frames").tag(n)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            if let warn = frameRAMWarning {
                Text(warn).font(.caption2).foregroundStyle(.orange)
            }
        }
    }

    /// Always show every option up to the model's hard cap. The user can
    /// pick longer than RAM suggests — we just hint at it in the warning
    /// below the dropdown rather than removing the option.
    private var availableFrameOptions: [Int] {
        model.frameOptions
    }

    /// Soft hint when the chosen length looks too aggressive for the Mac's
    /// total RAM at the current resolution. Doesn't block — the user might
    /// know better (e.g. they just freed memory).
    private var frameRAMWarning: String? {
        let cap = RAMChecker.safeFrameCap(
            model: model,
            width: resolution.width,
            height: resolution.height,
            available: RAMChecker.totalGB
        )
        if numFrames > cap {
            return "May exceed your Mac's RAM (\(RAMChecker.totalGB) GB total) at this length."
        }
        return nil
    }

    private var advancedToggle: some View {
        Button {
            withAnimation { showAdvanced = true }
        } label: {
            Label("Advanced options", systemImage: "chevron.right")
                .font(.caption)
        }
        .buttonStyle(.plain)
        .foregroundStyle(.secondary)
    }

    private var advancedSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Advanced (overrides Quality preset)").font(.caption.weight(.semibold))
                Spacer()
                Button {
                    withAnimation { showAdvanced = false }
                } label: { Image(systemName: "chevron.down") }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
            }
            HStack {
                numberField("Steps", value: $steps, step: 1)
                VStack(alignment: .leading, spacing: 2) {
                    Text("CFG scale").font(.caption)
                    Stepper(value: $cfgScale, in: 0...20, step: 0.5) {
                        Text(String(format: "%.1f", cfgScale))
                    }
                }
            }
            HStack {
                numberField("Seed", value: $seed, step: 1)
                Spacer()
            }
        }
    }

    private func numberField(_ label: String, value: Binding<Int>, step: Int) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption)
            Stepper(value: value, step: step) {
                Text(String(value.wrappedValue))
            }
        }
    }

    private var actionRow: some View {
        HStack {
            if service.isRunning {
                Button(role: .destructive) {
                    service.cancel()
                } label: {
                    Label("Cancel", systemImage: "stop.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            } else {
                Button {
                    tryGenerate()
                } label: {
                    Label("Generate", systemImage: "wand.and.stars")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.return, modifiers: [.command])
                .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
    }

    private var previewArea: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.black.opacity(0.15))
            Group {
                switch service.phase {
                case .idle:
                    ContentUnavailableView("No generation yet", systemImage: "film", description: Text("Enter a prompt and press Generate."))
                case .running(let step, let total, let message):
                    VStack(spacing: 12) {
                        ProgressView(value: Double(step), total: max(1, Double(total)))
                            .progressViewStyle(.linear)
                            .frame(width: 240)
                        Text(message).font(.footnote).foregroundStyle(.secondary)
                    }
                case .completed(let path):
                    completedPreview(path: path)
                case .failed(let msg):
                    ContentUnavailableView {
                        Label("Failed", systemImage: "exclamationmark.triangle")
                    } description: {
                        Text(msg)
                    } actions: {
                        Button("Show log") { showLogWindow() }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func completedPreview(path: String) -> some View {
        VStack(spacing: 8) {
            if let player {
                VideoPlayer(player: player)
                    .frame(minHeight: 240)
            }
            HStack(spacing: 8) {
                Text(URL(fileURLWithPath: path).lastPathComponent)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                } label: { Image(systemName: "folder") }
                .buttonStyle(.borderless)
                .help("Reveal in Finder")
            }
        }
        .padding(8)
    }

    private var historyShelf: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(service.recent, id: \.self) { path in
                    Button {
                        player = AVPlayer(url: URL(fileURLWithPath: path))
                        player?.play()
                    } label: {
                        RoundedRectangle(cornerRadius: 6)
                            .fill(Color.secondary.opacity(0.2))
                            .frame(width: 96, height: 72)
                            .overlay(
                                Image(systemName: "play.rectangle.fill")
                                    .font(.title2)
                                    .foregroundStyle(.secondary)
                            )
                    }
                    .buttonStyle(.plain)
                    .help(URL(fileURLWithPath: path).lastPathComponent)
                }
            }
            .padding(.horizontal, 4)
        }
        .frame(height: 80)
    }

    // MARK: - Actions

    private func applyModelDefaults() {
        quality = model.defaultQuality
        resolution = model.defaultResolution
        fps = model.fps
        applyQualityDefaults()
    }

    private func applyQualityDefaults() {
        let s = model.settings(quality)
        mode = s.mode
        steps = s.steps
        cfgScale = s.cfgScale
        numFrames = s.numFrames
        clampFramesToRAM()
    }

    /// Resolution change still snaps frame count down to the model's hard
    /// cap (`8N+1` ladder) — but no RAM-based clamping anymore. The user
    /// gets a soft warning instead.
    private func clampFramesToRAM() {
        if numFrames > model.maxFrames,
           let snap = model.frameOptions.last(where: { $0 <= model.maxFrames }) {
            numFrames = snap
        }
    }

    /// Soft gate: only warn when the model needs more RAM than the Mac has
    /// total. macOS's "available" reading is misleading on unified memory
    /// (idle apps get paged out under pressure) — using it as a hard gate
    /// blocked legitimate runs, so we let the user override.
    private func tryGenerate() {
        let req = VideoGenRequest(
            model: model,
            prompt: prompt,
            seed: seed,
            width: resolution.width,
            height: resolution.height,
            numFrames: numFrames,
            fps: fps,
            mode: mode,
            steps: steps,
            cfgScale: cfgScale
        )

        let total = RAMChecker.totalGB
        let needed = model.approxRAMGB
        if total < needed {
            ramWarningMessage = "This model needs about \(needed) GB of RAM, but your Mac has \(total) GB total. It may run very slowly or fail. Continue?"
            pendingRequest = req
            showRAMWarning = true
            return
        }

        service.generate(req)
    }

    private func showLogWindow() {
        let text = service.log.joined(separator: "\n")
        let alert = NSAlert()
        alert.messageText = "Video generation log"
        alert.informativeText = text.isEmpty ? "(no output)" : text
        alert.runModal()
    }
}

// MARK: - Shared install pane

/// Shown by both ImageGen and VideoGen when the shared venv / packages are
/// missing. One-click install button kicks `PythonManager.install()` and
/// tails the log below.
struct GenInstallPane: View {
    let feature: String
    @EnvironmentObject var python: PythonManager

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "shippingbox")
                .font(.system(size: 56))
                .foregroundStyle(.secondary)
            Text(feature).font(.title2.weight(.semibold))
            Text(reasonText)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .frame(maxWidth: 560)

            switch python.status {
            case .missingPython:
                VStack(alignment: .leading, spacing: 6) {
                    Text("Install Python 3 first:").font(.callout.weight(.semibold))
                    Text("brew install python")
                        .font(.system(.callout, design: .monospaced))
                        .padding(.horizontal, 10).padding(.vertical, 4)
                        .background(Color.secondary.opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
                    Text("Then quit and reopen MLX Core.").font(.caption).foregroundStyle(.secondary)
                }
            case .needsFFmpeg:
                VStack(alignment: .leading, spacing: 6) {
                    Text("Install ffmpeg:").font(.callout.weight(.semibold))
                    Text("brew install ffmpeg")
                        .font(.system(.callout, design: .monospaced))
                        .padding(.horizontal, 10).padding(.vertical, 4)
                        .background(Color.secondary.opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
                    Text("Then reopen this pane — ltx-2-mlx shells out to ffmpeg for audio/video muxing.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            default:
                Button {
                    Task { await python.install() }
                } label: {
                    if python.isInstalling {
                        ProgressView().controlSize(.small)
                        Text("Installing...").padding(.leading, 8)
                    } else {
                        Label("Install (≈3 GB, ~3 minutes)", systemImage: "arrow.down.circle")
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(python.isInstalling)
            }

            if let err = python.lastError {
                Text(err).font(.caption).foregroundStyle(.red).multilineTextAlignment(.center)
            }

            if !python.installLog.isEmpty {
                logTail
            }

            Spacer()
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var reasonText: String {
        switch python.status {
        case .missingPython:
            return "Python 3 isn't installed on this Mac. Image and video generation need Python to host the mflux / ltx-2-mlx pipelines."
        case .needsVenv:
            return "A dedicated Python environment at ~/.mlx-serve/venv needs to be created. Click Install to set it up — no effect on your system Python."
        case .needsPackages:
            return "The Python environment exists but is missing required packages. Click Install to finish the setup. ~50 GB of model weights download on first generation."
        case .needsFFmpeg:
            return "Python packages are installed, but the system ffmpeg binary is missing. ltx-2-mlx muxes audio into the mp4 via system ffmpeg."
        default:
            return ""
        }
    }

    private var logTail: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(python.installLog.enumerated()), id: \.offset) { idx, line in
                        Text(line)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .id(idx)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(8)
            }
            .frame(maxWidth: 640, maxHeight: 200)
            .background(Color.black.opacity(0.15), in: RoundedRectangle(cornerRadius: 6))
            .onChange(of: python.installLog.count) { _, n in
                if n > 0 {
                    withAnimation { proxy.scrollTo(n - 1, anchor: .bottom) }
                }
            }
        }
    }
}
