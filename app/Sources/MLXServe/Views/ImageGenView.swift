import SwiftUI
import AppKit

/// Image generation window — FLUX.2 via the shared Python venv.
///
/// UI layering: a Quality picker drives steps + CFG, a Resolution picker
/// pins to model-trained buckets, and Advanced lets the user override
/// individual fields if they really want to.
struct ImageGenView: View {
    @EnvironmentObject var python: PythonManager
    @EnvironmentObject var service: ImageGenService
    @EnvironmentObject var server: ServerManager

    @State private var prompt: String = ""
    @State private var negative: String = ""
    @State private var showAdvanced: Bool = false
    @State private var model: ImageModelPreset = .flux2Klein4B_Q4
    @State private var quality: QualityPreset = .good
    @State private var resolution: ResolutionOption = ImageModelPreset.flux2Klein4B_Q4.defaultResolution
    @State private var steps: Int = 8
    @State private var guidance: Double = 0.5
    @State private var seed: Int = -1
    @State private var showRAMWarning: Bool = false
    @State private var ramWarningMessage: String = ""
    @State private var pendingRequest: ImageGenRequest? = nil

    var body: some View {
        Group {
            switch python.status {
            case .unknown:
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            case .missingPython, .needsVenv, .needsPackages:
                GenInstallPane(feature: "Image Generation (FLUX.2)")
            case .needsFFmpeg, .ready:
                // mflux doesn't need ffmpeg — image gen works even when only
                // the video pipeline's system deps are missing.
                readyView
            }
        }
        .frame(minWidth: 880, minHeight: 640)
        .onAppear { applyModelDefaults() }
    }

    private var readyView: some View {
        HSplitView {
            VStack(alignment: .leading, spacing: 14) {
                promptSection
                modelSection
                qualitySection
                resolutionSection
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
                .frame(height: 110)
                .overlay(
                    RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5)
                )
        }
    }

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Model").font(.subheadline.weight(.semibold))
            Picker("", selection: $model) {
                ForEach(ImageModelPreset.all) { preset in
                    Text(preset.name).tag(preset)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .onChange(of: model) { _, _ in applyModelDefaults() }
            Text("~\(model.approxRAMGB) GB RAM")
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

    /// Tier-specific marketing-ish hint with the actual numbers, so users see
    /// the cost up front (e.g. "Super = 50 steps, CFG 4.5").
    private var qualityHint: String {
        let s = model.settings(quality)
        return "\(s.steps) steps, CFG \(String(format: "%.1f", s.guidance))"
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
        }
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
                } label: {
                    Image(systemName: "chevron.down")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
            }
            HStack {
                numberField("Steps", value: $steps, step: 1)
                guidanceField
            }
            HStack {
                numberField("Seed (-1 = random)", value: $seed, step: 1)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("Negative prompt").font(.caption)
                TextField("", text: $negative, prompt: Text("(optional, ignored by Klein)"))
                    .textFieldStyle(.roundedBorder)
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

    private var guidanceField: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("CFG / Guidance").font(.caption)
            Stepper(value: $guidance, in: 0...20, step: 0.5) {
                Text(String(format: "%.1f", guidance))
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
                    ContentUnavailableView("No generation yet", systemImage: "photo", description: Text("Enter a prompt and press Generate."))
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
            if let img = NSImage(contentsOfFile: path) {
                Image(nsImage: img)
                    .resizable()
                    .scaledToFit()
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
                        NSWorkspace.shared.open(URL(fileURLWithPath: path))
                    } label: {
                        if let img = NSImage(contentsOfFile: path) {
                            Image(nsImage: img)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 72, height: 72)
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                        } else {
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Color.secondary.opacity(0.2))
                                .frame(width: 72, height: 72)
                        }
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
        applyQualityDefaults()
    }

    private func applyQualityDefaults() {
        let s = model.settings(quality)
        steps = s.steps
        guidance = s.guidance
    }

    /// Soft gate: only block if the model truly can't fit (needs more RAM
    /// than the Mac physically has) — and even then, just warn so the user
    /// can override. Available-RAM was misleading: macOS aggressively pages
    /// out idle apps under unified-memory pressure, so a "5 GB free" reading
    /// rarely means the system can't allocate the working set.
    private func tryGenerate() {
        let req = ImageGenRequest(
            model: model,
            prompt: prompt,
            negativePrompt: negative,
            seed: seed,
            width: resolution.width,
            height: resolution.height,
            steps: steps,
            guidance: guidance
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
        alert.messageText = "Image generation log"
        alert.informativeText = text.isEmpty ? "(no output)" : text
        alert.runModal()
    }
}
