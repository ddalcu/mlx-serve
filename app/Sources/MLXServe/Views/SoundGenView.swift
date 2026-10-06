import SwiftUI
import AppKit

/// Sound Effects tab (Stable Audio 3): a description in, a short WAV out.
/// Built from the shared Create-pane pieces; everything shown is sticky
/// through `SoundGenSettings` (the pane unmounts on navigation).
struct SoundGenView: View {
    @EnvironmentObject var service: SoundGenService
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @Environment(\.openWindow) private var openWindow
    @ObservedObject private var clipPlayer = AudioClipPlayer.shared

    @State private var model: SoundModelPreset = .stableAudio3SmallSFX
    @State private var lanModel: String? = nil
    @State private var prompt: String = ""
    @State private var durationSeconds: Double = 10
    @State private var steps: Int? = nil
    @State private var seed: Int = -1
    @State private var keepResident: Bool = false
    @State private var showAdvanced: Bool = true
    @State private var showRAMWarning = false
    @State private var hydrating = false
    @State private var didHydrate = false

    var body: some View {
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    modelSection
                    promptSection
                    durationSection
                    advancedSection.padding(.top, 8)
                    actionRow.padding(.top, 14)
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minWidth: 340, idealWidth: 380)

            VStack(spacing: 12) {
                previewArea
                AudioHistoryShelf(title: "History", paths: service.recent,
                                  playingPath: clipPlayer.playingPath,
                                  onPlay: { clipPlayer.play($0) }, onStop: { clipPlayer.stop() })
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: MediaStorage.soundRoot)])
                } label: {
                    Label("Open output folder in Finder", systemImage: "folder").font(.app(.caption))
                }
                .buttonStyle(.borderless).foregroundStyle(.secondary).help(MediaStorage.soundRoot)
            }
            .padding(16)
            .frame(minWidth: 280)
        }
        .onAppear {
            if !didHydrate {
                hydrating = true
                hydrate()
                didHydrate = true
                DispatchQueue.main.async { hydrating = false }
            }
            if server.status == .running { Task { await server.refreshModels() } }
        }
        .onDisappear { clipPlayer.stop() }
        .onChange(of: snapshot) { _, s in if !hydrating { s.save() } }
        .onChange(of: service.phase) { _, phase in
            if case .running = phase { clipPlayer.stop() }
            if case .completed(let path) = phase { clipPlayer.play(path) }
        }
        .alert("Model exceeds your Mac's RAM", isPresented: $showRAMWarning) {
            Button(role: .cancel) {} label: { Text("Cancel").font(.app(.body)) }
            Button(role: .destructive) { service.generate(request, server: server) } label: {
                Text("Generate Anyway").font(.app(.body))
            }
        } message: {
            Text(L10n.format("This model needs about %d GB of RAM, but your Mac has %d GB total. It may run very slowly or fail. Continue?",
                             model.approxRAMGB, RAMChecker.totalGB)).font(.app(.body))
        }
    }

    // MARK: - Sections

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            MediaModelChooser.pane(
                all: SoundModelPreset.all,
                onThisMac: CustomMediaModels.soundPresets(from: server.allModels),
                capability: "sound",
                selected: $model, lanModel: $lanModel,
                capabilityOf: { _ in "Sound effects" },
                resolveCustom: { [models = server.allModels] in CustomMediaModels.soundPreset(for: $0, from: models) },
                bundleOf: { $0.bundle },
                downloads: downloads,
                onDownloadFinished: { appState.refreshModels() },
                persist: { snapshot.save() },
                accessory: AnyView(
                    Toggle(isOn: $keepResident) {
                        Text("Keep model loaded after generating").lineLimit(1).truncationMode(.tail)
                    }
                    .font(.app(.caption)).controlSize(.small)
                    .help("On: the model stays resident so the next sound is instant. Off (default): it's unloaded to free GPU memory.")))
            if lanModel == nil && !downloads.bundleReady(model.bundle) {
                BundleDownloadBar(bundle: model.bundle, showsStartButton: false)
            }
        }
    }

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("Describe the sound").font(.app(.headline).weight(.semibold))
                Spacer()
                Menu {
                    ForEach(SoundPrompt.examples, id: \.self) { p in
                        Button { prompt = p } label: { Text(verbatim: p).font(.app(.body)) }
                    }
                } label: {
                    HStack(spacing: 5) {
                        Text("Templates").font(.app(.body))
                        Image(systemName: "chevron.down")
                    }
                    .modifier(PaneChip())
                }
                .modifier(PaneChipMenu())
            }
            TextEditor(text: $prompt)
                .font(.app(.body))
                .frame(height: 80)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5))
            Text("What makes it, the material, the space — e.g. \"heavy wooden door creaking open in a stone hall\".")
                .font(.app(.caption2)).foregroundStyle(.secondary)
        }
    }

    private var durationSection: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Text("Duration").font(.app(.headline).weight(.semibold))
                Text(String(format: "%.1f sec", durationSeconds)).font(.app(.caption)).foregroundStyle(.secondary)
                Spacer()
            }
            Slider(value: $durationSeconds, in: model.durationRange, step: 0.5)
        }
    }

    private var advancedSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            FoldingSectionHeader(title: "Advanced options", isExpanded: $showAdvanced)
            if showAdvanced {
                FlowLayout(spacing: 14, rowSpacing: 10) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Steps").font(.app(.caption))
                        NumberField(range: model.stepsRange,
                                    value: Binding(get: { steps ?? model.defaultSteps }, set: { steps = $0 }),
                                    width: 52,
                                    help: "Sampler steps, \(model.stepsRange.lowerBound)–\(model.stepsRange.upperBound). The model is distilled for \(model.defaultSteps).")
                    }
                    SeedField(label: "Seed", placeholder: "Random", range: -1...Int(UInt32.max), value: $seed)
                }
                Text("Same seed + prompt + length reproduces the sound.")
                    .font(.app(.caption2)).foregroundStyle(.secondary)
            }
        }
    }

    private var actionRow: some View {
        HStack {
            if service.isRunning {
                Button(role: .destructive) { service.cancel() } label: {
                    Label("Cancel", systemImage: "stop.circle").font(.app(.body)).frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            } else {
                Button { generate() } label: {
                    Label("Generate", systemImage: "speaker.wave.3").font(.app(.body)).frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.return, modifiers: [.command])
                .disabled(prompt.isBlank || (lanModel == nil && !downloads.bundleReady(model.bundle)))
            }
        }
    }

    private var previewArea: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.15))
            switch service.phase {
            case .idle:
                ContentUnavailableView("No sounds yet", systemImage: "speaker.wave.3",
                                       description: Text("Describe a sound and press Generate.").font(.app(.body)))
            case .running(let step, let total, let message):
                VStack(spacing: 12) {
                    if total == 0 {
                        ProgressView().frame(width: 240)
                    } else {
                        ProgressView(value: Double(step), total: max(1, Double(total))).progressViewStyle(.linear).frame(width: 240)
                    }
                    Text(message).font(.app(.footnote)).foregroundStyle(.secondary)
                }
            case .completed(let path):
                let playing = clipPlayer.playingPath == path
                VStack(spacing: 12) {
                    Image(systemName: "speaker.wave.3.fill")
                        .font(.app(.largeTitle)).foregroundStyle(.tint)
                        .symbolEffect(.variableColor.iterative, options: .repeat(.continuous), isActive: playing)
                    Button { playing ? clipPlayer.stop() : clipPlayer.play(path) } label: {
                        Label(playing ? "Stop" : "Play", systemImage: playing ? "stop.fill" : "play.fill").font(.app(.body))
                    }
                    .buttonStyle(.bordered)
                    HStack(spacing: 8) {
                        Text(URL(fileURLWithPath: path).lastPathComponent)
                            .font(.app(.caption)).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                        Button { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)]) } label: {
                            Image(systemName: "folder")
                        }
                        .buttonStyle(.borderless).help("Reveal in Finder")
                    }
                }
                .padding(16)
            case .failed(let msg):
                ContentUnavailableView {
                    Label("Failed", systemImage: "exclamationmark.triangle").font(.app(.body))
                } description: {
                    Text(msg)
                } actions: {
                    Button { AppActivation.openWindow(id: "serverLog", using: openWindow) } label: {
                        Text("Show log").font(.app(.body))
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - State

    private var request: SoundGenRequest {
        SoundGenRequest(model: model, prompt: prompt, durationSeconds: durationSeconds,
                        steps: steps, seed: seed, keepResident: keepResident, lanModelId: lanModel)
    }

    private func generate() {
        if RAMChecker.totalGB < model.approxRAMGB {
            showRAMWarning = true
            return
        }
        service.generate(request, server: server)
    }

    private var snapshot: SoundGenSettings {
        var s = SoundGenSettings()
        s.modelId = LanPick.persisted(lanModel: lanModel, presetId: model.id)
        s.prompt = prompt
        s.durationSeconds = durationSeconds
        s.steps = steps
        s.seed = seed
        s.keepResident = keepResident
        s.showAdvanced = showAdvanced
        return s
    }

    private func hydrate() {
        let s = SoundGenSettings.load()
        model = s.resolvedModel(models: server.allModels)
        lanModel = LanPick.lanId(s.modelId)
        prompt = s.prompt
        durationSeconds = min(max(s.durationSeconds, model.durationRange.lowerBound), model.durationRange.upperBound)
        steps = s.steps
        seed = s.seed
        keepResident = s.keepResident
        showAdvanced = s.showAdvanced
    }
}

/// Starter descriptions, from Stability's own demo prompts for this model.
enum SoundPrompt {
    static let examples = [
        "Futuristic laser blast, sharp energy pulse, stereo movement, arcade style",
        "Dog barking next to a waterfall",
        "Sparkling fantasy energy swirl, mystical shimmer, rising magical burst",
        "Running footsteps on pavement, fast pace, urban street environment, energetic motion sound",
        "Chugging train coming into station with horn",
    ]
}
