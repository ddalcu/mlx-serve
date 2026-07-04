import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// The Avatar window (plan §7): a generated 3D model of a person talks back
/// using the local LLM + a cloned voice. A persona picker chooses the system
/// prompt, the clone voice clip, and the mesh; the big mic button (or the text
/// field) drives one hands-free turn, and the answer is spoken sentence-by-
/// sentence while it decodes. The SceneKit viewer and cloned-TTS pipeline are
/// reused from the 3D and Audio panes — this window is wiring, not new models.
struct AvatarView: View {
    @State private var emote: EmoteTrigger? = nil
    @EnvironmentObject var engine: AvatarEngine
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var model3d: Model3DGenService

    @State private var store = AvatarPersonaStore.load()
    @State private var typed = ""
    @State private var editing: AvatarPersona? = nil

    var body: some View {
        VStack(spacing: 0) {
            personaBar
            Divider()
            stage
            Divider()
            controls
        }
        .frame(minWidth: 520, minHeight: 640)
        .onAppear { engine.persona = store.selectedPersona }
        .sheet(item: $editing) { persona in
            AvatarPersonaEditor(persona: persona, recent: model3d.recent) { saved in
                store = store.upserting(saved)
                store.save()
                engine.persona = store.selectedPersona
            }
        }
    }

    // MARK: - Persona bar

    private var personaBar: some View {
        HStack(spacing: 10) {
            Menu {
                ForEach(store.personas) { p in
                    Button {
                        store.selectedId = p.id
                        store.save()
                        engine.persona = p
                    } label: {
                        Label(p.name, systemImage: p.id == store.selectedId ? "checkmark" : "person.crop.circle")
                    }
                }
                Divider()
                Button("New Persona…") {
                    editing = AvatarPersona(name: "New Persona")
                }
            } label: {
                Label(engine.persona.name, systemImage: "person.crop.circle")
                    .font(.headline)
            }
            .menuStyle(.borderlessButton)
            .fixedSize()

            if engine.persona.docFolderPath != nil {
                Label("Knowledge", systemImage: "books.vertical")
                    .labelStyle(.iconOnly)
                    .foregroundStyle(.secondary)
                    .help("This persona grounds answers in an attached document folder")
            }

            Spacer()

            Text(stateLabel)
                .font(.caption)
                .foregroundStyle(.secondary)

            Button {
                editing = engine.persona
            } label: {
                Label("Edit", systemImage: "slider.horizontal.3")
            }
            .help("Edit this persona's prompt, voice, and 3D model")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    // MARK: - Stage (3D model + subtitle)

    private var stage: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 0).fill(Color.black.opacity(0.12))
            if let path = engine.persona.glbPath, FileManager.default.fileExists(atPath: path) {
                Model3DSceneView(url: URL(fileURLWithPath: path), turntable: true,
                                 jawAmplitude: engine.speechAmplitude, emote: emote)
                    .overlay(alignment: .topTrailing) { emoteButtons }
            } else {
                ContentUnavailableView {
                    Label("No avatar model", systemImage: "cube.transparent")
                } description: {
                    Text("Pick a 3D model in this persona's settings — generate one from a photo in the 3D window first.")
                } actions: {
                    Button("Edit persona") { editing = engine.persona }
                }
            }

            // Streaming subtitle, pinned to the bottom.
            VStack {
                Spacer()
                if !engine.subtitle.isEmpty {
                    Text(engine.subtitle)
                        .font(.title3.weight(.medium))
                        .multilineTextAlignment(.center)
                        .foregroundStyle(.white)
                        .padding(.horizontal, 16).padding(.vertical, 10)
                        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 10))
                        .padding(.horizontal, 24)
                        .padding(.bottom, 16)
                        .transition(.opacity)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .animation(.easeInOut(duration: 0.15), value: engine.subtitle)
    }

    // MARK: - Controls (mic + text)

    private var controls: some View {
        VStack(spacing: 10) {
            if server.status != .running {
                Text("Start the server (menu-bar tray) with a chat model to talk to your avatar.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack(spacing: 12) {
                micButton
                TextField("Type a message…", text: $typed, onCommit: sendTyped)
                    .textFieldStyle(.roundedBorder)
                    .disabled(!engine.canSubmit || server.status != .running)
                Button(action: sendTyped) {
                    Image(systemName: "paperplane.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(typed.trimmingCharacters(in: .whitespaces).isEmpty || !engine.canSubmit || server.status != .running)
                .keyboardShortcut(.return, modifiers: [])
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var micButton: some View {
        Button {
            if engine.micActive {
                engine.stopListening()
            } else {
                Task { await engine.requestMicThenListen() }
            }
        } label: {
            Image(systemName: engine.micActive ? "stop.circle.fill" : "mic.circle.fill")
                .font(.system(size: 40))
                .foregroundStyle(engine.micActive ? Color.red : Color.accentColor)
        }
        .buttonStyle(.plain)
        .disabled(server.status != .running)
        .help(engine.micActive ? "Stop listening" : "Talk to your avatar")
    }

    private func sendTyped() {
        let text = typed.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        typed = ""
        engine.submit(text)
    }

    private var stateLabel: String {
        switch engine.state {
        case .idle: return "Ready"
        case .listening: return "Listening…"
        case .thinking: return "Thinking…"
        case .speaking: return "Speaking…"
        }
    }
}

// MARK: - Persona editor

/// Edit one persona: name, spoken system prompt, clone-voice clip (pick or
/// record, reusing the Audio pane's `AudioReference` normalization), and the 3D
/// model (pick from recent generations or a file). `onSave` upserts + persists.
private struct AvatarPersonaEditor: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var recorder = AudioRecorder()

    @State var persona: AvatarPersona
    let recent: [String]
    let onSave: (AvatarPersona) -> Void

    @State private var voiceError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Persona").font(.headline)

            VStack(alignment: .leading, spacing: 4) {
                Text("Name").font(.subheadline.weight(.semibold))
                TextField("Name", text: $persona.name).textFieldStyle(.roundedBorder)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("System prompt").font(.subheadline.weight(.semibold))
                TextEditor(text: $persona.systemPrompt)
                    .font(.body)
                    .frame(minHeight: 120)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
                Text("What the avatar is. It's read aloud, so keep answers short and speech-friendly.")
                    .font(.caption2).foregroundStyle(.secondary)
            }

            voiceSection
            modelSection
            documentsSection

            if let voiceError {
                Text(voiceError).font(.caption).foregroundStyle(.red)
            }

            Spacer()
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Save") { onSave(persona); dismiss() }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: [.command])
            }
        }
        .padding(18)
        .frame(width: 460, height: 540)
    }

    // MARK: Voice clone clip

    private var voiceSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Voice clone clip").font(.subheadline.weight(.semibold))
            HStack(spacing: 8) {
                if let path = persona.voiceClipPath {
                    Image(systemName: "waveform").foregroundStyle(.secondary)
                    Text((path as NSString).lastPathComponent)
                        .font(.caption).lineLimit(1).truncationMode(.middle)
                    Button { clearVoice() } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.borderless).foregroundStyle(.secondary)
                } else {
                    Text("None — the avatar uses the model's default voice.")
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
        }
    }

    // MARK: 3D model

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("3D model").font(.subheadline.weight(.semibold))
            HStack(spacing: 8) {
                if let path = persona.glbPath {
                    Image(systemName: "cube.transparent").foregroundStyle(.secondary)
                    Text((path as NSString).lastPathComponent)
                        .font(.caption).lineLimit(1).truncationMode(.middle)
                    Button { persona.glbPath = nil } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.borderless).foregroundStyle(.secondary)
                } else {
                    Text("None — a placeholder shows until you pick a mesh.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            HStack(spacing: 8) {
                if !recent.isEmpty {
                    Menu {
                        ForEach(recent, id: \.self) { path in
                            Button((path as NSString).lastPathComponent) { persona.glbPath = path }
                        }
                    } label: { Label("Recent generations", systemImage: "clock") }
                    .fixedSize()
                }
                Button { chooseModelFile() } label: { Label("Choose file…", systemImage: "folder") }
            }
            .font(.caption)
        }
    }

    // MARK: Knowledge (mini-RAG)

    private var documentsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Knowledge folder").font(.subheadline.weight(.semibold))
            HStack(spacing: 8) {
                if let path = persona.docFolderPath {
                    Image(systemName: "folder").foregroundStyle(.secondary)
                    Text((path as NSString).lastPathComponent)
                        .font(.caption).lineLimit(1).truncationMode(.middle)
                    Button { persona.docFolderPath = nil } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.borderless).foregroundStyle(.secondary)
                } else {
                    Text("None — the avatar answers from the model's own knowledge.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Button { chooseDocumentFolder() } label: { Label("Attach folder…", systemImage: "folder.badge.plus") }
                .font(.caption)
            Text("Documents (txt, md, pdf, json…) the avatar retrieves from to ground its answers.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func chooseDocumentFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false; panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url { persona.docFolderPath = url.path }
    }

    // MARK: Actions

    private func chooseVoiceFile() {
        voiceError = nil
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio, .wav, .mp3, .mpeg4Audio, .aiff]
        panel.canChooseFiles = true; panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let normalized = try AudioReference.normalizedReferenceWav(fromFile: url)
            persona.voiceClipPath = persistVoiceClip(normalized)
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
            persona.voiceClipPath = persistVoiceClip(normalized)
        } catch {
            voiceError = error.localizedDescription
        }
    }

    private func clearVoice() { persona.voiceClipPath = nil }

    private func chooseModelFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "glb") ?? .data]
        panel.canChooseFiles = true; panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url { persona.glbPath = url.path }
    }

    /// Copy the normalized clip out of the OS temp dir (which gets swept) into a
    /// stable per-persona location so a saved persona's voice survives relaunch.
    private func persistVoiceClip(_ url: URL) -> String {
        let dir = NSString(string: "~/.mlx-serve/avatar-voices").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let dest = (dir as NSString).appendingPathComponent("\(persona.id.uuidString).wav")
        try? FileManager.default.removeItem(atPath: dest)
        do {
            try FileManager.default.copyItem(at: url, to: URL(fileURLWithPath: dest))
            return dest
        } catch {
            return url.path   // fall back to the temp path rather than losing the clip
        }
    }
}


extension AvatarView {
    /// P3.3 emote triggers — tiny one-shot clips layered over the skeletal idle.
    var emoteButtons: some View {
        HStack(spacing: 6) {
            Button { emote = EmoteTrigger(kind: .nod, startedAt: CACurrentMediaTime()) } label: {
                Image(systemName: "checkmark.bubble")
            }
            .help("Nod")
            Button { emote = EmoteTrigger(kind: .sway, startedAt: CACurrentMediaTime()) } label: {
                Image(systemName: "hands.and.sparkles")
            }
            .help("Wave hello")
        }
        .buttonStyle(.borderless)
        .padding(8)
    }
}
