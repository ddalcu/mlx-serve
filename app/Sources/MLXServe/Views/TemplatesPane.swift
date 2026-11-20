import AVKit
import SwiftUI

/// Create ▸ Quick Create: a gallery of one-tap videos, pictures and songs, and the page that makes one.
struct TemplatesPane: View {
    @ObservedObject var run: TemplateRunner
    @ObservedObject var cast: CastStore

    var body: some View {
        if let t = run.template, !run.onGallery {
            TemplatePage(template: t, run: run, cast: cast)
        } else {
            TemplateGallery(run: run, cast: cast)
        }
    }
}

// MARK: - Readiness

private extension URL {
    /// A picture template's result; everything else plays as a video.
    var isPicture: Bool { ["png", "jpg", "jpeg"].contains(pathExtension.lowercased()) }
}

@MainActor
private enum TemplateCheck {
    static func sampleURL(_ t: MediaTemplate) -> URL? {
        Bundle.main.url(forResource: t.sampleName, withExtension: "mp4")
            ?? Bundle.main.url(forResource: t.sampleName, withExtension: "jpg")
    }

    /// How long it takes here, where that is known: "10 s · 3:40 on this Mac", "45 s song".
    static func readyNote(_ t: MediaTemplate, seconds: Double?) -> String {
        switch t.kind {
        case .video:
            L10n.format("%lld s · %@ on this Mac", t.seconds, seconds.map(H3TimeEstimate.duration) ?? "")
        case .image:
            L10n.text("A few seconds on this Mac")
        case .audio:
            L10n.format("%lld s song · a few seconds on this Mac", t.seconds)
        }
    }

    static func sampleURL(_ t: MediaTemplate?, choice: String) -> URL? {
        t.flatMap { Bundle.main.url(forResource: $0.sampleName(choice: choice), withExtension: "mp4") }
    }
}

private struct ReadinessBadge: View {
    let template: MediaTemplate
    let readiness: TemplateReadiness

    var body: some View {
        switch readiness {
        case .ready(let seconds):
            Label(TemplateCheck.readyNote(template, seconds: seconds), systemImage: "clock")
                .font(.app(.annotation)).foregroundStyle(.secondary)
        case .onPeer(let id):
            Label(template.seconds > 0 ? L10n.format("%lld s · made on %@", template.seconds, LanPick.peer(of: id))
                                       : L10n.format("Made on %@", LanPick.peer(of: id)), systemImage: "network")
                .font(.app(.annotation)).foregroundStyle(.secondary)
        case .needsDownload:
            Label(L10n.format("Needs a one-time %@ download", template.bundle.approxSizeLabel),
                  systemImage: "arrow.down.circle")
                .font(.app(.annotation)).foregroundStyle(.secondary)
        case .tooBig(let gb):
            Label(L10n.format("Needs a Mac with about %lld GB for its model", gb), systemImage: "memorychip")
                .font(.app(.annotation)).foregroundStyle(.orange)
        }
    }
}

// MARK: - Gallery

private struct TemplateGallery: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var server: ServerManager
    @ObservedObject var run: TemplateRunner
    @ObservedObject var cast: CastStore
    @State private var editing: CastEditTarget?

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            catalogue
            Divider()
            HistoryColumn(history: run.history, run: run, cast: cast)
                .frame(width: 300)
        }
        .sheet(item: $editing) { target in
            CastEditorSheet(cast: cast, target: target) { _ in editing = nil }
        }
    }

    private var catalogue: some View {
        let readiness = { (t: MediaTemplate) in TemplateRunner.readiness(t, shape: t.defaultShape, app: appState) }
        return ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Quick Create").font(.app(.pageTitle))
                    Text("Pick one, add a photo, and get a video, a picture or a song.")
                        .font(.app(.explainer)).foregroundStyle(.secondary)
                }
                if let t = run.template, run.isBusy || run.phase == .done {
                    ContinueBanner(template: t, run: run)
                }
                CastStrip(cast: cast, onEdit: { editing = .edit($0) }, onAdd: { editing = .add(slot: nil) })
                if let hero = MediaTemplate.allCases.first {
                    HeroCard(template: hero, readiness: readiness(hero), busy: run.isBusy && run.template != hero) {
                        run.open(hero)
                    }
                }
                ForEach(TemplateKind.allCases, id: \.self) { kind in
                    VStack(alignment: .leading, spacing: 12) {
                        Text(L10n.text(kind.title)).font(.app(.sectionTitle))
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 190, maximum: 230), spacing: 18, alignment: .top)],
                                  alignment: .leading, spacing: 20) {
                            ForEach(MediaTemplate.allCases.dropFirst().filter { $0.kind == kind }) { t in
                                TemplateCard(template: t, readiness: readiness(t), busy: run.isBusy && run.template != t) {
                                    run.open(t)
                                }
                            }
                        }
                    }
                }
            }
            .padding(24)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

// MARK: - History

/// The videos made here, newest first. A tile opens a player that works while
/// another video renders; opening one back on its template waits for that.
private struct HistoryColumn: View {
    @EnvironmentObject var appState: AppState
    @ObservedObject var history: TemplateHistory
    @ObservedObject var run: TemplateRunner
    @ObservedObject var cast: CastStore
    @State private var playing: UUID?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Made by you").font(.app(.sectionTitle))
            if history.entries.isEmpty {
                Text("What you make shows up here.").font(.app(.explainer)).foregroundStyle(.secondary)
                Spacer()
            } else {
                ScrollView {
                    LazyVGrid(columns: [GridItem(.flexible(), spacing: 12, alignment: .top),
                                        GridItem(.flexible(), spacing: 12, alignment: .top)],
                              alignment: .leading, spacing: 16) {
                        ForEach(history.entries) { e in
                            Button { playing = e.id } label: { HistoryTile(entry: e, cast: cast) }
                                .buttonStyle(.plain)
                                .popover(isPresented: Binding(get: { playing == e.id },
                                                              set: { if !$0 { playing = nil } }),
                                         arrowEdge: .leading) {
                                    HistoryPlayer(entry: e, canOpen: !run.isBusy,
                                                  onOpen: { playing = nil; run.restore(e, cast: cast) },
                                                  onDelete: { playing = nil; history.delete(e.id) })
                                }
                        }
                    }
                }
            }
        }
        .padding(.top, 24)
        .padding(.horizontal, 16)
        .frame(maxHeight: .infinity, alignment: .top)
    }
}

/// Who and what a history video is, for its tile and its player.
@MainActor
private enum HistoryText {
    static func title(_ e: TemplateHistoryEntry) -> String {
        e.videoTemplate.map { L10n.text($0.title) } ?? e.template
    }

    static func names(_ e: TemplateHistoryEntry, _ cast: CastStore) -> String {
        let order = e.videoTemplate?.slots.map(\.key) ?? Array(e.cast.keys)
        let names = order.compactMap { key in e.cast[key].flatMap { id in cast.members.first { $0.id == id }?.name } }
        return ListFormatter.localizedString(byJoining: names)
    }
}

private struct HistoryTile: View {
    let entry: TemplateHistoryEntry
    let cast: CastStore
    @State private var hovering = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            VideoPoster(url: URL(fileURLWithPath: entry.clip))
                .aspectRatio(CGFloat(entry.shareShape.width) / CGFloat(entry.shareShape.height), contentMode: .fit)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay {
                    Image(systemName: "play.circle.fill").font(.system(size: 28)).foregroundStyle(.white.opacity(hovering ? 0.95 : 0.7))
                }
            Text(verbatim: HistoryText.title(entry)).font(.app(.annotation, weight: .semibold)).lineLimit(1)
            let names = HistoryText.names(entry, cast)
            if !names.isEmpty {
                Text(verbatim: names).font(.app(.annotation)).foregroundStyle(.secondary).lineLimit(1)
            }
            Text(verbatim: entry.createdAt.formatted(.relative(presentation: .named)))
                .font(.app(.annotation)).foregroundStyle(.secondary)
        }
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
    }
}

/// A video's first moment as a still, drawn once per file.
private struct VideoPoster: View {
    let url: URL
    @State private var image: NSImage?

    var body: some View {
        Color.secondary.opacity(0.15)
            .overlay { if let image { Image(nsImage: image).resizable().scaledToFill() } }
            .clipped()
            .task(id: url) { image = await VideoPosters.poster(url) }
    }
}

@MainActor
private enum VideoPosters {
    private static let cache = NSCache<NSURL, NSImage>()

    static func poster(_ url: URL) async -> NSImage? {
        if let hit = cache.object(forKey: url as NSURL) { return hit }
        if url.isPicture {
            let image = NSImage(contentsOf: url)
            if let image { cache.setObject(image, forKey: url as NSURL) }
            return image
        }
        let gen = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: 400, height: 400)
        guard let still = try? await gen.image(at: CMTime(seconds: 1, preferredTimescale: 600)) else { return nil }
        let image = NSImage(cgImage: still.image, size: .zero)
        cache.setObject(image, forKey: url as NSURL)
        return image
    }
}

private struct HistoryPlayer: View {
    let entry: TemplateHistoryEntry
    let canOpen: Bool
    let onOpen: () -> Void
    let onDelete: () -> Void
    @State private var player: AVPlayer?
    @State private var confirmingDelete = false

    var body: some View {
        let shape = entry.shareShape
        let width: CGFloat = shape == .wide ? 400 : shape == .square ? 320 : 270
        VStack(alignment: .leading, spacing: 10) {
            Group {
                if entry.playURL.isPicture {
                    VideoPoster(url: entry.playURL)
                } else if let player {
                    AVPlayerViewRepresentable(player: player)
                } else {
                    Color.black
                }
            }
            .frame(width: width, height: (width * CGFloat(shape.height) / CGFloat(shape.width)).rounded())
            .clipShape(RoundedRectangle(cornerRadius: 10))
            Text(verbatim: HistoryText.title(entry)).font(.app(.rowTitle, weight: .semibold))
            Text(verbatim: entry.createdAt.formatted(date: .abbreviated, time: .shortened))
                .font(.app(.annotation)).foregroundStyle(.secondary)
            HStack(spacing: 8) {
                ShareLink(item: entry.playURL) { Label("Share…", systemImage: "square.and.arrow.up").font(.app(.value)) }
                if let gif = entry.gif {
                    ShareLink(item: URL(fileURLWithPath: gif)) { Label("GIF…", systemImage: "photo.on.rectangle").font(.app(.value)) }
                }
                Button { NSWorkspace.shared.activateFileViewerSelecting([entry.playURL]) } label: {
                    Label("Show in Finder", systemImage: "folder").font(.app(.value))
                }
            }
            HStack(spacing: 8) {
                Button(action: onOpen) { Text("Open as template").font(.app(.value)) }
                    .disabled(!canOpen)
                    .help(canOpen ? "" : L10n.text("Wait for the one being made now, or cancel it."))
                Spacer()
                if confirmingDelete {
                    Button(role: .destructive, action: onDelete) { Text("Delete").font(.app(.value)) }
                    Button { confirmingDelete = false } label: { Text("Keep").font(.app(.value)) }
                } else {
                    Button { confirmingDelete = true } label: { Text("Delete…").font(.app(.value)) }
                }
            }
        }
        .padding(14)
        .frame(width: width + 28)
        .task {
            guard !entry.playURL.isPicture else { return }
            player = AVPlayer(url: entry.playURL)
            player?.play()
        }
        .onDisappear { player?.pause() }
    }
}

private struct ContinueBanner: View {
    let template: MediaTemplate
    @ObservedObject var run: TemplateRunner

    var body: some View {
        Button { run.open(template) } label: {
            HStack(spacing: 10) {
                if run.isBusy { ProgressView().controlSize(.small) } else { Image(systemName: "checkmark.circle.fill").foregroundStyle(.green) }
                Text(run.isBusy ? L10n.format("Making %@…", L10n.text(template.title))
                                : L10n.format("%@ is ready", L10n.text(template.title)))
                    .font(.app(.rowTitle, weight: .medium))
                Spacer()
                Text("Open").font(.app(.value)).foregroundStyle(Color.accentColor)
            }
            .padding(12)
            .background(RoundedRectangle(cornerRadius: 10).fill(Color.accentColor.opacity(0.10)))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

private struct HeroCard: View {
    let template: MediaTemplate
    let readiness: TemplateReadiness
    let busy: Bool
    let open: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 20) {
            SamplePoster(template: template).frame(width: 170, height: 302)
            VStack(alignment: .leading, spacing: 10) {
                Label("This month", systemImage: "snowflake").font(.app(.annotation, weight: .semibold))
                    .foregroundStyle(.blue)
                Text(L10n.text(template.title)).font(.app(.sectionTitle))
                Text(L10n.text(template.blurb)).font(.app(.explainer)).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                ReadinessBadge(template: template, readiness: readiness)
                Button(action: open) { Text("Make one").font(.app(.value, weight: .semibold)) }
                    .buttonStyle(.borderedProminent)
                    .disabled(busy)
                    .padding(.top, 4)
                if busy { Text("One at a time: the current one is still being made.").font(.app(.annotation)).foregroundStyle(.secondary) }
            }
            .frame(maxWidth: 380, alignment: .leading)
        }
        .padding(16)
        .background(RoundedRectangle(cornerRadius: 14).fill(Color.secondary.opacity(0.08)))
    }
}

private struct TemplateCard: View {
    let template: MediaTemplate
    let readiness: TemplateReadiness
    let busy: Bool
    let open: () -> Void
    @State private var hovering = false

    var body: some View {
        Button(action: open) {
            VStack(alignment: .leading, spacing: 8) {
                SamplePoster(template: template).frame(height: 300)
                Text(L10n.text(template.title)).font(.app(.rowTitle, weight: .semibold))
                Text(L10n.text(template.blurb)).font(.app(.annotation)).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                ReadinessBadge(template: template, readiness: readiness)
            }
            .padding(10)
            .background(RoundedRectangle(cornerRadius: 14).fill(Color.secondary.opacity(hovering ? 0.14 : 0.08)))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(busy)
        .onHover { hovering = $0 }
    }
}

/// The template's own sample, looping and muted; its symbol when the bundle has none.
private struct SamplePoster: View {
    let template: MediaTemplate

    var body: some View {
        ZStack {
            if let url = TemplateCheck.sampleURL(template) {
                if url.isPicture { VideoPoster(url: url) } else { LoopingVideo(url: url) }
            } else {
                LinearGradient(colors: [.orange.opacity(0.35), .pink.opacity(0.35)], startPoint: .top, endPoint: .bottom)
                Image(systemName: template.symbol).font(.system(size: 44)).foregroundStyle(.white.opacity(0.9))
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

/// A muted loop that shows its still until the first frame is ready, inside
/// the one view: a still stacked under it shows through when the view is dimmed.
private struct LoopingVideo: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> LoopView { LoopView() }
    func updateNSView(_ view: LoopView, context: Context) { view.play(url) }

    final class LoopView: NSView {
        private let player = AVQueuePlayer()
        private let still = CALayer()
        private var looper: AVPlayerLooper?
        private var ready: NSKeyValueObservation?
        private var url: URL?

        override init(frame: NSRect) {
            super.init(frame: frame)
            wantsLayer = true
            player.isMuted = true
            still.contentsGravity = .resizeAspectFill
            still.masksToBounds = true
            layer?.addSublayer(still)
            ready = (layer as? AVPlayerLayer)?.observe(\.isReadyForDisplay, options: [.initial, .new]) { [weak self] l, _ in
                let shown = l.isReadyForDisplay
                DispatchQueue.main.async { self?.still.isHidden = shown }
            }
        }

        required init?(coder: NSCoder) { nil }

        override func makeBackingLayer() -> CALayer {
            let layer = AVPlayerLayer(player: player)
            layer.videoGravity = .resizeAspectFill
            return layer
        }

        override func layout() {
            super.layout()
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            still.frame = bounds
            CATransaction.commit()
        }

        func play(_ url: URL) {
            guard url != self.url else { return }
            self.url = url
            still.contents = nil
            Task { @MainActor [weak self] in
                let image = await VideoPosters.poster(url)
                guard let self, self.url == url else { return }
                self.still.contents = image
            }
            looper = AVPlayerLooper(player: player, templateItem: AVPlayerItem(url: url))
            player.play()
        }
    }
}

// MARK: - Cast

private enum CastEditTarget: Identifiable {
    /// A new member; `slot` is the blank to fill with them.
    case add(slot: String?)
    case edit(CastMember)

    var id: String {
        switch self {
        case .add(let slot): "add-\(slot ?? "")"
        case .edit(let m): m.id.uuidString
        }
    }
}

/// Photos decode once per file: names are unique per import.
@MainActor
private enum CastPhotos {
    private static let cache = NSCache<NSURL, NSImage>()

    static func image(_ url: URL) -> NSImage? {
        if let hit = cache.object(forKey: url as NSURL) { return hit }
        guard let img = NSImage(contentsOf: url) else { return nil }
        cache.setObject(img, forKey: url as NSURL)
        return img
    }
}

private struct CastAvatar: View {
    let member: CastMember
    let url: URL
    var size: CGFloat = 56
    var selected = false

    var body: some View {
        VStack(spacing: 4) {
            Group {
                if let img = CastPhotos.image(url) {
                    Image(nsImage: img).resizable().scaledToFill()
                } else {
                    Image(systemName: member.kind.isPerson ? "person.fill" : "pawprint.fill").foregroundStyle(.secondary)
                }
            }
            .frame(width: size, height: size)
            .background(Circle().fill(Color.secondary.opacity(0.15)))
            .clipShape(Circle())
            .overlay(Circle().stroke(selected ? Color.accentColor : .clear, lineWidth: 3))
            Text(verbatim: member.name).font(.app(.annotation)).lineLimit(1).frame(maxWidth: size + 16)
        }
    }
}

private struct AddCastButton: View {
    var size: CGFloat = 56
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: "plus")
                    .frame(width: size, height: size)
                    .overlay(Circle().strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [4, 3])).foregroundStyle(.secondary))
                Text("Add").font(.app(.annotation))
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

private struct CastStrip: View {
    @ObservedObject var cast: CastStore
    let onEdit: (CastMember) -> Void
    let onAdd: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Your cast").font(.app(.sectionTitle))
            Text("The people and pets your videos, pictures and songs star. Their photos stay on this Mac.")
                .font(.app(.explainer)).foregroundStyle(.secondary)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(alignment: .top, spacing: 14) {
                    ForEach(cast.members) { m in
                        Button { onEdit(m) } label: { CastAvatar(member: m, url: cast.photoURL(m)) }
                            .buttonStyle(.plain)
                    }
                    AddCastButton(action: onAdd)
                }
                .padding(.vertical, 2)
            }
        }
    }
}

private struct CastEditorSheet: View {
    @ObservedObject var cast: CastStore
    let target: CastEditTarget
    let onDone: (CastMember?) -> Void

    @State private var name = ""
    @State private var kind: CastKind = .dog
    @State private var voice: CastVoice = .goofy
    @State private var looks = ""
    @State private var consent = false
    @State private var photo: URL?
    @State private var dropTargeted = false
    @State private var error: String?

    private var existing: CastMember? {
        if case .edit(let m) = target { return m }
        return nil
    }

    private var preview: NSImage? {
        if let photo { return NSImage(contentsOf: photo) }
        return existing.flatMap { CastPhotos.image(cast.photoURL($0)) }
    }

    private var canSave: Bool {
        !name.trimmingCharacters(in: .whitespaces).isEmpty && (photo != nil || existing != nil)
            && (!kind.isPerson || consent)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(existing == nil ? "Add to your cast" : "Edit cast member").font(.app(.sectionTitle))
            HStack(alignment: .top, spacing: 18) {
                photoWell.frame(width: 170)
                CastFields(name: $name, kind: $kind, voice: $voice, looks: $looks, consent: $consent)
            }
            if let error { Text(verbatim: error).font(.app(.annotation)).foregroundStyle(.red) }
            HStack {
                if let existing {
                    Button(role: .destructive) { cast.delete(existing.id); onDone(nil) } label: {
                        Text("Remove").font(.app(.value))
                    }
                }
                Spacer()
                Button { onDone(nil) } label: { Text("Cancel").font(.app(.value)) }
                    .keyboardShortcut(.cancelAction)
                Button { save() } label: { Text("Save").font(.app(.value)) }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!canSave)
            }
        }
        .padding(20)
        .frame(width: 560)
        .onAppear(perform: load)
        .onChange(of: kind) { _, k in if existing == nil { voice = .default(for: k) } }
    }

    @ViewBuilder private var photoWell: some View {
        if let img = preview {
            VStack(spacing: 8) {
                Image(nsImage: img).resizable().scaledToFill()
                    .frame(width: 170, height: 170).clipShape(RoundedRectangle(cornerRadius: 12))
                Button { pick() } label: { Text("Change photo…").font(.app(.value)) }
            }
            .mediaDrop(.image, isTargeted: $dropTargeted) { photo = $0.first }
        } else {
            MediaDropWell(title: "Choose a photo", systemImage: "photo", isTargeted: dropTargeted) { pick() }
                .frame(height: 170)
                .mediaDrop(.image, isTargeted: $dropTargeted) { photo = $0.first }
        }
    }

    private func load() {
        if let m = existing {
            name = m.name; kind = m.kind; voice = m.voice; looks = m.looks; consent = m.consent
        }
    }

    private func pick() {
        let panel = OpenPanel.make()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        if AppActivation.runModal(panel) == .OK, let url = panel.url { photo = url }
    }

    private func save() {
        var m = existing ?? CastMember(name: "", kind: kind, voice: voice, photo: "")
        m.name = name.trimmingCharacters(in: .whitespaces)
        m.kind = kind
        m.voice = voice
        m.looks = looks
        m.consent = kind.isPerson && consent
        do {
            if existing == nil, let photo {
                onDone(try cast.add(m, photo: photo))
            } else {
                try cast.update(m, photo: photo)
                onDone(m)
            }
        } catch {
            self.error = error.localizedDescription
        }
    }
}

private struct CastFields: View {
    @Binding var name: String
    @Binding var kind: CastKind
    @Binding var voice: CastVoice
    @Binding var looks: String
    @Binding var consent: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            TextField("Name", text: $name).textFieldStyle(.roundedBorder).font(.app(.value))
            Picker(selection: $kind) {
                ForEach(CastKind.allCases) { k in Text(L10n.text(k.label)).tag(k) }
            } label: { Text("They are a").font(.app(.value)) }
            if !kind.isPerson {
                Picker(selection: $voice) {
                    ForEach(CastVoice.allCases) { v in Text(L10n.text(v.label)).tag(v) }
                } label: { Text("Voice").font(.app(.value)) }
            }
            TextField("What they look like (optional)", text: $looks).textFieldStyle(.roundedBorder).font(.app(.value))
            if kind.isPerson {
                Toggle(isOn: $consent) {
                    Text("This is me, or I have their permission").font(.app(.value))
                }
                .toggleStyle(.checkbox)
            }
        }
    }
}

// MARK: - Template page

private struct TemplatePage: View {
    let template: MediaTemplate
    @ObservedObject var run: TemplateRunner
    @ObservedObject var cast: CastStore
    @State private var editing: CastEditTarget?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Button { run.onGallery = true } label: {
                    Label("Quick Create", systemImage: "chevron.left").font(.app(.value))
                }
                .buttonStyle(.plain)
                .foregroundStyle(Color.accentColor)
                Text(L10n.text(template.title)).font(.app(.pageTitle))
            }
            HStack(alignment: .top, spacing: 28) {
                ScrollView {
                    TemplateForm(template: template, run: run, cast: cast, editing: $editing)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.trailing, 8)
                }
                .frame(maxWidth: 560)
                TemplateStage(template: template, run: run, cast: cast)
            }
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .sheet(item: $editing) { target in
            CastEditorSheet(cast: cast, target: target) { saved in
                if let saved, case .add(let slot?) = target { run.fill.cast[slot] = saved }
                editing = nil
            }
        }
    }
}

private struct TemplateForm: View {
    @EnvironmentObject var appState: AppState
    let template: MediaTemplate
    @ObservedObject var run: TemplateRunner
    @ObservedObject var cast: CastStore
    @Binding var editing: CastEditTarget?

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            Text(L10n.text(template.blurb)).font(.app(.explainer)).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 18) {
                ForEach(template.slots) { slot in
                    SlotRow(slot: slot, run: run, cast: cast, editing: $editing)
                }
                VStack(alignment: .leading, spacing: 6) {
                    Text("Where will you share it?").font(.app(.rowTitle, weight: .semibold))
                    Picker(selection: $run.shape) {
                        ForEach(ShareShape.allCases) { s in Text(L10n.text(s.label)).tag(s) }
                    } label: { EmptyView() }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .fixedSize()
                }
            }
            .disabled(run.isBusy)
            if run.phase == .filling || run.phase == .writing {
                WriteStep(template: template, run: run)
            } else {
                if !template.lineKeys(run.fill).isEmpty { LinesStep(template: template, run: run) }
                MakeStep(template: template, run: run)
            }
        }
    }
}

private struct SlotRow: View {
    let slot: TemplateSlot
    @ObservedObject var run: TemplateRunner
    @ObservedObject var cast: CastStore
    @Binding var editing: CastEditTarget?

    private var answer: Binding<String> {
        Binding(get: { run.fill.answers[slot.key] ?? "" }, set: { run.fill.answers[slot.key] = $0 })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text(L10n.text(slot.label)).font(.app(.rowTitle, weight: .semibold))
                if slot.optional { Text("optional").font(.app(.annotation)).foregroundStyle(.secondary) }
            }
            switch slot.kind {
            case .cast:
                CastPicker(slot: slot, run: run, cast: cast, editing: $editing)
            case .text(let example):
                TextField(example, text: answer).textFieldStyle(.roundedBorder).font(.app(.value))
            case .choice(let options):
                HStack(spacing: 12) {
                    ForEach(options, id: \.self) { o in
                        Button { answer.wrappedValue = o } label: {
                            ChoiceTile(label: o, url: TemplateCheck.sampleURL(run.template, choice: o),
                                       shape: run.template?.defaultShape ?? .square, selected: answer.wrappedValue == o)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }
}

/// One option of a picker, shown as the muted loop of how it looks, in the template's own shape.
private struct ChoiceTile: View {
    let label: String
    let url: URL?
    let shape: ShareShape
    let selected: Bool

    var body: some View {
        VStack(spacing: 6) {
            Group {
                if let url { LoopingVideo(url: url) } else { Color.secondary.opacity(0.12) }
            }
                .frame(width: shape.height > shape.width ? 104 : 120,
                       height: ((shape.height > shape.width ? 104 : 120) * CGFloat(shape.height) / CGFloat(shape.width)).rounded())
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(selected ? Color.accentColor : .clear, lineWidth: 3))
                .contentShape(RoundedRectangle(cornerRadius: 12))
            Text(L10n.text(label)).font(.app(.annotation, weight: selected ? .semibold : .regular))
                .foregroundStyle(selected ? .primary : .secondary)
        }
    }
}

/// The cast as faces to tap, minus anyone in another blank, plus Add.
private struct CastPicker: View {
    let slot: TemplateSlot
    @ObservedObject var run: TemplateRunner
    @ObservedObject var cast: CastStore
    @Binding var editing: CastEditTarget?

    var body: some View {
        let taken = Set(run.fill.cast.filter { $0.key != slot.key }.map(\.value.id))
        let offered = cast.members.filter { !taken.contains($0.id) }
        FlowLayout(spacing: 12, rowSpacing: 10) {
            if slot.optional {
                Button { run.fill.cast[slot.key] = nil } label: { NoneChip(selected: run.fill.cast[slot.key] == nil) }
                    .buttonStyle(.plain)
            }
            ForEach(offered) { m in
                Button { run.fill.cast[slot.key] = m } label: {
                    CastAvatar(member: m, url: cast.photoURL(m), size: 48, selected: run.fill.cast[slot.key]?.id == m.id)
                }
                .buttonStyle(.plain)
            }
            AddCastButton(size: 48) { editing = .add(slot: slot.key) }
        }
    }
}

private struct NoneChip: View {
    let selected: Bool

    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: "nosign").foregroundStyle(.secondary)
                .frame(width: 48, height: 48)
                .background(Circle().fill(Color.secondary.opacity(0.12)))
                .overlay(Circle().stroke(selected ? Color.accentColor : .clear, lineWidth: 3))
            Text("Nobody").font(.app(.annotation))
        }
    }
}

private struct WriteStep: View {
    @EnvironmentObject var appState: AppState
    let template: MediaTemplate
    @ObservedObject var run: TemplateRunner

    var body: some View {
        let missing = template.missing(run.fill)
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Button { run.writeLines(app: appState) } label: {
                    Text(template.writesOwnLines ? "Next" : template.kind == .audio ? "Write the lyrics" : "Write the lines")
                        .font(.app(.value, weight: .semibold))
                }
                .buttonStyle(.borderedProminent)
                .disabled(missing != nil || run.isBusy)
                if run.phase == .writing {
                    ProgressView().controlSize(.small)
                    Text("Writing…").font(.app(.annotation)).foregroundStyle(.secondary)
                    Button { run.cancel() } label: { Text("Cancel").font(.app(.value)) }
                }
            }
            if let missing {
                Text(L10n.format("Still needed: %@", L10n.text(missing))).font(.app(.annotation)).foregroundStyle(.secondary)
            }
        }
    }
}

private struct LinesStep: View {
    @EnvironmentObject var appState: AppState
    let template: MediaTemplate
    @ObservedObject var run: TemplateRunner

    var body: some View {
        let keys = template.lineKeys(run.fill)
        let budget = TemplateText.wordBudget(seconds: template.seconds)
        let used = TemplateText.words(in: run.lines)
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(template.kind == .audio ? "The lyrics" : "The lines").font(.app(.sectionTitle))
                Spacer()
                // A song is as long as it is: only a clip's lines have to fit its seconds.
                if template.kind == .video {
                    Text(used <= budget ? L10n.format("%lld of %lld words · fits", used, budget)
                                        : L10n.format("%lld of %lld words · too long for %lld seconds", used, budget, template.seconds))
                        .font(.app(.annotation)).foregroundStyle(used <= budget ? Color.secondary : Color.orange)
                }
            }
            ForEach(keys, id: \.self) { key in
                VStack(alignment: .leading, spacing: 4) {
                    Text(verbatim: template.speaker(of: key, run.fill)).font(.app(.annotation, weight: .semibold))
                        .foregroundStyle(.secondary)
                    TextField("", text: Binding(get: { run.lines[key] ?? "" }, set: { run.lines[key] = $0 }), axis: .vertical)
                        .lineLimit(key == "lyrics" ? 8...24 : 1...6)
                        .textFieldStyle(.roundedBorder).font(.app(.value))
                }
            }
            .disabled(run.isBusy)
            if let note = run.linesNote { Text(verbatim: note).font(.app(.annotation)).foregroundStyle(.orange) }
            HStack(spacing: 10) {
                Button { run.editFill() } label: { Text("Change the details").font(.app(.value)) }
                if !template.writesOwnLines {
                    Button { run.writeLines(app: appState) } label: { Label("Write again", systemImage: "arrow.clockwise").font(.app(.value)) }
                }
            }
            .disabled(run.isBusy)
        }
    }
}

private struct MakeStep: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var server: ServerManager
    let template: MediaTemplate
    @ObservedObject var run: TemplateRunner

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            switch run.phase {
            case .rendering, .captioning:
                RenderProgress(run: run)
            case .done:
                DoneActions(run: run)
            default:
                start
            }
        }
    }

    @ViewBuilder private var start: some View {
        let readiness = TemplateRunner.readiness(template, shape: run.shape, app: appState)
        if case .failed(let message) = run.phase {
            Text(verbatim: message).font(.app(.annotation)).foregroundStyle(.red)
        }
        switch readiness {
        case .tooBig(let gb):
            Text(L10n.format("This Mac doesn't have the memory for this model (about %lld GB).", gb))
                .font(.app(.explainer)).foregroundStyle(.orange)
        case .needsDownload:
            BundleDownloadBar(bundle: template.bundle)
        case .ready(let seconds):
            makeButton(seconds.map { L10n.format("%@ on this Mac", H3TimeEstimate.duration($0)) } ?? L10n.text("On this Mac"))
        case .onPeer(let id):
            makeButton(L10n.format("Made on %@, over your network", LanPick.peer(of: id)))
        }
        if template.captions { CaptionsStatus(run: run) }
    }

    private var makeLabel: String {
        switch template.kind {
        case .video: L10n.text("Make the video")
        case .image: L10n.text("Make the picture")
        case .audio: L10n.text("Make the song")
        }
    }

    private func makeButton(_ note: String) -> some View {
        HStack(spacing: 12) {
            Button { run.make(app: appState) } label: {
                Text(makeLabel).font(.app(.value, weight: .semibold))
            }
            .buttonStyle(.borderedProminent)
            .disabled(run.isBusy || template.linesMissing(run.fill, run.lines))
            Button { run.openInEditor(app: appState) } label: {
                Label("Advanced Editor", systemImage: "slider.horizontal.3").font(.app(.value))
            }
            .help(L10n.text("Open this in the full editor with every option"))
            Text(verbatim: note).font(.app(.annotation)).foregroundStyle(.secondary)
        }
    }
}

/// What captions still need on this Mac, before a video is made and after: a
/// video made without the speech models gets its captions once they arrive.
private struct CaptionsStatus: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var downloads: DownloadManager
    @ObservedObject var run: TemplateRunner

    var body: some View {
        let done = run.phase == .done
        switch CaptionModels.of(appState.localModels, engine: appState.serverOptions.transcribeEngine) {
        case .ready:
            if done, run.captioned == nil {
                HStack(spacing: 10) {
                    if let note = run.captionNote {
                        Text(verbatim: note).font(.app(.annotation)).foregroundStyle(.secondary)
                    }
                    Button { run.addCaptions(app: appState) } label: {
                        Label(run.captionNote == nil ? "Add captions" : "Try again", systemImage: "captions.bubble")
                            .font(.app(.value))
                    }
                }
            }
        case .download(let repos):
            if !done || run.captioned == nil {
                let fetching = (repos + [TrayTranscript.asrRepo, DownloadManager.alignerRepo])
                    .contains { downloads.downloads[$0]?.status == .downloading }
                HStack(spacing: 10) {
                    Text(done ? "No captions: they need the speech models, which aren't on this Mac."
                              : "Captions need the speech models, which aren't on this Mac.")
                        .font(.app(.annotation)).foregroundStyle(.secondary)
                    if fetching {
                        ProgressView().controlSize(.small)
                        Text("Downloading…").font(.app(.annotation)).foregroundStyle(.secondary)
                    } else {
                        Button {
                            for repo in repos {
                                downloads.start(repoId: repo) {
                                    appState.refreshModels()
                                    appState.server.rescanModels()
                                }
                            }
                        } label: { Text("Download").font(.app(.value)) }
                    }
                }
            }
        case .notConverted:
            HStack(spacing: 10) {
                Text("Captions need Nemotron Transcribe, which isn't on this Mac.")
                    .font(.app(.annotation)).foregroundStyle(.secondary)
                Button { appState.showModels(.myModels) } label: { Text("Models").font(.app(.value)) }
            }
        }
    }
}

private struct RenderProgress: View {
    @ObservedObject var run: TemplateRunner

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 14) {
                let kind = run.template?.kind ?? .video
                StepMark(title: L10n.text(kind == .audio ? "Lyrics" : kind == .image ? "Photo" : "Lines"), state: .done)
                if kind == .video { StepMark(title: L10n.text("Photos"), state: .done) }
                StepMark(title: animating(kind), state: run.phase == .rendering ? .now : .done)
                StepMark(title: L10n.text(kind == .audio ? "Video" : kind == .image ? "Meme text" : "Captions"),
                         state: run.phase == .captioning ? .now : .later)
            }
            if let p = run.progress, let f = p.fraction {
                ProgressView(value: f)
            } else {
                ProgressView().progressViewStyle(.linear)
            }
            HStack(spacing: 10) {
                if let p = run.progress {
                    TimelineView(.periodic(from: .now, by: 1)) { ctx in
                        Text(verbatim: p.elapsedText(now: ctx.date))
                            .font(.app(.annotation).monospacedDigit()).foregroundStyle(.secondary)
                    }
                    Text(verbatim: L10n.text(p.message)).font(.app(.annotation)).foregroundStyle(.secondary)
                }
                Spacer()
                Button { run.cancel() } label: { Text("Cancel").font(.app(.value)) }
            }
            Text("You can leave this page; it keeps going.").font(.app(.annotation)).foregroundStyle(.secondary)
        }
    }

    private func animating(_ kind: TemplateKind) -> String {
        let verb = L10n.text(kind == .audio ? "Singing" : kind == .image ? "Painting" : "Animating")
        guard run.phase == .rendering, let p = run.progress, p.total > 0, p.step > 0 else { return verb }
        return L10n.format("%@ · step %lld of %lld", verb, p.step, p.total)
    }
}

private struct StepMark: View {
    enum State { case done, now, later }
    let title: String
    let state: State

    var body: some View {
        HStack(spacing: 5) {
            switch state {
            case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            case .now: ProgressView().controlSize(.mini)
            case .later: Image(systemName: "circle").foregroundStyle(.secondary)
            }
            Text(verbatim: title).font(.app(.annotation, weight: state == .now ? .semibold : nil))
        }
    }
}

private struct DoneActions: View {
    @EnvironmentObject var appState: AppState
    @ObservedObject var run: TemplateRunner

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                if let url = run.result {
                    ShareLink(item: url) { Label("Share…", systemImage: "square.and.arrow.up").font(.app(.value)) }
                        .buttonStyle(.borderedProminent)
                    Button { NSWorkspace.shared.activateFileViewerSelecting([url]) } label: {
                        Label("Show in Finder", systemImage: "folder").font(.app(.value))
                    }
                }
                if run.captioned != nil {
                    Toggle(isOn: $run.showsCaptions) {
                        Text(run.template?.captions == false ? "Meme text" : "Captions").font(.app(.value))
                    }
                    .toggleStyle(.switch).controlSize(.small)
                }
            }
            if let gif = run.gif { GIFActions(gif: gif) }
            if run.template?.captions == true { CaptionsStatus(run: run) }
            HStack(spacing: 10) {
                Button { run.anotherTake(app: appState) } label: { Label("Another take", systemImage: "dice").font(.app(.value)) }
                Button { run.editFill() } label: { Text("Change the details").font(.app(.value)) }
                Button { run.openInEditor(app: appState) } label: {
                    Label("Advanced Editor", systemImage: "slider.horizontal.3").font(.app(.value))
                }
            }
        }
    }
}

/// A meme's GIF: shared from the menu, or copied to paste straight into a chat.
private struct GIFActions: View {
    let gif: URL
    @State private var copied = false

    var body: some View {
        HStack(spacing: 10) {
            ShareLink(item: gif) { Label("Share GIF…", systemImage: "square.and.arrow.up").font(.app(.value)) }
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.writeObjects([gif as NSURL])
                copied = true
            } label: {
                Label(copied ? "Copied" : "Copy GIF", systemImage: copied ? "checkmark" : "doc.on.doc").font(.app(.value))
            }
        }
    }
}

// MARK: - Stage

/// The phone-shaped picture on the right: the sample, then your photos, then
/// the live preview, then your video.
private struct TemplateStage: View {
    let template: MediaTemplate
    @ObservedObject var run: TemplateRunner
    let cast: CastStore
    @State private var player: AVPlayer?
    static let width: CGFloat = 300

    var body: some View {
        // Sized from the share shape, never from its content.
        let height = (Self.width * CGFloat(run.shape.height) / CGFloat(run.shape.width)).rounded()
        ZStack {
            Color.black.opacity(0.85)
            content.frame(width: Self.width, height: height).clipped()
        }
        .frame(width: Self.width, height: height)
        .clipShape(RoundedRectangle(cornerRadius: 18))
        .contentShape(RoundedRectangle(cornerRadius: 18))
        .overlay(RoundedRectangle(cornerRadius: 18).stroke(Color.secondary.opacity(0.25)))
        .task(id: run.playable) {
            player?.pause()
            player = run.playable.flatMap { $0.isPicture ? nil : AVPlayer(url: $0) }
            player?.play()
        }
        .onDisappear { player?.pause() }
    }

    private func photos(_ members: [CastMember]) -> [NSImage] {
        members.compactMap { CastPhotos.image(cast.photoURL($0)) }
    }

    @ViewBuilder private var content: some View {
        if run.phase == .done, let url = run.playable, url.isPicture {
            VideoPoster(url: url)
        } else if run.phase == .done, let player {
            AVPlayerViewRepresentable(player: player)
        } else if run.phase == .rendering || run.phase == .captioning, let preview = run.preview {
            Image(nsImage: preview).resizable().scaledToFill()
        } else if let name = template.choiceSample(run.fill),
                  let url = Bundle.main.url(forResource: name, withExtension: "mp4") {
            LoopingVideo(url: url)
                .overlay(alignment: .bottomTrailing) { CastFaces(images: photos(template.cast(run.fill))) }
        } else if let first = template.cast(run.fill).first, let img = CastPhotos.image(cast.photoURL(first)) {
            CastCollage(lead: img, others: photos(Array(template.cast(run.fill).dropFirst())))
        } else {
            SamplePoster(template: template)
        }
    }
}

private struct CastCollage: View {
    let lead: NSImage
    let others: [NSImage]

    var body: some View {
        // `Color.clear` takes exactly the proposed size; a fill-scaled image alone reports its own.
        Color.clear
            .overlay { Image(nsImage: lead).resizable().scaledToFill() }
            .clipped()
            .overlay(alignment: .bottomTrailing) { CastFaces(images: others) }
    }
}

/// The cast as overlapping circles in the stage's corner.
private struct CastFaces: View {
    let images: [NSImage]

    var body: some View {
        HStack(spacing: -10) {
            ForEach(images.indices, id: \.self) { i in
                Image(nsImage: images[i]).resizable().scaledToFill()
                    .frame(width: 64, height: 64).clipShape(Circle())
                    .overlay(Circle().stroke(.white, lineWidth: 2))
            }
        }
        .padding(12)
    }
}
