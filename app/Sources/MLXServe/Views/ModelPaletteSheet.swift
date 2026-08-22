import SwiftUI

/// ⌘L: the model switcher. A search field over every model this chat can be
/// pointed at — the composer's pill is the same list for the mouse, this is the
/// one for the keyboard: type to narrow, ↑/↓ to move, Return to load, Esc out.
///
/// Every decision it makes is `ModelPalette`'s (rows, filtering, where the
/// selection opens, how the arrows behave); this draws them and reports a pick.
struct ModelPaletteSheet: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    @State private var query = ""
    /// Index into the FILTERED rows. Read through `ModelPalette.clamped`, never
    /// raw: typing shrinks the list under it.
    @State private var selection: Int?
    @FocusState private var searchFocused: Bool

    private var rows: [ModelPaletteRow] {
        ModelPalette.rows(local: appState.localModels,
                          lan: server.lanModels(capability: "chat"))
    }

    private var matches: [ModelPaletteRow] { ModelPalette.filtered(rows, query: query) }

    /// The row the chat is on right now, so it opens on the model already
    /// answering and ⌘L + Return is a no-op.
    private var currentTag: String {
        ChatModelSelection.tag(localPath: appState.selectedModelPath,
                               lanChatModelId: server.lanChatModelId)
    }

    var body: some View {
        VStack(spacing: 0) {
            searchField
            Divider()
            if matches.isEmpty {
                emptyState
            } else {
                list
            }
            Divider()
            footer
        }
        .frame(width: 520, height: 420)
        // Esc closes. The sheet is not blocking — it configures the next
        // message, it does not gate it — so SwiftUI's own dismissal is the
        // right behaviour and nothing intercepts the binding.
        .onExitCommand { close() }
        .onAppear {
            selection = ModelPalette.selection(in: matches, current: currentTag)
            // Async so the focus lands after the sheet becomes key — the same
            // turn the quick launcher's field needs.
            DispatchQueue.main.async { searchFocused = true }
            // The list is read from `localModels`; a model that landed while
            // another window was up must be in it.
            appState.refreshModels()
        }
        .onChange(of: query) { _, _ in
            // A new filter re-aims at the current model when it survived the
            // narrowing, and at the top row otherwise — never at whatever
            // index the old list happened to be on.
            selection = ModelPalette.selection(in: matches, current: currentTag)
        }
    }

    // MARK: - Pieces

    private var searchField: some View {
        HStack(spacing: 10) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField("Search models…", text: $query)
                .textFieldStyle(.plain)
                .font(.system(size: 17))
                .focused($searchFocused)
                .onSubmit { pickSelected() }
                .onKeyPress(.upArrow) { moveSelection(-1) }
                .onKeyPress(.downArrow) { moveSelection(1) }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }

    private var list: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0, pinnedViews: [.sectionHeaders]) {
                    ForEach(sections, id: \.self) { section in
                        Section {
                            ForEach(indexedRows(in: section), id: \.row.id) { entry in
                                rowView(entry.row, index: entry.index)
                                    .id(entry.index)
                            }
                        } header: {
                            Text(section)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 6)
                                .background(.bar)
                        }
                    }
                }
            }
            .onChange(of: selection) { _, new in
                // Keyboard movement has to bring its row with it, or ↓ walks
                // the highlight off the bottom of a scrolled list.
                if let new { proxy.scrollTo(new, anchor: .center) }
            }
        }
    }

    private func rowView(_ row: ModelPaletteRow, index: Int) -> some View {
        let selected = ModelPalette.clamped(selection, count: matches.count) == index
        return Button { pick(row) } label: {
            HStack(spacing: 10) {
                // The checkmark says which model is answering NOW; the
                // highlight says which one Return would load. Two different
                // facts, so they are two different marks.
                Image(systemName: row.tag == currentTag ? "checkmark" : "cpu")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(row.tag == currentTag ? Color.accentColor : .secondary)
                    .frame(width: 16)
                VStack(alignment: .leading, spacing: 1) {
                    Text(row.title)
                        .font(.callout.weight(.medium))
                        .lineLimit(1)
                    if !row.detail.isEmpty {
                        Text(row.detail)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 7)
            .contentShape(Rectangle())
            .background(selected ? Color.accentColor.opacity(0.18) : Color.clear)
        }
        .buttonStyle(.plain)
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Spacer()
            Text(rows.isEmpty ? "No chat models on this Mac" : "No models match “\(query)”")
                .foregroundStyle(.secondary)
            // With nothing to pick, the one useful thing is the way to get a
            // model — the same door the pill's last row opens.
            if rows.isEmpty {
                Button("Manage Models…") {
                    close()
                    appState.showModels()
                }
            }
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    private var footer: some View {
        HStack(spacing: 12) {
            Text("↑↓ move · ↩ switch · esc close")
                .font(.caption2)
                .foregroundStyle(.tertiary)
            Spacer()
            Button("Manage Models…") {
                close()
                appState.showModels()
            }
            .buttonStyle(.plain)
            .font(.caption)
            .foregroundStyle(Color.accentColor)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    // MARK: - Sections

    /// Section headings in row order, deduped — the row list is already in the
    /// order the palette decided, so this must not sort.
    private var sections: [String] {
        var seen = Set<String>()
        return matches.compactMap { seen.insert($0.section).inserted ? $0.section : nil }
    }

    /// Rows of one section, carrying their index into the FLAT filtered list —
    /// which is what the arrow keys move over, so a per-section index would
    /// highlight the wrong row the moment there are two groups.
    private func indexedRows(in section: String) -> [(index: Int, row: ModelPaletteRow)] {
        matches.enumerated()
            .filter { $0.element.section == section }
            .map { (index: $0.offset, row: $0.element) }
    }

    // MARK: - Actions

    private func moveSelection(_ delta: Int) -> KeyPress.Result {
        selection = ModelPalette.move(ModelPalette.clamped(selection, count: matches.count),
                                      by: delta, count: matches.count)
        return .handled
    }

    private func pickSelected() {
        guard let tag = ModelPalette.tag(at: ModelPalette.clamped(selection, count: matches.count),
                                         in: matches) else { return }
        apply(tag)
    }

    private func pick(_ row: ModelPaletteRow) { apply(row.tag) }

    private func apply(_ tag: String) {
        // What a pick MEANS is `AppState.applyChatModelPick` — the palette
        // decides which row, never what loading it involves.
        appState.applyChatModelPick(tag)
        close()
    }

    private func close() { appState.modelPalettePresented = false }
}
