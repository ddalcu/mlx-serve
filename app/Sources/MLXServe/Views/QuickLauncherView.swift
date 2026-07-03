import SwiftUI

/// Content of the ⌃Space quick-launcher panel: a Spotlight-style prompt field
/// on top, the streamed answer below once a conversation exists. The panel has
/// two fixed sizes (compact / expanded — see `QuickLauncherLogic.panelHeight`),
/// so this view top-aligns and lets the answer area flex.
struct QuickLauncherView: View {
    @EnvironmentObject var appState: AppState
    @ObservedObject var controller: QuickLauncherController
    @ObservedObject var engine: ChatTurnEngine

    @State private var query = ""
    @FocusState private var focused: Bool

    private var conversation: ChatSession? {
        guard let sid = controller.sessionId else { return nil }
        return appState.chatSessions.first { $0.id == sid }
    }

    /// The launcher owns the in-flight turn (Stop button / spinner state).
    private var generatingHere: Bool {
        guard let sid = controller.sessionId else { return false }
        return engine.composerState(for: sid) == .generatingHere
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            inputRow
                .frame(height: 76)

            if let notice = controller.statusMessage {
                noticeRow(notice)
                    .frame(height: 36)
            }

            if let convo = conversation {
                Divider()
                answerArea(convo)
                footer
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .onExitCommand { controller.hide() }
        .onAppear { focused = true }
        .onChange(of: controller.focusTick) { _, _ in
            // Re-focus on every summon; async so it lands after the panel
            // becomes key.
            DispatchQueue.main.async { focused = true }
        }
    }

    // MARK: Input

    private var inputRow: some View {
        HStack(spacing: 12) {
            Image(systemName: "bolt.fill")
                .font(.system(size: 18, weight: .medium))
                .foregroundStyle(Color.accentColor)
            TextField("Ask the local model anything…", text: $query)
                .textFieldStyle(.plain)
                .font(.system(size: 20, weight: .regular))
                .focused($focused)
                .onSubmit {
                    if controller.submit(query) { query = "" }
                }
            if generatingHere {
                ProgressView()
                    .controlSize(.small)
            }
        }
        .padding(.horizontal, 20)
    }

    private func noticeRow(_ message: String) -> some View {
        Label(message, systemImage: "exclamationmark.triangle.fill")
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(.horizontal, 20)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: Answer

    private func answerArea(_ convo: ChatSession) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                if let question = convo.messages.last(where: { $0.role == .user })?.content,
                   !question.isEmpty {
                    Text(question)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                if let answer = convo.messages.last(where: { $0.role == .assistant }) {
                    if answer.content.isEmpty && answer.isStreaming {
                        GeneratingIndicator()
                    } else {
                        MarkdownText(answer.content)
                            .textSelection(.enabled)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
        }
        .defaultScrollAnchor(.bottom)
    }

    // MARK: Footer

    private var footer: some View {
        HStack(spacing: 12) {
            Text("↩ ask · esc close")
                .font(.caption2)
                .foregroundStyle(.tertiary)
            Spacer()
            if generatingHere {
                Button("Stop") { engine.stop() }
                    .keyboardShortcut(".", modifiers: .command)
                    .controlSize(.small)
            }
            Button("New  ⌘N") {
                controller.newConversation()
                query = ""
            }
            .keyboardShortcut("n", modifiers: .command)
            .controlSize(.small)
            Button("Open in Chat  ⌘↩") { controller.openInChat() }
                .keyboardShortcut(.return, modifiers: .command)
                .controlSize(.small)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
        .background(.black.opacity(0.06))
    }
}

/// Tray row for the launcher toggle — sits directly under the Voice section in
/// the menu-bar popover. Enabling registers the global ⌃Space hotkey (Carbon,
/// no permission prompt); disabling unregisters it and hides the panel.
struct QuickLauncherTrayRow: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "bolt.fill")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(appState.quickLauncherEnabled ? Color.accentColor : .secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text("Quick Launcher")
                    .font(.subheadline.weight(.medium))
                Text("\(QuickLauncherHotKey.display) — ask from anywhere")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Toggle("", isOn: $appState.quickLauncherEnabled)
                .labelsHidden()
                .toggleStyle(.switch)
                .controlSize(.small)
        }
        .help("Spotlight-style prompt panel on \(QuickLauncherHotKey.display): summon it from any app, ask the local model, and press ⌘↩ to continue in the chat window. If nothing happens on \(QuickLauncherHotKey.display), macOS may be using it for input-source switching (System Settings → Keyboard → Keyboard Shortcuts → Input Sources).")
    }
}
