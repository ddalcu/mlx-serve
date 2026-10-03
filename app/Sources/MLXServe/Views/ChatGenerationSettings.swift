import SwiftUI

struct ChatGenerationSettings: View {
    @EnvironmentObject var appState: AppState
    let sessionId: UUID
    @State private var presented = false

    private var profile: Binding<GenerationDefaults> {
        Binding(
            get: { appState.chatSessions.first { $0.id == sessionId }?.generationParams ?? .init() },
            set: { value in
                guard let index = appState.chatSessions.firstIndex(where: { $0.id == sessionId }) else { return }
                appState.chatSessions[index].generationParams = value
                appState.saveChatHistory()
            })
    }

    var body: some View {
        Button { presented.toggle() } label: {
            Image(systemName: "gearshape")
                .font(.app(.body, weight: .medium))
                .foregroundStyle(profile.wrappedValue.rules.isEmpty ? Color.secondary : Color.accentColor)
                .frame(width: ChatMetrics.composerIconSize, height: ChatMetrics.composerIconSize)
                .background(Color.secondary.opacity(0.15), in: Circle())
                .frame(width: ChatMetrics.composerControlSize, height: ChatMetrics.composerControlSize)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Chat generation settings")
        .composerTip(.generationSettings)
        .popover(isPresented: $presented, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Chat generation settings").font(.app(.headline))
                    Spacer()
                    Button { presented = false } label: { Text("Done").font(.app(.body)) }
                }
                Text("Saved only for this chat. Explicit values override agent and server defaults; server locks still win. Changes apply to the next turn.")
                    .font(.app(.caption)).foregroundStyle(.secondary)
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        GenerationDefaultsRows(profile: profile, inheritance: "Default",
                                               fields: GenerationField.clientFields, showsClientLocks: false)
                    }
                }
                Text("Thinking and reasoning effort use the brain control in the composer.")
                    .font(.app(.caption)).foregroundStyle(.secondary)
            }
            .padding(16)
            .frame(width: 600, height: 600)
        }
        .onChange(of: sessionId) { _, _ in presented = false }
    }
}
