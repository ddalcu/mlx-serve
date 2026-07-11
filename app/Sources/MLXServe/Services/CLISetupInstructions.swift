import SwiftUI
import AppKit

/// Copy-paste terminal setup for the supported coding-agent CLIs.
///
enum CLISetupInstructions {

    struct Tab: Identifiable, Equatable {
        let id: String
        let title: String
        /// Where to get the CLI if it isn't installed — shown as a caption.
        let installHint: String
        /// The full copy-paste block (config + launch).
        let command: String
    }

    /// MAS has no detection/launch, so the instructions panel takes the
    /// launcher's place in the tray — never both, never neither.
    static func replacesLauncher(features: BuildFeatures = BuildFeatures.current) -> Bool {
        !features.cliLauncher
    }

    /// Same CLIs, same order as the DMG launcher dropdown.
    static func tabs(baseURL: String, servedModelId: String,
                     budget: AgentBudget.Budget) -> [Tab] {
        [
            Tab(id: "claude",
                title: "Claude Code",
                installHint: "Requires the claude CLI: npm install -g @anthropic-ai/claude-code",
                command: """
                \(AgentConfigs.claudeCodeExports(baseURL: baseURL, model: servedModelId, budget: budget))
                claude --model \(servedModelId)
                """),
            // pi has no env-var/flag route for a custom base URL (a models.json
            // is required), but PI_CODING_AGENT_DIR relocates the config dir —
            // so we use a dedicated one and never overwrite the user's real
            // ~/.pi/agent/models.json. Same isolation move as OPENCODE_CONFIG.
            Tab(id: "pi",
                title: "pi",
                installHint: "Requires the pi CLI (pi.dev): curl -fsSL https://pi.dev/install.sh | sh",
                command: """
                mkdir -p ~/.mlx-serve/pi
                cat > ~/.mlx-serve/pi/models.json <<'EOF'
                \(AgentConfigs.piModelsJSON(baseURL: baseURL, model: servedModelId, budget: budget))
                EOF
                export PI_CODING_AGENT_DIR="$HOME/.mlx-serve/pi"
                pi --provider mlx --model \(servedModelId)
                """),
            // opencode needs no file at all: OPENCODE_CONFIG_CONTENT carries
            // the config inline and MERGES over the user's global/project
            // config, so their own settings and plugins keep working.
            Tab(id: "opencode",
                title: "OpenCode",
                installHint: "Requires the opencode CLI: curl -fsSL https://opencode.ai/install | bash",
                command: """
                export OPENCODE_CONFIG_CONTENT='\(AgentConfigs.opencodeJSON(baseURL: baseURL, model: servedModelId, budget: budget))'
                opencode --model mlx/\(servedModelId)
                """),
        ]
    }
}

// MARK: - UI

/// Tray "Code" button for builds without the one-click launcher: opens a
/// popover with per-CLI copy-paste instructions.
@MainActor
struct CLISetupInstructionsButton: View {
    let baseURL: String
    let servedModelId: String
    /// The running server's EFFECTIVE context (`/v1/models` meta.context_length)
    /// — the numbers written into the user's CLI config derive from it, never
    /// hardcoded. See `AgentBudget`.
    let serverContextLength: Int?
    let isEnabled: Bool

    @State private var showPanel = false

    var body: some View {
        Button {
            showPanel = true
        } label: {
            HStack(spacing: TrayFooterMetrics.iconSpacing) {
                Image(systemName: "terminal")
                Text("Code")
            }
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.bordered)
        .disabled(!isEnabled)
        .help("Connect a coding agent CLI (Claude Code, pi, OpenCode) to this server — shows the terminal commands to run")
        .popover(isPresented: $showPanel, arrowEdge: .bottom) {
            CLISetupInstructionsView(
                tabs: CLISetupInstructions.tabs(
                    baseURL: baseURL,
                    servedModelId: servedModelId,
                    budget: AgentBudget.forServerContext(serverContextLength)))
        }
    }
}

/// The panel: one tab per CLI, a monospaced command block, a Copy button.
/// Static content only — no spinners or animations (tray-popover freeze class).
struct CLISetupInstructionsView: View {
    let tabs: [CLISetupInstructions.Tab]

    @State private var selectedId: String = "claude"
    @State private var copied = false

    private var selected: CLISetupInstructions.Tab? {
        tabs.first { $0.id == selectedId } ?? tabs.first
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Use with a coding agent")
                .font(.headline)
            Text("Run these commands in Terminal from your project folder. They point the CLI at this Mac's local server — nothing leaves your machine.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Picker("", selection: $selectedId) {
                ForEach(tabs) { tab in
                    Text(tab.title).tag(tab.id)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            if let tab = selected {
                ScrollView([.vertical, .horizontal]) {
                    Text(tab.command)
                        .font(.system(size: 11, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(8)
                }
                .frame(height: 220)
                .background(Color(.textBackgroundColor).opacity(0.6))
                .clipShape(RoundedRectangle(cornerRadius: 6))

                HStack {
                    Text(tab.installHint)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                    Spacer()
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(tab.command, forType: .string)
                        copied = true
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { copied = false }
                    } label: {
                        Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                    }
                }
            }
        }
        .padding(14)
        .frame(width: 460)
        .onChange(of: selectedId) { _, _ in copied = false }
    }
}
