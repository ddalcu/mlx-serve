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
            // omp (oh-my-pi) — pi fork, own config tree: models.yml under the
            // agent dir. The env read is still pi's PI_CODING_AGENT_DIR
            // spelling (measured on omp v17); OMP_ exported too for when the
            // rename completes. Same isolation move as pi.
            Tab(id: "omp",
                title: "oh-my-pi",
                installHint: "Requires the omp CLI: curl -fsSL https://omp.sh/install | sh",
                command: """
                mkdir -p ~/.mlx-serve/omp
                cat > ~/.mlx-serve/omp/models.yml <<'EOF'
                \(AgentConfigs.ompModelsYML(baseURL: baseURL, model: servedModelId, budget: budget))
                EOF
                export PI_CODING_AGENT_DIR="$HOME/.mlx-serve/omp"
                export OMP_CODING_AGENT_DIR="$HOME/.mlx-serve/omp"
                omp --model mlx/\(servedModelId)
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
            // codex honors CODEX_HOME for its whole config tree; the dir must
            // exist before codex runs. Responses wire API — our /v1/responses.
            // The resolver line also finds the CLI the ChatGPT/Codex desktop
            // app bundles, for installs with no codex on PATH.
            Tab(id: "codex",
                title: "Codex",
                installHint: "Requires the codex CLI (npm install -g @openai/codex) or the ChatGPT desktop app, which bundles it",
                command: """
                mkdir -p ~/.mlx-serve/codex
                cat > ~/.mlx-serve/codex/config.toml <<'EOF'
                \(AgentConfigs.codexConfigTOML(baseURL: baseURL, model: servedModelId, budget: budget))
                EOF
                export CODEX_HOME="$HOME/.mlx-serve/codex"
                \(AgentConfigs.codexBinResolver)
                "$CODEX_BIN"
                """),
            // hermes reads its whole tree from HERMES_HOME; the .env is the
            // first-run wizard kill switch (OPENAI_BASE_URL set = configured).
            Tab(id: "hermes",
                title: "hermes",
                installHint: "Requires the hermes CLI: curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash",
                command: """
                mkdir -p ~/.mlx-serve/hermes
                cat > ~/.mlx-serve/hermes/config.yaml <<'EOF'
                \(AgentConfigs.hermesConfigYAML(baseURL: baseURL, apiKey: "mlx-serve",
                                                model: servedModelId, budget: budget, entries: []))
                EOF
                cat > ~/.mlx-serve/hermes/.env <<'ENVEOF'
                \(AgentConfigs.hermesEnvFile(baseURL: baseURL))
                ENVEOF
                export HERMES_HOME="$HOME/.mlx-serve/hermes"
                hermes
                """),
            // aider is pure env vars plus a litellm metadata file that tells
            // it the real context window for openai/<id>.
            Tab(id: "aider",
                title: "aider",
                installHint: "Requires the aider CLI: curl -LsSf https://aider.chat/install.sh | sh",
                command: """
                mkdir -p ~/.mlx-serve/aider
                cat > ~/.mlx-serve/aider/model-metadata.json <<'EOF'
                \(AgentConfigs.aiderModelMetadataJSON(model: servedModelId, budget: budget, entries: []))
                EOF
                export OPENAI_API_BASE='\(baseURL)/v1'
                export OPENAI_API_KEY=mlx-serve
                aider --model openai/\(servedModelId) --weak-model openai/\(servedModelId) --model-metadata-file ~/.mlx-serve/aider/model-metadata.json
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
        .help("Connect a coding agent CLI (Claude Code, pi, oh-my-pi, OpenCode, Codex, hermes, aider) to this server — shows the terminal commands to run")
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

            // Menu picker — seven CLIs no longer fit a segmented control in
            // this popover's width.
            Picker("", selection: $selectedId) {
                ForEach(tabs) { tab in
                    Text(tab.title).tag(tab.id)
                }
            }
            .pickerStyle(.menu)
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
