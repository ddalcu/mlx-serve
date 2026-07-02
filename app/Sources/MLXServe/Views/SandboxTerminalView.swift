import SwiftUI

/// A live terminal into the running agent-sandbox guest. Shows a unified
/// transcript of commands the AGENT runs (peeked from the sandbox) AND commands
/// the USER types here — both go through the same guest shell, so state (cwd,
/// env, installed packages) is shared. Lets the user poke around the exact
/// environment the agent is working in.
struct SandboxTerminalView: View {
    @ObservedObject private var sandbox = AgentSandbox.shared
    /// Per-command appends publish through the dedicated store (NOT through
    /// AgentSandbox, which the tray menu observes — churn class).
    @ObservedObject private var transcriptStore = AgentSandbox.shared.transcriptStore
    @State private var input = ""
    @State private var running = false
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if sandbox.isEnabled {
                transcript
                Divider()
                inputBar
            } else {
                disabledHint
            }
        }
        .frame(minWidth: 580, minHeight: 420)
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: "shippingbox.fill")
                .foregroundStyle(sandbox.guestRunning ? Color.green : .secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text("Sandbox Terminal").font(.headline)
                Text(sandbox.guestRunning ? "Guest running — commands run in an isolated Linux VM"
                                          : "Idle — the guest boots on the first command")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            Spacer()
            if sandbox.guestRunning {
                Button {
                    sandbox.teardown()
                } label: {
                    Label("Stop guest", systemImage: "stop.circle")
                }
                .controlSize(.small)
                .help("Shut the guest down. It re-boots on the next command.")
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
    }

    // MARK: Transcript

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(transcriptStore.entries) { entry in
                        entryView(entry).id(entry.id)
                    }
                    Color.clear.frame(height: 1).id(bottomID)
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(NSColor.textBackgroundColor))
            .onChange(of: transcriptStore.entries.count) { _, _ in
                withAnimation(.easeOut(duration: 0.15)) { proxy.scrollTo(bottomID, anchor: .bottom) }
            }
        }
    }

    private let bottomID = "sandbox-terminal-bottom"

    @ViewBuilder
    private func entryView(_ e: AgentSandbox.Entry) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            if e.source == .system {
                Text(e.output)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            } else {
                HStack(spacing: 6) {
                    Text(promptLabel(e.source))
                        .font(.caption.monospaced().weight(.semibold))
                        .foregroundStyle(e.source == .user ? Color.green : Color.blue)
                    Text(e.command)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                }
                let body = e.output.trimmingCharacters(in: .newlines)
                if !body.isEmpty {
                    Text(body)
                        .font(.caption.monospaced())
                        .foregroundStyle(.primary.opacity(0.9))
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if e.timedOut {
                    Text("[timed out]").font(.caption2.monospaced()).foregroundStyle(.orange)
                } else if let code = e.exitCode, code != 0 {
                    Text("[exit \(code)]").font(.caption2.monospaced()).foregroundStyle(.red)
                }
            }
        }
    }

    private func promptLabel(_ s: AgentSandbox.Entry.Source) -> String {
        switch s {
        case .agent: return "agent $"
        case .user:  return "you $"
        case .system: return "•"
        }
    }

    // MARK: Input

    private var inputBar: some View {
        HStack(spacing: 8) {
            Text("$").font(.body.monospaced()).foregroundStyle(.secondary)
            TextField("Run a command in the sandbox…", text: $input)
                .textFieldStyle(.plain)
                .font(.body.monospaced())
                .focused($inputFocused)
                .onSubmit(run)
                .disabled(running)
            if running {
                ProgressView().controlSize(.small)
            } else {
                Button("Run", action: run)
                    .keyboardShortcut(.return, modifiers: [])
                    .disabled(input.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
        .onAppear { inputFocused = true }
    }

    private func run() {
        let cmd = input
        guard !cmd.trimmingCharacters(in: .whitespaces).isEmpty, !running else { return }
        input = ""
        running = true
        Task {
            await sandbox.runUserCommand(cmd)
            running = false
            inputFocused = true
        }
    }

    // MARK: Disabled

    private var disabledHint: some View {
        VStack(spacing: 10) {
            Image(systemName: "shippingbox").font(.largeTitle).foregroundStyle(.secondary)
            Text("The Agent Sandbox is off.").font(.headline)
            Text("Turn it on in Settings → Agent Sandbox to run commands in an isolated Linux VM.")
                .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
