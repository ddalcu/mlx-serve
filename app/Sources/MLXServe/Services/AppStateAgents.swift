import Foundation
import SwiftUI

/// The app-side half of agents (personas): the defaults every turn falls back
/// to, and what actually happens when the user picks one — model, workspace,
/// voice.
///
/// Kept out of `AppState.swift` because it is one feature's worth of wiring, and
/// every method here is a thin adapter over a PURE decision (`AgentResolution`,
/// `AgentModelSwitch`, `AgentWorkspaceSwitch`, `ActiveAgentVoice`) that is tested
/// on its own.
@MainActor
extension AppState {

    /// The app's global voice as an `AgentVoice`, so resolution can treat "the
    /// agent didn't pick one" and "the user's setting" uniformly.
    var globalVoice: AgentVoice {
        switch serverOptions.voiceEngine {
        case .system: return .system(voice.selectedVoiceId ?? "")
        case .kokoro: return .kokoro(serverOptions.kokoroVoice)
        case .clone:  return .clone(serverOptions.voiceClonePath)
        }
    }

    /// Fold an agent's overrides into the app defaults for ONE turn.
    ///
    /// The surface passes its own toggle state (a chat tab's Think/Tools/MCP, the
    /// tray's, a task's) as the defaults; the agent overrides what it declared.
    /// With `agentId == nil` this returns those defaults verbatim — the
    /// unchanged-on-upgrade guarantee.
    func resolvedAgentSettings(agentId: UUID?,
                               toolsEnabled: Bool = false,
                               mcpEnabled: Bool = false,
                               thinkingEnabled: Bool = false,
                               autoApprove: Bool = false,
                               workingDirectory: String? = nil,
                               modelPath: String? = nil,
                               disabledTools: Set<AgentToolKind> = [],
                               reasoningEffort: ReasoningEffort = .low) -> ResolvedAgentSettings {
        let defaults = AppDefaultsSnapshot(
            toolsEnabled: toolsEnabled,
            mcpEnabled: mcpEnabled,
            thinkingEnabled: thinkingEnabled,
            autoApprove: autoApprove,
            tools: Set(AgentToolKind.allCases),
            // The chat tab's own Tools menu. Surfaces without one (tasks,
            // Telegram, the tray) pass nothing and are unaffected.
            disabledTools: disabledTools,
            workingDirectory: workingDirectory,
            // nil = whatever model is selected right now. A surface with its own
            // pin (a scheduled task) passes it here and the agent's pin overrides.
            modelPath: modelPath,
            temperature: serverOptions.defaultTemperature,
            maxTokens: maxTokens,
            voice: globalVoice,
            wakePhrase: WakeWord.normalizePhrase(serverOptions.wakePhrase) ?? WakeWord.defaultPhrase,
            reasoningEffort: reasoningEffort)
        return AgentResolution.resolve(agent: agents.agent(id: agentId), defaults: defaults)
    }

    /// The agent a chat tab is talking to (nil = none).
    func agent(forSession id: UUID?) -> Agent? {
        guard let id, let session = chatSessions.first(where: { $0.id == id }) else { return nil }
        return agents.agent(id: session.agentId)
    }

    /// Start a NEW chat as `agentId` (nil = the app's own defaults).
    ///
    /// A session's agent is decided here and never again: there is deliberately
    /// no `setAgent(_:forSession:)` any more. Switching mid-thread left half a
    /// conversation running under someone else's prompt, tools, model and voice,
    /// with nothing but the transcript to show where the seam was — and the
    /// composer's Think/Tools/MCP discs flipping under you as it happened. The
    /// choice belongs next to New Chat, which is where the button lives.
    ///
    /// Editing the agent still applies live: every turn re-reads
    /// `AgentResolution`, so turning its thinking on in the editor turns it on
    /// for the conversation already in progress.
    @discardableResult
    func startChat(withAgent agentId: UUID?) -> UUID {
        let id = newChatSession(agentId: agentId)
        // Model, workspace and voice all live OUTSIDE the turn — a session that
        // only carried the id would run the persona against whatever model
        // happened to be loaded.
        let workingDirectory = chatSessions.first { $0.id == id }?.workingDirectory
        Task { await applyAgentSelection(agentId, previousWorkingDirectory: workingDirectory) }
        return id
    }

    /// Open the Agents window ON a specific agent.
    ///
    /// The window is a single reused instance that otherwise lands on whoever
    /// sorts first, which is the wrong agent every time the user got here from a
    /// locked composer disc that just named a different one.
    func openAgentSettings(_ agentId: UUID, using openWindow: OpenWindowAction) {
        pendingAgentSelection = agentId
        AppActivation.openWindow(id: "agents", using: openWindow)
    }

    /// Everything that has to happen OUTSIDE the turn when an agent becomes
    /// active: its voice, its workspace, its model.
    func applyAgentSelection(_ agentId: UUID?, previousWorkingDirectory: String?) async {
        let agent = agents.agent(id: agentId)

        // Voice: from the very next sentence (the synthesizer re-reads per
        // utterance), nil = follow the app's own voice settings.
        ActiveAgentVoice.set(agent?.resolvedVoice)

        // Workspace: an agent switch is EXPLICIT intent, so it remounts even
        // under a live pinned CLI session — restarting those sessions in place.
        // It deliberately does NOT touch the global default workspace.
        if let path = agent?.workingDirectory {
            SecurityScopedBookmark.startAccessOnce(
                name: SecurityScopedBookmark.agentWorkspaceName(agent!.id))
            if case .remount(let target, let restart) =
                AgentWorkspaceSwitch.decide(from: previousWorkingDirectory, to: path) {
                AgentSandbox.shared.noteWorkspaceChanged(target, restartPinnedSessions: restart)
            }
        }

        // Model: load a pinned one, pass a LAN id straight through, and never
        // start a multi-GB download on our own initiative.
        switch agentModelDecision(for: agent) {
        case .noChange, .needsDownload, .unavailable:
            break
        case .load(let path):
            await useModelAndAwaitReady(atPath: path)
        case .lan(let id):
            server.lanChatModelId = id
        }
    }

    /// Is this agent usable right now, and what would picking it do?
    func agentModelDecision(for agent: Agent?) -> AgentModelSwitch.Decision {
        AgentModelSwitch.decide(modelPath: agent?.modelPath,
                                selectedModelPath: selectedModelPath,
                                downloadedPaths: localModels.map(\.path),
                                lanModelIds: server.lanModels(capability: "chat").map(\.name))
    }
}
