import Foundation

/// One field a section's "Reset" restores.
///
/// The `name` is the `ServerOptions` stored-property label; `apply` is what
/// actually runs. They're checked against each other by
/// `SettingsSectionResetTests` — the name feeds a reflection guard proving the
/// section→fields map covers every field exactly once, and a second test proves
/// the closure really resets the field its name claims.
struct SettingsFieldReset {
    let name: String
    let apply: (inout ServerOptions, ServerOptions) -> Void
}

/// Scoped "Reset to Defaults" for the Settings screen.
///
/// Before the sidebar there was one global button. With a sidebar, that button
/// sits at the bottom of whatever single section you're viewing, where "Reset to
/// Defaults" unavoidably reads as "reset THIS section" — so it now does exactly
/// that, and the confirmation says which scope it's about to apply
/// (`confirmMessage`). "All Settings" keeps the original whole-screen behavior.
///
/// The map below must stay COMPLETE: a field no section claims could never be
/// reset from the UI again, and one claimed twice would be clobbered from the
/// wrong pane. That's enforced by reflection, not by review.
enum SettingsReset {

    /// Fields restored when `category`'s Reset button is pressed. Empty for
    /// sections with nothing of their own to reset.
    static func fields(for category: SettingsCategory) -> [SettingsFieldReset] {
        switch category {
        // Model Folders' extra scan path lives on `DownloadManager.customRoot`,
        // not in ServerOptions; Updates holds no settings. Neither offers a
        // Reset button (`isResettable`), rather than a button that does nothing.
        case .modelFolders, .updates:
            return []

        case .server:
            return [
                f("host") { $0.host = $1.host },
                f("port") { $0.port = $1.port },
                f("ctxSize") { $0.ctxSize = $1.ctxSize },
                f("noVision") { $0.noVision = $1.noVision },
                f("logLevel") { $0.logLevel = $1.logLevel },
                // No UI row of its own (CLI-only), but it IS a server flag —
                // some section has to own it or it can never be restored.
                f("requestTimeout") { $0.requestTimeout = $1.requestTimeout },
                f("enableMetrics") { $0.enableMetrics = $1.enableMetrics },
                f("apiKey") { $0.apiKey = $1.apiKey },
                f("toolAutocorrect") { $0.toolAutocorrect = $1.toolAutocorrect },
                f("skipMemPreflight") { $0.skipMemPreflight = $1.skipMemPreflight },
            ]

        case .specDecode:
            return [
                f("enablePLD") { $0.enablePLD = $1.enablePLD },
                f("pldDraftLen") { $0.pldDraftLen = $1.pldDraftLen },
                f("pldKeyLen") { $0.pldKeyLen = $1.pldKeyLen },
                f("drafterPath") { $0.drafterPath = $1.drafterPath },
                f("draftBlockSize") { $0.draftBlockSize = $1.draftBlockSize },
                f("enableMTP") { $0.enableMTP = $1.enableMTP },
                f("mtpDepth") { $0.mtpDepth = $1.mtpDepth },
            ]

        // One section, so one reset: the universal knob and the MLX-only ones.
        case .performance:
            return [
                f("tokenizeCacheEntries") { $0.tokenizeCacheEntries = $1.tokenizeCacheEntries },
                f("maxConcurrent") { $0.maxConcurrent = $1.maxConcurrent },
                f("kvQuant") { $0.kvQuant = $1.kvQuant },
                f("prefixCacheEntries") { $0.prefixCacheEntries = $1.prefixCacheEntries },
                f("prefixCacheMem") { $0.prefixCacheMem = $1.prefixCacheMem },
                f("enablePrefixCacheDisk") { $0.enablePrefixCacheDisk = $1.enablePrefixCacheDisk },
                f("prefixCacheDisk") { $0.prefixCacheDisk = $1.prefixCacheDisk },
            ]

        case .ggufPerformance:
            return [
                f("llamaKvQuant") { $0.llamaKvQuant = $1.llamaKvQuant },
                f("llamaCacheEntries") { $0.llamaCacheEntries = $1.llamaCacheEntries },
            ]

        case .ds4:
            return [
                f("ssdStreaming") { $0.ssdStreaming = $1.ssdStreaming },
            ]

        case .requestDefaults:
            return [
                f("defaultMaxTokens") { $0.defaultMaxTokens = $1.defaultMaxTokens },
                f("defaultTemperature") { $0.defaultTemperature = $1.defaultTemperature },
                f("defaultTopP") { $0.defaultTopP = $1.defaultTopP },
                f("defaultTopK") { $0.defaultTopK = $1.defaultTopK },
                f("defaultRepeatPenalty") { $0.defaultRepeatPenalty = $1.defaultRepeatPenalty },
                f("defaultPresencePenalty") { $0.defaultPresencePenalty = $1.defaultPresencePenalty },
                f("defaultReasoningBudget") { $0.defaultReasoningBudget = $1.defaultReasoningBudget },
                // The next three have no row on this screen (thinking lives on
                // the chat toolbar, the per-request spec-decode overrides
                // duplicate the Speculative Decoding toggles) — but they're
                // per-request semantics, so this is their home.
                f("defaultEnableThinking") { $0.defaultEnableThinking = $1.defaultEnableThinking },
                f("perRequestEnablePLD") { $0.perRequestEnablePLD = $1.perRequestEnablePLD },
                f("perRequestEnableDrafter") { $0.perRequestEnableDrafter = $1.perRequestEnableDrafter },
            ]

        case .voice:
            return [
                f("voiceClonePath") { $0.voiceClonePath = $1.voiceClonePath },
                f("voiceCloneEnabled") { $0.voiceCloneEnabled = $1.voiceCloneEnabled },
                f("voiceCloneLabel") { $0.voiceCloneLabel = $1.voiceCloneLabel },
                f("wakePhrase") { $0.wakePhrase = $1.wakePhrase },
            ]

        case .sandbox:
            return [
                f("sandbox") { $0.sandbox = $1.sandbox },
            ]

        case .messaging:
            return [
                // The bot token is the one thing a reset must NOT take: only
                // @BotFather can reissue it, and nothing on this Mac can
                // re-derive it. Same carve-out the global reset has always made.
                f("telegram") { current, fresh in
                    let token = current.telegram.botToken
                    current.telegram = fresh.telegram
                    current.telegram.botToken = token
                },
            ]
        }
    }

    private static func f(_ name: String,
                          _ apply: @escaping (inout ServerOptions, ServerOptions) -> Void) -> SettingsFieldReset {
        SettingsFieldReset(name: name, apply: apply)
    }

    /// Does this section have anything of its own to restore? A Reset button on
    /// a section with no fields would be a button that lies.
    static func isResettable(_ category: SettingsCategory) -> Bool {
        !fields(for: category).isEmpty
    }

    /// Restore one section's fields; every other field is left exactly as it is.
    static func apply(_ category: SettingsCategory, to current: ServerOptions) -> ServerOptions {
        let fresh = ServerOptions()
        var out = current
        for field in fields(for: category) {
            field.apply(&out, fresh)
        }
        return out
    }

    /// Restore everything — the original whole-screen behavior, Telegram token kept.
    static func applyAll(to current: ServerOptions) -> ServerOptions {
        ServerOptions.resetToDefaults(preserving: current)
    }

    /// Apply whatever the sidebar has selected.
    static func apply(_ selection: SettingsSelection, to current: ServerOptions) -> ServerOptions {
        switch selection {
        case .all:                 return applyAll(to: current)
        case .category(let c):     return apply(c, to: current)
        }
    }

    // MARK: - Copy (the scope has to be stated BEFORE anything is wiped)

    static func buttonLabel(_ selection: SettingsSelection) -> String {
        switch selection {
        case .all:               return "Reset All Settings"
        case .category(let c):   return "Reset \(c.sidebarLabel)"
        }
    }

    static func confirmTitle(_ selection: SettingsSelection) -> String {
        switch selection {
        case .all:               return "Reset ALL settings to defaults?"
        case .category(let c):   return "Reset \(c.sidebarLabel) to defaults?"
        }
    }

    static func confirmMessage(_ selection: SettingsSelection) -> String {
        switch selection {
        case .all:
            return "This resets every section — Server, Speculative Decoding, Performance, Per-Request Defaults, Voice, Agent Sandbox and Messaging — not just the one you're looking at. Your Telegram bot token is kept, since only @BotFather can reissue it. The running server keeps its current flags until you hit Restart Now."
        case .category(let c):
            let scope = "This resets only the \(c.sidebarLabel) section. Every other section is left untouched."
            let tail = c == .messaging
                ? " Your Telegram bot token is kept, since only @BotFather can reissue it."
                : ""
            return scope + tail + " The running server keeps its current flags until you hit Restart Now."
        }
    }
}
