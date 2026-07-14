import XCTest
@testable import MLXCore

/// Unit tests for `SettingsReset` — per-section resets, introduced with the
/// Settings sidebar.
///
/// The sidebar made the old single global "Reset to Defaults" button dangerous:
/// sitting alone at the bottom of the Voice pane, a button labelled "Reset to
/// Defaults" reads as "reset Voice" while it actually wiped EVERY field. So a
/// reset is now scoped to whatever the sidebar has selected, and the
/// confirmation names that scope out loud.
///
/// The load-bearing guard is `testEveryServerOptionsFieldBelongsToExactlyOneSection`:
/// a per-section reset is only safe if the section→fields map is COMPLETE. A
/// field nobody claims would survive "Reset Server" forever (a stale value the
/// user believes they cleared); a field claimed twice means one section silently
/// resets another's setting.
final class SettingsSectionResetTests: XCTestCase {

    /// Every stored property of `ServerOptions`, by reflection — so a field
    /// added next year is covered without anyone remembering to update a list.
    private var allFieldNames: Set<String> {
        Set(Mirror(reflecting: ServerOptions()).children.compactMap { $0.label })
    }

    private func value(_ options: ServerOptions, _ field: String) -> String {
        let child = Mirror(reflecting: options).children.first { $0.label == field }
        return String(describing: child?.value ?? "<missing>")
    }

    /// A ServerOptions with every field moved off its default, so a reset has
    /// something to undo.
    private func mutated() -> ServerOptions {
        var o = ServerOptions()
        o.host = "127.0.0.1"; o.port = 9999; o.ctxSize = 32768; o.noVision = true
        o.logLevel = .debug; o.requestTimeout = 5; o.enableMetrics = true
        o.apiKey = "hunter2"; o.toolAutocorrect = false; o.skipMemPreflight = true
        o.enablePLD = false; o.pldDraftLen = 9; o.pldKeyLen = 9
        o.drafterPath = "/tmp/drafter"; o.draftBlockSize = 9
        o.enableMTP = false; o.mtpDepth = 6
        o.maxConcurrent = 8; o.kvQuant = .int4; o.prefixCacheEntries = 99
        o.prefixCacheMem = "9GB"; o.enablePrefixCacheDisk = true; o.prefixCacheDisk = "99GB"
        o.llamaKvQuant = .q4; o.llamaCacheEntries = 9
        o.ssdStreaming = true
        o.tokenizeCacheEntries = 9
        o.defaultMaxTokens = 99; o.defaultTemperature = 1.9; o.defaultTopP = 0.1
        o.defaultTopK = 9; o.defaultRepeatPenalty = 1.5; o.defaultPresencePenalty = 0.5
        o.defaultReasoningBudget = 99; o.defaultEnableThinking = true
        o.perRequestEnablePLD = .on; o.perRequestEnableDrafter = .off
        o.telegram.enabled = true; o.telegram.botToken = "123:secret"
        o.telegram.allowedChatIds = [42]; o.telegram.agentMode = true
        o.sandbox.enabled = true
        o.voiceClonePath = "/tmp/clip.wav"; o.voiceCloneEnabled = false
        o.voiceCloneLabel = "My clip"; o.wakePhrase = "hey robot"
        return o
    }

    // MARK: - The completeness guard

    /// CLASS GUARD. Without this, a per-section reset is a silent-staleness
    /// machine: any field no section claims can never be reset from the UI, and
    /// any field two sections claim gets clobbered from the wrong pane.
    ///
    /// Fields with no UI row of their own still need a home — `requestTimeout`
    /// (CLI-only) and the per-request spec-decode overrides are stored but never
    /// rendered, and they must still be restored by *some* section's reset.
    func testEveryServerOptionsFieldBelongsToExactlyOneSection() {
        var owners: [String: [SettingsCategory]] = [:]
        for category in SettingsCategory.allCases {
            for field in SettingsReset.fields(for: category) {
                owners[field.name, default: []].append(category)
            }
        }

        let claimed = Set(owners.keys)
        XCTAssertEqual(claimed, allFieldNames, """
            Every ServerOptions field must be reset by exactly one section.
            Unclaimed (would survive a reset forever): \(allFieldNames.subtracting(claimed).sorted())
            Unknown names in the map: \(claimed.subtracting(allFieldNames).sorted())
            """)

        let doubled = owners.filter { $0.value.count > 1 }
        XCTAssertTrue(doubled.isEmpty, "claimed by two sections: \(doubled)")
    }

    /// The map pairs a NAME (what the guard above checks) with a CLOSURE (what
    /// actually runs). This proves they agree: resetting a section really does
    /// restore each field it claims by name.
    func testEachSectionResetActuallyRestoresTheFieldsItClaims() {
        let defaults = ServerOptions()
        for category in SettingsCategory.allCases {
            let reset = SettingsReset.apply(category, to: mutated())
            for field in SettingsReset.fields(for: category) {
                // The Telegram token is deliberately preserved — see below.
                if field.name == "telegram" { continue }
                XCTAssertEqual(value(reset, field.name), value(defaults, field.name),
                               "\(category).\(field.name) was claimed but not reset")
            }
        }
    }

    // MARK: - Scoping: a section reset touches nothing else

    func testResettingOneSectionLeavesEveryOtherFieldAlone() {
        let before = mutated()
        let after = SettingsReset.apply(.voice, to: before)

        // Voice fields went back to default…
        XCTAssertEqual(after.wakePhrase, ServerOptions().wakePhrase)
        XCTAssertEqual(after.voiceClonePath, "")
        // …and nothing else moved.
        XCTAssertEqual(after.host, "127.0.0.1")
        XCTAssertEqual(after.kvQuant, .int4)
        XCTAssertEqual(after.defaultTemperature, 1.9)
        XCTAssertTrue(after.sandbox.enabled)
        XCTAssertEqual(after.telegram.botToken, "123:secret")
    }

    /// Resetting Messaging still keeps the bot token — @BotFather is the only
    /// place it can come from, exactly as the global reset has always done.
    func testResettingMessagingKeepsTheBotToken() {
        let after = SettingsReset.apply(.messaging, to: mutated())
        XCTAssertEqual(after.telegram.botToken, "123:secret", "the token is not re-derivable")
        XCTAssertFalse(after.telegram.enabled, "everything else in the section resets")
        XCTAssertEqual(after.telegram.allowedChatIds, [])
    }

    // MARK: - Global reset is unchanged

    /// "All Settings" keeps doing exactly what the button always did.
    func testGlobalResetStillMatchesTheOldBehavior() {
        let current = mutated()
        XCTAssertEqual(SettingsReset.applyAll(to: current),
                       ServerOptions.resetToDefaults(preserving: current))
    }

    // MARK: - Sections with nothing to reset

    /// Model Folders' custom path lives on `DownloadManager`, not `ServerOptions`,
    /// and Updates has no settings at all — offering a dead "Reset" button on
    /// those panes would be a lie.
    func testSectionsWithNoFieldsAreNotResettable() {
        XCTAssertFalse(SettingsReset.isResettable(.modelFolders))
        XCTAssertFalse(SettingsReset.isResettable(.updates))
        XCTAssertTrue(SettingsReset.isResettable(.server))
        XCTAssertTrue(SettingsReset.isResettable(.voice))
    }

    // MARK: - The warning must name its scope

    /// The whole point of the user's ask: before anything is wiped, say whether
    /// this resets ONE section or EVERYTHING.
    func testConfirmationNamesItsScope() {
        let global = SettingsReset.confirmMessage(.all)
        XCTAssertTrue(global.localizedCaseInsensitiveContains("every section"),
                      "a global reset must say it goes beyond the current pane: \(global)")

        let voice = SettingsReset.confirmMessage(.category(.voice))
        XCTAssertTrue(voice.contains("Voice"))
        XCTAssertTrue(voice.localizedCaseInsensitiveContains("only"),
                      "a section reset must say it's scoped: \(voice)")
        XCTAssertFalse(voice.localizedCaseInsensitiveContains("every section"))

        XCTAssertEqual(SettingsReset.buttonLabel(.all), "Reset All Settings")
        XCTAssertEqual(SettingsReset.buttonLabel(.category(.voice)), "Reset Voice")
        XCTAssertTrue(SettingsReset.confirmTitle(.category(.server)).contains("Server"))
    }
}
