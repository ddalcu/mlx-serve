import XCTest
@testable import MLXCore

/// The `computer` tool: one chat tool that drives the sandbox desktop through
/// `mlx-computer` in the guest. The argv mapping and every gate are pure, so
/// they are pinned here without a VM; `SANDBOX_DESKTOP_SMOKE=1` is the live
/// bar.
final class ComputerToolTests: XCTestCase {

    // MARK: kind / group / schema

    func testComputerIsAToggleableToolInItsOwnGroup() {
        XCTAssertEqual(AgentToolKind.computer.rawValue, "computer")
        XCTAssertEqual(AgentToolKind.computer.icon, "desktopcomputer")
        XCTAssertTrue(AgentToolKind.chatToggleable.contains(.computer))
        XCTAssertEqual(AgentToolGroup.computer.tools, [.computer])
        XCTAssertEqual(AgentToolGroup.computer.title, "Computer")
    }

    @MainActor
    func testSchemaHasOneToolWithTheActionVocabulary() throws {
        let def = try XCTUnwrap(AgentPrompt.toolDefinitions.first {
            ($0["function"] as? [String: Any])?["name"] as? String == "computer"
        })
        let fn = def["function"] as! [String: Any]
        let desc = fn["description"] as! String
        for action in ["observe", "read", "navigate", "research", "screenshot", "click", "double_click", "right_click",
                       "type", "key", "scroll", "drag", "open"] {
            XCTAssertTrue(desc.contains(action), "description must name \(action)")
        }
        XCTAssertTrue(desc.contains("Loop: look"), "small models need the loop spelled out")
        XCTAssertTrue(desc.contains("Research recipe") && desc.contains("Install recipe"), "the two recipes small models need")
        let props = (fn["parameters"] as! [String: Any])["properties"] as! [String: Any]
        for key in ["action", "url", "query", "id", "x", "y", "text", "keys", "direction", "amount", "command"] {
            XCTAssertNotNil(props[key], "missing parameter \(key)")
        }
        XCTAssertEqual((fn["parameters"] as! [String: Any])["required"] as? [String], ["action"])
        XCTAssertTrue(AgentEngine.toolExample(for: "computer").hasPrefix("{"))
    }

    // MARK: argv mapping (pure)

    private func argv(_ p: [String: String]) -> Result<[String], ComputerHandler.ArgError> {
        ComputerHandler.argv(for: p)
    }

    func testEveryActionMapsToOneGuestCommand() {
        XCTAssertEqual(try argv(["action": "observe"]).get(), ["observe"])
        XCTAssertEqual(try argv(["action": "observe", "window": "all"]).get(), ["observe", "--window", "all"])
        XCTAssertEqual(try argv(["action": "screenshot"]).get(), ["screenshot"])
        XCTAssertEqual(try argv(["action": "click", "id": "12"]).get(), ["click", "12"])
        XCTAssertEqual(try argv(["action": "click", "x": "640", "y": "400"]).get(), ["click", "640,400"])
        XCTAssertEqual(try argv(["action": "double_click", "id": "3"]).get(), ["click", "3", "--double"])
        XCTAssertEqual(try argv(["action": "right_click", "id": "3"]).get(), ["click", "3", "--button", "right"])
        XCTAssertEqual(try argv(["action": "type", "text": "uname -a"]).get(), ["type", "uname -a"])
        XCTAssertEqual(try argv(["action": "key", "keys": "ctrl+l"]).get(), ["key", "ctrl+l"])
        XCTAssertEqual(try argv(["action": "scroll", "direction": "down", "amount": "5", "id": "7"]).get(),
                       ["scroll", "down", "5", "--at", "7"])
        XCTAssertEqual(try argv(["action": "scroll", "direction": "up"]).get(), ["scroll", "up", "3"])
        XCTAssertEqual(try argv(["action": "drag", "from": "1", "to": "2"]).get(), ["drag", "1", "2"])
        XCTAssertEqual(try argv(["action": "drag", "x": "1", "y": "2", "to_x": "3", "to_y": "4"]).get(),
                       ["drag", "1,2", "3,4"])
        XCTAssertEqual(try argv(["action": "open", "command": "xfce4-terminal"]).get(), ["open", "xfce4-terminal"])
    }

    /// Small models send `id` as a bare number or with a `#`, `x`/`y` as
    /// floats, and the odd key under a synonym — all read leniently, but a
    /// call with no usable target is refused by name rather than clicked at 0,0.
    func testTargetsAreReadLenientlyAndAMissingOneIsNamed() {
        XCTAssertEqual(try argv(["action": "click", "id": "#12"]).get(), ["click", "12"])
        XCTAssertEqual(try argv(["action": "click", "element": "4"]).get(), ["click", "4"])
        XCTAssertEqual(try argv(["action": "click", "x": "10.7", "y": "20.2"]).get(), ["click", "10,20"])
        guard case .failure(let why) = argv(["action": "click"]) else { return XCTFail("no target must fail") }
        XCTAssertTrue(why.message.contains("id") && why.message.contains("x"), why.message)
        guard case .failure(let why2) = argv(["action": "type"]) else { return XCTFail("type needs text") }
        XCTAssertTrue(why2.message.contains("text"), why2.message)
        guard case .failure(let why3) = argv(["action": "dance"]) else { return XCTFail("unknown action") }
        XCTAssertTrue(why3.message.contains("dance"), why3.message)
        guard case .failure(let why4) = argv(["action": "scroll"]) else { return XCTFail("scroll needs direction") }
        XCTAssertTrue(why4.message.contains("direction"), why4.message)
        guard case .failure = argv([:]) else { return XCTFail("action is required") }
    }

    /// The argv becomes ONE guest shell line; every argument is single-quoted
    /// so typed text with quotes, `$` or `;` reaches xdotool as text.
    func testGuestCommandQuotesEveryArgument() {
        let cmd = ComputerHandler.guestCommand(["type", "echo 'hi'; $HOME"])
        XCTAssertTrue(cmd.hasPrefix("DISPLAY=:0 mlx-computer "), cmd)
        XCTAssertTrue(cmd.contains(VzGuest.shellQuote("echo 'hi'; $HOME")), cmd)
    }

    // MARK: gates, in order, each a named refusal

    private func handler(sandbox: Bool = true, desktop: String? = nil, vision: Bool = true,
                         run: @escaping (String, Double) async throws -> String = { _, _ in "ok" }) -> ComputerHandler {
        var h = ComputerHandler()
        h.sandboxEnabled = { sandbox }
        h.desktopRefusal = { desktop }
        h.desktopPending = { false }
        h.modelHasVision = { vision }
        h.ensureDesktop = { }
        h.runInGuest = run
        return h
    }

    func testSandboxOffIsRefusedFirst() async {
        do {
            _ = try await handler(sandbox: false, desktop: "desktop off").execute(parameters: ["action": "observe"], workingDirectory: nil)
            XCTFail("must throw")
        } catch {
            XCTAssertTrue("\(error)".contains("Agent Sandbox"), "\(error)")
        }
    }

    func testDesktopRefusalIsForwardedByName() async {
        do {
            _ = try await handler(desktop: "the sandbox desktop is still installing (Get:3 xfwm4)").execute(parameters: ["action": "observe"], workingDirectory: nil)
            XCTFail("must throw")
        } catch {
            XCTAssertTrue("\(error)".contains("still installing"), "\(error)")
        }
    }

    /// A screenshot is VISION INPUT. On a text-only model it would be a base64
    /// blob the model cannot read, so it is refused by name and the model is
    /// pointed back at observe.
    func testScreenshotRefusesOnATextOnlyModelButObserveStillWorks() async throws {
        let h = handler(vision: false)
        do {
            _ = try await h.execute(parameters: ["action": "screenshot"], workingDirectory: nil)
            XCTFail("must throw")
        } catch {
            XCTAssertTrue("\(error)".contains("vision"), "\(error)")
            XCTAssertTrue("\(error)".contains("observe"), "\(error)")
        }
        let ok = try await h.execute(parameters: ["action": "observe"], workingDirectory: nil)
        XCTAssertEqual(ok, "ok")
    }

    /// A refusal that only means "not up yet" gets ONE provisioning attempt
    /// (a fresh launch with the desktop on), then re-reads the state.
    func testAPendingDesktopIsProvisionedOnceBeforeActing() async throws {
        var refusal: String? = "the sandbox desktop is not set up yet"
        var provisioned = 0
        var h = handler()
        h.desktopRefusal = { refusal }
        h.desktopPending = { true }
        h.ensureDesktop = { provisioned += 1; refusal = nil }
        let out = try await h.execute(parameters: ["action": "observe"], workingDirectory: nil)
        XCTAssertEqual(out, "ok")
        XCTAssertEqual(provisioned, 1)
    }

    func testAnAllowedActionRunsTheGuestCommandAndReturnsItsOutput() async throws {
        var seen: String?
        let h = handler { cmd, _ in seen = cmd; return "[cwd: /Users/x/ws]\n[1] frame \"Terminal\"\n[exit code: 1]" }
        let out = try await h.execute(parameters: ["action": "click", "id": "4"], workingDirectory: nil)
        XCTAssertEqual(out, "[1] frame \"Terminal\"", "the shell framing is stripped")
        XCTAssertEqual(seen, ComputerHandler.guestCommand(["click", "4"]))
        XCTAssertEqual(ComputerHandler.stripShellFraming("[cwd: /a]\nOK"), "OK")
        XCTAssertEqual(ComputerHandler.stripShellFraming("plain"), "plain")
    }

    /// The two research primitives a 3B model can drive: go somewhere, read
    /// what is there. `navigate` takes a url or plain words (a web search).
    func testNavigateAndReadMapToTheHelper() {
        XCTAssertEqual(try argv(["action": "navigate", "url": "https://wikipedia.org"]).get(), ["navigate", "https://wikipedia.org"])
        XCTAssertEqual(try argv(["action": "navigate", "query": "raspberry pi 5 price"]).get(), ["navigate", "raspberry pi 5 price"])
        XCTAssertEqual(try argv(["action": "search", "text": "doom"]).get(), ["navigate", "doom"])
        XCTAssertEqual(try argv(["action": "read"]).get(), ["read"])
        XCTAssertEqual(try argv(["action": "read", "max": "800"]).get(), ["read", "--max", "800"])
        guard case .failure(let why) = argv(["action": "navigate"]) else { return XCTFail("navigate needs a target") }
        XCTAssertTrue(why.message.contains("url") && why.message.contains("query"), why.message)
    }

    /// `compute`, `observ`, `computr`: a clipped name gets the real one named
    /// with its example, never a bare "Unknown tool" dead end.
    @MainActor
    func testNearestToolNameForClippedNames() {
        XCTAssertEqual(AgentEngine.nearestToolName("compute"), "computer")
        XCTAssertEqual(AgentEngine.nearestToolName("computr"), "computer")
        XCTAssertEqual(AgentEngine.nearestToolName("readfile"), "readFile")
        XCTAssertNil(AgentEngine.nearestToolName("dance"))
        XCTAssertNil(AgentEngine.nearestToolName("ls"), "too short to guess")
    }

    func testProvisioningShipsFirefoxWithQuietPolicies() {
        XCTAssertTrue(SandboxDesktop.packages.contains("firefox-esr"))
        XCTAssertEqual(SandboxDesktop.provisionVersion, 2, "firefox joined the set: existing rootfs must re-run apt")
        XCTAssertTrue(SandboxDesktop.provisionScript.contains("/etc/firefox-esr/policies/policies.json"))
        XCTAssertTrue(SandboxDesktop.firefoxPolicies.contains("\"DisableSessionRestore\": true"))
        XCTAssertNotNil(try? JSONSerialization.jsonObject(with: Data(SandboxDesktop.firefoxPolicies.utf8)), "policies must be valid JSON")
        XCTAssertTrue(SandboxDesktop.promptLine.contains("navigate"), "the prompt teaches the research recipe")
        XCTAssertTrue(SandboxDesktop.promptLine.contains("apt-get install"), "and the install recipe")
    }

    /// `research` is the multi-site loop in code: the model names the topic
    /// and a site count, the guest visits them; the timeout scales with it.
    func testResearchMapsWithACappedSiteCountAndAScaledTimeout() {
        XCTAssertEqual(try argv(["action": "research", "query": "raspberry pi 5", "sites": "5"]).get(),
                       ["research", "raspberry pi 5", "--sites", "5"])
        XCTAssertEqual(try argv(["action": "research", "query": "doom"]).get(), ["research", "doom", "--sites", "3"])
        XCTAssertEqual(try argv(["action": "research", "topic": "x", "sites": "99"]).get(), ["research", "x", "--sites", "40"])
        guard case .failure = argv(["action": "research"]) else { return XCTFail("research needs a query") }
        XCTAssertEqual(ComputerHandler.timeout(for: ["research", "x", "--sites", "40"]), 660)
        XCTAssertEqual(ComputerHandler.timeout(for: ["research", "x", "--sites", "3"]), 105)
        XCTAssertEqual(ComputerHandler.timeout(for: ["click", "1"]), 60)
    }

    func testAnInventedInstallActionPointsAtTheShellRecipe() {
        guard case .failure(let why) = argv(["action": "install", "command": "chocolate-doom"]) else { return XCTFail() }
        XCTAssertTrue(why.message.contains("apt-get install -y chocolate-doom"), why.message)
        XCTAssertTrue(why.message.contains("shell"), why.message)
    }

    /// Not a computer-tool rule, but the same live run: `apt-get install doom`
    /// fails by name and a small model retries it verbatim. The shell tool
    /// appends the real candidates from apt-cache.
    func testAptMissingPackageGetsCandidates() {
        XCTAssertEqual(ShellHandler.missingAptPackage(in: "Reading state information...\nE: Unable to locate package doom\n[exit code: 100]"), "doom")
        XCTAssertNil(ShellHandler.missingAptPackage(in: "Setting up doom (1.0)"))
        XCTAssertEqual(ShellHandler.aptSuggestCommand("doom"), "apt-cache search --names-only 'doom' 2>/dev/null | head -8")
        let note = ShellHandler.aptSuggestionNote(missing: "doom", searchOutput: "chocolate-doom - Doom engine\nfreedoom - free content")
        XCTAssertTrue(note.contains("chocolate-doom") && note.contains("apt-get install -y <name>"), note)
        XCTAssertTrue(ShellHandler.aptSuggestionNote(missing: "x", searchOutput: "").contains("apt-cache search"))
        XCTAssertEqual(ShellHandler.stripShellFrame("[cwd: /a]\nchocolate-doom - x\n[exit code: 0]"), "chocolate-doom - x")
    }

    func testASuccessfulAptInstallNamesTheCommandsItBrought() {
        XCTAssertEqual(ShellHandler.aptInstalledPackages(inCommand: "apt-get update && sudo apt-get install -y chocolate-doom freedoom 2>&1"),
                       ["chocolate-doom", "freedoom"])
        XCTAssertEqual(ShellHandler.aptInstalledPackages(inCommand: "apt install --no-install-recommends libreoffice-calc; echo done"),
                       ["libreoffice-calc"])
        XCTAssertEqual(ShellHandler.aptInstalledPackages(inCommand: "ls -la"), [])
        XCTAssertEqual(ShellHandler.aptInstalledPackages(inCommand: "pip install requests"), [])
        XCTAssertTrue(ShellHandler.aptCommandsCommand(["chocolate-doom"]).contains("dpkg -L"))
        let note = ShellHandler.aptCommandsNote(packages: ["chocolate-doom"], listing: "/usr/games/chocolate-doom\n/usr/games/chocolate-setup")
        XCTAssertTrue(note.contains("chocolate-doom, chocolate-setup"), note)
        XCTAssertTrue(note.contains("computer open"), note)
        XCTAssertEqual(ShellHandler.aptCommandsNote(packages: ["x"], listing: ""), "")
    }

    /// `libreoffice-calc` alone installs a LibreOffice whose windows the tool
    /// cannot read (no GTK plugin → no a11y tree): the install note names the
    /// missing package; with it installed, nothing is added.
    func testALibreOfficeInstallWithoutTheGtkPluginIsNamed() {
        let note = ShellHandler.aptCommandsNote(packages: ["libreoffice-calc"], listing: "/usr/bin/libreoffice")
        XCTAssertTrue(note.contains("libreoffice-gtk3"), note)
        let ok = ShellHandler.aptCommandsNote(packages: ["libreoffice-calc", "libreoffice-gtk3"], listing: "/usr/bin/libreoffice")
        XCTAssertFalse(ok.contains("needs libreoffice-gtk3"), ok)
        XCTAssertNil(ShellHandler.missingLibreOfficeGtk(packages: ["chocolate-doom"]))
    }

    func testTypeAcceptsTheCommandSynonym() {
        XCTAssertEqual(try argv(["action": "type", "command": "uname -a"]).get(), ["type", "uname -a"])
    }

    // MARK: approval + repetition

    /// Not read-only, not path-confinable: the shell bucket. Auto under
    /// fullAuto/yolo, asks under readOnly/workspace.
    func testApprovalPolicyPutsComputerInTheShellBucket() {
        func decide(_ a: TaskAutonomy) -> ApprovalDecision {
            ApprovalPolicy.decide(tool: "computer", autonomy: a, arguments: ["action": "click", "id": "1"],
                                  rawArguments: "", workingDirectory: "/tmp/x")
        }
        XCTAssertEqual(decide(.yolo), .allow)
        XCTAssertEqual(decide(.fullAuto), .allow)
        guard case .ask = decide(.workspace) else { return XCTFail("workspace asks") }
        guard case .ask = decide(.readOnly) else { return XCTFail("readOnly asks") }
        XCTAssertFalse(ApprovalPolicy.readOnlyTools.contains("computer"))
    }

    /// `observe` after every action is the SAME call over and over by design;
    /// the repetition guard must not block it.
    func testComputerIsExemptFromTheRepetitionGuard() {
        XCTAssertTrue(AgentEngine.exemptTools.contains("computer"))
    }

    // MARK: prompt

    func testPromptLineNamesTheScreenAndTheLoop() {
        let s = AgentPrompt.executionEnvironmentSection(sandboxed: true, desktop: true)
        XCTAssertTrue(s.contains("1280x800"), s)
        XCTAssertTrue(s.contains("`computer`"), s)
        XCTAssertTrue(s.contains("observe first"), s)
        XCTAssertFalse(AgentPrompt.executionEnvironmentSection(sandboxed: true, desktop: false).contains("computer"))
        XCTAssertFalse(AgentPrompt.executionEnvironmentSection(sandboxed: false, desktop: true).contains("computer"),
                       "no desktop without the sandbox")
    }

    // MARK: guest install

    func testComputerScriptResolvesFromTheBundleThenTheRepo() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let bundled = tmp.appendingPathComponent("guest/mlx-computer.py")
        try FileManager.default.createDirectory(at: bundled.deletingLastPathComponent(), withIntermediateDirectories: true)
        try "x".write(to: bundled, atomically: true, encoding: .utf8)
        XCTAssertEqual(AgentSandbox.computerScriptPath(environment: [:], bundleResourceURL: tmp, executableURL: nil),
                       bundled.path)
        XCTAssertEqual(AgentSandbox.computerScriptPath(environment: ["MLX_COMPUTER_PATH": bundled.path],
                                                        bundleResourceURL: nil, executableURL: nil), bundled.path)
        XCTAssertNil(AgentSandbox.computerScriptPath(environment: [:], bundleResourceURL: nil, executableURL: nil))
        XCTAssertEqual(AgentSandbox.computerGuestPath, "/usr/local/bin/mlx-computer")
    }
}

/// A screenshot only helps if the model SEES it. Tool results are emitted as
/// text-only `tool` messages, and the server reads images from the last USER
/// message — so a browse/computer screenshot attached to the tool message
/// was never sent (pre-existing gap, found live 2026-09-05 with the desktop).
/// The fix: when the window ends on a tool result carrying images, the nudge
/// user message that follows every tool result carries them.
@MainActor
final class ToolScreenshotVisionTests: XCTestCase {
    private func history(_ msgs: [ChatMessage]) -> [[String: Any]] {
        AgentEngine.buildAgentHistory(messages: msgs, contextLength: 32000, maxTokens: 1024,
                                      buildMultimodalContent: { text, images in
                                          ["text": text, "images": images.count]
                                      })
    }

    func testATrailingToolScreenshotRidesTheNudgeUserMessage() {
        var call = ChatMessage(role: .assistant, content: "")
        call.toolCalls = [MLXCore.SerializedToolCall(id: "c1", name: "computer", arguments: "{\"action\":\"screenshot\"}")]
        var tool = ChatMessage(role: .system, content: "[screenshot captured]")
        tool.toolCallId = "c1"; tool.toolName = "computer"
        tool.images = [ChatImage(data: Data([0xFF, 0xD8, 0xFF]))]
        let h = history([ChatMessage(role: .user, content: "look"), call, tool])
        XCTAssertEqual(h.last?["role"] as? String, "user", "the nudge carries the picture: \(h)")
        let content = h.last?["content"] as? [String: Any]
        XCTAssertEqual(content?["images"] as? Int, 1)
        XCTAssertTrue((content?["text"] as? String ?? "").contains("screenshot"), "\(String(describing: content))")
        XCTAssertEqual(h.dropLast().last?["role"] as? String, "tool", "the tool result itself stays text")
    }

    func testAnOlderScreenshotIsNotResent() {
        var call = ChatMessage(role: .assistant, content: "")
        call.toolCalls = [MLXCore.SerializedToolCall(id: "c1", name: "computer", arguments: "{}")]
        var tool = ChatMessage(role: .system, content: "[screenshot captured]")
        tool.toolCallId = "c1"; tool.toolName = "computer"
        tool.images = [ChatImage(data: Data([1]))]
        let h = history([ChatMessage(role: .user, content: "look"), call, tool,
                         ChatMessage(role: .assistant, content: "I see a terminal.")])
        XCTAssertEqual(h.last?["role"] as? String, "assistant")
        XCTAssertFalse(h.contains { ($0["content"] as? [String: Any])?["images"] != nil })
    }

    /// A message typed mid-turn lands after the round's tool result
    /// (`ChatTurnEngine.deliverMidTurnMessages`), so the window ends on a USER
    /// message: it is emitted last, as the model's newest instruction, and no
    /// synthetic nudge (screenshot or "Continue.") is appended after it.
    func testATrailingMidTurnUserMessageIsTheLastItemWithNoNudge() {
        var call = ChatMessage(role: .assistant, content: "")
        call.toolCalls = [MLXCore.SerializedToolCall(id: "c1", name: "computer", arguments: "{\"action\":\"screenshot\"}")]
        var tool = ChatMessage(role: .system, content: "[screenshot captured]")
        tool.toolCallId = "c1"; tool.toolName = "computer"
        tool.images = [ChatImage(data: Data([0xFF, 0xD8, 0xFF]))]
        let h = history([ChatMessage(role: .user, content: "look"), call, tool,
                         ChatMessage(role: .user, content: "I solved the captcha, go on")])
        XCTAssertEqual(h.last?["role"] as? String, "user")
        let text = (h.last?["content"] as? String) ?? ((h.last?["content"] as? [String: Any])?["text"] as? String) ?? ""
        XCTAssertTrue(text.contains("captcha"), "the user's own words are the last item: \(h)")
        XCTAssertFalse(text.contains("screenshot"), "no synthetic nudge rides a real user message")
        XCTAssertEqual(h.filter { ($0["role"] as? String) == "user" }.count, 2)
    }

    func testNoMultimodalBuilderMeansNoSyntheticUserMessage() {
        var tool = ChatMessage(role: .system, content: "x")
        tool.toolCallId = "c1"; tool.images = [ChatImage(data: Data([1]))]
        let h = AgentEngine.buildAgentHistory(messages: [ChatMessage(role: .user, content: "look"), tool],
                                              contextLength: 32000, maxTokens: 1024)
        XCTAssertEqual(h.last?["role"] as? String, "tool")
    }
}
