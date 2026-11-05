import Foundation
import Testing
import AppKit
import SwiftUI
@testable import MLXCore

@MainActor
@Suite(.serialized)
struct ContextCompactionTests {
    private final class SummaryStub: APIClient {
        let events: [SSEEvent]
        let failure: Bool
        init(events: [SSEEvent], failure: Bool = false) { self.events = events; self.failure = failure }
        override func streamChat(port: UInt16, messages: [[String: Any]], maxTokens: Int = 2048,
            temperature: Double = 0.8, enableThinking: Bool = false, reasoningEffort: String? = nil,
            tools: [[String: Any]]? = nil, toolsJSON: String? = nil, defaults: RequestDefaults = .none,
            retryPolicy: RetryPolicy = .default, modelId: String? = nil,
            continueFinalMessage: Bool = false) -> AsyncThrowingStream<SSEEvent, Error> {
            AsyncThrowingStream { continuation in
                for event in events { continuation.yield(event) }
                if failure { continuation.finish(throwing: URLError(.networkConnectionLost)) }
                else { continuation.finish() }
            }
        }
    }
    private final class CountingStub: APIClient {
        var inputs: [String] = []
        var count = 200
        var onCount: (() -> Void)?
        var countFails = false
        var onSummary: (() -> Void)?
        override func contextTokenCount(port: UInt16, text: String) async throws -> Int {
            inputs.append(text)
            onCount?()
            if countFails { throw URLError(.networkConnectionLost) }
            return count
        }
        override func streamChat(port: UInt16, messages: [[String: Any]], maxTokens: Int = 2048,
            temperature: Double = 0.8, enableThinking: Bool = false, reasoningEffort: String? = nil,
            tools: [[String: Any]]? = nil, toolsJSON: String? = nil, defaults: RequestDefaults = .none,
            retryPolicy: RetryPolicy = .default, modelId: String? = nil,
            continueFinalMessage: Bool = false) -> AsyncThrowingStream<SSEEvent, Error> {
            onSummary?()
            return AsyncThrowingStream { $0.finish(throwing: URLError(.networkConnectionLost)) }
        }
    }

    @Test func testCompactionCannotSendEditedOrCancelledHistory() async throws {
        guard NSHomeDirectory().contains("mlx-compaction-test-home") else { return }
        _ = NSApplication.shared
        let state = AppState(startServerAtLaunch: false)
        var session = ChatSession(title: "measurement race")
        session.messages = [ChatMessage(role: .user, content: "original")]
        state.chatSessions.append(session)
        let api = CountingStub()
        for failure in [false, true] {
            state.chatSessions[state.chatSessions.count - 1].messages[0].content = "original"
            api.countFails = failure
            api.onCount = {
                let index = state.chatSessions.firstIndex { $0.id == session.id }!
                state.chatSessions[index].messages[0].content = "edited"
            }
            do {
                _ = try await state.chatEngine.prepareCompactedContext(api: api, sessionId: session.id,
                    config: .init(agentMode: false, mcpMode: false, enableThinking: false, voiceStyle: false,
                                  workingDirectory: nil, tools: [], maxTokens: 256),
                    context: 32768, tools: String(repeating: "x", count: 90000)) { _ in
                        [["role": "user", "content": "original"]]
                    }
                Issue.record("Edited history must not send the pre-measurement request")
            } catch ContextCompaction.Failure.changedHistory { }
        }
        api.countFails = false
        api.onCount = { withUnsafeCurrentTask { $0?.cancel() } }
        let task = Task {
            try await state.chatEngine.prepareCompactedContext(api: api, sessionId: session.id,
                config: .init(agentMode: false, mcpMode: false, enableThinking: false, voiceStyle: false,
                              workingDirectory: nil, tools: [], maxTokens: 256),
                context: 32768, tools: String(repeating: "x", count: 90000)) { _ in
                    [["role": "user", "content": "edited"]]
                }
        }
        do { _ = try await task.value; Issue.record("Cancelled measurement must not send") }
        catch is CancellationError { }
        let index = state.chatSessions.firstIndex { $0.id == session.id }!
        state.chatSessions[index].messages = (0..<30).map { ChatMessage(role: .user, content: "fact \($0)") }
        api.count = 30000
        api.onCount = nil
        api.onSummary = { state.chatSessions[index].messages[0].content = "edited during summary" }
        do {
            _ = try await state.chatEngine.prepareCompactedContext(api: api, sessionId: session.id,
                config: .init(agentMode: false, mcpMode: false, enableThinking: false, voiceStyle: false,
                              workingDirectory: nil, tools: [], maxTokens: 256),
                context: 32768, tools: String(repeating: "x", count: 90000)) { _ in
                    [["role": "user", "content": "original"]]
                }
            Issue.record("Failed summary must not send history edited while it was running")
        } catch ContextCompaction.Failure.changedHistory { }
        #expect(!state.chatSessions[index].messages.contains { $0.contextCheckpoint != nil })
    }

    @Test func testUncompactableHistoryStillSendsExistingBuilderOutput() async throws {
        guard NSHomeDirectory().contains("mlx-compaction-test-home") else { return }
        _ = NSApplication.shared
        let state = AppState(startServerAtLaunch: false)
        let long = (0..<30).map { ChatMessage(role: $0 % 2 == 0 ? .user : .assistant, content: "message \($0)") }
        let plan = try #require(ContextCompaction.plan(long))
        let checkpoint = ContextCompaction.checkpoint(summary: "prior summary", source: long, boundary: plan.boundary)
        for source in [[ChatMessage(role: .user, content: "one prompt")], long, long + [checkpoint]] {
            var session = ChatSession(title: "compaction fallback")
            session.messages = source
            state.chatSessions.append(session)
            let api = CountingStub()
            api.count = 30000
            let messages = try await state.chatEngine.prepareCompactedContext(api: api, sessionId: session.id,
                config: .init(agentMode: true, mcpMode: false, enableThinking: false, voiceStyle: false,
                              workingDirectory: nil, tools: [], maxTokens: 256),
                context: 32768, tools: String(repeating: "x", count: 90000)) { _ in
                    [["role": "user", "content": "one prompt"]]
                }
            #expect(messages.last?["content"] as? String == "one prompt")
            #expect(state.chatSessions.last?.messages == source)
        }
    }

    @Test func testSmallRequestsSkipTokenizerAndLargeOnesMeasureBuiltMessages() async throws {
        let api = CountingStub()
        let small: [[String: Any]] = [["role": "user", "content": "hello"]]
        let estimate = try await ContextCompactionRunner.tokensIfNearLimit(api: api, port: 1,
            messages: small, tools: nil, context: 32768, output: 4096)
        #expect(estimate == nil)
        #expect(api.inputs.isEmpty)
        let large: [[String: Any]] = [["role": "tool", "content": String(repeating: "x", count: 90000)]]
        let measured = try await ContextCompactionRunner.tokensIfNearLimit(api: api, port: 1,
            messages: large, tools: "tool-schema", context: 32768, output: 4096)
        #expect(measured == 1240)
        #expect(api.inputs.count == 1)
        #expect(api.inputs.first?.contains("tool-schema") == true)
    }

    @Test func testEmptyTruncatedFailedAndCancelledSummaryCannotCommit() async throws {
        let source = (0..<30).map { ChatMessage(role: .user, content: "fact \($0)") }
        let plan = try #require(ContextCompaction.plan(source))
        let cases = [SummaryStub(events: []),
                     SummaryStub(events: [.content("partial"), .truncated(.maxTokens)]),
                     SummaryStub(events: [.content("partial")], failure: true)]
        for api in cases {
            do {
                _ = try await ContextCompactionRunner.summarize(source: source, boundary: plan.boundary,
                    api: api, port: 1, model: nil, context: 32768)
                Issue.record("Invalid summary must fail before a checkpoint is returned")
            } catch { }
        }
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await ContextCompactionRunner.summarize(source: source, boundary: plan.boundary,
                api: SummaryStub(events: [.content("summary")]), port: 1, model: nil, context: 32768)
        }
        do { _ = try await task.value; Issue.record("Cancelled summary must not commit") }
        catch is CancellationError { }
        #expect(!source.contains { $0.contextCheckpoint != nil })
    }
    @Test func testIncompleteAndServerErrorStreamsCannotBecomeSummaries() async throws {
        let source = (0..<30).map { ChatMessage(role: .user, content: "fact \($0)") }
        let plan = try #require(ContextCompaction.plan(source))
        let cases: [[SSEEvent]] = [[.content("partial")], [.content("[Server Error: inference failed]"), .done]]
        for events in cases {
            do {
                _ = try await ContextCompactionRunner.summarize(source: source, boundary: plan.boundary,
                    api: SummaryStub(events: events), port: 1, model: nil, context: 32768)
                Issue.record("An unfinished or failed stream must not replace the history")
            } catch { }
        }
    }

    @Test func testWireDropsIncompleteToolRoundsAndLegacyWarnings() {
        var call = ChatMessage(role: .assistant, content: "Reading")
        call.toolCalls = [MLXCore.SerializedToolCall(id: "aborted", name: "readFile", arguments: "{}")]
        let legacy = ChatMessage(role: .assistant, content: "answer" + TruncationNotice.text(cause: .maxTokens, maxTokens: 256))
        let wire = AgentEngine.buildAgentHistory(messages: [call, ChatMessage(role: .user, content: "Continue"), legacy], contextLength: 32768, maxTokens: 4096)
        #expect(wire.allSatisfy { $0["tool_calls"] == nil })
        #expect(wire.last?["content"] as? String == "answer")
    }

    @Test func testMeasuredTokensTriggerBeforeWindowIsExhausted() {
        #expect(ContextCompaction.needsCompaction(tokens: 220_000, context: 262_144, output: 16_384))
        #expect(!ContextCompaction.needsCompaction(tokens: 20_000, context: 262_144, output: 16_384))
        #expect(ContextCompaction.needsCompaction(tokens: 260_555, context: 262_144, output: 0))
    }

    @Test func testSingleUserThirtyToolRoundsCanCompactWithinTheTurn() throws {
        let user = ChatMessage(role: .user, content: "Keep working on TASK-771")
        var source = [user]
        for i in 0..<30 {
            var call = ChatMessage(role: .assistant, content: "")
            call.toolCalls = [MLXCore.SerializedToolCall(id: "call-\(i)", name: "readFile", arguments: "{}")]
            var result = ChatMessage(role: .system, content: "result \(i)")
            result.toolCallId = "call-\(i)"
            source += [call, result]
        }
        let plan = try #require(ContextCompaction.plan(source))
        #expect(plan.boundary > 0)
        #expect(source[plan.boundary + 1].toolCalls?.isEmpty == false)
        source.append(ContextCompaction.checkpoint(summary: "TASK-771 remains unfinished", source: source, boundary: plan.boundary))
        let active = ContextCompaction.activeMessages(source)
        #expect(active[1].id == user.id)
        let wire = AgentEngine.buildAgentHistory(messages: source, contextLength: 32768, maxTokens: 4096)
        #expect(wire.first?["role"] as? String == "user")
        #expect(wire.first?["content"] as? String == user.content)
        #expect(wire.compactMap { $0["tool_call_id"] as? String } == (24..<30).map { "call-\($0)" })
        for i in 30..<40 {
            var call = ChatMessage(role: .assistant, content: "")
            call.toolCalls = [MLXCore.SerializedToolCall(id: "call-\(i)", name: "readFile", arguments: "{}")]
            var result = ChatMessage(role: .system, content: "result \(i)")
            result.toolCallId = "call-\(i)"
            source += [call, result]
        }
        let next = try #require(ContextCompaction.plan(source))
        #expect(next.boundary > plan.boundary)
        source.append(ContextCompaction.checkpoint(summary: "TASK-771 still unfinished", source: source, boundary: next.boundary))
        let continued = AgentEngine.buildAgentHistory(messages: source, contextLength: 32768, maxTokens: 4096)
        #expect(continued.first?["content"] as? String == user.content)
        #expect(continued.compactMap { $0["tool_call_id"] as? String } == (34..<40).map { "call-\($0)" })
    }

    @Test func testSummaryExcludesFailedAttemptsErrorsAndEmptyPlaceholders() throws {
        var failed = ChatMessage(role: .assistant, content: "failed fiction")
        failed.failedRetry = true
        var error = ChatMessage(role: .assistant, content: "error fiction")
        error.errorNotice = ChatErrorNotice(kind: .contextOverflow, message: "failure", neededTokens: 300, contextLength: 256)
        var placeholder = ChatMessage(role: .assistant, content: "")
        placeholder.isStreaming = true
        let legacy = ChatMessage(role: .assistant, content: "couldn't generate a response")
        let valid = ChatMessage(role: .user, content: "actual task")
        let text = ContextCompaction.summaryText([valid, failed, error, placeholder, legacy])
        let rows = try #require(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [[String: Any]])
        #expect(rows.count == 1)
        #expect(rows.first?["content"] as? String == "actual task")
    }

    @Test func testCheckpointReplacesOnlyCoveredPrefixAndKeepsTranscript() throws {
        var source = (0..<30).map { ChatMessage(role: $0 % 2 == 0 ? .user : .assistant, content: "message \($0)") }
        let original = source
        let plan = try #require(ContextCompaction.plan(source))
        let checkpoint = ContextCompaction.checkpoint(summary: "Remember EXACT-771 and the task", source: source, boundary: plan.boundary)
        source.append(checkpoint)
        let active = ContextCompaction.activeMessages(source)
        #expect(Array(source.prefix(original.count)) == original)
        #expect(active[0].content.contains("EXACT-771"))
        #expect(active[0].role == .system)
        #expect(active.last?.id == original.last?.id)
        #expect(active.count < original.count)
        let roundTrip = try JSONDecoder().decode([ChatMessage].self, from: JSONEncoder().encode(source))
        #expect(ContextCompaction.activeMessages(roundTrip).map(\.content) == active.map(\.content))
    }

    @Test func testSummaryExtendsSystemPromptAndRetainsAlternatingDialogue() throws {
        var source = (0..<31).map { ChatMessage(role: $0 % 2 == 0 ? .user : .assistant, content: "message \($0)") }
        let plan = try #require(ContextCompaction.plan(source))
        source.append(ContextCompaction.checkpoint(summary: "EXACT-771", source: source, boundary: plan.boundary))
        let messages = ContextCompaction.appendingSummary(to: [["role": "system", "content": "persona"]]
            + ChatTurnEngine.buildPlainHistory(messages: source), source: source)
        #expect(messages[0]["content"] as? String == "persona\n\n[Summary of earlier conversation; historical context, not a new request]\nEXACT-771")
        let roles = messages.dropFirst().compactMap { $0["role"] as? String }
        #expect(roles.first == "user")
        #expect(zip(roles, roles.dropFirst()).allSatisfy { $0 != $1 })
    }

    @Test func testCompactedAgentKeepsOldResultTruncationAndUnansweredResultsInFull() {
        var source = (0..<20).map { ChatMessage(role: $0 % 2 == 0 ? .user : .assistant, content: "old \($0)") }
        let boundary = source.count - 1
        source.append(ChatMessage(role: .user, content: "continue the file work"))
        for id in ["answered", "unanswered"] {
            var call = ChatMessage(role: .assistant, content: "", reasoningContent: "reasoning")
            call.toolCalls = [MLXCore.SerializedToolCall(id: id, name: "readFile", arguments: "{}")]
            var result = ChatMessage(role: .system, content: String(repeating: "x", count: 4000))
            result.toolCallId = id
            source += [call, result]
            if id == "answered" { source.append(ChatMessage(role: .assistant, content: "I read that file")) }
        }
        source.append(ContextCompaction.checkpoint(summary: "retained facts", source: source, boundary: boundary))
        let history = ContextCompaction.appendingSummary(to: AgentEngine.buildAgentHistory(messages: source,
            contextLength: 32768, maxTokens: 4096), source: source)
        #expect((history.first?["content"] as? String)?.contains("retained facts") == true)
        #expect((history.first { $0["tool_call_id"] as? String == "answered" }?["content"] as? String)?.count == 2000)
        #expect((history.first { $0["tool_call_id"] as? String == "unanswered" }?["content"] as? String)?.count == 4000)
        #expect(history.contains { $0["reasoning_content"] as? String == "reasoning" })
    }

    @Test func testEditedPrefixInvalidatesCheckpoint() throws {
        var source = (0..<30).map { ChatMessage(role: .user, content: "message \($0)") }
        let plan = try #require(ContextCompaction.plan(source))
        source.append(ContextCompaction.checkpoint(summary: "old facts", source: source, boundary: plan.boundary))
        source[0].content = "corrected fact"
        #expect(ContextCompaction.activeMessages(source).first?.content == "corrected fact")
    }

    @Test func testCheckpointRendersAsAMessageAlongsideToolCards() throws {
        let user = ChatMessage(role: .user, content: "Keep the project marker EXACT-771 and output path /tmp/pelican.html.")
        let reply = ChatMessage(role: .assistant, content: "I’ll preserve those details while continuing the task.")
        let marker = ContextCompaction.checkpoint(
            summary: "Project marker: EXACT-771. Output: /tmp/pelican.html. The animation task remains unfinished.",
            source: [user, reply], boundary: 0)
        var toolCard = ChatMessage(role: .assistant, content: "**readFile**(path: /tmp/pelican.html)")
        toolCard.isAgentSummary = true
        let rows = ChatRowBuilder.rows(from: [toolCard, marker])
        guard rows.count == 2, case .toolCall = rows[0], case .message(let rendered) = rows[1] else {
            Issue.record("A context checkpoint must remain a message instead of joining a tool card")
            return
        }
        #expect(rendered.id == marker.id)
        guard let directory = ProcessInfo.processInfo.environment["MLX_COMPACTION_UI_SNAPSHOTS"] else { return }
        for (name, messages) in [("before", [user, reply]), ("after", [user, marker, reply])] {
            let view = VStack(alignment: .leading, spacing: 20) {
                ForEach(messages) { MessageBubble(message: $0) }
            }.padding(24).frame(width: 760).background(Color(nsColor: .windowBackgroundColor))
            let host = NSHostingView(rootView: view.environment(\.colorScheme, .light))
            host.appearance = NSAppearance(named: .aqua)
            host.frame = NSRect(origin: .zero, size: host.fittingSize)
            let window = NSWindow(contentRect: host.frame, styleMask: [.borderless], backing: .buffered, defer: false)
            window.contentView = host
            host.layoutSubtreeIfNeeded()
            let bitmap = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
            host.cacheDisplay(in: host.bounds, to: bitmap)
            let png = try #require(bitmap.representation(using: .png, properties: [:]))
            try png.write(to: URL(fileURLWithPath: directory).appendingPathComponent("context-summary-\(name).png"))
        }
    }

    @Test func testBoundaryNeverSplitsToolRound() throws {
        var source = (0..<20).map { ChatMessage(role: .user, content: "message \($0)") }
        var call = ChatMessage(role: .assistant, content: "")
        call.toolCalls = [MLXCore.SerializedToolCall(id: "call-a", name: "readFile", arguments: "{}")]
        var result = ChatMessage(role: .system, content: "file result")
        result.toolCallId = "call-a"
        source.append(call)
        source.append(result)
        source += (0..<10).map { ChatMessage(role: .assistant, content: "recent \($0)") }
        let plan = try #require(ContextCompaction.plan(source))
        let suffix = source.suffix(from: plan.boundary + 1)
        let hasResult = suffix.contains { $0.toolCallId == "call-a" }
        let hasCall = suffix.contains { $0.toolCalls?.contains { $0.id == "call-a" } == true }
        #expect(hasCall == hasResult)
    }

    @Test func testTinyHistoryCannotBeSilentlyDiscarded() {
        #expect(ContextCompaction.plan([ChatMessage(role: .user, content: String(repeating: "x", count: 100_000))]) == nil)
    }

    @Test func testMediaCountingDoesNotTokenizeBase64() throws {
        let payload = String(repeating: "A", count: 200_000)
        let input = ContextCompactionRunner.countingInput([["role": "user", "content": [
            ["type": "text", "text": "inspect this"],
            ["type": "image_url", "image_url": ["url": "data:image/jpeg;base64," + payload]]
        ]]])
        let data = try JSONSerialization.data(withJSONObject: input.messages)
        #expect(data.count < 500)
        #expect(input.mediaTokens == 8192)
    }

    @Test func testRepeatedCheckpointAndDeletedPrefix() throws {
        var source = (0..<30).map { ChatMessage(role: .user, content: "fact \($0)") }
        let first = try #require(ContextCompaction.plan(source))
        source.append(ContextCompaction.checkpoint(summary: "first summary", source: source, boundary: first.boundary))
        source += (0..<20).map { ChatMessage(role: .user, content: "new fact \($0)") }
        let second = try #require(ContextCompaction.plan(source))
        source.append(ContextCompaction.checkpoint(summary: "second summary includes first", source: source, boundary: second.boundary))
        #expect(ContextCompaction.activeMessages(source).first?.content.contains("second summary") == true)
        source.removeFirst()
        #expect(ContextCompaction.activeMessages(source).first?.content == "fact 1")
    }

    @Test func testWirePreservesReasoningToolArgumentsAndFullResults() {
        var call = ChatMessage(role: .assistant, content: "read", reasoningContent: "reasoning")
        call.toolCalls = [MLXCore.SerializedToolCall(id: "parallel-a", name: "readFile", arguments: "{\"path\":\"/tmp/a\"}")]
        var result = ChatMessage(role: .system, content: String(repeating: "完整结果", count: 2000))
        result.toolCallId = "parallel-a"
        var placeholder = ChatMessage(role: .assistant, content: "")
        placeholder.isStreaming = true
        let wire = AgentEngine.buildAgentHistory(messages: [call, result, placeholder], contextLength: 32768, maxTokens: 4096)
        #expect(wire.count == 2)
        #expect(wire[0]["reasoning_content"] as? String == "reasoning")
        #expect(wire[1]["content"] as? String == result.content)
        #expect(wire[1]["tool_call_id"] as? String == "parallel-a")
    }

    @Test func testDisplayOnlyRowsDoNotSplitParallelToolResults() {
        var call = ChatMessage(role: .assistant, content: "")
        call.toolCalls = ["a", "b"].map { MLXCore.SerializedToolCall(id: $0, name: "generate_image", arguments: "{}") }
        var first = ChatMessage(role: .system, content: "first image generated")
        first.toolCallId = "a"
        var second = ChatMessage(role: .system, content: "second image generated")
        second.toolCallId = "b"
        let displayOnly = ChatMessage(role: .assistant, content: "")
        let wire = AgentEngine.buildAgentHistory(messages: [call, first, displayOnly, second], contextLength: 32768, maxTokens: 4096)
        #expect((wire.first?["tool_calls"] as? [[String: Any]])?.count == 2)
        #expect(wire.compactMap { $0["tool_call_id"] as? String } == ["a", "b"])
    }

    @Test func testLiveModelSummaryAndContinuation() async throws {
        guard ProcessInfo.processInfo.environment["MLX_COMPACTION_LIVE_TEST"] == "1" else { return }
        let api = APIClient()
        var source = [ChatMessage(role: .user, content: "记住项目唯一验收标记 EXACT-771，输出路径 /tmp/pelican.html。任务是单文件 HTML，鹈鹕骑车，脚掌必须跟随踏板。")]
        source += (0..<29).map { ChatMessage(role: $0 % 2 == 0 ? .user : .assistant,
            content: "第 \($0) 轮：" + String(repeating: "我们核对海边动画、车轮转动和脚掌联动，任务仍未完成。", count: 30)) }
        let plan = try #require(ContextCompaction.plan(source))
        let before = try await ContextCompactionRunner.measuredTokens(api: api, port: 11234,
            messages: ChatTurnEngine.buildPlainHistory(messages: source), tools: nil)
        let marker = try await ContextCompactionRunner.summarize(source: source, boundary: plan.boundary,
            api: api, port: 11234, model: nil, context: 32768)
        source.append(marker)
        let restored = try JSONDecoder().decode([ChatMessage].self, from: JSONEncoder().encode(source))
        var messages = ContextCompaction.appendingSummary(to: ChatTurnEngine.buildPlainHistory(messages: restored), source: restored)
        let after = try await ContextCompactionRunner.measuredTokens(api: api, port: 11234, messages: messages, tools: nil)
        #expect(after < before)
        #expect(marker.content.contains("EXACT-771"))
        #expect(marker.content.contains("/tmp/pelican.html"))
        messages.append(["role": "user", "content": "只回答最早约定的验收标记和输出文件绝对路径。"])
        var response = ""
        for try await event in api.streamChat(port: 11234, messages: messages, maxTokens: 256,
                                             temperature: 0, enableThinking: false) {
            if case .content(let delta) = event { response += delta }
        }
        #expect(response.contains("EXACT-771"))
        #expect(response.contains("/tmp/pelican.html"))
        print("COMPACTION_LIVE before=\(before) after=\(after) retained=EXACT-771 response=\(response)")
    }

    @Test func testLiveTurnEngineAutomaticallyCompactsBothModes() async throws {
        guard ProcessInfo.processInfo.environment["MLX_COMPACTION_ENGINE_TEST"] == "1" else { return }
        #expect(NSHomeDirectory().contains("mlx-compaction-test-home"))
        guard NSHomeDirectory().contains("mlx-compaction-test-home") else { return }
        _ = NSApplication.shared
        let state = AppState(startServerAtLaunch: false)
        state.server.port = 11234
        state.server.status = .running
        state.selectedModelPath = ""
        state.contextSize = 32768
        for (agent, oversized) in [(false, false), (true, false), (true, true)] {
            var session = ChatSession(title: "automatic compaction integration")
            session.messages = [ChatMessage(role: .user, content: "任务标记 ENGINE-771。只记住它，不要调用任何工具。")]
            session.messages += (0..<29).map { ChatMessage(role: $0 % 2 == 0 ? .user : .assistant,
                content: "历史 \($0): " + String(repeating: "我们讨论了车轮和脚掌动画，已确认需求，还没有生成文件。", count: 70)) }
            if oversized {
                session.messages[0].content += String(repeating: "需求尚未完成，继续保留任务标记。", count: 3000)
            }
            let ids = session.messages.map(\.id)
            state.chatSessions.append(session)
            let config = ChatTurnEngine.TurnConfig(agentMode: agent, mcpMode: false, enableThinking: false,
                voiceStyle: false, workingDirectory: nil, tools: [], maxTokens: 256)
            state.chatEngine.runTurn(sessionId: session.id, userText: "只回答最早的任务标记，不要调用工具。",
                images: nil, videos: nil, audio: nil, config: config, approval: { _ in false })
            let deadline = Date().addingTimeInterval(360)
            while !state.chatEngine.activeTurnSessionIds.isEmpty && Date() < deadline {
                try await Task.sleep(nanoseconds: 100_000_000)
            }
            #expect(state.chatEngine.activeTurnSessionIds.isEmpty)
            let completed = try #require(state.chatSessions.first { $0.id == session.id })
            let shouldCompact = !agent || oversized
            #expect(completed.messages.contains { $0.contextCheckpoint != nil } == shouldCompact)
            #expect(Array(completed.messages.prefix(ids.count)).map(\.id) == ids)
            #expect(completed.messages.last?.content.contains("ENGINE-771") == true)
            #expect(completed.messages.last?.errorNotice == nil)
            let disk = try ChatStore(path: ChatStore.defaultPath, legacyHistoryPath: nil, legacySidebar: nil).load()
            #expect(disk.first(where: { $0.id == session.id })?.messages.contains { $0.contextCheckpoint != nil } == shouldCompact)
            print("ENGINE_COMPACTION mode=\(agent ? "agent" : "plain") oversized=\(oversized) retained=ENGINE-771 compacted=\(shouldCompact)")
        }
    }
}
