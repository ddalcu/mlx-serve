import Combine
import Foundation
import SwiftUI

@MainActor
class AppState: ObservableObject {
    @Published var server = ServerManager()
    private var cancellables = Set<AnyCancellable>()
    @Published var downloads = DownloadManager()
    @Published var localModels: [LocalModel] = []
    @Published var selectedModelPath: String = "" {
        didSet { UserDefaults.standard.set(selectedModelPath, forKey: "selectedModelPath") }
    }
    @Published var chatSessions: [ChatSession] = []
    @Published var activeChatId: UUID?
    @Published var agentMemory = AgentMemory()
    @Published var toolExecutor = ToolExecutor()
    let testServer = TestServer()
    @Published var python = PythonManager()
    lazy var imageGen = ImageGenService(python: python)
    lazy var videoGen = VideoGenService(python: python)
    @Published var hasSeenWelcome: Bool {
        didSet { UserDefaults.standard.set(hasSeenWelcome, forKey: "hasSeenWelcome") }
    }
    @Published var autoStartServer: Bool {
        didSet { UserDefaults.standard.set(autoStartServer, forKey: "autoStartServer") }
    }
    @Published var maxTokens: Int {
        didSet { UserDefaults.standard.set(maxTokens, forKey: "maxTokens") }
    }
    @Published var contextSize: Int {
        didSet { UserDefaults.standard.set(contextSize, forKey: "contextSize") }
    }

    private let historyPath: String = {
        let dir = NSString(string: "~/.mlx-serve").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return (dir as NSString).appendingPathComponent("chat-history.json")
    }()

    init() {
        self.hasSeenWelcome = UserDefaults.standard.bool(forKey: "hasSeenWelcome")
        self.autoStartServer = UserDefaults.standard.bool(forKey: "autoStartServer")
        self.selectedModelPath = UserDefaults.standard.string(forKey: "selectedModelPath") ?? ""
        let stored = UserDefaults.standard.integer(forKey: "maxTokens")
        self.maxTokens = stored > 0 ? stored : 32768
        self.contextSize = UserDefaults.standard.integer(forKey: "contextSize") // 0 = auto
        server.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &cancellables)

        refreshModels()
        loadChatHistory()
        if ProcessInfo.processInfo.environment["TESTING_MODE"] != nil {
            testServer.start(appState: self)
        }
        AgentEngine.cleanupOverflowFiles()

        // Show welcome window on first launch
        if !hasSeenWelcome {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                Self.showWelcomeWindow {
                    self?.hasSeenWelcome = true
                }
            }
        }

        // Auto-start server if enabled and a model is available
        if autoStartServer, !selectedModelPath.isEmpty {
            server.start(modelPath: selectedModelPath, contextSize: contextSize)
        }

        // Fallback health detection — runs detached to avoid blocking MainActor
        if autoStartServer {
            let checkPort = server.port
            let mgr = server
            Task.detached {
                let api = APIClient()
                for _ in 0..<120 {
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                    if let ok = try? await api.checkHealth(port: checkPort), ok {
                        await mgr.forceRunning()
                        return
                    }
                }
            }
        }
    }

    func refreshModels() {
        localModels = downloads.discoverLocalModels()
        // Auto-select a model if none selected or current selection is invalid
        if localModels.first(where: { $0.path == selectedModelPath }) == nil,
           let first = localModels.first {
            selectedModelPath = first.path
        }
    }

    // MARK: - Chat Session Management

    func newChatSession() -> UUID {
        let session = ChatSession()
        chatSessions.insert(session, at: 0)
        activeChatId = session.id
        saveChatHistory()
        return session.id
    }

    func deleteSession(_ id: UUID) {
        chatSessions.removeAll { $0.id == id }
        if activeChatId == id {
            activeChatId = chatSessions.first?.id
        }
        saveChatHistory()
    }

    var activeSession: ChatSession? {
        get { chatSessions.first { $0.id == activeChatId } }
        set {
            if let newValue, let idx = chatSessions.firstIndex(where: { $0.id == newValue.id }) {
                chatSessions[idx] = newValue
            }
        }
    }

    func appendMessage(to sessionId: UUID, message: ChatMessage) {
        guard let idx = chatSessions.firstIndex(where: { $0.id == sessionId }) else { return }
        chatSessions[idx].messages.append(message)
        chatSessions[idx].updatedAt = Date()
        // Auto-title from first user message
        if chatSessions[idx].title == "New Chat",
           message.role == .user,
           !message.content.isEmpty {
            let title = String(message.content.prefix(40))
            chatSessions[idx].title = title + (message.content.count > 40 ? "..." : "")
        }
    }

    func updateLastMessage(in sessionId: UUID, content: String? = nil, reasoning: String? = nil, streaming: Bool? = nil, usage: TokenUsage? = nil) {
        guard let sIdx = chatSessions.firstIndex(where: { $0.id == sessionId }),
              !chatSessions[sIdx].messages.isEmpty else { return }
        let mIdx = chatSessions[sIdx].messages.count - 1
        if let content { chatSessions[sIdx].messages[mIdx].content += content }
        if let usage {
            chatSessions[sIdx].messages[mIdx].promptTokens = usage.promptTokens
            chatSessions[sIdx].messages[mIdx].completionTokens = usage.completionTokens
            chatSessions[sIdx].messages[mIdx].tokensPerSecond = usage.tokensPerSecond
        }
        if let reasoning { chatSessions[sIdx].messages[mIdx].reasoningContent = (chatSessions[sIdx].messages[mIdx].reasoningContent ?? "") + reasoning }
        if let streaming { chatSessions[sIdx].messages[mIdx].isStreaming = streaming }
    }

    // MARK: - Agent Helpers

    func updatePlanStatus(in sessionId: UUID, planId: UUID, status: PlanStatus) {
        guard let sIdx = chatSessions.firstIndex(where: { $0.id == sessionId }) else { return }
        for mIdx in chatSessions[sIdx].messages.indices {
            if chatSessions[sIdx].messages[mIdx].agentPlan?.id == planId {
                chatSessions[sIdx].messages[mIdx].agentPlan?.status = status
                break
            }
        }
    }

    func appendToolResults(to sessionId: UUID, results: [StepResult]) {
        guard let sIdx = chatSessions.firstIndex(where: { $0.id == sessionId }) else { return }
        for mIdx in chatSessions[sIdx].messages.indices.reversed() {
            if chatSessions[sIdx].messages[mIdx].role == .assistant {
                chatSessions[sIdx].messages[mIdx].toolResults = results
                break
            }
        }
    }

    // MARK: - Persistence

    func saveChatHistory() {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = .prettyPrinted
        guard let data = try? encoder.encode(chatSessions) else { return }
        try? data.write(to: URL(fileURLWithPath: historyPath))
    }

    private func loadChatHistory() {
        guard FileManager.default.fileExists(atPath: historyPath),
              let data = try? Data(contentsOf: URL(fileURLWithPath: historyPath)) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        chatSessions = (try? decoder.decode([ChatSession].self, from: data)) ?? []
        activeChatId = chatSessions.first?.id
    }

    // MARK: - Welcome Window

    private static func showWelcomeWindow(onDismiss: @escaping () -> Void) {
        let view = WelcomeView(onDismiss: onDismiss)
        let hostingView = NSHostingView(rootView: view)

        // Let SwiftUI compute the intrinsic size
        let fittingSize = hostingView.fittingSize
        hostingView.frame = NSRect(origin: .zero, size: fittingSize)

        let window = NSWindow(
            contentRect: NSRect(origin: .zero, size: fittingSize),
            styleMask: [.titled, .closable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.contentView = hostingView
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.isMovableByWindowBackground = true
        window.center()
        window.isReleasedWhenClosed = false
        window.level = .floating
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        _welcomeWindow = window
    }

    private static var _welcomeWindow: NSWindow?
}
