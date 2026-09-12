import Foundation

/// How the app handles an unexpected mlx-serve crash (#265).
enum CrashRecoveryMode: String, Codable, CaseIterable, Identifiable {
    /// Show a modal alert (today's behavior).
    case ask
    /// Auto-restart with backoff + macOS notification.
    case autoRestart
    /// Auto-restart with backoff, log only (headless-friendly).
    case silent

    var id: String { rawValue }

    var label: String {
        switch self {
        case .ask:         return "Ask (show alert)"
        case .autoRestart: return "Auto-restart"
        case .silent:      return "Auto-restart (silent)"
        }
    }

    static let defaultMode: CrashRecoveryMode = .ask
    static let defaultsKey = "crashRecoveryMode"
}

/// Pure decision logic for crash recovery — no side effects, fully testable.
enum CrashRecovery {
    static let maxRetries = 3
    static let crashWindow: TimeInterval = 300       // 5 minutes
    static let stableThreshold: TimeInterval = 300   // reset counter after 5 min uptime

    /// Whether the server should be auto-restarted after a crash.
    static func shouldAutoRestart(
        mode: CrashRecoveryMode,
        wasRunning: Bool,
        exitCode: Int32,
        isMemoryFailure: Bool,
        crashCount: Int,
        maxRetries: Int
    ) -> Bool {
        guard mode == .autoRestart || mode == .silent else { return false }
        guard wasRunning else { return false }
        guard exitCode != 0 else { return false }
        guard !isMemoryFailure else { return false }
        guard crashCount < maxRetries else { return false }
        return true
    }

    /// Exponential backoff delay in seconds: 1, 2, 4, 8 … capped at 30.
    static func backoffDelay(attempt: Int) -> TimeInterval {
        min(30.0, pow(2.0, Double(attempt - 1)))
    }

    /// Whether the crash window has expired (counter should reset).
    static func crashWindowExpired(lastCrash: Date?, window: TimeInterval) -> Bool {
        guard let last = lastCrash else { return true }
        return Date().timeIntervalSince(last) > window
    }
}
