import Foundation
import UserNotifications

/// What the app does when the server process dies on its own — a crash, an OOM
/// kill, a fatal Zig error — while it was supposed to be serving.
///
/// Before #265 every unexpected exit ended the same way: a modal crash alert and
/// a dead server. That is right when the user is sitting in front of the Mac,
/// and useless at 3 a.m. when the agent CLIs (and whatever scheduled task is
/// mid-run) just start failing every request. `.restartQuietly` and
/// `.restartAndTellMe` relaunch the same launch with a bounded backoff
/// (`CrashRestartPolicy`) instead; `.ask` keeps the old behavior and stays the
/// default, so nothing changes for anyone who never opens Settings.
enum CrashRecoveryMode: String, CaseIterable, Identifiable {
    case ask
    case restartQuietly
    case restartAndTellMe

    var id: String { rawValue }

    var label: String {
        switch self {
        case .ask: return "Ask"
        case .restartQuietly: return "Restart"
        case .restartAndTellMe: return "Restart + Notify"
        }
    }

    /// False for `.ask`: the caller presents the crash alert instead of
    /// relaunching.
    var restartsAutomatically: Bool { self != .ask }

    /// True for the mode that also posts a system notification. Best effort:
    /// notifications are only delivered once the user has granted them (the
    /// task scheduler asks at startup), and an ungranted mode degrades to
    /// `.restartQuietly`, never to a lost restart.
    var postsNotification: Bool { self == .restartAndTellMe }

    /// `@AppStorage`-backed display pref, like the Interface rows: there is no
    /// server-side flag for "what the app does when the server dies", so this
    /// deliberately does NOT ride `ServerOptions` / the CLI.
    static let prefKey = "crashRecoveryMode"

    static var current: CrashRecoveryMode {
        CrashRecoveryMode(rawValue: UserDefaults.standard.string(forKey: prefKey) ?? "") ?? .ask
    }
}

/// What to do about one unexpected exit.
enum CrashReaction: Equatable {
    case alert
    case restart(after: TimeInterval)
}

/// Bounded exponential backoff for relaunching a server that keeps dying.
///
/// Two independent guards against a relaunch treadmill:
///
/// * the delay grows (1, 2, 4, 8, 16, 30 s), so a crash that resolves itself
///   gets a fast retry while a spin gets progressively less noisy;
/// * the budget is finite (6 attempts in a streak). Once it is spent the app
///   falls back to the crash alert, so a model that CANNOT load (corrupt
///   weights, too big for this Mac, a bad flag) surfaces its log instead of
///   restart-looping forever behind a spinner.
///
/// A run that stayed healthy for `stableUptime` is not a crash loop, so the
/// next crash gets the whole budget back — otherwise a long-lived server that
/// crashed twice in a week would run out of attempts.
struct CrashRestartPolicy {
    /// 120 s: longer than any model load on a supported Mac, shorter than a
    /// coffee break. Below this the previous run counts as part of the streak.
    static let stableUptime: TimeInterval = 120

    static let standard = CrashRestartPolicy(delays: [1, 2, 4, 8, 16, 30])

    let delays: [TimeInterval]

    /// Attempts spent in the CURRENT streak. Reset by a stable run, by a
    /// user-initiated start, and by Stop.
    private(set) var attempts = 0

    init(delays: [TimeInterval]) { self.delays = delays }

    var budget: Int { delays.count }

    /// True when `uptime` counts as a healthy run rather than part of a streak.
    static func countsAsStable(uptime: TimeInterval) -> Bool { uptime >= stableUptime }

    /// Spend the next attempt. nil = this streak has used up its budget.
    mutating func nextDelay() -> TimeInterval? {
        guard attempts < delays.count else { return nil }
        defer { attempts += 1 }
        return delays[attempts]
    }

    mutating func reset() { attempts = 0 }

    /// The whole reaction to an unexpected exit, with the bookkeeping folded in
    /// so the loop-breaking rules are testable without a process to crash.
    ///
    /// `uptime` is how long the dead process had been healthy (nil = it never
    /// reached `.running`, e.g. it died during model load).
    mutating func reaction(mode: CrashRecoveryMode, canRelaunch: Bool,
                           uptime: TimeInterval?) -> CrashReaction {
        guard mode.restartsAutomatically, canRelaunch else { return .alert }
        if let uptime, Self.countsAsStable(uptime: uptime) { reset() }
        guard let delay = nextDelay() else { return .alert }
        return .restart(after: delay)
    }
}

/// Best-effort system notification for `.restartAndTellMe`.
///
/// Not routed through `TaskNotifier`: that one is scoped to the task scheduler
/// (it registers approval categories and deep-links into a run), and a crash
/// notice has nothing to link to. Guarded the same way, because
/// `UNUserNotificationCenter` traps with no bundle identifier (i.e. under
/// `swift test`).
enum CrashRecoveryNotifier {
    static func postRestart(exitCode: Int32, attempt: Int, budget: Int) {
        guard Bundle.main.bundleIdentifier != nil else { return }
        let content = UNMutableNotificationContent()
        content.title = L10n.text("mlx-serve exited; restarting it")
        content.body = L10n.format("Exit code %lld. Relaunch attempt %lld of %lld.",
                                   Int64(exitCode), Int64(attempt), Int64(budget))
        // No sound: a restart loop must not turn into an alarm.
        let request = UNNotificationRequest(identifier: "mlx-serve-crash-restart",
                                           content: content, trigger: nil)
        UNUserNotificationCenter.current().add(request)
    }
}
