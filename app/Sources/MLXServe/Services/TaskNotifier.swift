import Foundation
import AppKit
import UserNotifications

/// System-notification surface for scheduled tasks: completion, failure, and the
/// actionable "needs approval" prompt that drives the pause/resume flow. Also the
/// `UNUserNotificationCenterDelegate` so taps deep-link into the Tasks window and
/// the Approve/Deny actions resume a paused run.
///
/// Guarded so it is a no-op outside a real app bundle (UNUserNotificationCenter
/// crashes with no bundle identifier — e.g. under `swift test`).
final class TaskNotifier: NSObject, UNUserNotificationCenterDelegate {
    static let shared = TaskNotifier()

    /// Set at startup so notification taps can route back into the scheduler.
    weak var appState: AppState?

    // Category + action identifiers.
    private static let approvalCategory = "TASK_NEEDS_APPROVAL"
    private static let completedCategory = "TASK_COMPLETED"
    private static let failedCategory = "TASK_FAILED"
    private static let approveAction = "APPROVE"
    private static let denyAction = "DENY"

    private var available: Bool { Bundle.main.bundleIdentifier != nil }

    func requestAuthorization() {
        guard available else { return }
        let center = UNUserNotificationCenter.current()
        center.delegate = self
        center.setNotificationCategories([
            UNNotificationCategory(identifier: Self.approvalCategory, actions: [
                UNNotificationAction(identifier: Self.approveAction, title: String(localized: "Approve"),
                                     options: [.authenticationRequired]),
                UNNotificationAction(identifier: Self.denyAction, title: String(localized: "Deny"),
                                     options: [.destructive]),
            ], intentIdentifiers: [], options: []),
            UNNotificationCategory(identifier: Self.completedCategory, actions: [],
                                   intentIdentifiers: [], options: []),
            UNNotificationCategory(identifier: Self.failedCategory, actions: [],
                                   intentIdentifiers: [], options: []),
        ])
        center.requestAuthorization(options: [.alert, .sound]) { _, _ in }
    }

    // MARK: - Posting

    func notifyCompleted(task: ScheduledTask, run: TaskRun) {
        post(title: "✓ \(task.title)",
             body: run.summary ?? String(localized: "Task completed."),
             category: Self.completedCategory, task: task, run: run)
    }

    func notifyFailed(task: ScheduledTask, run: TaskRun) {
        post(title: String(localized: "⚠ \(task.title) failed"),
             body: run.summary ?? String(localized: "The task run failed."),
             category: Self.failedCategory, task: task, run: run)
    }

    func notifyNeedsApproval(task: ScheduledTask, run: TaskRun) {
        let tool = run.pendingApproval?.toolName ?? String(localized: "a tool")
        post(title: String(localized: "\(task.title) needs approval"),
             body: String(localized: "Wants to run “\(tool)”. \(run.pendingApproval?.reason ?? "")"),
             category: Self.approvalCategory, task: task, run: run)
    }

    private func post(title: String, body: String, category: String, task: ScheduledTask, run: TaskRun) {
        guard available else { return }
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = String(body.prefix(240))
        content.sound = .default
        content.categoryIdentifier = category
        content.userInfo = ["taskId": task.id.uuidString, "runId": run.id.uuidString]
        let request = UNNotificationRequest(identifier: run.id.uuidString, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(request)
    }

    // MARK: - Delegate

    func userNotificationCenter(_ center: UNUserNotificationCenter,
                                willPresent notification: UNNotification,
                                withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void) {
        completionHandler([.banner, .sound])
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter,
                                didReceive response: UNNotificationResponse,
                                withCompletionHandler completionHandler: @escaping () -> Void) {
        let info = response.notification.request.content.userInfo
        let taskId = (info["taskId"] as? String).flatMap(UUID.init)
        let runId = (info["runId"] as? String).flatMap(UUID.init)
        let action = response.actionIdentifier
        Task { @MainActor in
            guard let appState = self.appState else { completionHandler(); return }
            switch action {
            case Self.approveAction:
                if let runId { appState.taskScheduler.resume(runId: runId, approved: true) }
            case Self.denyAction:
                if let runId { appState.taskScheduler.resume(runId: runId, approved: false) }
            default:
                // Tap on the body — deep-link to the task.
                if let taskId {
                    appState.pendingTaskDeepLink = taskId
                    NSApplication.shared.activate(ignoringOtherApps: true)
                }
            }
            completionHandler()
        }
    }
}
