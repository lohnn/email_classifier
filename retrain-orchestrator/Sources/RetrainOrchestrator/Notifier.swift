import Foundation
import UserNotifications
import RetrainKit

/// Native notifications with per-category switches and PINNED failures
/// (design §6): a failure is delivered even when other categories are
/// muted. Category switches live in state.json (Settings window).

enum NotificationCategory: String, CaseIterable {
    case runStarted
    case runFinished
    case runFailed
    case prOpened
    case deployVerified
    case skipped
}

final class Notifier {
    // lazy: UNUserNotificationCenter.current() requires a bundled app —
    // creating one in a bare-binary dev launch raises. The bundle guards in
    // requestAuthorization/post keep it uncreated in that case.
    private lazy var center = UNUserNotificationCenter.current()
    private var authorized = false

    /// Ask once; the answer is remembered for this session only (macOS
    /// tracks it system-wide) — a "no" simply downgrades signal to the log.
    func requestAuthorization(completion: ((Bool) -> Void)? = nil) {
        // UNUserNotificationCenter is only meaningful inside a real bundle
        // (never crash in a bare-binary dev launch).
        guard Bundle.main.bundleIdentifier != nil else {
            authorized = false
            completion?(false)
            return
        }
        center.requestAuthorization(options: [.alert, .sound]) { [weak self] granted, _ in
            DispatchQueue.main.async {
                self?.authorized = granted
                completion?(granted)
            }
        }
    }

    /// Menu-bar apps must present banners themselves while alive — the
    /// presentation delegate lives here, owned by the AppDelegate.
    func setPresentationDelegate(_ delegate: UNUserNotificationCenterDelegate) {
        guard Bundle.main.bundleIdentifier != nil else { return }
        center.delegate = delegate
    }

    /// Deliver unless the category is muted in Settings. Failures bypass
    /// the switches entirely (pinned).
    func postIfEnabled(_ category: NotificationCategory, state: AppState,
                       title: String, body: String) {
        if category != .runFailed {
            let enabled: Bool
            switch category {
            case .runStarted: enabled = state.notifications.runStarted
            case .runFinished: enabled = state.notifications.runFinished
            case .runFailed: enabled = true
            case .prOpened: enabled = state.notifications.prOpened
            case .deployVerified: enabled = state.notifications.deployVerified
            case .skipped: enabled = state.notifications.skipped
            }
            if !enabled { return }
        }
        post(title: title, body: body, failure: category == .runFailed)
    }

    /// Pinned failure path: delivered regardless of any switch state.
    func postFailure(title: String, body: String) {
        post(title: title, body: body, failure: true)
    }

    /// Category-less signal (e.g. erase-complete): always delivered.
    func postAlways(title: String, body: String) {
        post(title: title, body: body, failure: false)
    }

    private func post(title: String, body: String, failure: Bool) {
        guard authorized, Bundle.main.bundleIdentifier != nil else { return }
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        if failure {
            // Pinned semantics = delivered even when its category switch is
            // muted (handled in postIfEnabled). Critical-alert TIME SENSITIVE
            // is an Apple-granted entitlement — plain .default is the
            // sandboxed app's correct loudness.
            content.sound = .default
            content.userInfo = ["pinned": true]
        }
        let request = UNNotificationRequest(
            identifier: "retrain.\(UUID().uuidString)",
            content: content,
            trigger: nil  // deliver now
        )
        center.add(request) { error in
            if let error = error {
                NSLog("Notification failed: \(error)")
            }
        }
    }
}
