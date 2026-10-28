import AppKit
import RetrainKit
import UserNotifications

/// Status item + menu (design §3): Run now / Settings… / Open log /
/// Erase all data… / Quit. Run now evaluates the locked trigger policy;
/// pipeline EXECUTION arrives at P2 (never half-implemented here).

final class AppDelegate: NSObject, NSApplicationDelegate, UNUserNotificationCenterDelegate {
    private var statusItem: NSStatusItem?
    private var ledger: Ledger!
    private var state: AppState!
    private var runLog: RunLog!
    private var notifier: Notifier!
    private var settingsController: SettingsWindowController?

    // MARK: lifecycle

    func applicationDidFinishLaunching(_ notification: Notification) {
        let dir = Ledger.defaultDirectory()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true, attributes: nil)
        ledger = Ledger(directory: dir)
        state = ledger.load()
        runLog = RunLog(directory: dir)
        notifier = Notifier()
        runLog.info("App started (P1 skeleton, bundle \(Bundle.main.bundleIdentifier ?? "none"))")
        state = ledger.update { _ in }
        buildStatusItem()
        notifier.setPresentationDelegate(self)
        notifier.requestAuthorization { [weak self] granted in
            if !granted {
                self?.runLog.warning("Notification authorization not granted — signal falls back to the log only")
            }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        if let runLog = runLog {
            runLog.info("App terminating")
        }
    }

    // MARK: menu bar

    private func buildStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = item.button {
            if let image = NSImage(systemSymbolName: "arrow.trianglehead.2.clockwise.rotate.90",
                                   accessibilityDescription: "Retrain orchestrator") {
                image.isTemplate = true
                button.image = image
                button.title = " \(statusLabel())"
            } else {
                button.title = "↻ \(statusLabel())"
            }
            button.font = NSFont.menuBarFont(ofSize: 0)
        }
        let menu = NSMenu()
        menu.autoenablesItems = false
        menu.addItem(withTitle: "Run now", action: #selector(runNow(_:)), keyEquivalent: "")
        menu.addItem(withTitle: "Settings…", action: #selector(openSettings(_:)), keyEquivalent: ",")
        menu.addItem(.separator())
        menu.addItem(withTitle: "Open log", action: #selector(openLog(_:)), keyEquivalent: "")
        menu.addItem(withTitle: "Erase all data…", action: #selector(eraseAllData(_:)), keyEquivalent: "")
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "Quit Retrain Orchestrator",
                              action: #selector(NSApplication.terminate(_:)),
                              keyEquivalent: "q")
        menu.addItem(quit)
        item.menu = menu
        statusItem = item
    }

    private func statusLabel() -> String {
        guard let finished = state.lastTrainFinishedAt ?? state.lastTrainStartedAt else {
            return "idle"
        }
        let f = DateFormatter()
        f.dateFormat = "d MMM HH:mm"
        return "idle · last \(f.string(from: finished))"
    }

    private func refreshMenuTitle() {
        guard let button = statusItem?.button else { return }
        let label = statusLabel()
        if button.image != nil {
            button.title = " \(label)"
        } else {
            button.title = "↻ \(label)"
        }
    }

    // MARK: actions

    @objc private func runNow(_ sender: Any?) {
        // P1: evaluate the locked trigger policy and report. No pipeline, no
        // network — the corrections delta becomes real when P2 wires the
        // classifier's counts (ledger baseline field already exists).
        let current = ledger.load()
        let decision = TriggerPolicy.evaluate(state: current, newCorrections: 0, now: Date())
        runLog.info("Run now: \(describe(decision)) [pipeline execution lands in P2]")
        switch decision {
        case .fire(let reason):
            notifier.postIfEnabled(
                .runStarted, state: current,
                title: "Retrain trigger: would fire",
                body: "\(reason). The pipeline itself lands in P2."
            )
        case .skip(let reason, let detail):
            let title = reason == .noNewTrainingData
                ? "Retrain skip: no new training data"
                : "Retrain not due"
            notifier.postIfEnabled(
                .skipped, state: current,
                title: title, body: "\(detail). (Pipeline lands in P2.)"
            )
        }
        refreshMenuTitle()
    }

    private func describe(_ decision: TriggerDecision) -> String {
        switch decision {
        case .fire(let reason):
            return "fire (\(reason))"
        case .skip(let reason, let detail):
            return "skip: \(reason.rawValue) (\(detail))"
        }
    }

    @objc private func openSettings(_ sender: Any?) {
        if settingsController == nil {
            settingsController = SettingsWindowController(
                ledger: ledger,
                runLog: runLog,
                onSaved: { [weak self] in
                    self?.applySavedSettings()
                }
            )
        }
        settingsController?.present()
    }

    @objc private func openLog(_ sender: Any?) {
        if !FileManager.default.fileExists(atPath: runLog.fileURL.path) {
            runLog.info("Log opened by the operator")
        }
        NSWorkspace.shared.open(runLog.fileURL)
    }

    @objc private func eraseAllData(_ sender: Any?) {
        let alert = NSAlert()
        alert.messageText = "Erase all data?"
        alert.informativeText = "This empties the app's sandboxed data directory (state.json, logs, everything). Settings reset to defaults. The login-item toggle is not affected."
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Erase")
        alert.addButton(withTitle: "Cancel")
        alert.buttons.first?.hasDestructiveAction = true
        if alert.runModal() != .alertFirstButtonReturn {
            runLog.info("Erase all data: cancelled")
            return
        }
        do {
            let n = try Erase.wipe(directory: ledger.directory)
            state = ledger.load()  // fresh defaults, persisted again
            runLog.warning("Erase all data: wiped \(n) item(s) in \(ledger.directory.path); fresh state created")
            notifier.postAlways(
                title: "Erase complete",
                body: "All retrain-orchestrator data was erased; settings are back to defaults."
            )
            settingsController?.reload()
        } catch {
            runLog.error("Erase all data FAILED: \(error)")
            let fail = NSAlert()
            fail.messageText = "Erase failed"
            fail.informativeText = "\(error)"
            fail.runModal()
        }
    }

    // MARK: settings plumbing

    private func applySavedSettings() {
        state = ledger.load()
        runLog.info("Settings saved (threshold \(state.correctionsThreshold), cadence \(state.weeklyCadenceEnabled ? "\(state.weeklyCadenceDays)d" : "off"), autonomy \(state.autonomy.rawValue))")
        refreshMenuTitle()
    }

    // MARK: notification presentation while running

    func userNotificationCenter(_ center: UNUserNotificationCenter,
                                willPresent notification: UNNotification) async -> UNNotificationPresentationOptions {
        // Menu-bar apps must present banners themselves while alive.
        return [.banner, .sound]
    }
}
