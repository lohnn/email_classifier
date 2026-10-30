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
    /// Menu-bar indicator (T3): nil = not probed yet (gray), true/false =
    /// last explicit probe verdict. Probes happen ONLY on explicit presses
    /// (Settings Test button, launcher outcomes) — no background poller;
    /// P2 wires the automatic cadence.
    private var connection: (reachable: Bool?, detail: String) = (nil, "not probed yet")
    /// M6: launcher + bootstrap orchestration.
    private let launcher = PipelineLauncher()
    private let runtimeEnsuring: RuntimeEnsuring = RuntimeBootstrapper(
        commands: ProcessCommandRunner(), fetcher: URLArtifactFetcher())
    private var passInFlight = false
    private var runtimeProvisioning = false
    private var appearanceObservation: NSKeyValueObservation?

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
        if !state.firstRunCompleted {
            // Operator decision 2026-10-29: no default classifier address —
            // Settings IS the onboarding. It reopens at every launch until
            // the first successful Save flips firstRunCompleted.
            runLog.info("First run — opening Settings for initial setup (classifier URL unset by design)")
            openSettings(nil)
        }
        // Composite dot must regenerate when the menubar appearance flips.
        appearanceObservation = NSApp.observe(\.effectiveAppearance, options: []) { [weak self] _, _ in
            DispatchQueue.main.async { self?.refreshStatusImage() }
        }
        refreshStatusImage()
        // Consented first-launch runtime bootstrap (receipt short-circuits
        // when already provisioned). Provisioned state gates Run now.
        runtimeProvisioning = true
        Task { [weak self] in
            guard let self = self else { return }
            do {
                let manifest = try RuntimeManifest.bundled()
                let receipt = try await self.runtimeEnsuring.ensureRuntime(
                    manifest: manifest, containerDir: Ledger.defaultDirectory(), log: self.runLog)
                self.runtimeProvisioning = false
                self.runLog.info("Runtime provisioning complete (uv \(receipt.uvVersion), llama \(receipt.llamaVersion), torch \(receipt.torchVersion))")
                self.notifier.postAlways(title: "Runtime ready",
                                         body: "uv \(receipt.uvVersion), python \(receipt.pythonVersion), llama-server \(receipt.llamaVersion), torch \(receipt.torchVersion) — Run now can go.")
            } catch {
                self.runtimeProvisioning = false
                self.runLog.error("Runtime provisioning FAILED: \(error) — Run now will retry it")
                self.notifier.postFailure(title: "Runtime bootstrap failed",
                                          body: "Run now will retry. Check the log (Open log) for the full error.")
            }
        }
    }

    /// The launcher assembles-and-runs; failures NEVER throw — they come
    /// back as pinned outcomes. Dot + notifications render here (app-side).
    private func executeLaunch(phase: LaunchPhase) async -> LaunchOutcome {
        await launcher.execute(phase: phase, ledger: ledger,
                               containerDir: Ledger.defaultDirectory(),
                               runID: runIDForPhase(phase),
                               log: runLog)
    }

    private func runIDForPhase(_ phase: LaunchPhase) -> String {
        if phase == .resume, let pending = launcher.pendingRunID(runsDir: Ledger.defaultDirectory().appendingPathComponent("runs")) {
            return pending
        }
        return PipelineLauncher.newRunID()
    }

    private func launchPhaseForNow() -> LaunchPhase {
        launcher.phase(runsDir: Ledger.defaultDirectory().appendingPathComponent("runs"))
    }

    private func applyOutcome(_ outcome: LaunchOutcome) {
        if let reachable = outcome.dotReachable {
            connection = (reachable, outcome.headline)
        }
        refreshStatusImage()
        if outcome.pinnedFailure {
            notifier.postFailure(title: outcome.headline, body: outcome.detail)
        } else {
            notifier.postAlways(title: outcome.headline, body: outcome.detail)
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
        refreshStatusImage()
    }

    private func statusLabel() -> String {
        guard let finished = state.lastTrainFinishedAt ?? state.lastTrainStartedAt else {
            return "idle"
        }
        let f = DateFormatter()
        f.dateFormat = "d MMM HH:mm"
        return "idle · last \(f.string(from: finished))"
    }

    /// One writer for the status-bar visuals: the composite image (round-14
    /// lesson — menubar ignores attributed-text colors, so the dot is drawn
    /// as pixels) + adaptive title + tooltip. Regenerated on appearance
    /// change via the effectiveAppearance observer.
    private func refreshStatusImage() {
        guard let button = statusItem?.button else { return }
        let dotState: ConnectionState
        switch connection.reachable {
        case .some(true): dotState = .reachable
        case .some(false): dotState = .failed
        case .none: dotState = .unknown
        }
        let isDark = NSApp.effectiveAppearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        button.image = StatusIconImage.composite(
            glyphName: "arrow.trianglehead.2.clockwise.rotate.90",
            dotState: dotState,
            isDark: isDark
        )
        button.title = " \(statusLabel())"
        button.toolTip = "Retrain orchestrator — classifier: \(connection.detail)"
    }

    // MARK: actions

    @objc private func runNow(_ sender: Any?) {
        // M6: Run now = provision runtime (receipt short-circuits) →
        // launcher pass. Run-now-only until the operator validates one
        // full manual run; the unattended timer is a follow-up commit.
        guard !passInFlight else {
            runLog.warning("Run now ignored — a pass is already in flight")
            notifier.postAlways(title: "Already running",
                                body: "A retrain pass is in flight — check the menu bar for the live state.")
            return
        }
        if runtimeProvisioning {
            runLog.warning("Run now deferred — runtime still provisioning (first run downloads ~170 MB)")
            notifier.postAlways(title: "Runtime still installing",
                                body: "The embedded runtime (~170 MB) is downloading into the sandbox. Try Run now again in a few minutes.")
            return
        }
        passInFlight = true
        let phase = launchPhaseForNow()
        Task { [weak self] in
            guard let self = self else { return }
            self.runLog.info("Run now: executing \(phase == .resume ? "ship pass (resume)" : "training pass")")
            let outcome = await self.executeLaunch(phase: phase)
            self.passInFlight = false
            self.applyOutcome(outcome)
        }
    }

    @objc private func openSettings(_ sender: Any?) {
        if settingsController == nil {
            settingsController = SettingsWindowController(
                ledger: ledger,
                runLog: runLog,
                onSaved: { [weak self] in
                    self?.applySavedSettings()
                },
                onProbe: { [weak self] reachable, detail in
                    guard let self = self else { return }
                    self.connection = (reachable, detail)
                    self.refreshStatusImage()
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
        let pre = Erase.inventory(directory: ledger.directory)
        let alert = NSAlert()
        alert.messageText = "Erase all data?"
        alert.informativeText = """
            This empties the app's sandboxed data directory:
            \(ledger.directory.path)

            \(pre.files) file(s), \(pre.bytes) byte(s) would be removed; fresh defaults will be rewritten afterwards. The login-item toggle is not affected.
            """
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
            // The dialog promised "fresh defaults will be rewritten" — make
            // that an actual write (update, not load: load alone never
            // persists anything).
            state = ledger.update { _ in }
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
        refreshStatusImage()
    }

    // MARK: notification presentation while running

    func userNotificationCenter(_ center: UNUserNotificationCenter,
                                willPresent notification: UNNotification) async -> UNNotificationPresentationOptions {
        // Menu-bar apps must present banners themselves while alive.
        return [.banner, .sound]
    }
}
