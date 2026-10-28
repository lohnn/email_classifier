import AppKit
import RetrainKit

/// Settings window (design §3/§4/§6): trigger threshold, weekly-cadence
/// toggle + interval, autonomy dial, notification category switches,
/// login-item toggle. Everything persists to state.json (atomic) on Save;
/// the login item applies immediately (it is a system registration).

final class SettingsWindowController: NSObject {
    private let ledger: Ledger
    private let runLog: RunLog
    private let onSaved: () -> Void
    /// Fired after every Test-connection probe: (reachable, short detail).
    /// Drives the menu-bar dot/tooltip — same no-background-poller rule.
    private let onProbe: (Bool, String) -> Void
    private var window: NSWindow?

    // controls
    private let thresholdField = NSTextField(string: "10")
    private let cadenceCheckbox = NSButton(checkboxWithTitle: "Weekly cadence",
                                           target: nil, action: nil)
    private let cadenceDaysField = NSTextField(string: "7")
    private let autonomyPopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private var categoryChecks: [NotificationCategory: NSButton] = [:]
    private let loginItemCheckbox = NSButton(
        checkboxWithTitle: "Start with the system (login item)",
        target: nil, action: nil
    )
    // connection controls (operator request T2)
    private let urlField = NSTextField(string: "http://jarvis:8008")
    private let testButton = NSButton(title: "Test connection", target: nil, action: nil)
    private let probeResult = NSTextField(wrappingLabelWithString: "Not probed yet.")

    enum URLValidation {
        case ok(String)
        case invalid(String)
    }

    init(ledger: Ledger, runLog: RunLog, onSaved: @escaping () -> Void,
         onProbe: @escaping (Bool, String) -> Void) {
        self.ledger = ledger
        self.runLog = runLog
        self.onSaved = onSaved
        self.onProbe = onProbe
        super.init()
        loginItemCheckbox.target = self
        loginItemCheckbox.action = #selector(loginItemToggled(_:))
        buildWindow()
    }

    // MARK: presentation

    func present() {
        pullFromState()
        guard let window = window else { return }
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    /// Re-sync after the underlying state changed underneath (e.g. Erase
    /// all data while a settings window happens to be open).
    func reload() {
        pullFromState()
    }

    private func pullFromState() {
        let state = ledger.load()
        thresholdField.stringValue = String(state.correctionsThreshold)
        cadenceCheckbox.state = state.weeklyCadenceEnabled ? .on : .off
        cadenceDaysField.stringValue = String(state.weeklyCadenceDays)
        if let index = AutonomyMode.allCases.firstIndex(of: state.autonomy) {
            autonomyPopup.selectItem(at: index)
        }
        categoryChecks[.runStarted]?.state = state.notifications.runStarted ? .on : .off
        categoryChecks[.runFinished]?.state = state.notifications.runFinished ? .on : .off
        categoryChecks[.runFailed]?.state = .on  // pinned — never user-editable
        categoryChecks[.prOpened]?.state = state.notifications.prOpened ? .on : .off
        categoryChecks[.deployVerified]?.state = state.notifications.deployVerified ? .on : .off
        categoryChecks[.skipped]?.state = state.notifications.skipped ? .on : .off
        loginItemCheckbox.state = LoginItem.isEnabled() ? .on : .off
        urlField.stringValue = state.classifierURL
        // probeResult intentionally NOT reset here: it reflects the last
        // actual probe, which a state reload does not invalidate.
    }

    // MARK: UI assembly

    private func buildWindow() {
        for mode in AutonomyMode.allCases {
            autonomyPopup.addItem(withTitle: mode.label)
        }

        cadenceCheckbox.target = self
        cadenceCheckbox.action = #selector(cadenceToggled(_:))

        // Failure switch exists but is pinned: rendered checked + disabled.
        for category in NotificationCategory.allCases {
            let check = NSButton(checkboxWithTitle: categoryLabel(category),
                                 target: nil, action: nil)
            if category == .runFailed {
                check.state = .on
                check.isEnabled = false
            }
            categoryChecks[category] = check
        }

        let thresholdRow = row([
            label("New confirmed corrections \u{2265}"),
            width(thresholdField, 60),
        ])
        let cadenceRow = row([
            cadenceCheckbox,
            label("interval (days)"),
            width(cadenceDaysField, 48),
        ])
        let autonomyRow = row([
            label("Autonomy"),
            autonomyPopup,
        ])

        testButton.target = self
        testButton.action = #selector(testConnectionClicked(_:))
        probeResult.isSelectable = true  // operator copies the truthful render

        let urlRow = row([
            label("Classifier URL"),
            width(urlField, 260),
        ])
        let testRow = row([testButton])

        let notifGrid = NSGridView(views: [
            [categoryChecks[.runStarted]!, categoryChecks[.runFinished]!, categoryChecks[.prOpened]!],
            [categoryChecks[.deployVerified]!, categoryChecks[.skipped]!, categoryChecks[.runFailed]!],
        ])
        notifGrid.rowSpacing = 6
        notifGrid.columnSpacing = 14

        let save = NSButton(title: "Save", target: self, action: #selector(saveClicked(_:)))
        save.keyEquivalent = "\r"
        let cancel = NSButton(title: "Cancel", target: self, action: #selector(cancelClicked(_:)))
        cancel.keyEquivalent = "\u{1b}"
        let spacer = NSView()
        spacer.setContentHuggingPriority(.init(1), for: .horizontal)
        let buttons = row([spacer, cancel, save])

        let stack = NSStackView(views: [
            sectionLabel("Run the retrain pipeline when"),
            thresholdRow,
            cadenceRow,
            note("The cadence fires only when at least 1 new correction exists. Zero new corrections means the pipeline skips and waits (a skipped event is mirrored once running, P2)."),
            sectionLabel("Autonomy"),
            autonomyRow,
            sectionLabel("Classifier connection"),
            urlRow,
            testRow,
            probeResult,
            note("Probes GET <url>/health only when you press Test connection — no background polling in P1. Give host AND port, e.g. http://jarvis:8008 (MagicDNS) or http://100.79.150.106:8008 (raw Tailscale IP)."),
            sectionLabel("Notifications"),
            notifGrid,
            note("Failures are pinned: always delivered, even when every other category is muted."),
            sectionLabel("System"),
            loginItemCheckbox,
            note("Data lives only in the sandboxed container dir (~/Library/Containers/<app id>/). Pipeline execution arrives with P2 — this window is fully functional already."),
            buttons,
        ])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 8
        stack.edgeInsets = NSEdgeInsets(top: 14, left: 16, bottom: 12, right: 16)
        stack.translatesAutoresizingMaskIntoConstraints = false

        let content = NSView(frame: NSRect(x: 0, y: 0, width: 520, height: 640))
        content.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor),
            stack.topAnchor.constraint(equalTo: content.topAnchor),
            stack.bottomAnchor.constraint(lessThanOrEqualTo: content.bottomAnchor, constant: -8),
        ])

        // Give the flexible row room to push buttons right.
        spacer.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        stack.setCustomSpacing(16, after: cadenceRow)
        stack.setCustomSpacing(16, after: autonomyRow)
        stack.setCustomSpacing(16, after: probeResult)
        stack.setCustomSpacing(16, after: notifGrid)

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 520, height: 640),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.title = "Retrain Orchestrator Settings"
        window.isReleasedWhenClosed = false
        window.contentView = content
        self.window = window
    }

    private func label(_ text: String) -> NSTextField {
        return NSTextField(labelWithString: text)
    }

    private func sectionLabel(_ text: String) -> NSTextField {
        let f = label(text)
        f.font = NSFont.boldSystemFont(ofSize: NSFont.systemFontSize)
        return f
    }

    private func note(_ text: String) -> NSTextField {
        let f = label(text)
        f.font = NSFont.systemFont(ofSize: NSFont.smallSystemFontSize)
        f.textColor = .secondaryLabelColor
        f.maximumNumberOfLines = 0
        f.preferredMaxLayoutWidth = 480
        return f
    }

    private func categoryLabel(_ category: NotificationCategory) -> String {
        switch category {
        case .runStarted: return "Run started"
        case .runFinished: return "Run finished"
        case .runFailed: return "Run failed (pinned)"
        case .prOpened: return "Vocabulary PR opened"
        case .deployVerified: return "Deploy verified"
        case .skipped: return "Skipped"
        }
    }

    private func row(_ views: [NSView]) -> NSStackView {
        let s = NSStackView(views: views)
        s.orientation = .horizontal
        s.alignment = .firstBaseline
        s.spacing = 8
        return s
    }

    private func width(_ field: NSTextField, _ w: CGFloat) -> NSView {
        let box = NSView()
        field.translatesAutoresizingMaskIntoConstraints = false
        box.addSubview(field)
        NSLayoutConstraint.activate([
            field.leadingAnchor.constraint(equalTo: box.leadingAnchor),
            field.trailingAnchor.constraint(equalTo: box.trailingAnchor),
            field.topAnchor.constraint(equalTo: box.topAnchor),
            field.bottomAnchor.constraint(equalTo: box.bottomAnchor),
            field.widthAnchor.constraint(equalToConstant: w),
        ])
        return box
    }

    // MARK: actions

    @objc private func cadenceToggled(_ sender: NSButton?) {
        // Visual only; the value is persisted on Save.
    }

    @objc private func loginItemToggled(_ sender: NSButton?) {
        let enable = loginItemCheckbox.state == .on
        if let error = LoginItem.setEnabled(enable) {
            runLog.warning(error)
            loginItemCheckbox.state = enable ? .off : .on
            showError(error)
            return
        }
        // Applies immediately — persist as such (independent of Save).
        ledger.update { $0.loginItemEnabled = enable }
        runLog.info("Login item \(enable ? "enabled" : "disabled")")
    }

    @objc private func saveClicked(_ sender: NSButton?) {
        let classifierURLValue: String
        switch Self.validateURL(urlField.stringValue) {
        case .ok(let value):
            classifierURLValue = value
        case .invalid(let why):
            showError(why)
            return
        }
        let threshold = clampInt(thresholdField.stringValue, range: 1...500, fallback: 10)
        let days = clampInt(cadenceDaysField.stringValue, range: 1...90, fallback: 7)
        let autonomyIndex = min(max(autonomyPopup.indexOfSelectedItem, 0), AutonomyMode.allCases.count - 1)
        let autonomy = AutonomyMode.allCases[autonomyIndex]

        ledger.update { s in
            s.correctionsThreshold = threshold
            s.weeklyCadenceEnabled = cadenceCheckbox.state == .on
            s.weeklyCadenceDays = days
            s.autonomy = autonomy
            s.classifierURL = classifierURLValue
            s.notifications.runStarted = categoryChecks[.runStarted]?.state == .on
            s.notifications.runFinished = categoryChecks[.runFinished]?.state == .on
            s.notifications.runFailed = true  // pinned by contract
            s.notifications.prOpened = categoryChecks[.prOpened]?.state == .on
            s.notifications.deployVerified = categoryChecks[.deployVerified]?.state == .on
            s.notifications.skipped = categoryChecks[.skipped]?.state == .on
        }
        onSaved()
        closeWindow()
    }

    // MARK: classifier connection probe

    /// Normalize + validate the classifier base URL: scheme must be
    /// http/https, host must be present; port is recommended but not
    /// forced (operator may serve on 80/443). Returns the trimmed value
    /// with trailing slashes stripped so "…/health" joins cleanly.
    static func validateURL(_ raw: String) -> URLValidation {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        while text.hasSuffix("/") { text.removeLast() }
        guard !text.isEmpty else {
            return .invalid("Classifier URL must not be empty.")
        }
        guard let components = URLComponents(string: text),
              let scheme = components.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              let host = components.host, !host.isEmpty else {
            return .invalid("Classifier URL must be a full http(s) URL with a host (and ideally a port), e.g. http://jarvis:8008")
        }
        return .ok(text)
    }

    private static let stampFormat: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f
    }()

    @objc private func testConnectionClicked(_ sender: NSButton?) {
        let value: String
        switch Self.validateURL(urlField.stringValue) {
        case .ok(let v): value = v
        case .invalid(let why):
            probeResult.stringValue = why
            return
        }

        // The "did my URL change take" check probes what is IN THE FIELD,
        // not what was last saved — exactly the operator's use case.
        testButton.isEnabled = false
        probeResult.stringValue = "Probing \(value)/health …"
        guard let healthURL = URL(string: value + "/health") else {
            testButton.isEnabled = true
            probeResult.stringValue = "Invalid URL: \(value)"
            return
        }
        var request = URLRequest(url: healthURL)
        request.timeoutInterval = 5.0
        request.httpMethod = "GET"
        let started = Date()
        let task = URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                guard let self = self else { return }
                self.testButton.isEnabled = true
                self.renderProbe(data: data, response: response, error: error, at: started)
            }
        }
        task.resume()
    }

    /// Render whatever /health truthfully returned (W-121 discipline:
    /// reachability by body truth, never by status code alone), then hand
    /// the verdict to the menu-bar indicator.
    private func renderProbe(data: Data?, response: URLResponse?, error: Error?, at started: Date) {
        let stamp = Self.stampFormat.string(from: started)
        if let error = error {
            probeResult.stringValue = "Unreachable (\(stamp)): \(error.localizedDescription)"
            onProbe(false, "unreachable, \(stamp)")
            return
        }
        guard let http = response as? HTTPURLResponse else {
            probeResult.stringValue = "Non-HTTP response (\(stamp))"
            onProbe(false, "non-HTTP response, \(stamp)")
            return
        }

        var bodySummary = ""
        var statusSaysOk = true
        if let data = data,
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let status = json["status"] as? String {
                bodySummary = "status=\(status)"
                statusSaysOk = (status == "ok")
            }
            if let checks = json["checks"] as? [String: Any] {
                let flat = checks
                    .map { "\($0.key)=\($0.value)" }
                    .sorted()
                    .joined(separator: ", ")
                if !flat.isEmpty {
                    bodySummary += (bodySummary.isEmpty ? "" : "; ") + flat
                }
            }
        }

        // Reachable-and-healthy = HTTP 200 AND /health's own status ok.
        // A 503 or a body saying the model is not loaded renders truthfully
        // and counts as NOT healthy for the indicator.
        let reachable = http.statusCode == 200 && statusSaysOk
        probeResult.stringValue = "HTTP \(http.statusCode)\(bodySummary.isEmpty ? "" : " — \(bodySummary)") (\(stamp))"
        let detail = "HTTP \(http.statusCode)\(bodySummary.isEmpty ? "" : ", \(bodySummary)"), \(stamp)"
        onProbe(reachable, detail)
        if reachable {
            runLog.info("Connection probe OK: \(probeResult.stringValue)")
        } else {
            runLog.warning("Connection probe FAILED: \(probeResult.stringValue)")
        }
    }

    @objc private func cancelClicked(_ sender: NSButton?) {
        closeWindow()
    }

    private func closeWindow() {
        window?.orderOut(nil)
    }

    private func showError(_ text: String) {
        let alert = NSAlert()
        alert.messageText = "Settings problem"
        alert.informativeText = text
        alert.alertStyle = .warning
        alert.runModal()
    }

    private func clampInt(_ text: String, range: ClosedRange<Int>, fallback: Int) -> Int {
        guard let v = Int(text.trimmingCharacters(in: .whitespaces)) else { return fallback }
        return min(max(v, range.lowerBound), range.upperBound)
    }
}
