import Foundation

/// The app's ONE state file: the run ledger plus the operator-editable
/// settings, persisted as `state.json` (atomic writes) inside the sandbox
/// container dir only (design §3).
///
/// Trigger-policy fields exist from P1; pipeline EXECUTION arrives at P2
/// (design §10) — this ledger is what makes P2 resumable and the cadence
/// auditable ("last_train timestamp, corrections-count baseline").
///
/// Decode is deliberately TOLERANT (missing keys fall back to defaults):
/// an operator restarting an older binary against a newer state.json must
/// never lose their Settings to a hard decode failure.

public struct NotificationSettings: Codable, Equatable {
    public var runStarted: Bool = true
    public var runFinished: Bool = true
    public var runFailed: Bool = true       // failures are PINNED (§6): always delivered
    public var prOpened: Bool = true
    public var deployVerified: Bool = true
    public var skipped: Bool = true
}

public enum AutonomyMode: String, Codable, CaseIterable {
    case askBeforeRun = "ask_before_run"
    case gatedShip = "gated_ship"
    case fullAutomatic = "full_automatic"

    public var label: String {
        switch self {
        case .askBeforeRun: return "Ask before each run"
        case .gatedShip: return "Auto-train, gated shipping"
        case .fullAutomatic: return "Fully automatic"
        }
    }
}

public struct AppState: Codable, Equatable {
    // --- ledger ---
    public var createdAt: Date = Date()
    public var updatedAt: Date = Date()
    public var lastTrainStartedAt: Date?
    public var lastTrainFinishedAt: Date?
    public var lastTrainStatus: String?      // "ok" | "failed" | "skipped" | nil (never ran)
    public var lastRunID: String?
    /// Confirmed-corrections count at the time of the last train (§4).
    /// P2 resets this after each successful train; until then it stays 0.
    public var correctionsBaseline: Int = 0
    public var runCounter: Int = 0

    // --- trigger policy (§4; Settings-editable) ---
    public var correctionsThreshold: Int = 10
    public var weeklyCadenceEnabled: Bool = true
    public var weeklyCadenceDays: Int = 7
    public var autonomy: AutonomyMode = .fullAutomatic
    public var notifications: NotificationSettings = NotificationSettings()
    public var loginItemEnabled: Bool = false

    /// Classifier base URL (host AND port) — the probe + P2 pre-flight
    /// target. Never hardcoded elsewhere: Settings carries and edits it.
    /// Operator decision 2026-10-29: NO default ADDRESS — every setup is
    /// different, so this starts EMPTY and the app opens Settings on first
    /// launch until it is set. The port suggestion (8008, the service's
    /// default) travels in the field placeholder and docs, not as a value.
    public var classifierURL: String = ""

    /// Onboarding gate: Settings auto-opens at launch until the FIRST
    /// successful Save (which flips this). Cancel/close does not — the
    /// preferences are not set up yet. A fresh install and Erase-all-data
    /// both land here; an older state.json without the key also decodes
    /// to false, re-running the check once (harmless, deliberate).
    public var firstRunCompleted: Bool = false

    private enum Keys: String, CodingKey {
        case createdAt, updatedAt, lastTrainStartedAt, lastTrainFinishedAt
        case lastTrainStatus, lastRunID, correctionsBaseline, runCounter
        case correctionsThreshold, weeklyCadenceEnabled, weeklyCadenceDays
        case autonomy, notifications, loginItemEnabled, classifierURL
        case firstRunCompleted
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Keys.self)
        createdAt = try c.decodeIfPresent(Date.self, forKey: .createdAt) ?? Date()
        updatedAt = try c.decodeIfPresent(Date.self, forKey: .updatedAt) ?? Date()
        lastTrainStartedAt = try c.decodeIfPresent(Date.self, forKey: .lastTrainStartedAt)
        lastTrainFinishedAt = try c.decodeIfPresent(Date.self, forKey: .lastTrainFinishedAt)
        lastTrainStatus = try c.decodeIfPresent(String.self, forKey: .lastTrainStatus)
        lastRunID = try c.decodeIfPresent(String.self, forKey: .lastRunID)
        correctionsBaseline = try c.decodeIfPresent(Int.self, forKey: .correctionsBaseline) ?? 0
        runCounter = try c.decodeIfPresent(Int.self, forKey: .runCounter) ?? 0
        correctionsThreshold = try c.decodeIfPresent(Int.self, forKey: .correctionsThreshold) ?? 10
        weeklyCadenceEnabled = try c.decodeIfPresent(Bool.self, forKey: .weeklyCadenceEnabled) ?? true
        weeklyCadenceDays = try c.decodeIfPresent(Int.self, forKey: .weeklyCadenceDays) ?? 7
        autonomy = try c.decodeIfPresent(AutonomyMode.self, forKey: .autonomy) ?? .fullAutomatic
        notifications = try c.decodeIfPresent(NotificationSettings.self, forKey: .notifications)
            ?? NotificationSettings()
        loginItemEnabled = try c.decodeIfPresent(Bool.self, forKey: .loginItemEnabled) ?? false
        classifierURL = try c.decodeIfPresent(String.self, forKey: .classifierURL) ?? ""
        firstRunCompleted = try c.decodeIfPresent(Bool.self, forKey: .firstRunCompleted) ?? false
    }
}

/// Atomic `state.json` persistence inside the sandbox container dir.
public final class Ledger {
    /// Dates on disk are EPOCH DOUBLES — full Double precision, lossless
    /// through JSON's shortest-round-trip serialization (build-round-9
    /// lesson: ISO8601 — even WITH fractional seconds — is millisecond-
    /// granular while Date() carries sub-microsecond precision, so a
    /// formatted round-trip silently loses the delta; state.json is
    /// machine-only and the epoch doubles still sort/diff fine).
    /// ISO8601 strings exist ONLY as legacy-DECODE fallbacks: the ladder is
    /// Double → ISO8601-with-fractional (a9bf3ad builds) → seconds-only
    /// ISO8601 (the oldest live installs). Human-legibility tradeoff of
    /// epoch doubles is a conscious call. This codec is for the ON-DISK
    /// machine format only — the mirror's `at` wire format
    /// (RetrainMirrorEvent) is a separate explicit contract with the brain
    /// and stays seconds-only ISO8601 strings, unchanged.
    enum JSONDateCodec {
        static let fractional: ISO8601DateFormatter = {
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            return formatter
        }()
        static let plain = ISO8601DateFormatter()

        /// Lossless: everything date-ish on disk is an epoch Double.
        static func epochDouble(_ date: Date) -> Double {
            date.timeIntervalSince1970
        }

        /// Decode ladder (oldest → newest formats all readable):
        /// Double(epoch) → ISO8601-fractional → ISO8601-seconds-only.
        static func parseWire(_ container: SingleValueDecodingContainer) throws -> Date {
            if let epoch = try? container.decode(Double.self), epoch.isFinite {
                return Date(timeIntervalSince1970: epoch)
            }
            let text = try container.decode(String.self)
            if let date = fractional.date(from: text) { return date }
            if let date = plain.date(from: text) { return date }
            throw DecodingError.dataCorrupted(
                DecodingError.Context(codingPath: container.codingPath,
                                      debugDescription: "unparseable ledger date: \(text)"))
        }
    }

    /// The app's data dir: Application Support inside the sandbox container.
    /// Everything the app writes lives here (design §3: container-dir-only
    /// footprint); "Erase all data" wipes exactly this directory.
    public static func defaultDirectory() -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("RetrainOrchestrator", isDirectory: true)
    }

    public let directory: URL
    public var stateFileURL: URL { directory.appendingPathComponent("state.json") }
    private let lock = NSLock()

    public init(directory: URL) {
        self.directory = directory
    }

    public func load() -> AppState {
        lock.lock()
        defer { lock.unlock() }
        return loadLocked()
    }

    private func loadLocked() -> AppState {
        guard let data = try? Data(contentsOf: stateFileURL) else {
            return AppState()
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            return try JSONDateCodec.parseWire(container)
        }
        do {
            return try decoder.decode(AppState.self, from: data)
        } catch {
            // Corrupt ledger: do not take the app down; start fresh and let
            // the next save overwrite the bad file.
            return AppState()
        }
    }

    /// Atomically persist a state (write-to-temp + rename via .atomic).
    public func save(_ state: AppState) throws {
        lock.lock()
        defer { lock.unlock() }
        try saveLocked(state)
    }

    private func saveLocked(_ state: AppState) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .custom { date, encoder in
            var container = encoder.singleValueContainer()
            try container.encode(JSONDateCodec.epochDouble(date))
        }
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(state)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // .atomic: the temp file lands in the same dir and os.replace's over
        // the target — no torn state.json is ever observable.
        try data.write(to: stateFileURL, options: .atomic)
    }

    /// Load → mutate → save in one locked step. Returns the new state.
    /// Discarding the result is a legitimate call pattern (the login-item
    /// toggle and Settings' Save both re-read via their own paths), hence
    /// @discardableResult — build round 3's only warnings were both this.
    @discardableResult
    public func update(_ mutate: (inout AppState) -> Void) -> AppState {
        lock.lock()
        defer { lock.unlock() }
        var state = loadLocked()
        mutate(&state)
        state.updatedAt = Date()
        do {
            try saveLocked(state)
        } catch {
            // Persist failure is logged by the caller layer; never crash.
            NSLog("Ledger save failed: \(error)")
        }
        return state
    }
}
