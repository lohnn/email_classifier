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

struct NotificationSettings: Codable, Equatable {
    var runStarted: Bool = true
    var runFinished: Bool = true
    var runFailed: Bool = true       // failures are PINNED (§6): always delivered
    var prOpened: Bool = true
    var deployVerified: Bool = true
    var skipped: Bool = true
}

enum AutonomyMode: String, Codable, CaseIterable {
    case askBeforeRun = "ask_before_run"
    case gatedShip = "gated_ship"
    case fullAutomatic = "full_automatic"

    var label: String {
        switch self {
        case .askBeforeRun: return "Ask before each run"
        case .gatedShip: return "Auto-train, gated shipping"
        case .fullAutomatic: return "Fully automatic"
        }
    }
}

struct AppState: Codable, Equatable {
    // --- ledger ---
    var createdAt: Date = Date()
    var updatedAt: Date = Date()
    var lastTrainStartedAt: Date?
    var lastTrainFinishedAt: Date?
    var lastTrainStatus: String?      // "ok" | "failed" | "skipped" | nil (never ran)
    var lastRunID: String?
    /// Confirmed-corrections count at the time of the last train (§4).
    /// P2 resets this after each successful train; until then it stays 0.
    var correctionsBaseline: Int = 0
    var runCounter: Int = 0

    // --- trigger policy (§4; Settings-editable) ---
    var correctionsThreshold: Int = 10
    var weeklyCadenceEnabled: Bool = true
    var weeklyCadenceDays: Int = 7
    var autonomy: AutonomyMode = .fullAutomatic
    var notifications: NotificationSettings = NotificationSettings()
    var loginItemEnabled: Bool = false

    private enum Keys: String, CodingKey {
        case createdAt, updatedAt, lastTrainStartedAt, lastTrainFinishedAt
        case lastTrainStatus, lastRunID, correctionsBaseline, runCounter
        case correctionsThreshold, weeklyCadenceEnabled, weeklyCadenceDays
        case autonomy, notifications, loginItemEnabled
    }

    init() {}

    init(from decoder: Decoder) throws {
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
    }
}

/// Atomic `state.json` persistence inside the sandbox container dir.
final class Ledger {
    /// The app's data dir: Application Support inside the sandbox container.
    /// Everything the app writes lives here (design §3: container-dir-only
    /// footprint); "Erase all data" wipes exactly this directory.
    static func defaultDirectory() -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("RetrainOrchestrator", isDirectory: true)
    }

    let directory: URL
    var stateFileURL: URL { directory.appendingPathComponent("state.json") }
    private let lock = NSLock()

    init(directory: URL) {
        self.directory = directory
    }

    func load() -> AppState {
        lock.lock()
        defer { lock.unlock() }
        return loadLocked()
    }

    private func loadLocked() -> AppState {
        guard let data = try? Data(contentsOf: stateFileURL) else {
            return AppState()
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        do {
            return try decoder.decode(AppState.self, from: data)
        } catch {
            // Corrupt ledger: do not take the app down; start fresh and let
            // the next save overwrite the bad file.
            return AppState()
        }
    }

    /// Atomically persist a state (write-to-temp + rename via .atomic).
    func save(_ state: AppState) throws {
        lock.lock()
        defer { lock.unlock() }
        try saveLocked(state)
    }

    private func saveLocked(_ state: AppState) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(state)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // .atomic: the temp file lands in the same dir and os.replace's over
        // the target — no torn state.json is ever observable.
        try data.write(to: stateFileURL, options: .atomic)
    }

    /// Load → mutate → save in one locked step. Returns the new state.
    func update(_ mutate: (inout AppState) -> Void) -> AppState {
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
