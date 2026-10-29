import Foundation

/// The retrain-event status mirror body (design §6) — the Swift twin of the
/// brain's strictly-validated `RetrainEventRequest` (api/models.py). The
/// brain rejects unknown fields, so this encoder emits EXACTLY the contract
/// keys; event/status enums are closed on both sides.
///
/// Idempotency contract (brain): the (run_id, event) pair is the uniqueness
/// key — a retried POST returns the first record and changes nothing, so
/// "already_recorded" responses are SUCCESS for the app (at-least-once
/// delivery degrades to exactly-once recording on the mirror).
public enum RetrainEventType: String, Codable, Equatable {
    case runStarted = "run_started"
    case runFinished = "run_finished"
    case runFailed = "run_failed"
    case prOpened = "pr_opened"
    case deployVerified = "deploy_verified"
    case skipped
}

public enum RetrainEventStatus: String, Codable, Equatable {
    case ok
    case failed
    case skipped
}

public struct RetrainMirrorEvent: Codable, Equatable {
    public var event: RetrainEventType
    /// ISO8601 on the wire (brain expects a parseable timestamp).
    public var at: Date
    public var runID: String
    public var status: RetrainEventStatus
    /// Metrics the brain's human-log renders (e.g. accuracy). Kept numeric.
    public var eval: [String: Double]?
    /// The TRAINING-DATA repo commit the run was built on (X-Snapshot-Data-Commit).
    public var dataSHA: String?
    /// Shipped artifact → full lowercase sha256.
    public var artifactSHAs: [String: String]?
    /// Cumulative seconds per pipeline stage.
    public var durations: [String: Double]?
    /// Failure text — and, by documented contract, the skip reason for
    /// skipped events and the "manual merge required" pointer on the
    /// merge-checkpoint run_finished.
    public var error: String?

    private enum CodingKeys: String, CodingKey {
        case event, at
        case runID = "run_id"
        case status
        case eval
        case dataSHA = "data_sha"
        case artifactSHAs = "artifact_shas"
        case durations
        case error
    }

    public init(event: RetrainEventType, at: Date, runID: String, status: RetrainEventStatus,
                eval: [String: Double]? = nil, dataSHA: String? = nil,
                artifactSHAs: [String: String]? = nil, durations: [String: Double]? = nil,
                error: String? = nil) {
        self.event = event
        self.at = at
        self.runID = runID
        self.status = status
        self.eval = eval
        self.dataSHA = dataSHA
        self.artifactSHAs = artifactSHAs
        self.durations = durations
        self.error = error
    }

    /// Wire bytes — ISO8601 dates, exactly the contract keys (the brain
    /// 422s surprises).
    public func mirrorJSONData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(self)
    }
}

/// The brain's POST /admin/retrain-events response body (the embedded event
/// echo is never introspected by the app, so it is not modeled).
public struct MirrorReceipt: Codable, Equatable {
    public let status: String           // "recorded" | "already_recorded"
    public let duplicate: Bool
}
