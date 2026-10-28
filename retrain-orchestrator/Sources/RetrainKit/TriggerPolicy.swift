import Foundation

/// Pure evaluation of the LOCKED trigger policy (design §4).
///
/// Fire when:
///   1. new confirmed corrections >= correctionsThreshold (default 10), OR
///   2. weekly cadence: >= weeklyCadenceDays since the last train
///      AND >= 1 new confirmed correction.
///
/// Zero-corrections rule (operator-set): with NO new confirmed corrections
/// the pipeline never runs — P2 mirrors a "skipped" retrain-event with the
/// "no new training data" reason. A quiet wait (some corrections, neither
/// threshold nor cadence) is NOT an event — nothing triggered.

enum SkipReason: String, Equatable {
    /// Zero-corrections rule violated: mirrored as a skipped event in P2.
    case noNewTrainingData
    /// Threshold not reached and cadence not due: plain waiting, no event.
    case notDue
}

enum TriggerDecision: Equatable {
    case fire(reason: String)
    case skip(reason: SkipReason, detail: String)
}

struct TriggerPolicy {
    static func evaluate(
        state: AppState,
        newCorrections: Int,
        now: Date = Date()
    ) -> TriggerDecision {
        // Zero-corrections rule — a precondition on every branch.
        guard newCorrections >= 1 else {
            return .skip(
                reason: .noNewTrainingData,
                detail: "0 new confirmed corrections since the last train"
            )
        }

        // (1) threshold branch
        if newCorrections >= state.correctionsThreshold {
            return .fire(
                reason: "\(newCorrections) new corrections >= threshold \(state.correctionsThreshold)"
            )
        }

        // (2) weekly-cadence branch: enough days since last train AND >= 1 new correction
        if state.weeklyCadenceEnabled,
           let last = state.lastTrainFinishedAt ?? state.lastTrainStartedAt {
            let daysSince = now.timeIntervalSince(last) / 86400.0
            if daysSince >= Double(state.weeklyCadenceDays) {
                return .fire(
                    reason: "weekly cadence: \(Int(daysSince))d since last train >= \(state.weeklyCadenceDays)d, with \(newCorrections) new correction(s)"
                )
            }
        }

        return .skip(
            reason: .notDue,
            detail: "\(newCorrections) new correction(s); threshold \(state.correctionsThreshold), cadence \(state.weeklyCadenceDays)d \(state.weeklyCadenceEnabled ? "on" : "off")"
        )
    }
}
