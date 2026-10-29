import XCTest
@testable import RetrainKit

final class TriggerPolicyTests: XCTestCase {
    private func makeState(threshold: Int = 10,
                           cadenceOn: Bool = true,
                           cadenceDays: Int = 7,
                           lastTrainDaysAgo: Double? = nil) -> AppState {
        var s = AppState()
        s.correctionsThreshold = threshold
        s.weeklyCadenceEnabled = cadenceOn
        s.weeklyCadenceDays = cadenceDays
        if let ago = lastTrainDaysAgo {
            let stamp = Date().addingTimeInterval(-ago * 86400.0)
            s.lastTrainStartedAt = stamp
            s.lastTrainFinishedAt = stamp
        }
        return s
    }

    private func skipReason(_ d: TriggerDecision) -> SkipReason? {
        if case .skip(let reason, _) = d { return reason }
        return nil
    }

    func testZeroCorrectionsNeverFires() {
        let d = TriggerPolicy.evaluate(state: makeState(threshold: 1), newCorrections: 0)
        XCTAssertEqual(d, .skip(reason: .noNewTrainingData,
                                detail: "0 new confirmed corrections since the last train"))
    }

    func testZeroCorrectionsStillSkipsEvenWhenCadenceIsDue() {
        let d = TriggerPolicy.evaluate(state: makeState(lastTrainDaysAgo: 30), newCorrections: 0)
        XCTAssertEqual(skipReason(d), .noNewTrainingData)
    }

    func testThresholdFires() {
        let d = TriggerPolicy.evaluate(state: makeState(threshold: 10, lastTrainDaysAgo: 0),
                                       newCorrections: 10)
        guard case .fire(let reason) = d else { return XCTFail("expected fire, got \(d)") }
        XCTAssertTrue(reason.contains(">= threshold 10"))
    }

    func testBelowThresholdQuietlyWaits() {
        let d = TriggerPolicy.evaluate(state: makeState(threshold: 10, lastTrainDaysAgo: 0),
                                       newCorrections: 3)
        XCTAssertEqual(skipReason(d), .notDue)
    }

    func testCadenceFiresWithAtLeastOneCorrection() {
        let d = TriggerPolicy.evaluate(state: makeState(threshold: 10, lastTrainDaysAgo: 7.1),
                                       newCorrections: 1)
        guard case .fire(let reason) = d else { return XCTFail("expected fire, got \(d)") }
        XCTAssertTrue(reason.contains("weekly cadence"))
    }

    func testCadenceNeedsFreshData() {
        let d = TriggerPolicy.evaluate(state: makeState(lastTrainDaysAgo: 8), newCorrections: 0)
        XCTAssertEqual(skipReason(d), .noNewTrainingData)
    }

    func testCadenceDisabledDoesNotFire() {
        let d = TriggerPolicy.evaluate(
            state: makeState(threshold: 10, cadenceOn: false, lastTrainDaysAgo: 30),
            newCorrections: 1
        )
        XCTAssertEqual(skipReason(d), .notDue)
    }

    func testFallsBackToStartedAtWhenNoFinishTimestamp() {
        var s = makeState(threshold: 10)
        s.lastTrainFinishedAt = nil
        s.lastTrainStartedAt = Date().addingTimeInterval(-30 * 86400.0)
        let d = TriggerPolicy.evaluate(state: s, newCorrections: 1)
        guard case .fire = d else { return XCTFail("expected fire via lastTrainStartedAt, got \(d)") }
    }
}

final class AppStateTests: XCTestCase {
    /// PRODUCTION-PATH RULE (build-round-10 root cause of record): AppState
    /// round-trips here go through Ledger.save/load — the exact epoch-double
    /// codec + tolerant decode chain the app runs. A test-local JSONEncoder
    /// with its own date strategy drifts from production invisibly (three
    /// rounds of correct-but-inert fixes proved it).
    private func makeState() -> AppState {
        var s = AppState()
        s.correctionsThreshold = 12
        s.weeklyCadenceEnabled = false
        s.weeklyCadenceDays = 9
        s.autonomy = .gatedShip
        s.correctionsBaseline = 44
        s.runCounter = 3
        s.lastTrainFinishedAt = Date(timeIntervalSince1970: 1_770_000_000)
        s.lastRunID = "run-20261028-1"
        s.notifications.prOpened = false
        return s
    }

    private func freshLedger() -> (Ledger, URL) {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("appstate-tests-\(UUID().uuidString)", isDirectory: true)
        // Test setup states its own filesystem precondition EXPLICITLY
        // (build-round-12 harness lesson): the fixture-byte tests write
        // straight into stateFileURL, whose PARENT must exist — lazy
        // creation belongs to production paths (Ledger.save mkdirs), not
        // to coincidence.
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let ledger = Ledger(directory: dir)
        return (ledger, dir)
    }

    /// INSTRUMENTED for one round (build round 12): the defers reported
    /// removeItem ENOENT on dirs the tree says `try?` should have
    /// swallowed — this round makes the pre-removal truth visible in the
    /// paste. Revert to a plain `try? removeItem` in round 13 with the
    /// outcome noted; cleanup noise must never fail a green suite.
    private func remove(_ dir: URL) {
        let exists = FileManager.default.fileExists(atPath: dir.path)
        do {
            try FileManager.default.removeItem(at: dir)
        } catch {
            print("[test-cleanup] exists=\(exists) err=\(String(describing: error))")
        }
    }

    /// Round-13 INSTRUMENTATION (test-file-only; scheduled KEEP — cheap
    /// and load-bearing when renders-identical-but-unequal strikes). Walks
    /// EVERY field; dates render as %.17g epoch doubles — deliberately NOT
    /// Date's description (seconds-only truncation hides exactly the
    /// deltas this hunts). Empty string when field-wise equal.
    private func describeDeltas(_ original: AppState, _ restored: AppState) -> String {
        var lines: [String] = []
        func noteDate(_ field: String, _ a: Date?, _ b: Date?) {
            let da = a.map { String(format: "%.17g", $0.timeIntervalSince1970) } ?? "nil"
            let db = b.map { String(format: "%.17g", $0.timeIntervalSince1970) } ?? "nil"
            if da != db { lines.append("\(field): \(da) vs \(db)") }
        }
        func noteValue<T: Equatable>(_ field: String, _ a: T, _ b: T) {
            if a != b {
                lines.append("\(field): \(String(describing: a)) vs \(String(describing: b))")
            }
        }
        noteDate("createdAt", original.createdAt, restored.createdAt)
        noteDate("updatedAt", original.updatedAt, restored.updatedAt)
        noteDate("lastTrainStartedAt", original.lastTrainStartedAt, restored.lastTrainStartedAt)
        noteDate("lastTrainFinishedAt", original.lastTrainFinishedAt, restored.lastTrainFinishedAt)
        noteValue("lastTrainStatus", original.lastTrainStatus, restored.lastTrainStatus)
        noteValue("lastRunID", original.lastRunID, restored.lastRunID)
        noteValue("correctionsBaseline", original.correctionsBaseline, restored.correctionsBaseline)
        noteValue("runCounter", original.runCounter, restored.runCounter)
        noteValue("correctionsThreshold", original.correctionsThreshold, restored.correctionsThreshold)
        noteValue("weeklyCadenceEnabled", original.weeklyCadenceEnabled, restored.weeklyCadenceEnabled)
        noteValue("weeklyCadenceDays", original.weeklyCadenceDays, restored.weeklyCadenceDays)
        noteValue("autonomy", original.autonomy, restored.autonomy)
        noteValue("notifications", original.notifications, restored.notifications)
        noteValue("loginItemEnabled", original.loginItemEnabled, restored.loginItemEnabled)
        noteValue("classifierURL", original.classifierURL, restored.classifierURL)
        noteValue("firstRunCompleted", original.firstRunCompleted, restored.firstRunCompleted)
        return lines.joined(separator: "\n")
    }

    func testRoundTripPreservesValues() throws {
        // Exercises the PRODUCTION path: save → state.json (epoch doubles)
        // → load (tolerant decode chain) — exact equality is the contract.
        let (ledger, dir) = freshLedger()
        defer { remove(dir) }
        let original = makeState()
        try ledger.save(original)
        let restored = ledger.load()
        XCTAssertEqual(restored, original)
        // Round-13 truth instrumentation: the paste must name WHICH field
        // differs and its two raw double renders (bit-equal or not), plus
        // exactly what the encoder wrote on disk (integral vs fractional,
        // digit count). No further codec theory is acted on before that.
        let delta = describeDeltas(original, restored)
        let disk = (try? Data(contentsOf: ledger.stateFileURL))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "(unreadable)"
        let preview = String(disk.prefix(1200))
        XCTAssertTrue(delta.isEmpty,
                      "ROUND-13 DELTA (field: raw-double-doubles)\n\(delta.isEmpty ? "(field-wise EQUAL yet struct unequal — Date construction semantics suspected)" : delta)\n\nON-DISK STATE.JSON (first 1200 chars):\n\(preview)")
    }

    func testMissingKeysFallBackToDefaults() throws {
        // A state.json written by a NEWER app read by an OLDER binary must
        // degrade to defaults, never hard-fail. Raw bytes through the
        // PRODUCTION decode path.
        let (ledger, dir) = freshLedger()
        defer { remove(dir) }
        try Data("{}".utf8).write(to: ledger.stateFileURL)
        let restored = ledger.load()
        XCTAssertEqual(restored.correctionsThreshold, 10)
        XCTAssertEqual(restored.weeklyCadenceDays, 7)
        XCTAssertTrue(restored.weeklyCadenceEnabled)
        XCTAssertEqual(restored.autonomy, .fullAutomatic)
        XCTAssertNil(restored.lastTrainFinishedAt)
        XCTAssertFalse(restored.loginItemEnabled)
        // Operator decision 2026-10-29: NO default classifier address —
        // empty string, never a setup-specific host baked in.
        XCTAssertEqual(restored.classifierURL, "")
        XCTAssertFalse(restored.firstRunCompleted)
    }

    func testUnknownKeysAreIgnored() throws {
        let (ledger, dir) = freshLedger()
        defer { remove(dir) }
        let json = #"{"correctionsThreshold": 5, "nobodyKnowsThisYet": 123}"#
        try Data(json.utf8).write(to: ledger.stateFileURL)
        let restored = ledger.load()
        XCTAssertEqual(restored.correctionsThreshold, 5)
    }

    /// The legacy decode ladder end-to-end THROUGH the ledger: a
    /// date-carrying state.json in each historical format decodes exactly.
    func testLegacyDateLadderAllThreeFormats() throws {
        // (1) newest: epoch double — exact, sub-second preserved.
        let (l1, d1) = freshLedger()
        defer { remove(d1) }
        try Data(#"{"lastTrainFinishedAt": 1770000000.123456}"#.utf8).write(to: l1.stateFileURL)
        XCTAssertEqual(l1.load().lastTrainFinishedAt,
                       Date(timeIntervalSince1970: 1_770_000_000.123456))

        // (2) middle: a9bf3ad-era fractional ISO — millisecond-exact.
        let (l2, d2) = freshLedger()
        defer { remove(d2) }
        try Data(#"{"lastTrainFinishedAt": "2026-02-02T02:40:00.123Z"}"#.utf8).write(to: l2.stateFileURL)
        XCTAssertEqual(l2.load().lastTrainFinishedAt,
                       Date(timeIntervalSince1970: 1_770_000_000.123))

        // (3) oldest: seconds-only ISO (the oldest live installs) —
        // to-the-second.
        let (l3, d3) = freshLedger()
        defer { remove(d3) }
        try Data(#"{"lastTrainFinishedAt": "2026-02-02T02:40:00Z"}"#.utf8).write(to: l3.stateFileURL)
        XCTAssertEqual(l3.load().lastTrainFinishedAt,
                       Date(timeIntervalSince1970: 1_770_000_000))
    }
}
