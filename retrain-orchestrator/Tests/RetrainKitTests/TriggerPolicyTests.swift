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
        /// Bit-pattern comparison (round-14 upgrade, from the round-13
        /// analysis): %.17g STRINGS collide at epoch magnitude — one Double
        /// ULP ≈ 2.4e-7 s, the 17th significant decimal of a 10-integer-
        /// digit number — so a 1-ULP drift renders to the SAME text and the
        /// instrument could miss exactly the deltas it hunts. Bit patterns
        /// collide NOWHERE for distinct doubles; deltas print hex bits AND
        /// %.20g magnitude together.
        func noteDate(_ field: String, _ a: Date?, _ b: Date?) {
            let render: (Date?) -> String = { date in
                guard let date = date else { return "nil" }
                return String(format: "0x%016llx (%.20g)",
                              date.timeIntervalSince1970.bitPattern,
                              date.timeIntervalSince1970)
            }
            let bitEqual = a?.timeIntervalSince1970.bitPattern == b?.timeIntervalSince1970.bitPattern
            if !bitEqual {
                lines.append("\(field): \(render(a)) vs \(render(b))")
            }
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
        // → load (tolerant decode chain).
        let (ledger, dir) = freshLedger()
        defer { remove(dir) }
        let original = makeState()
        try ledger.save(original)
        let restored = ledger.load()

        // PRE-REGISTERED FALLBACK — FIRED (pastes of rounds 14/15): the
        // delta walker finds every field equal with all date BIT-PATTERNS
        // equal, while synthesized struct equality fails → Foundation
        // `Date ==` on this runtime (macOS 27 SDK) is not bit-faithful
        // (its internal representation carries more than
        // timeIntervalSince1970). The test CONTRACT moves to machine-truth:
        // per-date bit-pattern assertions + plain equality on every other
        // field. The ledger's persistence contract is DOUBLE-FIDELITY, not
        // Date-identity. (Round-13 instrumentation stays: it is what
        // proved this.)
        //
        // Round-13 truth instrumentation kept for future recurrence hunts:
        // the delta walker + raw on-disk JSON preview remain the first
        // responder if this ever regresses again.
        let delta = describeDeltas(original, restored)
        XCTAssertTrue(delta.isEmpty,
                      "ROUND-14 DELTA (hex bit patterns + %.20g)\n\(delta.isEmpty ? "(field-wise equal, all date bit-patterns equal — fired the pre-registered Date-== fallback)" : delta)")

        func assertBits(_ field: String, _ a: Date?, _ b: Date?) {
            XCTAssertEqual(a?.timeIntervalSince1970.bitPattern,
                           b?.timeIntervalSince1970.bitPattern, field)
        }
        assertBits("createdAt", original.createdAt, restored.createdAt)
        assertBits("updatedAt", original.updatedAt, restored.updatedAt)
        assertBits("lastTrainStartedAt", original.lastTrainStartedAt, restored.lastTrainStartedAt)
        assertBits("lastTrainFinishedAt", original.lastTrainFinishedAt, restored.lastTrainFinishedAt)
        XCTAssertEqual(restored.lastTrainStatus, original.lastTrainStatus)
        XCTAssertEqual(restored.lastRunID, original.lastRunID)
        XCTAssertEqual(restored.correctionsBaseline, original.correctionsBaseline)
        XCTAssertEqual(restored.runCounter, original.runCounter)
        XCTAssertEqual(restored.correctionsThreshold, original.correctionsThreshold)
        XCTAssertEqual(restored.weeklyCadenceEnabled, original.weeklyCadenceEnabled)
        XCTAssertEqual(restored.weeklyCadenceDays, original.weeklyCadenceDays)
        XCTAssertEqual(restored.autonomy, original.autonomy)
        XCTAssertEqual(restored.notifications, original.notifications)
        XCTAssertEqual(restored.loginItemEnabled, original.loginItemEnabled)
        XCTAssertEqual(restored.classifierURL, original.classifierURL)
        XCTAssertEqual(restored.firstRunCompleted, original.firstRunCompleted)
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

final class LedgerTests: XCTestCase {
    func testSaveLoadRoundTripInTempDir() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("retrain-ledger-tests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let ledger = Ledger(directory: dir)

        let before = ledger.load()
        XCTAssertEqual(before.correctionsThreshold, 10)  // defaults before first save

        _ = ledger.update { $0.correctionsThreshold = 42; $0.correctionsBaseline = 7 }
        let after = ledger.load()
        XCTAssertEqual(after.correctionsThreshold, 42)
        XCTAssertEqual(after.correctionsBaseline, 7)
        XCTAssertTrue(FileManager.default.fileExists(atPath: ledger.stateFileURL.path))
    }

    func testCorruptFileIsTolerated() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("retrain-ledger-tests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("not json at all".utf8).write(to: dir.appendingPathComponent("state.json"))
        let ledger = Ledger(directory: dir)
        XCTAssertEqual(ledger.load().correctionsThreshold, 10)  // fresh defaults win
    }
}

final class EraseTests: XCTestCase {
    func testWipeEmptiesDirectoryButKeepsIt() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("retrain-erase-tests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appendingPathComponent("state.json"))
        try Data("log".utf8).write(to: dir.appendingPathComponent("orchestrator.log"))
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("nested"), withIntermediateDirectories: true)
        try Data("x".utf8).write(to: dir.appendingPathComponent("nested/deep.txt"))

        let deleted = try Erase.wipe(directory: dir)
        XCTAssertEqual(deleted, 3)
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.path))
        let remaining = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        XCTAssertTrue(remaining.isEmpty)
    }
}
