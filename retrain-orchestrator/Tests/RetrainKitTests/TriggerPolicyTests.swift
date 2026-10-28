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

    private func encode(_ s: AppState) throws -> Data {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try e.encode(s)
    }

    private func decode(_ data: Data) throws -> AppState {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return try d.decode(AppState.self, from: data)
    }

    func testRoundTripPreservesValues() throws {
        let original = makeState()
        let restored = try decode(try encode(original))
        XCTAssertEqual(restored, original)
    }

    func testMissingKeysFallBackToDefaults() throws {
        // A state.json written by a NEWER app read by an OLDER binary must
        // degrade to defaults, never hard-fail.
        let restored = try decode(Data("{}".utf8))
        XCTAssertEqual(restored.correctionsThreshold, 10)
        XCTAssertEqual(restored.weeklyCadenceDays, 7)
        XCTAssertTrue(restored.weeklyCadenceEnabled)
        XCTAssertEqual(restored.autonomy, .fullAutomatic)
        XCTAssertNil(restored.lastTrainFinishedAt)
        XCTAssertFalse(restored.loginItemEnabled)
    }

    func testUnknownKeysAreIgnored() throws {
        let json = #"{"correctionsThreshold": 5, "nobodyKnowsThisYet": 123}"#
        let restored = try decode(Data(json.utf8))
        XCTAssertEqual(restored.correctionsThreshold, 5)
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
