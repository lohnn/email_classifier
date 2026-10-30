import XCTest
import CryptoKit
@testable import RetrainKit

/// M6 launcher tests — the bootstrap-first ordering, drift halting, setup
/// gate, and config assembly. All seams faked (RuntimeEnsuring,
/// PipelineEvolving, ClassifierServing); no network, no subprocesses.

final class FakeRuntimeEnsuring: RuntimeEnsuring {
    var receipt: RuntimeReceipt
    var error: Error?
    private(set) var ensureCalls = 0

    init(receipt: RuntimeReceipt, error: Error? = nil) {
        self.receipt = receipt
        self.error = error
    }

    func ensureRuntime(manifest: RuntimeManifest, containerDir: URL,
                       log: RunLog) async throws -> RuntimeReceipt {
        ensureCalls += 1
        if let error = error { throw error }
        return receipt
    }
}

final class FakePipeline: PipelineEvolving {
    private(set) var configs: [PipelineConfiguration] = []
    var result: PipelineRunResult

    init(result: PipelineRunResult) {
        self.result = result
    }

    func runTraining(_ config: PipelineConfiguration) async -> PipelineRunResult {
        configs.append(config)
        return result
    }

    func resumeAfterMerge(_ config: PipelineConfiguration) async -> PipelineRunResult {
        configs.append(config)
        return result
    }
}

final class PipelineLauncherTests: XCTestCase {
    private var tmp: URL!
    private var ledger: Ledger!
    private var log: RunLog!

    override func setUpWithError() throws {
        tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("launcher-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        ledger = Ledger(directory: tmp.appendingPathComponent("appdata"))
        log = RunLog(directory: tmp)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmp)
    }

    private func stubManifest(manifestVersion: Int = 1) -> RuntimeManifest {
        // STUB manifest matching makeReceipt versions — launcher tests must
        // never depend on the ambient filesystem search (round-31 lesson:
        // 'bundled manifest is v1' was an environment assumption).
        let json = """
        {
          "manifestVersion": \(manifestVersion),
          "generatedAt": "2026-10-30T00:00:00Z",
          "python": {"version": "3.14"},
          "minFreeGB": 0.0001,
          "layers": {
            "uv": {"name": "uv", "version": "0.13.6", "url": "u", "sha256": "1", "bytes": 1, "kind": "tar.gz", "innerPath": "u"},
            "llamaServer": {"name": "llama-server", "version": "b11982", "url": "l", "sha256": "2", "bytes": 1, "kind": "tar.gz", "binaryName": "llama-server"},
            "torch": {"name": "torch", "version": "2.15.0", "url": "t", "sha256": "3", "bytes": 1, "kind": "wheel"}
          }
        }
        """
        return try! JSONDecoder().decode(RuntimeManifest.self, from: Data(json.utf8))
    }

    private func makeReceipt(manifestVersion: Int = 1) -> RuntimeReceipt {
        // Versions match the COMMITTED manifest — RuntimeDrift.check tests
        // with these must pass; drift tests manufacture mismatches via
        // manifestVersion instead.
        RuntimeReceipt(manifestVersion: manifestVersion,
                       uvVersion: "0.13.6", pythonVersion: "3.14",
                       llamaVersion: "b11982", torchVersion: "2.15.0",
                       uvPath: tmp.appendingPathComponent("uv").path,
                       llamaServerPath: tmp.appendingPathComponent("llama-server").path,
                       venvPath: tmp.appendingPathComponent("venv").path,
                       installedAt: Date(), snapshotRequirementsSHA: nil)
    }

    private func configure(url: String) {
        _ = ledger.update { $0.classifierURL = url }
    }

    func testEmptyClassifierURLIsSetupOutcome_notError() async {
        // Operator decision 2026-10-29: empty URL = SETUP state. Nothing
        // (ensure/pipeline) may run before it is configured.
        configure(url: "")
        let ensure = FakeRuntimeEnsuring(receipt: makeReceipt())
        let pipeline = FakePipeline(result: .skipped(reason: "x"))
        let launcher = PipelineLauncher(ensure: ensure, pipeline: pipeline)
        let outcome = await launcher.execute(phase: .train, ledger: ledger,
                                             containerDir: tmp, runID: "run-1", log: log)
        XCTAssertEqual(outcome.headline, "Classifier URL not set")
        XCTAssertTrue(outcome.pinnedFailure)
        XCTAssertEqual(ensure.ensureCalls, 0)
        XCTAssertTrue(pipeline.configs.isEmpty)
    }

    func testDriftHaltsBeforePipeline() async {
        configure(url: "http://jarvis:8008")
        // bundled manifest is v1 — a v1 receipt + version-1 manifest matches;
        // force drift via an OLD/manufactured receipt version mismatch:
        let ensure = FakeRuntimeEnsuring(receipt: makeReceipt(manifestVersion: 99))
        let pipeline = FakePipeline(result: .completed(PipelineSummary(
            runID: "r", vocabBranch: "vocab/r", vocabPushed: true,
            stagedGGUFBytes: 1, dataSHA: "d", durations: [:])))
        // v1 stub vs v99 receipt: drift BY DESIGN in the stub, not ambient.
        let launcher = PipelineLauncher(ensure: ensure, pipeline: pipeline,
                                        manifestProvider: { try! stubManifest(manifestVersion: 1) })
        let outcome = await launcher.execute(phase: .train, ledger: ledger,
                                             containerDir: tmp, runID: "run-1", log: log)
        XCTAssertEqual(outcome.headline, "Runtime drift — run halted")
        XCTAssertTrue(outcome.pinnedFailure)
        XCTAssertTrue(pipeline.configs.isEmpty)
    }

    func testInsufficientDiskBecomesPinnedOutcome() async {
        configure(url: "http://jarvis:8008")
        let stub = stubManifest()
        let ensure = FakeRuntimeEnsuring(receipt: makeReceipt(),
                                         error: BootstrapError.insufficientDisk(neededGB: 8, availableGB: 1.2))
        let launcher = PipelineLauncher(ensure: ensure, pipeline: FakePipeline(result: .skipped(reason: "x")),
                                        manifestProvider: { stub })
        let outcome = await launcher.execute(phase: .train, ledger: ledger,
                                             containerDir: tmp, runID: "run-1", log: log)
        XCTAssertTrue(outcome.headline.contains("Not enough disk space"))
        XCTAssertTrue(outcome.pinnedFailure)
    }

    func testHappyTrainPassAssemblesConfigFromReceiptAndState() async throws {
        configure(url: "http://jarvis:8008")
        let receipt = makeReceipt()
        let ensure = FakeRuntimeEnsuring(receipt: receipt)
        let completed = PipelineSummary(runID: "run-1", vocabBranch: "vocab/run-1",
                                        vocabPushed: true, stagedGGUFBytes: 7,
                                        dataSHA: String(repeating: "d", count: 40),
                                        durations: ["training": 1.0])
        let pipeline = FakePipeline(result: .completed(completed))
        let launcher = PipelineLauncher(ensure: ensure, pipeline: pipeline)

        let outcome = await launcher.execute(phase: .train, ledger: ledger,
                                             containerDir: tmp, runID: "run-1", log: log,
                                             manifestProvider: { stubManifest() })
        XCTAssertEqual(outcome.headline, "Retrain run completed")
        XCTAssertFalse(outcome.pinnedFailure)
        XCTAssertEqual(outcome.dotReachable, true)
        XCTAssertEqual(outcome.runSummary?.vocabBranch, "vocab/run-1")

        // config assembly: paths derive from the receipt + container dir
        let config = try XCTUnwrap(pipeline.configs.first)
        XCTAssertEqual(config.runID, "run-1")
        XCTAssertEqual(config.pythonBin.path,
                       receipt.venvPath + "/bin/python")
        XCTAssertEqual(config.llamaServerBin.path, receipt.llamaServerPath)
        XCTAssertEqual(config.workspaceDir, tmp.appendingPathComponent("workspace"))
        XCTAssertEqual(config.runsDir, tmp.appendingPathComponent("runs"))
        // production shape carries the isDirectory trailing component
        XCTAssertEqual(config.trainingDataDir,
                       tmp.appendingPathComponent("workspace/training_data", isDirectory: true))
    }

    func testResumeExecutesShipPass() async {
        configure(url: "http://jarvis:8008")
        let ensure = FakeRuntimeEnsuring(receipt: makeReceipt())
        let completed = PipelineSummary(runID: "run-1", vocabBranch: "vocab/run-1",
                                        vocabPushed: true, stagedGGUFBytes: 7,
                                        dataSHA: "d", durations: [:])
        let pipeline = FakePipeline(result: .completed(completed))
        let launcher = PipelineLauncher(ensure: ensure, pipeline: pipeline)
        let outcome = await launcher.execute(phase: .resume, ledger: ledger,
                                             containerDir: tmp, runID: "run-1", log: log,
                                             manifestProvider: { stubManifest() })
        XCTAssertEqual(outcome.headline, "Retrain run completed")
        XCTAssertEqual(pipeline.configs.first?.runID, "run-1")
    }

    func testNewRunIDFormatAndPhaseHelpers() {
        let launcher = PipelineLauncher(ensure: FakeRuntimeEnsuring(receipt: makeReceipt()))
        let runsDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("launcher-phase-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: runsDir, withIntermediateDirectories: true)

        XCTAssertEqual(launcher.phase(runsDir: runsDir), .train)  // no run state → train

        let runID = PipelineLauncher.newRunID()
        XCTAssertTrue(runID.hasPrefix("run-") && runID.count == "run-20261029-120000".count)
    }
}
