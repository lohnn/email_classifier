import XCTest
@testable import RetrainKit

/// M4 pipeline-core tests: mirror-event wire shape, classifier client
/// parsing (URLProtocol-stubbed), and the runner's stage semantics with
/// fakes (no real subprocess, no real network). Deterministic: mirror
/// posts are AWAITED inline by the runner.

final class StubURLProtocol: URLProtocol {
    private static let lock = NSLock()
    private static var stored: ((URLRequest) -> (HTTPURLResponse, Data))?

    /// Set per-test BEFORE the session is created; returns the canned reply.
    static var handler: ((URLRequest) -> (HTTPURLResponse, Data))? {
        get {
            lock.lock()
            defer { lock.unlock() }
            return stored
        }
        set {
            lock.lock()
            stored = newValue
            lock.unlock()
        }
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let handler = Self.handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        let (response, data) = handler(request)
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

private func httpResponse(_ status: Int, _ url: URL) -> HTTPURLResponse {
    HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: nil)!
}

/// Runs an async throws body synchronously (tests never race the helper).
private func awaitSync<T>(_ body: @escaping () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    var value: T?
    var thrown: Error?
    Task {
        do { value = try await body() }
        catch { thrown = error }
        semaphore.signal()
    }
    semaphore.wait()
    if let thrown = thrown { throw thrown }
    return value!
}

/// Runs an async (non-throwing) body synchronously.
private func awaitSyncValue<T>(_ body: @escaping () async -> T) -> T {
    let semaphore = DispatchSemaphore(value: 0)
    var value: T?
    Task {
        value = await body()
        semaphore.signal()
    }
    semaphore.wait()
    return value!
}

final class ClassifierClientTests: XCTestCase {
    private func stubbedClient() throws -> ClassifierClient {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubURLProtocol.self]
        let session = URLSession(configuration: config)
        return try ClassifierClient(baseURL: "http://jarvis:8008/", session: session)
    }

    func testNormalizedBaseURL() {
        XCTAssertEqual(ClassifierClient.normalizedBaseURL(" http://jarvis:8008/ "), "http://jarvis:8008")
        XCTAssertEqual(ClassifierClient.normalizedBaseURL("http://100.79.150.106:8008"),
                       "http://100.79.150.106:8008")
        XCTAssertNil(ClassifierClient.normalizedBaseURL(""))                 // no defaults (operator rule)
        XCTAssertNil(ClassifierClient.normalizedBaseURL("jarvis:8008"))      // scheme required
        XCTAssertNil(ClassifierClient.normalizedBaseURL("ftp://x"))          // http/https only
        XCTAssertNil(ClassifierClient.normalizedBaseURL("http:///no-host"))  // host required
    }

    func testHealthParsing() throws {
        let client = try stubbedClient()
        StubURLProtocol.handler = { request in
            XCTAssertEqual(request.url?.path, "/health")
            return (httpResponse(200, request.url!),
                    Data(#"{"status":"ok","checks":{"database":"ok","model":"ok"}}"#.utf8))
        }
        let report = try awaitSync { try await client.health() }
        XCTAssertTrue(report.healthy)
        XCTAssertEqual(report.checks["model"], "ok")
    }

    func testHealth503IsUnhealthyButNotThrown() throws {
        let client = try stubbedClient()
        StubURLProtocol.handler = { request in
            (httpResponse(503, request.url!),
             Data(#"{"status":"error","checks":{"model":"not loaded"}}"#.utf8))
        }
        let report = try awaitSync { try await client.health() }
        XCTAssertFalse(report.healthy)
        XCTAssertEqual(report.status, "error")
    }

    func testMirrorEventWireShapeMatchesBrainContract() throws {
        let event = RetrainMirrorEvent(
            event: .runFinished, at: Date(timeIntervalSince1970: 1_770_000_000),
            runID: "run-1", status: .ok,
            dataSHA: String(repeating: "a", count: 40),
            durations: ["training": 640.0],
            error: "manual GGUF merge required"
        )
        let data = try event.mirrorJSONData()
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        // EXACTLY the brain's contract keys (unknown fields would 422)
        XCTAssertEqual(Set(json.keys),
                       ["event", "at", "run_id", "status", "data_sha", "durations", "error"])
        XCTAssertEqual(json["run_id"] as? String, "run-1")
        XCTAssertEqual(json["event"] as? String, "run_finished")
        // Wire 'at' = UTC ISO8601. Equality is checked THROUGH the same
        // formatter the encoder uses — a hand-typed wall-clock string
        // silently pinned a local timezone (build-round-7 lesson).
        let wire = json["at"] as? String
        let formatter = ISO8601DateFormatter()  // UTC by definition
        let expected = formatter.string(from: event.at)
        XCTAssertEqual(wire, expected)
        XCTAssertTrue(wire?.hasSuffix("Z") == true, "wire 'at' must be UTC-suffixed")
        XCTAssertNotNil(formatter.date(from: wire ?? ""), "wire 'at' must parse back")
    }

    func testSendMirrorEventParsesReceipt() throws {
        let client = try stubbedClient()
        StubURLProtocol.handler = { request in
            XCTAssertEqual(request.url?.path, "/admin/retrain-events")
            XCTAssertEqual(request.httpMethod, "POST")
            return (httpResponse(200, request.url!),
                    Data(#"{"status":"recorded","duplicate":false,"event":{}}"#.utf8))
        }
        let receipt = try awaitSync {
            try await client.sendMirrorEvent(
                RetrainMirrorEvent(event: .runStarted, at: Date(), runID: "r", status: .ok)
            )
        }
        XCTAssertEqual(receipt.status, "recorded")
        XCTAssertFalse(receipt.duplicate)
    }

    func testSnapshotHeadersAndTempArchive() throws {
        let client = try stubbedClient()
        StubURLProtocol.handler = { request in
            XCTAssertEqual(request.url?.path, "/admin/training-snapshot")
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
                headerFields: [
                    "X-Snapshot-Code-Commit": String(repeating: "a", count: 40),
                    "X-Snapshot-Data-Commit": String(repeating: "b", count: 40),
                    "X-Snapshot-Data-Repo": "git",
                    "X-Snapshot-Bytes": "7",
                ]
            )!
            return (response, Data("tar-bytes".utf8))
        }
        let snapshot = try awaitSync { try await client.fetchSnapshot() }
        XCTAssertEqual(snapshot.codeCommit, String(repeating: "a", count: 40))
        XCTAssertEqual(snapshot.dataRepoMode, "git")
        XCTAssertEqual(snapshot.bytes, 7)
        defer { try? FileManager.default.removeItem(at: snapshot.archiveFile) }
        XCTAssertEqual(try Data(contentsOf: snapshot.archiveFile), Data("tar-bytes".utf8))
    }

    func testPostCentroidsCarriesRunIDAndSha() throws {
        let client = try stubbedClient()
        var seenURL: URL?
        StubURLProtocol.handler = { request in
            seenURL = request.url
            return (httpResponse(200, request.url!),
                    Data(#"{"status":"committed","branch":"vocab/run-7","commit":"c","pushed":true}"#.utf8))
        }
        let payload = Data(#"{"centroids_l2":{}}"#.utf8)
        let receipt = try awaitSync { try await client.postCentroids(payload, runID: "run-7") }
        XCTAssertEqual(receipt.status, "committed")
        XCTAssertEqual(receipt.pushed, true)
        let query = seenURL?.query ?? ""
        XCTAssertTrue(query.contains("run_id=run-7"))
        XCTAssertTrue(query.contains("sha256_expected="))
    }

    func testHTTPErrorSurfacesAsClassifierError() throws {
        // W-121 discipline: non-2xx must be LOUD on every guarded path
        // (snapshot/mirror/centroids/upload all carry the 200-guard).
        // health() is the documented exception — see the next test.
        let client = try stubbedClient()
        StubURLProtocol.handler = { request in
            (httpResponse(403, request.url!), Data("nope".utf8))
        }
        XCTAssertThrowsError(try awaitSync { try await client.fetchSnapshot() }) { error in
            XCTAssertEqual(error as? ClassifierError, .http(status: 403, body: "nope"))
        }
    }

    func testHealthHTTPErrorIsDataNotException() throws {
        /// health() deliberately NEVER throws on HTTP-level errors: the
        /// pre-flight skip logic needs unhealthiness as DATA (a 503
        /// model-not-loaded skip is a normal outcome, mirrored — not an
        /// exception). Transport errors still throw. (Round-7 lesson: the
        /// 403-throw expectation was mis-wired onto this endpoint.)
        let client = try stubbedClient()
        StubURLProtocol.handler = { request in
            XCTAssertEqual(request.url?.path, "/health")
            // Multi-statement closure: explicit return required (round-8).
            return (httpResponse(403, request.url!), Data("nope".utf8))
        }
        let report = try awaitSync { try await client.health() }
        XCTAssertEqual(report.httpStatus, 403)
        XCTAssertFalse(report.healthy)   // unhealthiness as data, by contract
        XCTAssertNil(report.status)      // body was not JSON
    }
}

// ---------------------------------------------------------------------------
// Runner semantics with fakes
// ---------------------------------------------------------------------------

/// Thread-safe event collector (mirror posts are awaited, but the sink is
/// shared across fakes so locking stays honest).
final class MirrorEventSink {
    private let lock = NSLock()
    private var list: [RetrainMirrorEvent] = []
    func append(_ event: RetrainMirrorEvent) {
        lock.lock()
        list.append(event)
        lock.unlock()
    }
    func snapshot() -> [RetrainMirrorEvent] {
        lock.lock()
        defer { lock.unlock() }
        return list
    }
}

final class FakeClassifier: ClassifierServing {
    var healthReport = HealthReport(httpStatus: 200, status: "ok", checks: [:])
    var snapshotArchive: URL
    let events: MirrorEventSink
    var failSnapshot = false

    init(events: MirrorEventSink, snapshotArchive: URL) {
        self.events = events
        self.snapshotArchive = snapshotArchive
    }

    func health() async throws -> HealthReport { healthReport }
    func fetchSnapshot() async throws -> SnapshotHeaders {
        if failSnapshot { throw ClassifierError.http(status: 500, body: "boom") }
        return SnapshotHeaders(codeCommit: String(repeating: "c", count: 40),
                               dataCommit: String(repeating: "d", count: 40),
                               dataRepoMode: "git", bytes: 3, archiveFile: snapshotArchive)
    }
    func sendMirrorEvent(_ event: RetrainMirrorEvent) async throws -> MirrorReceipt {
        events.append(event)
        return MirrorReceipt(status: "recorded", duplicate: false)
    }
    func postCentroids(_ data: Data, runID: String) async throws -> VocabReceipt {
        VocabReceipt(status: "committed", branch: "vocab/\(runID)",
                     commit: "abc", pushed: true, pushError: nil)
    }
    func uploadArtifact(_ data: Data, filename: String, sha256: String) async throws -> UploadReceipt {
        UploadReceipt(status: "staged", stagedAs: filename, size: data.count, sha256: sha256)
    }
}

final class FakeCommands: CommandRunning {
    /// Observability hook: (stage, launchPath, arguments) before the fake success.
    var onRun: ((String, String, [String]) -> Void)?
    func run(stage: String, launchPath: String, arguments: [String],
             workingDirectory: URL, environment: [String: String],
             log: RunLog, timeout: TimeInterval) throws -> CommandResult {
        onRun?(stage, launchPath, arguments)
        return CommandResult(exitCode: 0, durationSeconds: 0.01, allOutput: "fake ok")
    }
}

final class FakeLlama: LlamaServing {
    var started = false
    var stopped = false
    func start(model: URL, port: UInt16, log: RunLog, waitTimeout: TimeInterval) async throws {
        started = true
    }
    func stop() { stopped = true }
}

final class PipelineRunnerTests: XCTestCase {
    /// Materializes the trainer-provision fixture: a snapshot-shape brain dir
    /// with the pinned requirements file + a runtime receipt whose recorded
    /// sha MATCHES the file (the satisfied state — the happy-path default, so
    /// no test silently installs). `requirements` + `receiptSHA` let tests
    /// manufacture the pin-change / stale cases.
    @discardableResult
    private func materializeSnapshotSkeleton(tmp: URL,
                                             requirements: String = PipelineRunnerTests.DEFAULT_REQS,
                                             receiptSHA: String? = nil) throws -> URL {
        let brain = tmp.appendingPathComponent("workspace/email_classifier_brain", isDirectory: true)
        try FileManager.default.createDirectory(at: brain, withIntermediateDirectories: true)
        let reqFile = brain.appendingPathComponent("train_lora_requirements.txt")
        try Data(requirements.utf8).write(to: reqFile)
        var sha = receiptSHA
        if receiptSHA == nil { sha = try PipelineRunner.fileSHA256(reqFile) }
        let paths = RuntimeBootstrapper.Paths.under(containerDir: tmp)
        // writeReceipt writes ATOMICALLY but does NOT create rt/ — unlike
        // ensureRuntime, which builds the tree first (round-37 fixture
        // lesson: the swallowed try? at call sites turned the missing dir
        // into the runner's honest "no runtime receipt" failure).
        try FileManager.default.createDirectory(at: paths.rtDir,
                                                withIntermediateDirectories: true)
        let receipt = RuntimeReceipt(
            manifestVersion: 1, uvVersion: "0.13.6T",
            pythonVersion: "3.14", llamaVersion: "b11982T", torchVersion: "2.15.0T",
            uvPath: paths.binDir.appendingPathComponent("uv").path,
            llamaServerPath: paths.binDir.appendingPathComponent("llama-server").path,
            venvPath: paths.rtDir.appendingPathComponent("venv").path,
            installedAt: Date(), snapshotRequirementsSHA: sha)
        try RuntimeBootstrapper.writeReceipt(receipt, rtDir: paths.rtDir)
        // fixture self-check: a fixture that silently fails LIES (never
        // again — this assert fires in-harness if the write didn't land)
        XCTAssert(FileManager.default.fileExists(
            paths.rtDir.appendingPathComponent("receipt.json").path),
            "materializeSnapshotSkeleton: receipt write did not land")
        return reqFile
    }

    private static let DEFAULT_REQS = "peft==0.21.2\nsentence-transformers==6.1.0\ntransformers==5.19.0\n"

    private func makeConfig(tmp: URL, events: MirrorEventSink,
                            classifier: ClassifierServing,
                            commands: CommandRunning,
                            llama: FakeLlama) -> PipelineConfiguration {
        let log = RunLog(directory: tmp)
        return PipelineConfiguration(
            runID: "run-test-1", classifier: classifier, commands: commands,
            llama: llama, log: log, containerDir: tmp,
            workspaceDir: tmp.appendingPathComponent("workspace"),
            runsDir: tmp.appendingPathComponent("runs"),
            pythonBin: URL(fileURLWithPath: "/usr/bin/python-fake"),
            llamaServerBin: URL(fileURLWithPath: "/usr/bin/llama-server-fake"),
            uvBin: tmp.appendingPathComponent("rt/bin/uv")
        )
    }

    private func freshTmp() -> URL {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("p2-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        return tmp
    }

    func testUnhealthyPreflightSkipsWithMirroredSkip() async {
        let tmp = freshTmp()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let sink = MirrorEventSink()
        let classifier = FakeClassifier(events: sink, snapshotArchive: tmp)
        classifier.healthReport = HealthReport(httpStatus: 503, status: "error",
                                               checks: ["model": "not loaded"])
        let config = makeConfig(tmp: tmp, events: sink, classifier: classifier,
                                commands: FakeCommands(), llama: FakeLlama())
        let outcome = awaitSyncValue { await PipelineRunner().runTraining(config) }
        guard case .skipped(let reason) = outcome else {
            return XCTFail("expected skipped, got \(outcome)")
        }
        XCTAssertTrue(reason.contains("503"))
        XCTAssertEqual(sink.snapshot().map { $0.event }, [.runStarted, .skipped])
    }

    func testSnapshotFailureFailsRun() async {
        let tmp = freshTmp()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let sink = MirrorEventSink()
        let classifier = FakeClassifier(events: sink, snapshotArchive: tmp)
        classifier.failSnapshot = true
        let config = makeConfig(tmp: tmp, events: sink, classifier: classifier,
                                commands: FakeCommands(), llama: FakeLlama())
        let outcome = awaitSyncValue { await PipelineRunner().runTraining(config) }
        guard case .failed(let stage, _) = outcome else {
            return XCTFail("expected failed, got \(outcome)")
        }
        XCTAssertTrue(stage.contains("preflight"))
        XCTAssertTrue(sink.snapshot().contains { $0.event == .runFailed })
    }

    func testTrainingPassStopsAtMergeCheckpoint() async {
        let tmp = freshTmp()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let sink = MirrorEventSink()
        let classifier = FakeClassifier(events: sink, snapshotArchive: tmp)
        let config = makeConfig(tmp: tmp, events: sink, classifier: classifier,
                                commands: FakeCommands(), llama: FakeLlama())
        _ = try? materializeSnapshotSkeleton(tmp: tmp)
        let outcome = awaitSyncValue { await PipelineRunner().runTraining(config) }
        guard case .waitingForManualMerge(let state) = outcome else {
            return XCTFail("expected waitingForManualMerge, got \(outcome)")
        }
        XCTAssertEqual(state.phase, "awaiting_manual_merge")
        XCTAssertEqual(state.dataCommit, String(repeating: "d", count: 40))
        XCTAssertNotNil(PipelineRunStore.load(runsDir: config.runsDir, runID: config.runID))
        XCTAssertTrue(FileManager.default.fileExists(atPath: config.mergedGGUFURL.deletingLastPathComponent().path))
        // EXACT event sequence: run_started + run_finished(merge required).
        // The dedupe key means run_finished can only fire once per run.
        let events = sink.snapshot()
        XCTAssertEqual(events.map { $0.event }, [.runStarted, .runFinished])
        XCTAssertEqual(events.last?.status, .ok)
        XCTAssertTrue(events.last?.error?.contains("merge required") == true)
        XCTAssertFalse(FileManager.default.fileExists(atPath: config.mergedGGUFURL.path))
    }

    func testResumeWithoutMergedGGUFFails() async {
        let tmp = freshTmp()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let sink = MirrorEventSink()
        let classifier = FakeClassifier(events: sink, snapshotArchive: tmp)
        let config = makeConfig(tmp: tmp, events: sink, classifier: classifier,
                                commands: FakeCommands(), llama: FakeLlama())
        _ = try? materializeSnapshotSkeleton(tmp: tmp)
        _ = awaitSyncValue { await PipelineRunner().runTraining(config) }
        let outcome = awaitSyncValue { await PipelineRunner().resumeAfterMerge(config) }
        guard case .failed(let stage, let message) = outcome else {
            return XCTFail("expected failed, got \(outcome)")
        }
        XCTAssertEqual(stage, "merge-checkpoint")
        XCTAssertTrue(message.contains("missing"))
    }

    func testFullResumeCompletesAndShips() async {
        let tmp = freshTmp()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let sink = MirrorEventSink()
        let classifier = FakeClassifier(events: sink, snapshotArchive: tmp)
        let llama = FakeLlama()
        let commands = FakeCommands()
        let config = makeConfig(tmp: tmp, events: sink, classifier: classifier,
                                commands: commands, llama: llama)
        _ = try? materializeSnapshotSkeleton(tmp: tmp)
        _ = awaitSyncValue { await PipelineRunner().runTraining(config) }

        // The operator's merge lands the GGUF at the checkpoint path; the
        // fake centroids "script" produces the artifact the ship stage reads.
        let ggufBytes = Data("fake-gguf".utf8)
        try? ggufBytes.write(to: config.mergedGGUFURL)
        commands.onRun = { _, _, args in
            if args.contains("build_label_centroids.py"),
               let outFlag = args.firstIndex(of: "--out"), outFlag + 1 < args.count {
                try? Data(#"{"centroids_l2":{"Jarvis/FOCUS":[0.1]}}"#.utf8)
                    .write(to: URL(fileURLWithPath: args[outFlag + 1]))
            }
        }

        let outcome = awaitSyncValue { await PipelineRunner().resumeAfterMerge(config) }
        guard case .completed(let summary) = outcome else {
            return XCTFail("expected completed, got \(outcome)")
        }
        XCTAssertEqual(summary.vocabBranch, "vocab/run-test-1")
        XCTAssertTrue(summary.vocabPushed)
        XCTAssertEqual(summary.stagedGGUFBytes, ggufBytes.count)
        XCTAssertTrue(llama.started)
        XCTAssertTrue(llama.stopped)  // failure-free path still stops the server
        // terminal event = pr_opened with the staged-GGUF note
        let events = sink.snapshot()
        XCTAssertEqual(events.last?.event, .prOpened)
        XCTAssertTrue(events.last?.error?.contains("staged") == true)
    }

    // MARK: trainer-requirements stage (snapshot-driven provisioning, 2026-10-30)

    func testTrainerRequirementsSatisfiedSkipsInstall() async {
        let tmp = freshTmp()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let sink = MirrorEventSink()
        let classifier = FakeClassifier(events: sink, snapshotArchive: tmp)
        let commands = FakeCommands()
        let config = makeConfig(tmp: tmp, events: sink, classifier: classifier,
                                commands: commands, llama: FakeLlama())
        _ = try? materializeSnapshotSkeleton(tmp: tmp)  // receipt sha == file sha
        var sawInstall = false
        commands.onRun = { stage, _, _ in
            if stage == "trainer-requirements" { sawInstall = true }
        }
        let outcome = awaitSyncValue { await PipelineRunner().runTraining(config) }
        guard case .waitingForManualMerge = outcome else {
            return XCTFail("expected waitingForManualMerge, got \(outcome)")
        }
        XCTAssertFalse(sawInstall, "satisfied pins must short-circuit the install")
        // the bootstrapper receipt still records the ORIGINAL sha
        let bootstrapper = RuntimeBootstrapper(commands: commands,
                                               fetcher: URLArtifactFetcher())
        let receipt = bootstrapper.readReceipt(
            paths: RuntimeBootstrapper.Paths.under(containerDir: tmp))
        XCTAssertEqual(receipt?.snapshotRequirementsSHA,
                       try PipelineRunner.fileSHA256(config.brainDir
                           .appendingPathComponent("train_lora_requirements.txt")))
    }

    func testTrainerRequirementsPinChangeReinstallsAndRecordsSHA() async throws {
        let tmp = freshTmp()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let sink = MirrorEventSink()
        let classifier = FakeClassifier(events: sink, snapshotArchive: tmp)
        let commands = FakeCommands()
        let config = makeConfig(tmp: tmp, events: sink, classifier: classifier,
                                commands: commands, llama: FakeLlama())
        let reqFile = try? materializeSnapshotSkeleton(tmp: tmp)
        // the snapshot lands CHANGED pins: rewrite the file after the receipt
        try? Data(Self.DEFAULT_REQS.appending("peft==0.22.0\n").utf8)
            .write(to: reqFile ?? config.brainDir
                .appendingPathComponent("train_lora_requirements.txt"))
        let uvBinURL = tmp.appendingPathComponent("rt/bin/uv")
        try? FileManager.default.createDirectory(at: uvBinURL.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        try? Data("uv".utf8).write(to: uvBinURL)

        var invoked: (stage: String, launchPath: String, args: [String])?
        commands.onRun = { stage, launchPath, args in
            if stage == "trainer-requirements" { invoked = (stage, launchPath, args) }
        }
        let outcome = awaitSyncValue { await PipelineRunner().runTraining(config) }
        guard case .waitingForManualMerge = outcome else {
            return XCTFail("expected waitingForManualMerge, got \(outcome)")
        }
        let call = try XCTUnwrap(invoked, "changed pins must reinstall")
        XCTAssertEqual(call.launchPath, config.uvBin.path)
        XCTAssertEqual(call.args,
                       ["pip", "install", "--python", config.pythonBin.path, "-r",
                        config.brainDir.appendingPathComponent("train_lora_requirements.txt").path])
        // the receipt recorded the INSTALLED file's sha (auto-heal anchor)
        let bootstrapper = RuntimeBootstrapper(commands: commands,
                                               fetcher: URLArtifactFetcher())
        let receipt = bootstrapper.readReceipt(
            paths: RuntimeBootstrapper.Paths.under(containerDir: tmp))
        XCTAssertEqual(receipt?.snapshotRequirementsSHA,
                       try PipelineRunner.fileSHA256(reqFile ?? URL(fileURLWithPath: "/dev/null")))
    }

    func testTrainerRequirementsMissingFileAbortsActionably() async {
        let tmp = freshTmp()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let sink = MirrorEventSink()
        let classifier = FakeClassifier(events: sink, snapshotArchive: tmp)
        let config = makeConfig(tmp: tmp, events: sink, classifier: classifier,
                                commands: FakeCommands(), llama: FakeLlama())
        // NO skeleton: the snapshot unpacked without the requirements file
        let outcome = awaitSyncValue { await PipelineRunner().runTraining(config) }
        guard case .failed(let stage, let message) = outcome else {
            return XCTFail("expected failed, got \(outcome)")
        }
        XCTAssertEqual(stage, "trainer-requirements")
        XCTAssertTrue(message.contains("snapshot is incomplete"), message)
        XCTAssertTrue(sink.snapshot().contains { $0.event == .runFailed })
    }

    func testTrainerRequirementsMissingUvIsPinnedError() async {
        let tmp = freshTmp()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let sink = MirrorEventSink()
        let classifier = FakeClassifier(events: sink, snapshotArchive: tmp)
        let config = makeConfig(tmp: tmp, events: sink, classifier: classifier,
                                commands: FakeCommands(), llama: FakeLlama())
        _ = try? materializeSnapshotSkeleton(tmp: tmp, receiptSHA: "stale-sha")
        // stale receipt sha ≠ file sha ⇒ install path ⇒ uv must exist; it does not
        let outcome = awaitSyncValue { await PipelineRunner().runTraining(config) }
        guard case .failed(let stage, let message) = outcome else {
            return XCTFail("expected failed, got \(outcome)")
        }
        XCTAssertEqual(stage, "trainer-requirements")
        XCTAssertTrue(message.contains("uv binary missing"), message)
        XCTAssertTrue(sink.snapshot().contains { $0.event == .runFailed })
    }
}
