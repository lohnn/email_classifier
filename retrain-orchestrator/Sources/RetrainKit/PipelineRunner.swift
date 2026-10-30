import Foundation
import CryptoKit

/// The P2 pipeline core (design §5, Mode A): pre-flight → snapshot → MPS
/// LoRA train → [HUMAN MERGE CHECKPOINT] → Metal centroids on the merged
/// GGUF → GGUF staged upload → vocabulary POST → mirror events.
///
/// Event-mirroring contract (the brain dedupes on (run_id, event)):
///   run_started        once, at training-pass start
///   run_finished       once, at the merge checkpoint — error carries the
///                      "manual GGUF merge required" pointer (the training
///                      pass is what finished; P3 removes this stop)
///   run_failed         once per run on any failure (either pass)
///   skipped            once, on pre-flight unreachable/unhealthy
///   pr_opened          once, on ship-pass completion (vocab branch pushed;
///                      error carries branch + staged-GGUF note)
/// Mirror POSTs are best-effort: a failed post logs a warning and does not
/// fail the pipeline (retry policy is P5 hardening). Stage timing is kept
/// OUT of closures (plain local vars, mutated in the owning async function
/// — no captured-var mutation in child tasks).

public struct PipelineRunState: Codable, Equatable {
    public var runID: String
    /// "preflight" | "snapshot" | "training" | "awaiting_manual_merge" |
    /// "embedding" | "shipping" | "completed" | "failed" | "skipped"
    public var phase: String
    public var startedAt: Date
    public var updatedAt: Date
    public var codeCommit: String?
    public var dataCommit: String?
    public var dataRepoMode: String?
    /// Where the operator must place the merged GGUF (merge instructions).
    public var mergeTargetPath: String?
    public var lastMessage: String?

    public init(runID: String, phase: String, startedAt: Date, updatedAt: Date,
                codeCommit: String?, dataCommit: String?, dataRepoMode: String?,
                mergeTargetPath: String?, lastMessage: String?) {
        self.runID = runID
        self.phase = phase
        self.startedAt = startedAt
        self.updatedAt = updatedAt
        self.codeCommit = codeCommit
        self.dataCommit = dataCommit
        self.dataRepoMode = dataRepoMode
        self.mergeTargetPath = mergeTargetPath
        self.lastMessage = lastMessage
    }
}

public struct PipelineSummary: Equatable {
    public let runID: String
    public let vocabBranch: String
    public let vocabPushed: Bool
    public let stagedGGUFBytes: Int
    public let dataSHA: String
    public let durations: [String: Double]

    public init(runID: String, vocabBranch: String, vocabPushed: Bool,
                stagedGGUFBytes: Int, dataSHA: String, durations: [String: Double]) {
        self.runID = runID
        self.vocabBranch = vocabBranch
        self.vocabPushed = vocabPushed
        self.stagedGGUFBytes = stagedGGUFBytes
        self.dataSHA = dataSHA
        self.durations = durations
    }
}

public enum PipelineRunResult {
    case completed(PipelineSummary)
    /// Training pass done — mirrors "run finished (merge required)" and
    /// stops. Resume via resumeAfterMerge.
    case waitingForManualMerge(PipelineRunState)
    case failed(stage: String, message: String)
    case skipped(reason: String)
}

public protocol LlamaServing {
    /// Start llama-server on 127.0.0.1:port serving `model`, wait until its
    /// /health answers 200 (throws on deadline).
    func start(model: URL, port: UInt16, log: RunLog, waitTimeout: TimeInterval) async throws
    func stop()
}

public struct PipelineConfiguration {
    public var runID: String
    public var classifier: ClassifierServing
    public var commands: CommandRunning
    public var llama: LlamaServing
    public var log: RunLog

    /// App data dir (state.json's home).
    public var containerDir: URL
    /// Unpacked snapshot root: workspace/email_classifier_brain, .../training_data.
    public var workspaceDir: URL
    /// runs/<run_id>/ scratch + artifacts.
    public var runsDir: URL
    /// venv python from the embedded runtime bootstrap (M5).
    public var pythonBin: URL
    /// Metal llama-server binary from the embedded runtime bootstrap (M5).
    public var llamaServerBin: URL
    /// Pinned uv binary (bootstrap layer 1) — drives the idempotent
    /// `uv pip install -r` of the snapshot's trainer-requirements pins.
    public var uvBin: URL

    /// Non-conflicting local port for the lab llama-server (the Pi serves
    /// :8090; the historical lab default was :4503 — 4599 avoids both).
    public var llamaPort: UInt16 = 4599
    public var trainTimeout: TimeInterval = 3 * 3600
    public var centroidsTimeout: TimeInterval = 2 * 3600
    public var llamaStartTimeout: TimeInterval = 900
    /// Snapshot-driven trainer-stack install (transformers/peft + transitive —
    /// hundreds-of-MB download class, consented like the bootstrap).
    public var trainerProvisionTimeout: TimeInterval = 1800

    /// Merged-GGUF filename shipped to the classifier (their LLAMA_MODEL_FILE
    /// default name; the Pi path alignment rule is a P4 ship concern).
    public static let ggufFilename = "merged_text_q8_0.gguf"

    public init(runID: String, classifier: ClassifierServing, commands: CommandRunning,
                llama: LlamaServing, log: RunLog, containerDir: URL,
                workspaceDir: URL, runsDir: URL, pythonBin: URL, llamaServerBin: URL,
                uvBin: URL) {
        self.runID = runID
        self.classifier = classifier
        self.commands = commands
        self.llama = llama
        self.log = log
        self.containerDir = containerDir
        self.workspaceDir = workspaceDir
        self.runsDir = runsDir
        self.pythonBin = pythonBin
        self.llamaServerBin = llamaServerBin
        self.uvBin = uvBin
    }

    public var runDir: URL { runsDir.appendingPathComponent(runID, isDirectory: true) }
    public var mergedGGUFURL: URL { runDir.appendingPathComponent(PipelineConfiguration.ggufFilename) }
    public var centroidsURL: URL { runDir.appendingPathComponent("label_centroids.json") }
    public var brainDir: URL { workspaceDir.appendingPathComponent("email_classifier_brain", isDirectory: true) }
    public var trainingDataDir: URL { workspaceDir.appendingPathComponent("training_data", isDirectory: true) }
}

enum PipelineRunStore {
    static func stateURL(runsDir: URL, runID: String) -> URL {
        runsDir.appendingPathComponent(runID, isDirectory: true)
            .appendingPathComponent("state.json")
    }

    static func save(_ state: PipelineRunState, runsDir: URL) throws {
        try FileManager.default.createDirectory(
            at: runsDir.appendingPathComponent(state.runID, isDirectory: true),
            withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(state)
        try data.write(to: stateURL(runsDir: runsDir, runID: state.runID), options: .atomic)
    }

    static func load(runsDir: URL, runID: String) -> PipelineRunState? {
        guard let data = try? Data(contentsOf: stateURL(runsDir: runsDir, runID: runID)) else {
            return nil
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(PipelineRunState.self, from: data)
    }
}

public final class PipelineRunner {
    public init() {}

    // MARK: training pass

    @discardableResult
    public func runTraining(_ config: PipelineConfiguration) async -> PipelineRunResult {
        var durations: [String: Double] = [:]
        var state = PipelineRunState(
            runID: config.runID, phase: "preflight",
            startedAt: Date(), updatedAt: Date(),
            codeCommit: nil, dataCommit: nil, dataRepoMode: nil,
            mergeTargetPath: config.mergedGGUFURL.path, lastMessage: nil
        )
        Self.saveQuietly(state, config)

        config.log.info("Run \(config.runID): mirror run_started")
        await mirror(.init(event: .runStarted, at: Date(), runID: config.runID, status: .ok), config)

        // --- pre-flight: target must exist (mirrored skip, no retry storm) ---
        do {
            var stageStart = Date()
            let report = try await config.classifier.health()
            durations["preflight"] = Date().timeIntervalSince(stageStart)
            guard report.healthy else {
                let reason = "classifier unhealthy: HTTP \(report.httpStatus), "
                    + "status=\(report.status ?? "?"), checks=\(report.checks)"
                return await finishSkipped(reason, state, config)
            }

            // --- snapshot: fetch + unpack (workspace refreshed each run) ---
            stageStart = Date()
            let headers = try await config.classifier.fetchSnapshot()
            state.codeCommit = headers.codeCommit
            state.dataCommit = headers.dataCommit
            state.dataRepoMode = headers.dataRepoMode
            try FileManager.default.createDirectory(at: config.workspaceDir,
                                                    withIntermediateDirectories: true)
            // System tar (macOS ships it; sandboxed apps may exec system
            // binaries). Overwrites the previous workspace in place.
            _ = try config.commands.run(
                stage: "snapshot-unpack",
                launchPath: "/usr/bin/tar",
                arguments: ["-xzf", headers.archiveFile.path, "-C", config.workspaceDir.path],
                workingDirectory: config.workspaceDir,
                environment: [:],
                log: config.log,
                timeout: 300
            )
            durations["snapshot"] = Date().timeIntervalSince(stageStart)
            config.log.info("Run \(config.runID): snapshot unpacked (code \(headers.codeCommit.prefix(12)), data repo \(headers.dataRepoMode))")
            state.phase = "training"
            state.updatedAt = Date()
            Self.saveQuietly(state, config)
        } catch {
            return await finishFailed("preflight/snapshot", error, config)
        }

        // --- trainer requirements: snapshot-driven provisioning (2026-10-30) ---
        // Idempotent install of the snapshot's train_lora_requirements.txt into
        // the app venv (trainer + pins move as ONE snapshot unit; pin changes
        // auto-heal here — never a halt). Runs BEFORE training, AFTER unpack.
        do {
            let stageStart = Date()
            try Self.provisionTrainerRequirements(config)
            durations["trainer-requirements"] = Date().timeIntervalSince(stageStart)
        } catch {
            return await finishFailed("trainer-requirements", error, config)
        }

        // --- train: MPS LoRA pass (trainer contract: M7's train_lora.py) ---
        do {
            let stageStart = Date()
            defer { durations["training"] = Date().timeIntervalSince(stageStart) }
            _ = try config.commands.run(
                stage: "training",
                launchPath: config.pythonBin.path,
                arguments: ["train_lora.py", "--output", config.runDir.appendingPathComponent("lora").path],
                workingDirectory: config.brainDir,
                environment: [
                    "TRAINING_DATA_DIR": config.trainingDataDir.path,
                    "PYTHONPATH": config.brainDir.path,
                ],
                log: config.log,
                timeout: config.trainTimeout
            )
        } catch {
            return await finishFailed("training", error, config)
        }

        // --- MERGE CHECKPOINT (Mode A): stop, mirror, hand over to human ---
        state.phase = "awaiting_manual_merge"
        state.updatedAt = Date()
        state.lastMessage = "Manual GGUF merge required; merged GGUF expected at \(config.mergedGGUFURL.path)"
        Self.saveQuietly(state, config)
        config.log.warning("Run \(config.runID): waiting for manual GGUF merge at \(config.mergedGGUFURL.path) (P3 automates this)")
        durations["total_training_pass"] = Date().timeIntervalSince(state.startedAt)
        await mirror(
            .init(event: .runFinished, at: Date(), runID: config.runID, status: .ok,
                  dataSHA: state.dataCommit,
                  durations: durations,
                  error: "manual GGUF merge required — merged GGUF expected at \(config.mergedGGUFURL.path)"),
            config
        )
        return .waitingForManualMerge(state)
    }

    // MARK: trainer requirements (snapshot-driven, decision 2026-10-30)

    /// Chunked file sha256 (stream-safe; same hashing discipline as the
    /// bootstrap layers — update() is MUTATING, `var` hasher).
    static func fileSHA256(_ url: URL) throws -> String {
        var hasher = SHA256()
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Idempotent trainer-stack provision (coordinator decision 2026-10-30):
    /// the requirements file RIDES THE SNAPSHOT (trainer + pins move as one
    /// unit — no receipt-short-circuit skew); the receipt records the sha of
    /// the INSTALLED file (`snapshotRequirementsSHA`); a later snapshot with
    /// changed pins REINSTALLS here (auto-heal, never a halt). Install
    /// failure = the run's pinned failure with the honest retry story (next
    /// Run now) — never a silent half-state.
    static func provisionTrainerRequirements(_ config: PipelineConfiguration) throws {
        let reqFile = config.brainDir.appendingPathComponent("train_lora_requirements.txt")
        guard FileManager.default.fileExists(atPath: reqFile.path) else {
            throw StageError.failed(
                stage: "trainer-requirements",
                message: "train_lora_requirements.txt missing from the snapshot brain dir (\(config.brainDir.path)) — the snapshot is incomplete; re-run, and a persistent miss means the classifier-side snapshot file set changed (report it)")
        }
        let sha = try fileSHA256(reqFile)

        let bootstrapper = RuntimeBootstrapper(commands: config.commands,
                                               fetcher: URLArtifactFetcher())
        let paths = RuntimeBootstrapper.Paths.under(containerDir: config.containerDir)
        guard let receipt = bootstrapper.readReceipt(paths: paths) else {
            throw StageError.failed(
                stage: "trainer-requirements",
                message: "no runtime receipt at \(paths.rtDir.appendingPathComponent("receipt.json").path) — the embedded runtime is not bootstrapped (Settings › Refresh runtime)")
        }
        if receipt.snapshotRequirementsSHA == sha {
            config.log.info("trainer requirements satisfied (sha \(sha.prefix(12))) — skipping install")
            return
        }

        guard FileManager.default.fileExists(atPath: config.uvBin.path) else {
            throw StageError.failed(
                stage: "trainer-requirements",
                message: "uv binary missing at \(config.uvBin.path) — runtime bootstrap incomplete (Settings › Refresh runtime)")
        }
        let previous = receipt.snapshotRequirementsSHA
        config.log.info(
            "trainer package pins \(previous.map { String($0.prefix(12)) } ?? "never installed") → \(sha.prefix(12)): installing the pinned trainer stack into the venv (transformers/peft + transitive; hundreds-of-MB download class, same consented embedded-runtime bucket)")
        _ = try config.commands.run(
            stage: "trainer-requirements",
            launchPath: config.uvBin.path,
            arguments: ["pip", "install", "--python", config.pythonBin.path, "-r", reqFile.path],
            workingDirectory: config.brainDir,
            environment: [:],
            log: config.log,
            timeout: config.trainerProvisionTimeout
        )
        try bootstrapper.recordRequirementsSHA(sha, containerDir: config.containerDir)
        config.log.info("trainer stack installed against snapshot pins (sha \(sha.prefix(12))) — recorded in the runtime receipt")
    }

    // MARK: ship pass (after the human merge)

    @discardableResult
    public func resumeAfterMerge(_ config: PipelineConfiguration) async -> PipelineRunResult {
        var durations: [String: Double] = [:]
        let passStart = Date()
        guard let state = PipelineRunStore.load(runsDir: config.runsDir, runID: config.runID) else {
            return .failed(stage: "resume", message: "No run state found for \(config.runID) — nothing to resume")
        }
        guard state.phase == "awaiting_manual_merge" else {
            return .failed(stage: "resume", message: "Run \(config.runID) is '\(state.phase)', not awaiting_manual_merge")
        }
        let merged = config.mergedGGUFURL
        guard FileManager.default.fileExists(atPath: merged.path) else {
            return .failed(stage: "merge-checkpoint",
                           message: "Merged GGUF missing at \(merged.path) — place it there first")
        }
        let dataSHA = state.dataCommit ?? ""

        // --- embed: local Metal llama-server on the MERGED GGUF ---
        do {
            let stageStart = Date()
            try await config.llama.start(model: merged, port: config.llamaPort,
                                         log: config.log,
                                         waitTimeout: config.llamaStartTimeout)
            defer { config.llama.stop() }
            _ = try config.commands.run(
                stage: "centroids",
                launchPath: config.pythonBin.path,
                arguments: [
                    "build_label_centroids.py",
                    "--data-dir", config.trainingDataDir.appendingPathComponent("Jarvis").path,
                    "--splits", config.workspaceDir
                        .appendingPathComponent("retrain_2026-10-13/corpus").path,
                    "--out", config.centroidsURL.path,
                    "--endpoint", "http://127.0.0.1:\(config.llamaPort)/v1/embeddings",
                ],
                workingDirectory: config.brainDir,
                environment: ["PYTHONPATH": config.brainDir.path],
                log: config.log,
                timeout: config.centroidsTimeout
            )
            durations["embedding+centroids"] = Date().timeIntervalSince(stageStart)
        } catch {
            config.llama.stop()  // failure path: never leave the server up
            return await finishFailed("embedding/centroids", error, config)
        }
        guard FileManager.default.fileExists(atPath: config.centroidsURL.path) else {
            config.llama.stop()
            return .failed(stage: "centroids",
                           message: "centroids script did not produce \(config.centroidsURL.lastPathComponent)")
        }

        // --- ship: GGUF staged upload + vocabulary POST ---
        let ggufData: Data
        let ggufSHA: String
        do {
            var stageStart = Date()
            ggufData = try Data(contentsOf: merged)
            ggufSHA = ClassifierClient.sha256Hex(ggufData)
            _ = try await config.classifier.uploadArtifact(
                ggufData, filename: PipelineConfiguration.ggufFilename, sha256: ggufSHA
            )
            durations["upload-gguf"] = Date().timeIntervalSince(stageStart)

            stageStart = Date()
            let vocabBytes = try Data(contentsOf: config.centroidsURL)
            _ = try await config.classifier.postCentroids(vocabBytes, runID: config.runID)
            durations["post-vocab"] = Date().timeIntervalSince(stageStart)
        } catch {
            return await finishFailed("shipping", error, config)
        }

        // --- terminal event: pr_opened is the ship-pass signal (the
        // (run_id, run_finished) pair is already used by the checkpoint) ---
        let stagedBytes = ggufData.count
        durations["total_ship_pass"] = Date().timeIntervalSince(passStart)
        await mirror(
            .init(event: .prOpened, at: Date(), runID: config.runID, status: .ok,
                  dataSHA: dataSHA.isEmpty ? nil : dataSHA,
                  artifactSHAs: [PipelineConfiguration.ggufFilename: ggufSHA],
                  durations: durations,
                  error: "vocabulary committed on vocab/\(config.runID); merged GGUF staged (\(stagedBytes) bytes)"),
            config
        )

        var finalState = state
        finalState.phase = "completed"
        finalState.updatedAt = Date()
        finalState.lastMessage = "completed; vocab branch pushed, GGUF staged (\(stagedBytes) bytes)"
        Self.saveQuietly(finalState, config)

        return .completed(PipelineSummary(
            runID: config.runID,
            vocabBranch: "vocab/\(config.runID)",
            vocabPushed: true,
            stagedGGUFBytes: stagedBytes,
            dataSHA: dataSHA,
            durations: durations
        ))
    }

    // MARK: plumbing

    private static func saveQuietly(_ state: PipelineRunState, _ config: PipelineConfiguration) {
        do {
            try PipelineRunStore.save(state, runsDir: config.runsDir)
        } catch {
            config.log.error("Run state save failed: \(error)")
        }
    }

    /// Mirror delivery — AWAITED inline (never detached): the run's own
    /// flow waits for the classifier to accept the event (at-least-once;
    /// the brain's (run_id, event) dedupe makes retransmissions harmless)
    /// and a failed post only logs (mirror retry policy is P5). Awaiting
    /// keeps the sequence deterministic and observable in tests.
    private func mirror(_ event: RetrainMirrorEvent, _ config: PipelineConfiguration) async {
        do {
            let receipt = try await config.classifier.sendMirrorEvent(event)
            config.log.info("Mirror event \(event.event.rawValue) \u{2192} \(receipt.status)")
        } catch {
            config.log.warning("Mirror post FAILED for \(event.event.rawValue) (P5 hardening will retry): \(error)")
        }
    }

    private func finishSkipped(_ reason: String, _ state: PipelineRunState,
                               _ config: PipelineConfiguration) async -> PipelineRunResult {
        config.log.warning("Run \(config.runID) SKIPPED: \(reason)")
        await mirror(.init(event: .skipped, at: Date(), runID: config.runID, status: .skipped,
                           error: reason), config)
        var marked = state
        marked.phase = "skipped"
        marked.updatedAt = Date()
        marked.lastMessage = reason
        Self.saveQuietly(marked, config)
        return .skipped(reason: reason)
    }

    private func finishFailed(_ stage: String, _ error: Error, _ config: PipelineConfiguration) async -> PipelineRunResult {
        let message: String
        switch error {
        case StageError.failed(let failedStage, let inner):
            message = "\(failedStage): \(inner)"
        case StageError.timeout(let timedOutStage, let timeout, let tail):
            message = "\(timedOutStage) timed out after \(timeout)s — tail: \(tail)"
        default:
            message = "\(error)"
        }
        config.log.error("Run \(config.runID) FAILED (\(stage)): \(message)")
        await mirror(.init(event: .runFailed, at: Date(), runID: config.runID, status: .failed,
                           error: message), config)
        if var state = PipelineRunStore.load(runsDir: config.runsDir, runID: config.runID) {
            state.phase = "failed"
            state.updatedAt = Date()
            state.lastMessage = message
            Self.saveQuietly(state, config)
        }
        return .failed(stage: stage, message: message)
    }
}

/// Real llama-server handle for the embed stage — a long-lived Process with
/// a readiness poll on /health (llama.cpp answers 503 until the model loads).
/// The binary comes from the embedded runtime bootstrap (container dir) and
/// is injected — never derived from the app bundle (sandbox writes only in
/// the container dir).
/// Async-safe state guard for the llama-server handle: the compiler
/// flags NSLock inside async functions because the hazard is holding a
/// lock ACROSS a suspension. The Class below keeps every critical section
/// synchronous and tiny (state swap / terminate — never an await), which
/// is genuinely-safe scoped locking, not a noop. Reserve/publish/cancel
/// preserves the idempotent-start contract ("already running for this
/// stage") and fences a cancelled start from terminating a process it
/// never launched.
final class LlamaProcessGuard {
    private enum State {
        case idle
        case starting          // reserved; launch in progress
        case running(Process)
    }

    private let lock = NSLock()
    private var state: State = .idle

    /// Reserve the start slot. false = already starting/running (caller
    /// returns early — the idempotent-start semantics).
    func tryBegin() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if case .idle = state {
            state = .starting
            return true
        }
        return false
    }

    /// Publish the launched process (start() succeeded).
    func publish(_ process: Process) {
        lock.lock()
        defer { lock.unlock() }
        state = .running(process)
    }

    /// Launch failed after reservation — back to idle.
    func cancelStart() {
        lock.lock()
        defer { lock.unlock() }
        if case .starting = state {
            state = .idle
        }
    }

    /// Stop-and-clear. Synchronous terminate inside the tiny critical
    /// section; the process is never left running and the slot reopens.
    func terminateIfRunning() {
        lock.lock()
        defer { lock.unlock() }
        if case .running(let process) = state, process.isRunning {
            process.terminate()
        }
        state = .idle
    }
}

/// Real llama-server handle for the embed stage — a long-lived Process with
/// a readiness poll on /health (llama.cpp answers 503 until the model
/// loads). The binary comes from the embedded runtime bootstrap (container
/// dir) and is injected — never derived from the app bundle (sandbox writes
/// only in the container dir). State is guarded by LlamaProcessGuard (see
/// its doc comment for the async-safety reasoning; round-6 warnings).
public final class ProcessLlamaServer: LlamaServing {
    private let binaryURL: URL
    private let guardState = LlamaProcessGuard()

    public init(binaryURL: URL) {
        self.binaryURL = binaryURL
    }

    public func start(model: URL, port: UInt16, log: RunLog, waitTimeout: TimeInterval) async throws {
        guard guardState.tryBegin() else {
            return  // already starting/running for this stage (idempotent)
        }

        guard FileManager.default.fileExists(atPath: binaryURL.path) else {
            guardState.cancelStart()
            throw StageError.failed(
                stage: "embedding",
                message: "llama-server binary missing at \(binaryURL.path) — runtime bootstrap incomplete (Settings › Refresh runtime)"
            )
        }

        let started = Process()
        started.executableURL = binaryURL
        started.arguments = [
            "-m", model.path,
            "--port", String(port),
            "--host", "127.0.0.1",
            "--embedding",
            "--pooling", "mean",
            // W-132 #3 / FINDINGS §16 serve flags: --embedding mode rejects any
            // single input over the default ubatch (512 tokens) with HTTP 500 —
            // the §16-validated serve block carries --ubatch-size 2048 for
            // email-length inputs; the centroids build embeds exactly those.
            "--ubatch-size", "2048",
        ]
        let pipe = Pipe()
        started.standardOutput = pipe
        started.standardError = pipe
        started.standardInput = FileHandle.nullDevice
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty { handle.readabilityHandler = nil; return }
            if let text = String(data: data, encoding: .utf8) {
                log.write(level: "LLAMA", text.trimmingCharacters(in: .newlines))
            }
        }
        do {
            try started.run()
        } catch {
            guardState.cancelStart()
            throw StageError.failed(stage: "embedding",
                                    message: "could not launch llama-server: \(error)")
        }
        guardState.publish(started)

        // Readiness poll: any HTTP answer means serving; 200 = model loaded.
        // (No inout params in async functions — pollHealth returns its note.)
        let deadline = Date().addingTimeInterval(waitTimeout)
        var lastNote = "not polled yet"
        while Date() < deadline {
            let (ready, note) = await Self.pollHealth(port: port)
            if ready { return }
            lastNote = note
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        }
        stop()
        throw StageError.timeout(stage: "llama-start", timeout: waitTimeout, tail: lastNote)
    }

    public func stop() {
        guardState.terminateIfRunning()
    }

    private static func pollHealth(port: UInt16) async -> (ready: Bool, note: String) {
        guard let url = URL(string: "http://127.0.0.1:\(port)/health") else {
            return (false, "bad health URL")
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 5
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            if let http = response as? HTTPURLResponse {
                if http.statusCode == 200 { return (true, "loaded") }
                return (false, "HTTP \(http.statusCode) (model still loading?)")
            }
            return (false, "non-HTTP response")
        } catch {
            return (false, "server not up yet: \(error.localizedDescription)")
        }
    }
}
