import Foundation

/// M6 glue: assembles the pipeline from Ledger state + runtime receipt and
/// executes a launch pass (train or ship). The app target (AppDelegate's
/// Run now / launch bootstrap) calls THIS — the Launcher owns the
/// bootstrap-first ordering, drift halting and the outcome text the app
/// renders into notifications + the menu-bar indicator.

public protocol RuntimeEnsuring {
    func ensureRuntime(manifest: RuntimeManifest, containerDir: URL,
                       log: RunLog) async throws -> RuntimeReceipt
}

extension RuntimeBootstrapper: RuntimeEnsuring {}

public protocol PipelineEvolving {
    func runTraining(_ config: PipelineConfiguration) async -> PipelineRunResult
    func resumeAfterMerge(_ config: PipelineConfiguration) async -> PipelineRunResult
}

extension PipelineRunner: PipelineEvolving {}

public enum LaunchPhase {
    case train      // Run now, fresh run
    case resume     // ship pass after the human merge
}

public struct LaunchOutcome: Equatable {
    public let headline: String
    public let detail: String
    /// Pinned = delivered even when the failure category is muted (design §6).
    public let pinnedFailure: Bool
    /// Menu-bar dot verdict: nil = stays unknown, true = reachable/healthy
    /// signal path succeeded, false = failed. nil for drift/setup outcomes.
    public let dotReachable: Bool?
    public let runSummary: PipelineSummary?

    public static func == (a: LaunchOutcome, b: LaunchOutcome) -> Bool {
        a.headline == b.headline && a.detail == b.detail
            && a.pinnedFailure == b.pinnedFailure && a.dotReachable == b.dotReachable
            && a.runSummary == b.runSummary
    }
}

public final class PipelineLauncher {
    /// Injections for testability: runtime ensure + pipeline evolve have
    /// protocol seams; the classifier is FACTORED from the state's URL so
    /// tests can fake it.
    public typealias ClassifierProvider = (String) throws -> ClassifierServing

    private let ensure: RuntimeEnsuring
    private let pipeline: PipelineEvolving
    private let provideClassifier: ClassifierProvider
    private let commands: CommandRunning
    /// Optional: nil → the launcher builds a ProcessLlamaServer from the
    /// receipt's llama-server path at execution time.
    private let llama: LlamaServing?
    private let attempts: Int

    public init(ensure: RuntimeEnsuring? = nil, pipeline: PipelineEvolving? = nil,
                commands: CommandRunning = ProcessCommandRunner(),
                llama: LlamaServing? = nil,
                provideClassifier: @escaping ClassifierProvider = { url in
                    try ClassifierClient(baseURL: url)
                },
                attempts: Int = 2) {
        self.ensure = ensure ?? RuntimeBootstrapper(commands: commands,
                                                    fetcher: URLArtifactFetcher(),
                                                    attempts: attempts)
        self.pipeline = pipeline ?? PipelineRunner()
        self.provideClassifier = provideClassifier
        self.commands = commands
        self.llama = llama
        self.attempts = attempts
    }


    // MARK: execution

    /// Execute a launch pass. NEVER throws: every failure maps into a
    /// LaunchOutcome with pinned semantics — the app renders + mirrors.
    public func execute(phase: LaunchPhase,
                        ledger: Ledger,
                        containerDir: URL,
                        runID: String,
                        log: RunLog) async -> LaunchOutcome {
        let state = ledger.load()

        // Setup gate (operator decision 2026-10-29: no default address —
        // an empty URL is a SETUP state, surfaced for Settings, not an error).
        if state.classifierURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return LaunchOutcome(
                headline: "Classifier URL not set",
                detail: "Open Settings and enter the classifier URL (host AND port, e.g. http://jarvis:8008). This app deliberately ships with no default address.",
                pinnedFailure: true, dotReachable: nil, runSummary: nil)
        }

        // Runtime first (downloads on the very first run — consented).
        let manifest: RuntimeManifest
        let receipt: RuntimeReceipt
        do {
            manifest = try RuntimeManifest.bundled()
            receipt = try await ensure.ensureRuntime(manifest: manifest,
                                                     containerDir: containerDir,
                                                     log: log)
        } catch BootstrapError.insufficientDisk(let needed, let available) {
            return LaunchOutcome(
                headline: "Not enough disk space for the runtime",
                detail: String(format: "The embedded runtime needs %.0f GB free; the disk has %.1f GB. Free space and press Run now again.", needed, available),
                pinnedFailure: true, dotReachable: nil, runSummary: nil)
        } catch {
            return LaunchOutcome(
                headline: "Runtime bootstrap failed",
                detail: "\(error) — check the log; pressing Run now retries.",
                pinnedFailure: true, dotReachable: nil, runSummary: nil)
        }

        // Drift halt (never silent behavior change).
        let driftIssues  = RuntimeDrift.check(receipt: receipt, manifest: manifest,
                                              requirementsSHA: nil)
        if !driftIssues.isEmpty {
            return LaunchOutcome(
                headline: "Runtime drift — run halted",
                detail: driftIssues.joined(separator: "; ") + ". Refresh the runtime before running.",
                pinnedFailure: true, dotReachable: nil, runSummary: nil)
        }

        // Classifier wiring (from the operator's URL — probe included in the
        // pipeline's own pre-flight stage).
        let classifier: ClassifierServing
        do {
            classifier = try provideClassifier(state.classifierURL)
        } catch {
            return LaunchOutcome(
                headline: "Classifier URL invalid",
                detail: "\(error) — fix it in Settings.",
                pinnedFailure: true, dotReachable: nil, runSummary: nil)
        }

        let llamaHandle = llama ?? ProcessLlamaServer(binaryURL: URL(fileURLWithPath: receipt.llamaServerPath))
        let config = PipelineConfiguration(
            runID: runID,
            classifier: classifier,
            commands: commands,
            llama: llamaHandle,
            log: log,
            containerDir: containerDir,
            workspaceDir: containerDir.appendingPathComponent("workspace"),
            runsDir: containerDir.appendingPathComponent("runs"),
            pythonBin: URL(fileURLWithPath: receipt.venvPath).appendingPathComponent("bin/python"),
            llamaServerBin: URL(fileURLWithPath: receipt.llamaServerPath)
        )

        let result: PipelineRunResult
        switch phase {
        case .train:
            result = await pipeline.runTraining(config)
        case .resume:
            result = await pipeline.resumeAfterMerge(config)
        }
        return Self.outcome(for: result,
                            runStatePath: PipelineRunStore.stateURL(runsDir: config.runsDir, runID: runID))
    }

    /// Map the runner's result into the user-facing outcome. Machine-true
    /// detail source: PipelineRunState in the run dir on failure/skip.
    static func outcome(for result: PipelineRunResult,
                        runStatePath: URL) -> LaunchOutcome {
        switch result {
        case .completed(let summary):
            return LaunchOutcome(
                headline: "Retrain run completed",
                detail: "Vocabulary branch \(summary.vocabBranch) pushed; merged GGUF staged (\(summary.stagedGGUFBytes) bytes). Next: merge checkpoint done — deploy lands with P4.",
                pinnedFailure: false, dotReachable: true, runSummary: summary)
        case .waitingForManualMerge(let state):
            return LaunchOutcome(
                headline: "Manual GGUF merge required",
                detail: "Training done. Merge the LoRA into the merged GGUF (llama.cpp CPU tooling; see the log for the recipe) and place it at: \(state.mergeTargetPath ?? "(unknown)") — then press Run now again to ship + publish the vocabulary.",
                pinnedFailure: true, dotReachable: nil, runSummary: nil)
        case .failed(let stage, let message):
            return LaunchOutcome(
                headline: "Retrain run failed (\(stage))",
                detail: message, pinnedFailure: true, dotReachable: false, runSummary: nil)
        case .skipped(let reason):
            return LaunchOutcome(
                headline: "Retrain skipped",
                detail: reason, pinnedFailure: false, dotReachable: nil, runSummary: nil)
        }
    }
}
