import Foundation

/// Subprocess supervisor for pipeline stages: streams stdout+stderr into
/// the run log, keeps a bounded tail for failure messages, and enforces a
/// timeout (terminate). Non-zero exit throws StageError with the tail —
/// a failed stage is a loud, attributable event, never a silent half-run.

public struct CommandResult: Equatable {
    public let exitCode: Int32
    public let durationSeconds: Double
    /// Last `limit` chars of combined output (failure attribution).
    public func tail(_ limit: Int = 4000) -> String {
        String(allOutput.suffix(limit))
    }
    public let allOutput: String

    public init(exitCode: Int32, durationSeconds: Double, allOutput: String) {
        self.exitCode = exitCode
        self.durationSeconds = durationSeconds
        self.allOutput = allOutput
    }
}

public enum StageError: Error {
    /// Stage name + the operator-focused failure text (tail excerpt included).
    case failed(stage: String, message: String)
    case timeout(stage: String, timeout: TimeInterval, tail: String)
}

public protocol CommandRunning {
    func run(stage: String, launchPath: String, arguments: [String],
             workingDirectory: URL, environment: [String: String],
             log: RunLog, timeout: TimeInterval) throws -> CommandResult
}

/// Thread-safe accumulating sink for a subprocess's combined streams.
final class OutputSink {
    private let lock = NSLock()
    private var buffer = ""

    func append(_ text: String) {
        lock.lock()
        buffer += text
        lock.unlock()
    }

    func snapshot() -> String {
        lock.lock()
        defer { lock.unlock() }
        return buffer
    }
}

public final class ProcessCommandRunner: CommandRunning {
    public init() {}

    public func run(stage: String, launchPath: String, arguments: [String],
                    workingDirectory: URL, environment: [String: String],
                    log: RunLog, timeout: TimeInterval) throws -> CommandResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: launchPath)
        process.arguments = arguments
        process.currentDirectoryURL = workingDirectory

        var env = ProcessInfo.processInfo.environment
        for (key, value) in environment {
            env[key] = value
        }
        process.environment = env

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        process.standardInput = FileHandle.nullDevice

        let sink = OutputSink()
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            guard let text = String(data: data, encoding: .utf8) else { return }
            sink.append(text)
            // Stream into the run log as it happens — a stuck stage stays
            // observable in the log the operator can open live.
            log.write(level: "STAGE", text.trimmingCharacters(in: .newlines))
        }

        let started = Date()
        do {
            try process.run()
        } catch {
            throw StageError.failed(stage: stage,
                                    message: "could not launch \(launchPath): \(error)")
        }

        // Watchdog: terminate on timeout (never leave a hung trainer).
        let timedOut = TimeoutFlag()
        let watchdog = DispatchWorkItem {
            if process.isRunning {
                timedOut.fire()
                process.terminate()
            }
        }
        DispatchQueue.global().asyncAfter(deadline: .now() + timeout, execute: watchdog)

        process.waitUntilExit()
        watchdog.cancel()

        let result = CommandResult(
            exitCode: process.terminationStatus,
            durationSeconds: Date().timeIntervalSince(started),
            allOutput: sink.snapshot()
        )
        if timedOut.isFired {
            throw StageError.timeout(stage: stage, timeout: timeout, tail: result.tail())
        }
        if result.exitCode != 0 {
            throw StageError.failed(stage: stage,
                                    message: "exit \(result.exitCode): \(result.tail())")
        }
        return result
    }
}

/// Watchdog flag safe to fire from a different thread than the waiter.
final class TimeoutFlag {
    private let lock = NSLock()
    private var fired = false
    func fire() {
        lock.lock()
        fired = true
        lock.unlock()
    }
    var isFired: Bool {
        lock.lock()
        defer { lock.unlock() }
        return fired
    }
}
