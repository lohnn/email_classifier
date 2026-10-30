import XCTest
import CryptoKit
@testable import RetrainKit

/// M5 bootstrap tests — fakes everywhere heavy (the fetcher serves local
/// fixture archives; commands are recorded), so no network, no real uv.
/// What is verified: disk pre-flight, download-verify-unpack with sha
/// enforcement + retry, uv/torch/venv invocation wiring, receipt +
/// drift contract.

/// Sync-state helper: the async fetch() calls these sync methods — the
/// NSLock critical sections never appear inside an async context (the
/// round-6/18 warning family).
final class FetchLedger {
    private let lock = NSLock()
    private var counts: [String: Int] = [:]
    private var failuresRemaining: [String: Int]

    init(failuresRemaining: [String: Int]) {
        self.failuresRemaining = failuresRemaining
    }

    /// Records one fetch attempt; returns true when this attempt is
    /// SUPPOSED to fail (consumes the planned failure).
    func recordAttempt(_ url: URL) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let key = url.absoluteString
        counts[key, default: 0] += 1
        let fails = failuresRemaining[key] ?? 0
        if fails > 0 {
            failuresRemaining[key] = fails - 1
            return true
        }
        return false
    }

    func totalFetches() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return counts.values.reduce(0, +)
    }

    func fetchCount(_ url: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        return counts[url] ?? 0
    }
}

final class FakeArtifactFetcher: ArtifactFetching {
    private let files: [String: URL]
    /// Introspection surface for the tests (intentionally not private —
    /// FetchLedger's methods are the sync-safe read side; round-19).
    let ledger: FetchLedger

    init(files: [String: URL], failuresRemaining: [String: Int] = [:]) {
        self.files = files
        self.ledger = FetchLedger(failuresRemaining: failuresRemaining)
    }

    func fetch(_ url: URL, expectedBytes: Int?, log: RunLog) async throws -> URL {
        if ledger.recordAttempt(url) {
            throw URLError(.cannotConnectToHost)
        }
        guard let file = files[url.absoluteString] else {
            throw URLError(.fileDoesNotExist)
        }
        let dest = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("fetch-\(UUID().uuidString)")
        try FileManager.default.copyItem(at: file, to: dest)
        return dest
    }
}

final class RecordingCommands: CommandRunning {
    struct Call: Equatable {
        let launchPath: String
        let arguments: [String]
    }
    private(set) var calls: [Call] = []

    func run(stage: String, launchPath: String, arguments: [String],
             workingDirectory: URL, environment: [String: String],
             log: RunLog, timeout: TimeInterval) throws -> CommandResult {
        calls.append(Call(launchPath: launchPath, arguments: arguments))
        return CommandResult(exitCode: 0, durationSeconds: 0.001, allOutput: "fake ok")
    }

    func hasCall(prefix: String, containing arg: String) -> Bool {
        calls.contains { $0.launchPath.hasPrefix(prefix) && $0.arguments.contains(arg) }
    }
}

/// Hybrid runner: executes REAL system binaries (/usr/bin/tar — the
/// unpack IS behavior under test), records everything else (uv/venv/torch
/// invocations). Round-20 lesson: a recording-only fake leaves the
/// archive contents un-materialized, and "expected binary inside the
/// archive" is then the CORRECT verdict — the fake lied, not the code.
final class HybridCommands: CommandRunning {
    private(set) var calls: [String] = []   // "launchPath|arg1|arg2…"

    func run(stage: String, launchPath: String, arguments: [String],
             workingDirectory: URL, environment: [String: String],
             log: RunLog, timeout: TimeInterval) throws -> CommandResult {
        calls.append(([launchPath] + arguments).joined(separator: "|"))
        if launchPath == "/usr/bin/tar" {
            return try ProcessCommandRunner().run(
                stage: stage, launchPath: launchPath, arguments: arguments,
                workingDirectory: workingDirectory, environment: environment,
                log: log, timeout: timeout
            )
        }
        return CommandResult(exitCode: 0, durationSeconds: 0.001, allOutput: "fake ok")
    }

    func hasCall(prefix: String, containing arg: String) -> Bool {
        calls.contains { $0.hasPrefix(prefix) && $0.contains(arg) }
    }
}

final class RuntimeBootstrapTests: XCTestCase {
    private var tmp: URL!
    private var log: RunLog!

    override func setUpWithError() throws {
        tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("bootstrap-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        log = RunLog(directory: tmp)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmp)
    }

    // MARK: fixture builders (real archives, tiny)

    /// Builds a gzipped tar with the given relative members — tar runs on
    /// the Mac, so the fixture exercises the real unpack path.
    private func makeTarGz(members: [String: Data], tag: String) throws -> URL {
        let staging = tmp.appendingPathComponent("staging-\(tag)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        for (rel, data) in members {
            let file = staging.appendingPathComponent(rel)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try data.write(to: file)
        }
        let archive = tmp.appendingPathComponent("fixture-\(tag).tar.gz")
        _ = try ProcessCommandRunner().run(
            stage: "fixture-build",
            launchPath: "/usr/bin/tar",
            arguments: ["-czf", archive.path, "-C", staging.path] + members.keys.sorted(),
            workingDirectory: staging,
            environment: [:],
            log: log,
            timeout: 60
        )
        try? FileManager.default.removeItem(at: staging)
        return archive
    }

    private func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func sha256Hex(at url: URL) throws -> String {
        try sha256Hex(Data(contentsOf: url))
    }

    private struct Fixtures {
        let files: [String: URL]
        let uvSHA: String
        let llamaSHA: String
        let torchSHA: String
    }

    private func buildFixtures() throws -> Fixtures {
        let uv = try makeTarGz(members: ["uv-aarch64-apple-darwin/uv": Data("#!/bin/fake-uv".utf8)], tag: "uv")
        let llama = try makeTarGz(
            members: ["llama-b11982T/bin-macos-arm64/llama-server": Data("server".utf8),
                      "llama-b11982T/bin-macos-arm64/libggml.dylib": Data("dylib".utf8)],
            tag: "llama")
        let wheel = tmp.appendingPathComponent("torch-2.15.0T-cp314-fake.whl")
        try Data("wheel-bytes".utf8).write(to: wheel)
        return Fixtures(files: ["fixture://uv.tar.gz": uv,
                                "fixture://llama.tar.gz": llama,
                                "fixture://wheelhost/torch-2.15.0T-cp314-fake.whl": wheel],
                        uvSHA: try sha256Hex(at: uv),
                        llamaSHA: try sha256Hex(at: llama),
                        torchSHA: try sha256Hex(at: wheel))
    }

    /// The manifest is built FROM the fixture shas — a verify mismatch can
    /// then only come from a deliberate wrong-sha override, never from an
    /// accidental default (round-18 crash root cause: default "1"*64
    /// shas mismatched the built fixtures and a try! turned it fatal).
    private func makeManifest(_ fixtures: Fixtures? = nil,
                              minFreeGB: Double = 0.0001,
                              manifestVersion: Int = 1,
                              uvSHAOverride: String? = nil,
                              uvVersion: String = "0.13.6T",
                              llamaVersion: String = "b11982T",
                              torchVersion: String = "2.15.0T") throws -> RuntimeManifest {
        let uvSHA = uvSHAOverride ?? fixtures?.uvSHA ?? String(repeating: "1", count: 64)
        let llamaSHA = fixtures?.llamaSHA ?? String(repeating: "2", count: 64)
        let torchSHA = fixtures?.torchSHA ?? String(repeating: "3", count: 64)
        let json = """
        {
          "manifestVersion": \(manifestVersion),
          "generatedAt": "2026-10-29T00:00:00Z",
          "python": {"version": "3.14"},
          "minFreeGB": \(minFreeGB),
          "layers": {
            "uv": {"name": "uv", "version": "\(uvVersion)", "url": "fixture://uv.tar.gz",
                   "sha256": "\(uvSHA)", "bytes": 0, "kind": "tar.gz",
                   "innerPath": "uv-aarch64-apple-darwin/uv"},
            "llamaServer": {"name": "llama-server", "version": "\(llamaVersion)", "url": "fixture://llama.tar.gz",
                   "sha256": "\(llamaSHA)", "bytes": 0, "kind": "tar.gz", "binaryName": "llama-server"},
            "torch": {"name": "torch", "version": "\(torchVersion)", "url": "fixture://wheelhost/torch-2.15.0T-cp314-fake.whl",
                   "sha256": "\(torchSHA)", "bytes": 0, "kind": "wheel"}
          }
        }
        """
        return try JSONDecoder().decode(RuntimeManifest.self, from: Data(json.utf8))
    }

    // MARK: tests

    func testInsufficientDiskFailsBeforeAnyDownload() async throws {
        let manifest = try makeManifest(minFreeGB: 1_000_000)  // impossible ask
        let fetcher = FakeArtifactFetcher(files: [:])
        let commands = HybridCommands()
        let bootstrapper = RuntimeBootstrapper(commands: commands, fetcher: fetcher)
        do {
            _ = try await bootstrapper.ensureRuntime(manifest: manifest, containerDir: tmp, log: log)
            XCTFail("expected insufficientDisk")
        } catch let BootstrapError.insufficientDisk(needed, _) {
            XCTAssertEqual(needed, 1_000_000)
            XCTAssertEqual(fetcher.ledger.totalFetches(), 0, "downloads must not start before disk pre-flight passes")
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }

    func testShaMismatchFailsTheLayer() async throws {
        // fixture bytes hashed ≠ manifest sha (deliberate override) →
        // verify must fail loudly — and NEVER via try! (a round-18 crash
        // killed the whole xctest process that way).
        let fixtures = try buildFixtures()
        let manifest = try makeManifest(fixtures, uvSHAOverride: String(repeating: "9", count: 64))
        let fetcher = FakeArtifactFetcher(files: fixtures.files)
        let bootstrapper = RuntimeBootstrapper(commands: RecordingCommands(), fetcher: fetcher, attempts: 1)
        do {
            _ = try await bootstrapper.ensureRuntime(manifest: manifest, containerDir: tmp, log: log)
            XCTFail("expected layerVerifyFailed")
        } catch let BootstrapError.layerVerifyFailed(layer, _, _) {
            XCTAssertEqual(layer, "uv")
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }

    func testRetriesPassOnSecondAttempt() async throws {
        let fixtures = try buildFixtures()
        let manifest = try makeManifest(fixtures)  // matches the fixture bytes
        let fetcher = FakeArtifactFetcher(files: fixtures.files,
                                          failuresRemaining: ["fixture://uv.tar.gz": 1])
        // Real tar required: this test runs all the way to a receipt, which
        // means the uv AND llama-server layers actually unpack.
        let commands = HybridCommands()
        let bootstrapper = RuntimeBootstrapper(commands: commands, fetcher: fetcher, attempts: 2)
        let receipt = try await bootstrapper.ensureRuntime(manifest: manifest, containerDir: tmp, log: log)
        XCTAssertNotNil(receipt, "second attempt should succeed (Mac-side retries live here)")
        XCTAssertEqual(fetcher.ledger.fetchCount("fixture://uv.tar.gz"), 2)
    }

    func testReceiptAndDriftContract() async throws {
        let fixtures = try buildFixtures()
        let manifest = try makeManifest(fixtures)
        let fetcher = FakeArtifactFetcher(files: fixtures.files)
        let commands = HybridCommands()
        let bootstrapper = RuntimeBootstrapper(commands: commands, fetcher: fetcher)
        let receipt = try await bootstrapper.ensureRuntime(manifest: manifest, containerDir: tmp, log: log)

        XCTAssertEqual(receipt.uvVersion, "0.13.6T")
        XCTAssertEqual(receipt.llamaVersion, "b11982T")
        XCTAssertEqual(receipt.torchVersion, "2.15.0T")
        XCTAssertEqual(receipt.pythonVersion, "3.14")

        // filesystem contract: uv binary, llama tree + symlink, venv python
        XCTAssertTrue(FileManager.default.fileExists(atPath: receipt.uvPath))
        XCTAssertTrue(FileManager.default.fileExists(atPath: receipt.llamaServerPath))
        let paths = RuntimeBootstrapper.Paths.under(containerDir: tmp)
        XCTAssertTrue(FileManager.default.fileExists(atPath: paths.rtDir.appendingPathComponent("receipt.json").path))

        // drift: matching manifest → silent
        XCTAssertEqual(RuntimeDrift.check(receipt: receipt, manifest: manifest, requirementsSHA: nil), [])
        // drift: manifest version moved
        let newer = try makeManifest(fixtures, manifestVersion: 2)
        XCTAssertFalse(RuntimeDrift.check(receipt: receipt, manifest: newer, requirementsSHA: nil).isEmpty)
        // drift: snapshot requirements changed since provisioning
        let receipt2 = RuntimeReceipt(manifestVersion: receipt.manifestVersion,
                                      uvVersion: receipt.uvVersion,
                                      pythonVersion: receipt.pythonVersion,
                                      llamaVersion: receipt.llamaVersion,
                                      torchVersion: receipt.torchVersion,
                                      uvPath: receipt.uvPath,
                                      llamaServerPath: receipt.llamaServerPath,
                                      venvPath: receipt.venvPath,
                                      installedAt: receipt.installedAt,
                                      snapshotRequirementsSHA: "old-sha")
        let issues = RuntimeDrift.check(receipt: receipt2, manifest: manifest, requirementsSHA: "new-sha")
        XCTAssertEqual(issues.count, 1)
        XCTAssertTrue(issues[0].contains("snapshot requirements changed"))
    }

    func testReceiptShortCircuitSkipsDownloads() async throws {
        let fixtures = try buildFixtures()
        let manifest = try makeManifest(fixtures)
        let fetcher = FakeArtifactFetcher(files: fixtures.files)
        let commands = HybridCommands()
        let bootstrapper = RuntimeBootstrapper(commands: commands, fetcher: fetcher)
        _ = try await bootstrapper.ensureRuntime(manifest: manifest, containerDir: tmp, log: log)
        let callsBefore = fetcher.ledger.totalFetches()
        _ = try await bootstrapper.ensureRuntime(manifest: manifest, containerDir: tmp, log: log)
        let callsAfter = fetcher.ledger.totalFetches()
        XCTAssertEqual(callsBefore, callsAfter, "matching receipt must short-circuit all downloads")
    }

    func testUvAndTorchAreInvokedThroughTheRecordedCommands() async throws {
        let fixtures = try buildFixtures()
        let manifest = try makeManifest(fixtures)
        let commands = HybridCommands()
        let bootstrapper = RuntimeBootstrapper(commands: commands, fetcher: FakeArtifactFetcher(files: fixtures.files))
        let receipt = try await bootstrapper.ensureRuntime(manifest: manifest, containerDir: tmp, log: log)
        XCTAssertTrue(commands.hasCall(prefix: receipt.uvPath, containing: "install"))   // uv python install 3.14
        XCTAssertTrue(commands.hasCall(prefix: receipt.uvPath, containing: "venv"))      // uv venv
        let torchCall = "torch-2.15.0T-cp314-fake.whl"
        XCTAssertTrue(commands.calls.contains { $0.contains(torchCall) && $0.contains("pip") })  // local verified wheel via uv pip
    }
}
