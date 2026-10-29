import XCTest
@testable import RetrainKit

/// M5 bootstrap tests — fakes everywhere heavy (the fetcher serves local
/// fixture archives; commands are recorded), so no network, no real uv.
/// What is verified: disk pre-flight, download-verify-unpack with sha
/// enforcement + retry, uv/torch/venv invocation wiring, receipt +
/// drift contract.

final class FakeArtifactFetcher: ArtifactFetching {
    /// url → pre-built local archive file (copied per fetch).
    private let mapping: [String: URL]
    /// Keyed by url string: return this error INSTEAD of the file for the
    /// first N attempts (retry verification).
    private var failuresRemaining: [String: Int]
    private(set) var fetchCount: [String: Int] = [:]
    private let lock = NSLock()

    init(mapping: [String: URL], failuresRemaining: [String: Int] = [:]) {
        self.mapping = mapping
        self.failuresRemaining = failuresRemaining
    }

    func fetch(_ url: URL, expectedBytes: Int?, log: RunLog) async throws -> URL {
        lock.lock()
        fetchCount[url.absoluteString, default: 0] += 1
        let fails = failuresRemaining[url.absoluteString] ?? 0
        if fails > 0 {
            failuresRemaining[url.absoluteString] = fails - 1
            lock.unlock()
            throw URLError(.cannotConnectToHost)
        }
        lock.unlock()
        guard let file = mapping[url.absoluteString] else {
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

    private func makeManifest(minFreeGB: Double = 0.0001,
                              manifestVersion: Int = 1,
                              uvSHA: String = String(repeating: "1", count: 64),
                              llamaSHA: String = String(repeating: "2", count: 64),
                              torchSHA: String = String(repeating: "3", count: 64),
                              uvVersion: String = "0.13.6T",
                              llamaVersion: String = "b11982T",
                              torchVersion: String = "2.15.0T") -> RuntimeManifest {
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
            "torch": {"name": "torch", "version": "\(torchVersion)", "url": "fixture://torch.whl",
                   "sha256": "\(torchSHA)", "bytes": 0, "kind": "wheel"}
          }
        }
        """
        return try! JSONDecoder().decode(RuntimeManifest.self, from: Data(json.utf8))
    }

    private func buildFixtures() throws -> [String: URL] {
        let uv = try makeTarGz(members: ["uv-aarch64-apple-darwin/uv": Data("#!/bin/fake-uv".utf8)], tag: "uv")
        let llama = try makeTarGz(
            members: ["llama-b11982T/bin-macos-arm64/llama-server": Data("server".utf8),
                      "llama-b11982T/bin-macos-arm64/libggml.dylib": Data("dylib".utf8)],
            tag: "llama")
        let wheel = tmp.appendingPathComponent("torch-2.15.0T-cp314-fake.whl")
        try Data("wheel-bytes".utf8).write(to: wheel)
        return ["fixture://uv.tar.gz": uv,
                "fixture://llama.tar.gz": llama,
                "fixture://torch.whl": wheel]
    }

    // MARK: tests

    func testInsufficientDiskFailsBeforeAnyDownload() async {
        let manifest = makeManifest(minFreeGB: 1_000_000)  // impossible ask
        let fetcher = FakeArtifactFetcher(mapping: [:])
        let bootstrapper = RuntimeBootstrapper(commands: RecordingCommands(), fetcher: fetcher)
        do {
            _ = try await bootstrapper.ensureRuntime(manifest: manifest, containerDir: tmp, log: log)
            XCTFail("expected insufficientDisk")
        } catch let BootstrapError.insufficientDisk(needed, _) {
            XCTAssertEqual(needed, 1_000_000)
            XCTAssertTrue(fetcher.fetchCount.isEmpty, "downloads must not start before disk pre-flight passes")
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }

    func testShaMismatchFailsTheLayer() async {
        // fixture bytes hashed ≠ manifest sha → verify must fail loudly.
        let fixtures = try! buildFixtures()
        let manifest = makeManifest(uvSHA: String(repeating: "9", count: 64))
        let fetcher = FakeArtifactFetcher(mapping: fixtures)
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

    func testRetriesPassOnSecondAttempt() async {
        let fixtures = try! buildFixtures()
        let manifest = makeManifest()  // matches the fixture bytes
        let fetcher = FakeArtifactFetcher(mapping: fixtures, failuresRemaining: ["fixture://uv.tar.gz": 1])
        let bootstrapper = RuntimeBootstrapper(commands: RecordingCommands(), fetcher: fetcher, attempts: 2)
        let receipt = try? await bootstrapper.ensureRuntime(manifest: manifest, containerDir: tmp, log: log)
        XCTAssertNotNil(receipt, "second attempt should succeed (Mac-side retries live here)")
        XCTAssertEqual(fetcher.fetchCount["fixture://uv.tar.gz"], 2)
    }

    func testReceiptAndDriftContract() async {
        let fixtures = try! buildFixtures()
        let manifest = makeManifest()
        let fetcher = FakeArtifactFetcher(mapping: fixtures)
        let bootstrapper = RuntimeBootstrapper(commands: RecordingCommands(), fetcher: fetcher)
        let receipt = try! await bootstrapper.ensureRuntime(manifest: manifest, containerDir: tmp, log: log)

        XCTAssertEqual(receipt.uvVersion, "0.13.6T")
        XCTAssertEqual(receipt.llamaVersion, "b11982T")
        XCTAssertEqual(receipt.torchVersion, "2.15.0T")
        XCTAssertEqual(receipt.pythonVersion, "3.14")

        // filesystem contract: uv binary, llama tree + symlink, torch wheel
        XCTAssertTrue(FileManager.default.fileExists(atPath: receipt.uvPath))
        XCTAssertTrue(FileManager.default.fileExists(atPath: receipt.llamaServerPath))
        XCTAssertTrue(FileManager.default.fileExists(atPath: receipt.venvPath + "/bin/python"))
        let paths = RuntimeBootstrapper.Paths.under(containerDir: tmp)
        XCTAssertTrue(FileManager.default.fileExists(atPath: paths.rtDir.appendingPathComponent("receipt.json").path))

        // drift: matching manifest → silent
        XCTAssertEqual(RuntimeDrift.check(receipt: receipt, manifest: manifest, requirementsSHA: nil), [])
        // drift: manifest version moved
        let newer = makeManifest(manifestVersion: 2)
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

    func testReceiptShortCircuitSkipsDownloads() async {
        let fixtures = try! buildFixtures()
        let manifest = makeManifest()
        let fetcher = FakeArtifactFetcher(mapping: fixtures)
        let bootstrapper = RuntimeBootstrapper(commands: RecordingCommands(), fetcher: fetcher)
        _ = try! await bootstrapper.ensureRuntime(manifest: manifest, containerDir: tmp, log: log)
        let callsBefore = fetcher.fetchCount.values.reduce(0, +)
        _ = try! await bootstrapper.ensureRuntime(manifest: manifest, containerDir: tmp, log: log)
        let callsAfter = fetcher.fetchCount.values.reduce(0, +)
        XCTAssertEqual(callsBefore, callsAfter, "matching receipt must short-circuit all downloads")
    }

    func testUvAndTorchAreInvokedThroughTheRecordedCommands() async {
        let fixtures = try! buildFixtures()
        let manifest = makeManifest()
        let commands = RecordingCommands()
        let bootstrapper = RuntimeBootstrapper(commands: commands, fetcher: FakeArtifactFetcher(mapping: fixtures))
        let receipt = try! await bootstrapper.ensureRuntime(manifest: manifest, containerDir: tmp, log: log)
        XCTAssertTrue(commands.hasCall(prefix: receipt.uvPath, containing: "install"))   // uv python install 3.14
        XCTAssertTrue(commands.hasCall(prefix: receipt.uvPath, containing: "venv"))      // uv venv
        XCTAssertTrue(commands.calls.contains { $0.arguments.contains("torch-2.15.0T-cp314-fake.whl") })  // local verified wheel
    }
}
