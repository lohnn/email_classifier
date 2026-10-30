import Foundation
import CryptoKit

/// M5 — embedded runtime bootstrap (design §5 flow, operator-consented
/// 2026-10-29): downloads INTO the sandbox container dir the pinned
/// uv-managed Python + PyTorch-MPS wheel + Metal llama.cpp server, with
/// authoring-time sha256 pins (RuntimeManifest.json ships in the Kit —
/// pins are never re-resolved at runtime; breaking-drift is a mirrored
/// event, not a silent upgrade).
///
/// Layout inside the app data dir (Erase-all-data wipes it all — the
/// consented contract):
///   rt/bin/uv               pinned uv binary
///   rt/venv/                uv-created venv with the pinned torch
///   rt/llama-cpp/           unpacked llama.cpp tree (Metal dylibs beside
///                           the server binary)
///   rt/bin/llama-server     symlink → the unpacked server binary
///   rt/wheels/              downloaded, sha-verified torch wheel
///   rt/receipt.json         what was installed, when, versions + paths
///
/// Retries live Mac-side (apple/container #1511 rounds down the platform's
/// own reliability — upstream's problem, our backoff).

public struct RuntimeManifestLayer: Codable, Equatable {
    public let name: String
    public let version: String
    public let url: String
    public let sha256: String
    public let bytes: Int
    public let kind: String          // "tar.gz" | "wheel"
    public let innerPath: String?    // uv single binary inside its tar
    public let binaryName: String?   // llama-server tree layout
}

public struct RuntimeManifest: Codable, Equatable {
    public let manifestVersion: Int
    public let generatedAt: String
    public let generatorNote: String?
    public let python: [String: String]          // version
    public let minFreeGB: Double
    public let layers: [String: RuntimeManifestLayer]  // uv | llamaServer | torch

    public var uvLayer: RuntimeManifestLayer? { layers["uv"] }
    public var llamaLayer: RuntimeManifestLayer? { layers["llamaServer"] }
    public var torchLayer: RuntimeManifestLayer? { layers["torch"] }
    public var pythonVersion: String { python["version"] ?? "" }

    /// Load the manifest shipped with the Kit (authoring-time pins — the
    /// load-bearing constant of the deterministic bootstrap).
    /// Locator with injectable roots: production uses defaultSearchRoots();
    /// Kit tests inject a fixture bundle so the defensive path is
    /// deterministic everywhere (and Bundle.module is never trusted).
    public static func bundled(searchRoots: [URL]? = nil) throws -> RuntimeManifest {
        let roots = searchRoots ?? Self.defaultSearchRoots()
        return try decode(fromSearchRoots: roots)
    }

    /// Never touches Bundle.module: the generated accessor cannot be
    /// caught (static let + fatalError). All roots here resolvable WITHOUT
    /// triggering it.
    static func defaultSearchRoots() -> [URL] {
        var roots: [URL] = []
        for bundle in Bundle.allBundles {
            roots.append(bundle.bundleURL)
        }
        if let resourceURL = Bundle.main.resourceURL {
            // resourceURL is already a URL — no path re-wrapping.
            roots.append(resourceURL)
        }
        roots.append(Bundle.main.bundleURL)
        if let executableURL = Bundle.main.executableURL?.deletingLastPathComponent() {
            roots.append(executableURL)
        }
        return roots
    }

    static func decode(fromSearchRoots roots: [URL]) throws -> RuntimeManifest {
        let fm = FileManager.default
        var tried: [String] = []
        for root in roots {
            // Bundle naming drifted across SPM releases — match ANY
            // RetrainKit_*.bundle directory under each root.
            guard let entries = try? fm.contentsOfDirectory(at: root,
                                                            includingPropertiesForKeys: nil) else {
                tried.append(root.path)
                continue
            }
            let bundles = entries.filter {
                let n = $0.lastPathComponent
                return n.hasPrefix("RetrainKit_") && n.hasSuffix(".bundle")
            }
            for bundleDir in bundles {
                tried.append(bundleDir.path)
                // Error-tolerant walk (round-31 lesson): the locator runs on
                // END-USER filesystems (DerivedData internals included) —
                // unreadable/failed subtrees are the NORMAL case. Skip them,
                // never surface a mid-walk IO error; only a COMPLETED-empty
                // search throws the catchable invalidManifest naming roots.
                if let files = fm.enumerator(
                    at: bundleDir,
                    includingPropertiesForKeys: [.isRegularFileKey],
                    errorHandler: { _, _ in .skipDescendants }
                ) {
                    while let item = files.nextObject() as? URL {
                        let values = try? item.resourceValues(forKeys: [.isRegularFileKey])
                        if values?.isRegularFile == false { continue }
                        if item.lastPathComponent == "RuntimeManifest.json" {
                            do {
                                let data = try Data(contentsOf: item)
                                return try JSONDecoder().decode(RuntimeManifest.self, from: data)
                            } catch {
                                throw BootstrapError.invalidManifest(
                                    "RuntimeManifest at \(item.path) failed to decode: \(error)")
                            }
                        }
                    }
                }
            }
        }
        throw BootstrapError.invalidManifest(
            "RuntimeManifest.json not found — searched RetrainKit_*.bundle under "
            + tried.joined(separator: ", ")
            + ". The build script must ship the SwiftPM resource bundle into the .app")
    }

    /// Legacy entry kept for API continuity (defensive locator beneath).
    public static func bundledLegacy(_ unused: Void = ()) throws -> RuntimeManifest {
        try decode(fromSearchRoots: defaultSearchRoots())
    }
}

public struct RuntimeReceipt: Codable, Equatable {
    public var manifestVersion: Int
    public var uvVersion: String
    public var pythonVersion: String
    public var llamaVersion: String
    public var torchVersion: String
    public var uvPath: String
    public var llamaServerPath: String
    public var venvPath: String
    public var installedAt: Date
    /// Recorded by the pipeline when the snapshot lands (drift input for
    /// later runs); nil until then.
    public var snapshotRequirementsSHA: String?
}

/// Download-layer errors owned by the fetcher (URLError codes are NOT
/// extended — .dataLengthExceeded doesn't exist on current SDKs, and
/// squatting Apple's enum for our own meanings was the round-17 smell).
public enum FetchError: Error, Equatable {
    case sizeMismatch(expected: Int, actual: Int)
}

public enum BootstrapError: Error, Equatable {
    case invalidManifest(String)
    case insufficientDisk(neededGB: Double, availableGB: Double)
    case layerVerifyFailed(layer: String, expected: String, actual: String)
    case unpackFailed(layer: String, message: String)
    case commandFailed(layer: String, message: String)
}

public protocol ArtifactFetching {
    /// Streams url → a local temp file; returns the temp URL (caller cleans).
    func fetch(_ url: URL, expectedBytes: Int?, log: RunLog) async throws -> URL
}

/// Real fetcher: URLSession download (streams to disk — torch is 132 MB;
/// in-memory fetch would be reckless at this size) with Mac-side retries.
public final class URLArtifactFetcher: ArtifactFetching {
    private let session: URLSession
    private let attempts: Int
    private let backoffSeconds: [TimeInterval]

    public init(session: URLSession = .shared, attempts: Int = 3,
                backoffSeconds: [TimeInterval] = [2, 4, 8]) {
        self.session = session
        self.attempts = attempts
        self.backoffSeconds = backoffSeconds
    }

    public func fetch(_ url: URL, expectedBytes: Int?, log: RunLog) async throws -> URL {
        var lastError: Error = URLError(.cannotConnectToHost)
        for attempt in 1...max(1, attempts) {
            do {
                let (temp, response) = try await session.download(from: url)
                guard let http = response as? HTTPURLResponse else {
                    throw URLError(.badServerResponse)
                }
                guard (200..<300).contains(http.statusCode) else {
                    throw URLError(.badServerResponse)
                }
                if let expected = expectedBytes,
                   let attrs = try? FileManager.default.attributesOfItem(atPath: temp.path),
                   let size = attrs[.size] as? Int, size != expected {
                    log.warning("Layer download size mismatch (got \(size), manifest says \(expected)) — re-fetching")
                    throw FetchError.sizeMismatch(expected: expected, actual: size)
                }
                return temp
            } catch {
                lastError = error
                log.warning("Fetch attempt \(attempt)/\(attempts) failed for \(url.lastPathComponent): \(error)")
                if attempt < attempts {
                    let backoff = backoffSeconds[min(attempt - 1, backoffSeconds.count - 1)]
                    try? await Task.sleep(nanoseconds: UInt64(backoff * 1_000_000_000))
                }
            }
        }
        throw lastError
    }
}

public final class RuntimeBootstrapper {
    private let commands: CommandRunning
    private let fetcher: ArtifactFetching
    private let attempts: Int

    public init(commands: CommandRunning, fetcher: ArtifactFetching, attempts: Int = 2) {
        self.commands = commands
        self.fetcher = fetcher
        self.attempts = attempts
    }

    public struct Paths {
        public let rtDir: URL
        public let binDir: URL
        public let llamaTreeDir: URL
        public let wheelsDir: URL
        public let venvBinPython: URL
        public static func under(containerDir: URL) -> Paths {
            let rt = containerDir.appendingPathComponent("rt", isDirectory: true)
            return Paths(
                rtDir: rt,
                binDir: rt.appendingPathComponent("bin", isDirectory: true),
                llamaTreeDir: rt.appendingPathComponent("llama-cpp", isDirectory: true),
                wheelsDir: rt.appendingPathComponent("wheels", isDirectory: true),
                venvBinPython: rt.appendingPathComponent("venv/bin/python")
            )
        }
    }

    // MARK: disk pre-flight — the honest part of the consented download

    static func availableGB(at dir: URL) -> Double {
        let values = try? dir.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        if let bytes = values?.volumeAvailableCapacityForImportantUsage {
            return Double(bytes) / 1_000_000_000
        }
        return .infinity  // cannot read the capacity — don't block on our own ignorance
    }

    static func assertDiskRoom(for manifest: RuntimeManifest, at containerDir: URL) throws {
        let free = availableGB(at: containerDir)
        guard free >= manifest.minFreeGB else {
            throw BootstrapError.insufficientDisk(neededGB: manifest.minFreeGB,
                                                  availableGB: free)
        }
    }

    // MARK: layers

    /// Download (with retries) → chunked sha256 → strict verify → position.
    private func fetchAndVerify(_ layer: RuntimeManifestLayer,
                                at destination: URL,
                                log: RunLog) async throws {
        guard let url = URL(string: layer.url) else {
            throw BootstrapError.invalidManifest("bad layer url for \(layer.name)")
        }
        log.info("Runtime layer \(layer.name) \(layer.version): downloading (\(layer.bytes) bytes)")
        let archive = try await fetcher.fetch(url, expectedBytes: layer.bytes, log: log)
        defer { try? FileManager.default.removeItem(at: archive) }

        // Chunked hashing (stream-safe for a 132 MB wheel).
        var hasher = SHA256()
        let handle = try FileHandle(forReadingFrom: archive)
        defer { try? handle.close() }
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        guard digest == layer.sha256 else {
            log.error("Layer \(layer.name) sha MISMATCH (expected \(layer.sha256.prefix(16)), got \(digest.prefix(16)))")
            throw BootstrapError.layerVerifyFailed(layer: layer.name,
                                                   expected: layer.sha256,
                                                   actual: digest)
        }
        log.info("Layer \(layer.name): sha256 VERIFIED (\(digest.prefix(16)))")

        // Guard (round-21): a scheme-host-only URL ("fixture://name.whl")
        // has an EMPTY path — lastPathComponent returns "/" and
        // appendingPathComponent("/") collapses the destination onto the
        // PARENT directory (here: rt/wheels itself → moveItem FileExists).
        // Refuse malformed destinations loudly instead.
        let destinationName = destination.lastPathComponent
        guard !destinationName.isEmpty, destinationName != "/" else {
            throw BootstrapError.invalidManifest(
                "layer \(layer.name) url has no usable path component for a destination filename")
        }
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: archive, to: destination)
    }

    struct UnpackedArchive {
        let tempDir: URL
        func cleanup() { try? FileManager.default.removeItem(at: tempDir) }
    }

    private func unpack(_ archive: URL, kind: String, layer: String,
                        commands: CommandRunning, log: RunLog) throws -> UnpackedArchive {
        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rt-unpack-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        switch kind {
        case "tar.gz":
            _ = try commands.run(stage: "bootstrap-unpack",
                                 launchPath: "/usr/bin/tar",
                                 arguments: ["-xzf", archive.path, "-C", tempDir.path],
                                 workingDirectory: tempDir,
                                 environment: [:],
                                 log: log,
                                 timeout: 300)
        case "wheel":
            // Wheels are zip containers — a wheel is NOT unpacked; it is
            // kept whole for `uv pip install <file>`.
            try FileManager.default.copyItem(at: archive, to: tempDir.appendingPathComponent(archive.lastPathComponent))
        default:
            try FileManager.default.removeItem(at: tempDir)
            throw BootstrapError.unpackFailed(layer: layer, message: "unknown archive kind: \(kind)")
        }
        return UnpackedArchive(tempDir: tempDir)
    }

    private func makeExecutable(_ url: URL) throws {
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
    }

    /// Bootstrap everything the pinned manifest asks for. Idempotent: an
    /// existing receipt whose versions match the manifest short-circuits.
    @discardableResult
    public func ensureRuntime(manifest: RuntimeManifest,
                              containerDir: URL,
                              log: RunLog) async throws -> RuntimeReceipt {
        let paths = Paths.under(containerDir: containerDir)

        // Idempotence: a matching receipt means the runtime is already in
        // place exactly as pinned.
        if let receipt = readReceipt(paths: paths), receipt.manifestVersion == manifest.manifestVersion {
            log.info("Runtime already bootstrapped (manifest v\(manifest.manifestVersion): uv \(receipt.uvVersion), llama \(receipt.llamaVersion), torch \(receipt.torchVersion)) — skipping downloads")
            return receipt
        }

        try Self.assertDiskRoom(for: manifest, at: containerDir)
        let free = Self.availableGB(at: containerDir)
        log.info("Bootstrap: disk pre-flight OK (\(String(format: "%.1f", free)) GB free, need \(manifest.minFreeGB) GB+)")

        guard let uv = manifest.uvLayer, let llama = manifest.llamaLayer, let torch = manifest.torchLayer else {
            throw BootstrapError.invalidManifest("manifest is missing a layer (uv/llamaServer/torch)")
        }

        try FileManager.default.createDirectory(at: paths.binDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: paths.wheelsDir, withIntermediateDirectories: true)

        // --- layer 1: uv (single binary out of a tar.gz) ---
        let uvArchiveURL = paths.wheelsDir.appendingPathComponent("uv-\(uv.version).tar.gz")
        var retry = 0
        while true {
            do { try await fetchAndVerify(uv, at: uvArchiveURL, log: log); break }
            catch {
                retry += 1
                guard retry <= attempts else { throw error }
                log.warning("Layer uv failed (attempt \(retry)/\(attempts)) — retrying: \(error)")
            }
        }
        let uvUnpack = try unpack(uvArchiveURL, kind: uv.kind, layer: "uv", commands: commands, log: log)
        let uvInner = uvUnpack.tempDir.appendingPathComponent(uv.innerPath ?? "")
        guard FileManager.default.fileExists(atPath: uvInner.path) else {
            throw BootstrapError.unpackFailed(layer: "uv", message: "expected binary at \(uv.innerPath ?? "?") inside the archive")
        }
        let uvBin = paths.binDir.appendingPathComponent("uv")
        if FileManager.default.fileExists(atPath: uvBin.path) {
            try FileManager.default.removeItem(at: uvBin)
        }
        try FileManager.default.copyItem(at: uvInner, to: uvBin)
        try makeExecutable(uvBin)
        uvUnpack.cleanup()
        log.info("Runtime layer uv \(uv.version): installed at \(uvBin.path)")

        // --- layer 2: python via pinned uv, then the venv ---
        _ = try commands.run(stage: "bootstrap-python",
                             launchPath: uvBin.path,
                             arguments: ["python", "install", manifest.pythonVersion],
                             workingDirectory: paths.rtDir,
                             environment: [:],
                             log: log,
                             timeout: 1800)
        let venvURL = paths.rtDir.appendingPathComponent("venv", isDirectory: true)
        if FileManager.default.fileExists(atPath: venvURL.path) {
            try FileManager.default.removeItem(at: venvURL)
        }
        _ = try commands.run(stage: "bootstrap-venv",
                             launchPath: uvBin.path,
                             arguments: ["venv", venvURL.path, "--python", manifest.pythonVersion],
                             workingDirectory: paths.rtDir,
                             environment: [:],
                             log: log,
                             timeout: 600)
        log.info("Runtime layer python \(manifest.pythonVersion): venv at \(venvURL.path)")

        // --- layer 3: llama.cpp tree (binary + Metal dylibs together) ---
        let llamaArchiveURL = paths.wheelsDir.appendingPathComponent("llama-\(llama.version).tar.gz")
        retry = 0
        while true {
            do { try await fetchAndVerify(llama, at: llamaArchiveURL, log: log); break }
            catch {
                retry += 1
                guard retry <= attempts else { throw error }
                log.warning("Layer \(llama.name) failed (attempt \(retry)/\(attempts)) — retrying: \(error)")
            }
        }
        if FileManager.default.fileExists(atPath: paths.llamaTreeDir.path) {
            try FileManager.default.removeItem(at: paths.llamaTreeDir)
        }
        try FileManager.default.createDirectory(at: paths.llamaTreeDir, withIntermediateDirectories: true)
        let llamaUnpack = try unpack(llamaArchiveURL, kind: llama.kind, layer: "llama-server", commands: commands, log: log)
        // Locate the server binary inside the unpacked tree (archive root
        // name changes with the tag — search, then assert exactly one).
        let enumerator = FileManager.default.enumerator(at: llamaUnpack.tempDir, includingPropertiesForKeys: nil)
        var found: [URL] = []
        while let item = enumerator?.nextObject() as? URL {
            if item.lastPathComponent == llama.binaryName { found.append(item) }
        }
        guard found.count == 1 else {
            llamaUnpack.cleanup()
            throw BootstrapError.unpackFailed(layer: "llama-server",
                        message: "expected exactly one \(llama.binaryName ?? "binary") in the archive, found \(found.count)")
        }
        let llamaBinTarget = paths.llamaTreeDir
            .appendingPathComponent(llama.version, isDirectory: true)
        if FileManager.default.fileExists(atPath: llamaBinTarget.path) {
            try FileManager.default.removeItem(at: llamaBinTarget)
        }
        let serverDir = found[0].deletingLastPathComponent()
        try FileManager.default.createDirectory(at: llamaBinTarget.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: serverDir, to: llamaBinTarget)
        let llamaServerBin = llamaBinTarget.appendingPathComponent(llama.binaryName ?? "llama-server")
        try makeExecutable(llamaServerBin)
        let llamaLink = paths.binDir.appendingPathComponent(llama.binaryName ?? "llama-server")
        if FileManager.default.fileExists(atPath: llamaLink.path) {
            try FileManager.default.removeItem(at: llamaLink)
        }
        try FileManager.default.createSymbolicLink(at: llamaLink, withDestinationURL: llamaServerBin)
        llamaUnpack.cleanup()
        log.info("Runtime layer llama-server \(llama.version): tree at \(llamaBinTarget.path), launcher symlinked")

        // --- layer 4: torch (MPS) — OUR download, sha-verified, then local
        // wheel install through uv (no floating PyPI resolution) ---
        let torchWheelURL = URL(string: torch.url) ?? URL(fileURLWithPath: "/dev/null")
        var wheelName = torchWheelURL.lastPathComponent
        if wheelName.isEmpty || wheelName == "/" {
            // scheme-host-only URL fallback: derive the on-disk name.
            wheelName = "torch-\(torch.version).whl"
        }
        let wheelDest = paths.wheelsDir.appendingPathComponent(wheelName)
        retry = 0
        while true {
            do { try await fetchAndVerify(torch, at: wheelDest, log: log); break }
            catch {
                retry += 1
                guard retry <= attempts else { throw error }
                log.warning("Layer torch failed (attempt \(retry)/\(attempts)) — retrying: \(error)")
            }
        }
        let venvPython = paths.venvBinPython
        _ = try commands.run(stage: "bootstrap-torch",
                             launchPath: uvBin.path,
                             arguments: ["pip", "install", "--python", venvPython.path, wheelDest.path],
                             workingDirectory: paths.rtDir,
                             environment: [:],
                             log: log,
                             timeout: 3600)
        log.info("Runtime layer torch \(torch.version): installed into the venv from the verified wheel")

        // --- receipt: the drift contract's anchor ---
        let receipt = RuntimeReceipt(
            manifestVersion: manifest.manifestVersion,
            uvVersion: uv.version,
            pythonVersion: manifest.pythonVersion,
            llamaVersion: llama.version,
            torchVersion: torch.version,
            uvPath: uvBin.path,
            llamaServerPath: llamaServerBin.path,
            venvPath: venvURL.path,
            installedAt: Date(),
            snapshotRequirementsSHA: nil
        )
        try Self.writeReceipt(receipt, rtDir: paths.rtDir)
        log.warning("Runtime bootstrap COMPLETE: uv \(uv.version), python \(manifest.pythonVersion), llama \(llama.version), torch \(torch.version). Erase all data removes all of it.")
        return receipt
    }

    public func readReceipt(paths: Paths) -> RuntimeReceipt? {
        let url = paths.rtDir.appendingPathComponent("receipt.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(RuntimeReceipt.self, from: data)
    }

    /// Persists an updated receipt (the pipeline records the installed
    /// trainer-requirements SHA here after a successful install). Same codec
    /// + atomic write as the bootstrap; throws if there is no receipt to
    /// update (no bootstrap record = structural anomaly, never silently
    /// fabricated).
    public func recordRequirementsSHA(_ sha: String,
                                      containerDir: URL) throws {
        let paths = Paths.under(containerDir: containerDir)
        guard var receipt = readReceipt(paths: paths) else {
            throw BootstrapError.commandFailed(
                layer: "trainer-requirements",
                message: "runtime receipt missing at \(paths.rtDir.appendingPathComponent("receipt.json").path) — refresh the runtime")
        }
        receipt.snapshotRequirementsSHA = sha
        try Self.writeReceipt(receipt, rtDir: paths.rtDir)
    }

    static func writeReceipt(_ receipt: RuntimeReceipt, rtDir: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(receipt)
        try data.write(to: rtDir.appendingPathComponent("receipt.json"), options: .atomic)
    }
}

/// Drift hook for later runs: runtime receipt vs pinned manifest. Mismatches
/// are mirrored events (M6 wires the emit) — never silent upgrades.
/// NOTE (2026-10-30, coordinator decision): the SNAPSHOT trainer-requirements
/// file is deliberately NOT a drift issue — the pipeline's
/// trainer-requirements stage auto-heals pin changes (idempotent reinstall);
/// it never halts the run. Only manifest-layer drift halts here.
/// (`requirementsSHA` stays in the signature for call-site continuity; the
/// mismatch is healed, and the sha recorded, by the runner's
/// trainer-requirements stage — a halt here would defeat the auto-heal.)
public enum RuntimeDrift {
    public static func check(receipt: RuntimeReceipt,
                             manifest: RuntimeManifest,
                             requirementsSHA: String?) -> [String] {
        var issues: [String] = []
        if receipt.manifestVersion != manifest.manifestVersion {
            issues.append("runtime manifest v\(receipt.manifestVersion) ≠ pinned v\(manifest.manifestVersion)")
        }
        if let uv = manifest.uvLayer, receipt.uvVersion != uv.version {
            issues.append("uv \(receipt.uvVersion) ≠ pinned \(uv.version)")
        }
        if let llama = manifest.llamaLayer, receipt.llamaVersion != llama.version {
            issues.append("llama-server \(receipt.llamaVersion) ≠ pinned \(llama.version)")
        }
        if let torch = manifest.torchLayer, receipt.torchVersion != torch.version {
            issues.append("torch \(receipt.torchVersion) ≠ pinned \(torch.version)")
        }
        _ = requirementsSHA
        return issues
    }
}
