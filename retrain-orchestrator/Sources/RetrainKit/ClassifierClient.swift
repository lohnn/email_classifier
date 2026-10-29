import Foundation
import CryptoKit

/// HTTP surface for the classifier's retrain-orchestration endpoints
/// (peer-only over Tailscale, keyless via the trusted-peer bypass).
/// Shapes are coded to api/models.py + api/routes/admin.py exactly:
/// unknown fields are the BRAIN's 422 risk, not ours — we send only
/// contract keys and parse loosely with optionals.

public enum ClassifierError: Error, Equatable {
    /// The configured base URL is not a usable http(s) URL with a host.
    case badBaseURL(String)
    case http(status: Int, body: String)
    case invalidResponse(String)
}

public struct HealthReport: Equatable {
    public let httpStatus: Int
    /// /health's own "status" field ("ok" when healthy).
    public let status: String?
    /// Per-check truth, rendered loosely (values stringified).
    public let checks: [String: String]

    public var healthy: Bool { httpStatus == 200 && status == "ok" }
}

public struct SnapshotHeaders: Equatable {
    /// Code commit the snapshot trains with (sanity: equals deployment HEAD).
    public let codeCommit: String
    /// The training-data repo commit — the mirror's data_sha.
    public let dataCommit: String
    /// "git" | "dir-fallback"
    public let dataRepoMode: String
    public let bytes: Int
    /// The downloaded tar.gz — caller unpacks (and deletes).
    public let archiveFile: URL
}

public struct VocabReceipt: Equatable {
    /// "committed" | "already_recorded"
    public let status: String
    public let branch: String
    public let commit: String?
    public let pushed: Bool?
    public let pushError: String?
}

public struct UploadReceipt: Equatable {
    public let status: String
    public let stagedAs: String?
    public let size: Int?
    public let sha256: String?
}

public protocol ClassifierServing {
    func health() async throws -> HealthReport
    func fetchSnapshot() async throws -> SnapshotHeaders
    func sendMirrorEvent(_ event: RetrainMirrorEvent) async throws -> MirrorReceipt
    func postCentroids(_ data: Data, runID: String) async throws -> VocabReceipt
    func uploadArtifact(_ data: Data, filename: String, sha256: String) async throws -> UploadReceipt
}

public final class ClassifierClient: ClassifierServing {
    /// Canonical base-URL normalization (mirror of the Settings validation):
    /// http/https scheme + host required, whitespace trimmed, trailing
    /// slashes stripped so "/health"-style joins stay clean. Port may be
    /// empty (operator may serve on 80/443) but host is mandatory.
    public static func normalizedBaseURL(_ raw: String) -> String? {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        while text.hasSuffix("/") { text.removeLast() }
        guard !text.isEmpty else { return nil }
        guard let components = URLComponents(string: text),
              let scheme = components.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              let host = components.host, !host.isEmpty else {
            return nil
        }
        return text
    }

    private let baseURL: String
    private let session: URLSession
    /// Snapshot is <1 MB today (0.2 MB code + ~0.2 MB data) — in-memory
    /// data(for:) instead of download-task streaming. Revisit only if the
    /// snapshot grows past web-scale sanity (design note, M4).
    private let requestTimeout: TimeInterval

    public init(baseURL: String, session: URLSession = .shared,
                requestTimeout: TimeInterval = 120) throws {
        guard let normalized = Self.normalizedBaseURL(baseURL) else {
            throw ClassifierError.badBaseURL(baseURL)
        }
        self.baseURL = normalized
        self.session = session
        self.requestTimeout = requestTimeout
    }

    // MARK: plumbing

    private func url(_ path: String, query: [String: String] = [:]) throws -> URL {
        var text = baseURL + path
        if !query.isEmpty {
            var components = URLComponents(string: text)!
            let items = query.sorted { $0.key < $1.key }.map {
                URLQueryItem(name: $0.key, value: $0.value)
            }
            components.queryItems = items
            text = components.string!
        }
        guard let url = URL(string: text) else {
            throw ClassifierError.badBaseURL(text)
        }
        return url
    }

    private func request(_ url: URL, method: String, body: Data?) -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.timeoutInterval = requestTimeout
        if let body = body {
            request.httpBody = body
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        return request
    }

    /// One invoke path for every call — endpoint verbs differ via the
    /// request only. Transport errors throw; non-2xx is returned for the
    /// caller to branch on.
    private func invoke(_ url: URL, method: String, body: Data?) async throws -> (HTTPURLResponse, Data) {
        let (data, urlResponse) = try await session.data(for: request(url, method: method, body: body))
        guard let http = urlResponse as? HTTPURLResponse else {
            throw ClassifierError.invalidResponse("not an HTTP response for \(url)")
        }
        return (http, data)
    }

    private static func stringBody(_ data: Data) -> String {
        String(data: data.prefix(2000), encoding: .utf8) ?? "(non-UTF8 body)"
    }

    // MARK: endpoints

    public func health() async throws -> HealthReport {
        let (http, data) = try await invoke(try url("/health"), method: "GET", body: nil)
        var status: String?
        var checks: [String: String] = [:]
        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            status = json["status"] as? String
            if let raw = json["checks"] as? [String: Any] {
                for (key, value) in raw {
                    checks[key] = value is String ? (value as? String) : String(describing: value)
                }
            }
        }
        return HealthReport(httpStatus: http.statusCode, status: status, checks: checks)
    }

    public func fetchSnapshot() async throws -> SnapshotHeaders {
        let (http, data) = try await invoke(try url("/admin/training-snapshot"), method: "GET", body: nil)
        guard http.statusCode == 200 else {
            throw ClassifierError.http(status: http.statusCode, body: Self.stringBody(data))
        }
        let header = { (name: String) -> String in
            (http.value(forHTTPHeaderField: name) ?? "").trimmingCharacters(in: .whitespaces)
        }
        let headers = SnapshotHeaders(
            codeCommit: header("X-Snapshot-Code-Commit"),
            dataCommit: header("X-Snapshot-Data-Commit"),
            dataRepoMode: header("X-Snapshot-Data-Repo"),
            bytes: Int(header("X-Snapshot-Bytes")) ?? data.count,
            archiveFile: try Self.writeTempArchive(data)
        )
        return headers
    }

    static func writeTempArchive(_ data: Data) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
        let url = dir.appendingPathComponent("training-snapshot-\(UUID().uuidString).tar.gz")
        try data.write(to: url, options: .atomic)
        return url
    }

    public func sendMirrorEvent(_ event: RetrainMirrorEvent) async throws -> MirrorReceipt {
        let (http, data) = try await invoke(try url("/admin/retrain-events"),
                                            method: "POST",
                                            body: try event.mirrorJSONData())
        guard http.statusCode == 200 else {
            throw ClassifierError.http(status: http.statusCode, body: Self.stringBody(data))
        }
        return try JSONDecoder().decode(MirrorReceipt.self, from: data)
    }

    public func postCentroids(_ data: Data, runID: String) async throws -> VocabReceipt {
        // Integrity echo matches the upload-endpoint convention.
        let sha = Self.sha256Hex(data)
        let url = try url("/admin/receive-centroids",
                          query: ["run_id": runID, "sha256_expected": sha])
        let (http, body) = try await invoke(url, method: "POST", body: data)
        guard http.statusCode == 200 else {
            throw ClassifierError.http(status: http.statusCode, body: Self.stringBody(body))
        }
        return try Self.decodeVocabReceipt(body)
    }

    /// Staged upload of the merged GGUF (pipeline step 5 — STAGED ONLY;
    /// activation is P4's pair). sha256_expected pins the transfer.
    public func uploadArtifact(_ data: Data, filename: String, sha256: String) async throws -> UploadReceipt {
        let url = try url("/admin/upload-model-artifact",
                          query: ["filename": filename, "sha256_expected": sha256])
        var request = request(url, method: "POST", body: nil)
        request.httpBody = data
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        // URLSession.data(for:) returns (Data, URLResponse) — Data FIRST
        // (build round 5: the swapped destructure here was the module's one
        // compile error; the invoke() site was already in the right order).
        let (responseData, urlResponse) = try await session.data(for: request)
        guard let http = urlResponse as? HTTPURLResponse else {
            throw ClassifierError.invalidResponse("not an HTTP response for upload")
        }
        guard http.statusCode == 200 else {
            throw ClassifierError.http(status: http.statusCode, body: Self.stringBody(responseData))
        }
        struct Lazy: Decodable {
            let status: String
            let staged_as: String?
            let size: Int?
            let sha256: String?
        }
        let decoded = try JSONDecoder().decode(Lazy.self, from: responseData)
        return UploadReceipt(status: decoded.status, stagedAs: decoded.staged_as,
                             size: decoded.size, sha256: decoded.sha256)
    }

    public static func sha256Hex(_ data: Data) -> String {
        CryptoKit.SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func decodeVocabReceipt(_ body: Data) throws -> VocabReceipt {
        struct Lazy: Decodable {
            let status: String
            let branch: String
            let commit: String?
            let pushed: Bool?
            let push_error: String?
        }
        let decoded = try JSONDecoder().decode(Lazy.self, from: body)
        return VocabReceipt(status: decoded.status, branch: decoded.branch,
                            commit: decoded.commit, pushed: decoded.pushed,
                            pushError: decoded.push_error)
    }
}
