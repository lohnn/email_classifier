import Foundation

/// Always-on structured log inside the sandbox container dir (design §6).
/// P1 writes one `orchestrator.log`; per-run files arrive with P2 execution
/// (log rotation is P5). The file is also what menu > Open log opens.

final class RunLog {
    let fileURL: URL
    private let queue = DispatchQueue(label: "retrain.orchestrator.log")

    private static let stamp: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss"
        f.locale = Locale(identifier: "en_US_POSIX")
        return f
    }()

    init(directory: URL) {
        fileURL = directory.appendingPathComponent("orchestrator.log")
    }

    func info(_ message: String) { write(level: "INFO", message) }
    func warning(_ message: String) { write(level: "WARN", message) }
    func error(_ message: String) { write(level: "ERROR", message) }

    func write(level: String, _ message: String) {
        let line = "\(Self.stamp.string(from: Date())) [\(level)] \(message)\n"
        let data = Data(line.utf8)
        queue.sync {
            do {
                try appendBytes(data)
            } catch {
                // A failed log-append in the log path must report, never
                // vanish: stderr is the fallback channel when the file
                // itself is the broken channel.
                let note = "[retrain-orchestrator] log append failed: \(error)\n"
                _ = try? FileHandle.standardError.write(contentsOf: Data(note.utf8))
            }
        }
    }

    /// One append; throws on any failure so `write` can route it to stderr.
    /// Era note (build round 1): modern SDKs declare
    /// `FileHandle(forWritingTo:)` THROWING + non-Optional — the old
    /// failable-Optional form is gone; value-returning `seekToEnd`/`write`
    /// results are discarded explicitly so every toolchain era compiles
    /// warning-free.
    private func appendBytes(_ data: Data) throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: fileURL.path) {
            try fm.createDirectory(at: directory(), withIntermediateDirectories: true)
            guard fm.createFile(atPath: fileURL.path, contents: data, attributes: nil) else {
                throw CocoaError(.fileWriteUnknown,
                                 userInfo: [NSFilePathErrorKey: fileURL.path])
            }
            return
        }
        let handle = try FileHandle(forWritingTo: fileURL)
        defer { _ = try? handle.close() }
        _ = try handle.seekToEnd()
        _ = try handle.write(contentsOf: data)
    }

    private func directory() -> URL {
        fileURL.deletingLastPathComponent()
    }

    /// Tail of the log (for a future in-app log window; "Open log" uses the file).
    func readTail(maxBytes: Int = 65536) -> String {
        guard let data = try? Data(contentsOf: fileURL) else { return "" }
        let start = data.count > maxBytes ? data.count - maxBytes : 0
        let chunk = data.subdata(in: start..<data.count)
        return String(data: chunk, encoding: .utf8) ?? ""
    }
}
