import Foundation

/// "Erase all data…" — the guaranteed wipe (design §3).
/// Removes every entry INSIDE the app's data dir (state.json, logs, and
/// anything future phases add) while keeping the directory itself. The
/// caller recreates fresh state afterwards.

public enum Erase {
    /// Empty a directory without removing the directory itself.
    /// Returns the count of top-level entries deleted.
    @discardableResult
    public static func wipe(directory: URL) throws -> Int {
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let contents = try fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: []
        )
        var deleted = 0
        for item in contents {
            try fm.removeItem(at: item)
            deleted += 1
        }
        return deleted
    }

    /// What a wipe WOULD remove: recursively counted files + total bytes.
    /// Shown in the Erase confirmation dialog so the operator always sees
    /// the resolved target and the real before-state (acceptance finding
    /// 2026-10-28: "no data in the specified folder to begin with" — the
    /// dialog must name exactly what it would touch).
    public static func inventory(directory: URL) -> (files: Int, bytes: Int64) {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
            options: []
        ) else {
            return (0, 0)
        }
        var files = 0
        var bytes: Int64 = 0
        for case let url as URL in enumerator {
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey])
            if values?.isDirectory == true { continue }
            files += 1
            bytes += Int64(values?.fileSize ?? 0)
        }
        return (files, bytes)
    }
}
