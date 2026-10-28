import Foundation

/// "Erase all data…" — the guaranteed wipe (design §3).
/// Removes every entry INSIDE the app's data dir (state.json, logs, and
/// anything future phases add) while keeping the directory itself. The
/// caller recreates fresh state afterwards.

enum Erase {
    /// Empty a directory without removing the directory itself.
    /// Returns the count of top-level entries deleted.
    @discardableResult
    static func wipe(directory: URL) throws -> Int {
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
}
