import Foundation
import ServiceManagement

/// "Start with the system" — the standard login-item toggle (design §3).
/// SMAppService needs macOS 13 (the package's minimum) and a real .app
/// bundle to register. This is a system-side registration, deliberately
/// NOT erased by "Erase all data…" (data ≠ launch settings).

enum LoginItem {
    /// Register/unregister immediately; returns nil on success, else a
    /// human-readable error for the Settings sheet/log.
    static func setEnabled(_ enabled: Bool) -> String? {
        let service = SMAppService.mainApp
        do {
            if enabled {
                if service.status == .enabled { return nil }
                try service.register()
            } else {
                if service.status == .notRegistered { return nil }
                try service.unregister()
            }
            return nil
        } catch {
            return "Login item \(enabled ? "registration" : "removal") failed: \(error)"
        }
    }

    static func isEnabled() -> Bool {
        SMAppService.mainApp.status == .enabled
    }
}
