import AppKit
import RetrainKit

// Menu-bar entry point. The .app bundle carries LSUIElement=true (no dock
// icon); setActivationPolicy(.accessory) keeps that contract even when the
// bare binary is launched from a terminal during development.

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
