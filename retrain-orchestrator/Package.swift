// swift-tools-version:5.9
// Retrain Orchestrator — native macOS menu-bar app (design §3/§4/§10).
// Built with Xcode Command Line Tools ONLY (no full Xcode) via ./build_app.sh.

import PackageDescription

let package = Package(
    name: "RetrainOrchestrator",
    platforms: [
        .macOS(.v13)  // SMAppService (login items) requires 13.0
    ],
    targets: [
        // Pure logic: ledger, trigger policy, log store, erase — unit-testable.
        .target(
            name: "RetrainKit",
            path: "Sources/RetrainKit"
        ),
        // AppKit surface: menu bar, settings window, notifications, login item.
        .executableTarget(
            name: "RetrainOrchestrator",
            dependencies: ["RetrainKit"],
            path: "Sources/RetrainOrchestrator"
        ),
        .testTarget(
            name: "RetrainKitTests",
            dependencies: ["RetrainKit"],
            path: "Tests/RetrainKitTests"
        ),
    ]
)
