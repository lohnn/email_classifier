import AppKit

/// Menu-bar status icon with the operator's state dot — the composite-image
/// fix for the round-20 finding: the status-bar button renderer IGNORES
/// attributed-text colors, so the dot MUST be drawn as pixels.
///
/// Drawing contract: the glyph resolves to label color for the CURRENT
/// appearing (menubar draws with its appearance set around the handler);
/// the dot uses fixed system colors that read on both light and dark bars.
/// The image is regenerated on appearance change (AppDelegate observes
/// NSApp.effectiveAppearance) — regeneration is cheap.

enum ConnectionState {
    case unknown      // not probed yet — gray dot, template-ish glyph
    case reachable    // last probe healthy — green dot
    case failed       // last probe failed/unhealthy — red dot
}

enum StatusIconImage {
    static let side: CGFloat = 18

    static func dotColor(for state: ConnectionState) -> NSColor {
        switch state {
        case .unknown: return .systemGray
        case .reachable: return .systemGreen
        case .failed: return .systemRed
        }
    }

    /// Glyph (retrain lifecycle symbol) + small state dot, drawn as ONE
    /// image so the menubar renders both in their true colors.
    /// isDark is INJECTED by the caller (derived from
    /// NSApp.effectiveAppearance, observed for changes) — NSAppearance
    /// .current is deprecated on this SDK era and the composite is
    /// regenerated on appearance change anyway.
    static func composite(glyphName: String, dotState: ConnectionState, isDark: Bool) -> NSImage {
        let size = NSSize(width: side, height: side)
        let image = NSImage(size: size, flipped: false) { rect in
            guard let base = NSImage(systemSymbolName: glyphName,
                                     accessibilityDescription: "Retrain orchestrator") else {
                return false
            }
            let configured = base.withSymbolConfiguration(
                NSImage.SymbolConfiguration(pointSize: 12, weight: .regular)
            ) ?? base
            // Glyph: resolved label color for the current draw appearance.
            let glyphColor = isDark ? NSColor.white : NSColor.black
            let colored = configured.withSymbolConfiguration(
                NSImage.SymbolConfiguration(paletteColors: [glyphColor])
            ) ?? configured
            let glyphRect = NSRect(x: rect.width - 15, y: 3, width: 13, height: 13)
            colored.draw(in: glyphRect)

            // The state dot — the operator's explicit ask (M6 fix).
            let dotRect = NSRect(x: 0, y: 1, width: 6, height: 6)
            dotColor(for: dotState).setFill()
            NSBezierPath(ovalIn: dotRect).fill()
            return true
        }
        image.isTemplate = false  // custom colors: template rendering would strip them
        return image
    }
}
