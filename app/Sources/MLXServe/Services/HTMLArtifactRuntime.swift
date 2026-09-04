import AppKit
import CoreGraphics
import Foundation
import SwiftUI

/// The stage an artifact's page stands on, and the report it sends back.
///
/// `HTMLArtifact` decides WHAT runs; this decides what it runs INSIDE. The
/// distinction earned its own file after the first shipping shape rendered a
/// model's page verbatim into a bordered card and produced exactly the thing
/// this rewrite is about: a dark 100vh slab, frozen at its placeholder height,
/// sitting inside a light rounded rectangle with a grey header strip on top —
/// technically a faithful rendering of the model's HTML, and visibly a foreign
/// object in the transcript.
///
/// Three ideas fix that, and all three live here because all three are pure:
///
/// 1. **A stage, not a costume.** The page is still loaded verbatim; nothing
///    rewrites the model's markup. The defaults arrive as an injected
///    stylesheet placed FIRST in the cascade, so a page with opinions keeps
///    every one of them and a page without opinions inherits the app's type,
///    palette and accent colour through `--mlx-*` custom properties. A model
///    that knows about them can write a widget that matches the chat exactly;
///    one that doesn't still lands on something that belongs.
/// 2. **The viewport is a lie inside a transcript.** A page written for a
///    browser window says `min-height: 100vh`. The web view's viewport is
///    whatever height we guessed, so such a page measures exactly our guess and
///    can never report anything else — the block sticks at its placeholder for
///    good. The clamp is the one rule allowed to shout.
/// 3. **A page that paints itself should paint the whole block.** The probe
///    reports the page's own computed background, and the card wears it. One
///    surface, one set of rounded corners, no rectangle inside a rectangle.
///
/// The probe also answers the question a blank preview used to leave hanging:
/// a page reaching for a CDN has every load blocked by design, and saying so is
/// the difference between a considered limitation and a broken feature.
enum HTMLArtifactRuntime {

    // MARK: - Colours

    /// A colour the PAGE computed, in the page's own terms.
    struct RGB: Equatable {
        var red: Double
        var green: Double
        var blue: Double

        /// Rec. 709 luma. A plain channel average calls pure blue "light" and
        /// then paints black chrome on it.
        var luminance: Double { 0.2126 * red + 0.7152 * green + 0.0722 * blue }
        var isDark: Bool { luminance < 0.5 }
    }

    /// `getComputedStyle` always answers in `rgb()` / `rgba()`, so that is the
    /// only shape parsed. Anything else is a value we did not ask for, and
    /// guessing at it paints the card a colour nobody chose — hence nil, never
    /// a fallback to black.
    static func parseCSSColor(_ text: String) -> (color: RGB, alpha: Double)? {
        guard let open = text.firstIndex(of: "("), let close = text.lastIndex(of: ")"),
              open < close else { return nil }
        let head = text[text.startIndex..<open].trimmingCharacters(in: .whitespaces).lowercased()
        guard head == "rgb" || head == "rgba" else { return nil }

        let parts = text[text.index(after: open)..<close]
            .replacingOccurrences(of: "/", with: " ")
            .replacingOccurrences(of: ",", with: " ")
            .split(separator: " ")
            .map { Double($0.trimmingCharacters(in: .whitespaces)) }
        guard parts.count >= 3 else { return nil }
        let channels = parts.prefix(3).compactMap { $0 }
        guard channels.count == 3 else { return nil }
        let alpha = parts.count > 3 ? (parts[3] ?? 1) : 1
        return (RGB(red: channels[0] / 255, green: channels[1] / 255, blue: channels[2] / 255),
                min(max(alpha, 0), 1))
    }

    // MARK: - Which surface the block wears

    /// How the floating controls should read against whatever is behind them.
    enum Chrome: Equatable {
        /// The page paints nothing: it floats on the transcript's own card and
        /// the chrome follows the app's appearance.
        case app
        case dark
        case light
    }

    struct Surface: Equatable {
        /// The colour the CARD should be painted, when the page has one to
        /// lend. `nil` leaves the app's own material alone.
        var fill: RGB?
        var chrome: Chrome

        static let unpainted = Surface(fill: nil, chrome: .app)
    }

    /// Below this the page is compositing over whatever is behind it, and
    /// hoisting the colour would paint it twice.
    static let opaqueAlpha = 0.9

    /// What the card should do with what the page painted.
    static func surface(background: String?, foreground: String?, hasBackgroundImage: Bool) -> Surface {
        if let background, let parsed = parseCSSColor(background), parsed.alpha >= opaqueAlpha {
            return Surface(fill: parsed.color, chrome: parsed.color.isDark ? .dark : .light)
        }
        // A gradient or an image cannot be reduced to one fill, so the page goes
        // on painting it edge to edge and the only question left is which way
        // the controls should read. The text colour answers it: a page whose
        // own text is light is a page with a dark surface under it.
        if hasBackgroundImage, let foreground, let text = parseCSSColor(foreground) {
            return Surface(fill: nil, chrome: text.color.isDark ? .light : .dark)
        }
        return .unpainted
    }

    // MARK: - The report

    /// The name the probe posts to, and the handler the coordinator registers.
    /// One spelling.
    static let messageHandler = "mlxArtifact"

    /// A page can produce an unbounded error string; the strip that shows it
    /// is one line in a chat transcript.
    static let maxDiagnosticLength = 200

    struct Report: Equatable {
        /// The page's own content height. `nil` when the page reported one that
        /// cannot be laid out — a NaN frame does not break one block, it breaks
        /// the whole chat column.
        var height: CGFloat?
        var surface: Surface
        var blockedRemoteLoads: Int
        var scriptError: String?
    }

    /// Reads what the probe posted. Everything crossing this boundary is
    /// written by a page a model wrote, so nothing is assumed about it.
    static func report(from body: Any) -> Report? {
        guard let payload = body as? [String: Any] else { return nil }
        let known = ["h", "bg", "fg", "img", "blocked", "err"]
        guard known.contains(where: { payload[$0] != nil }) else { return nil }

        var height: CGFloat?
        if let number = payload["h"] as? NSNumber {
            let value = CGFloat(truncating: number)
            if value.isFinite, value >= 0 { height = value }
        }

        var error = (payload["err"] as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if error?.isEmpty == true { error = nil }
        if let text = error, text.count > maxDiagnosticLength {
            error = String(text.prefix(maxDiagnosticLength))
        }

        return Report(height: height,
                      surface: surface(background: payload["bg"] as? String,
                                       foreground: payload["fg"] as? String,
                                       hasBackgroundImage: (payload["img"] as? NSNumber)?.boolValue ?? false),
                      blockedRemoteLoads: max(0, (payload["blocked"] as? NSNumber)?.intValue ?? 0),
                      scriptError: error)
    }

    /// The one line shown under the header when the page did not get what it
    /// asked for.
    ///
    /// Worth its own function because the alternative shipped once and was the
    /// worst part of the feature: a model reaches for a charting library on a
    /// CDN, every remote load is blocked exactly as designed, and the reader
    /// gets an empty box. A named limitation is a feature; a blank rectangle is
    /// a bug report.
    static func diagnostic(blockedRemoteLoads: Int, scriptError: String?) -> String? {
        var parts: [String] = []
        if let scriptError, !scriptError.isEmpty { parts.append(scriptError) }
        if blockedRemoteLoads > 0 {
            parts.append(blockedRemoteLoads == 1
                         ? "1 remote resource blocked — previews run offline"
                         : "\(blockedRemoteLoads) remote resources blocked — previews run offline")
        }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    // MARK: - The stage

    /// The app's own palette, handed to the page as custom properties.
    ///
    /// Colours as CSS strings rather than as `NSColor`s: this half of the
    /// renderer is what a test can hold, and a resolved hex string is a value
    /// two languages can agree on.
    struct Theme: Equatable {
        var foreground: String
        var background: String
        var accent: String
        var dark: Bool

        /// The app as it is drawn right now.
        ///
        /// Read through an explicit `NSAppearance` rather than whatever the
        /// calling thread happens to have current: a dynamic `NSColor` resolves
        /// against the appearance in effect at the moment it is asked, and a
        /// block laid out during a theme switch would otherwise hand the page
        /// the palette it is leaving.
        static func current(_ scheme: ColorScheme) -> Theme {
            let appearance = NSAppearance(named: scheme == .dark ? .darkAqua : .aqua)
            return Theme(foreground: hex(.labelColor, appearance),
                         background: hex(CodeTheme.backgroundNS, appearance),
                         accent: hex(.controlAccentColor, appearance),
                         dark: scheme == .dark)
        }

        /// sRGB, because a page has no other colour space. A colour that cannot
        /// be converted (a pattern or catalog colour with no components) falls
        /// back to the CSS system keyword rather than to an invented value.
        static func hex(_ color: NSColor, _ appearance: NSAppearance?) -> String {
            var resolved = color
            if let appearance {
                appearance.performAsCurrentDrawingAppearance { resolved = color.usingColorSpace(.sRGB) ?? color }
            }
            guard let srgb = resolved.usingColorSpace(.sRGB) else { return "CanvasText" }
            let channel = { (value: CGFloat) in Int((min(max(value, 0), 1) * 255).rounded()) }
            return String(format: "#%02x%02x%02x",
                          channel(srgb.redComponent), channel(srgb.greenComponent),
                          channel(srgb.blueComponent))
        }
    }

    /// Separates the defaults from the clamp, so a test can hold each half to
    /// its own standard: the defaults must never shout, the clamp must.
    static let clampMarker = "/*mlx-clamp*/"

    /// The id of the stylesheet the stage installs. Named so it can be REPLACED
    /// when the app changes appearance, rather than the page being reloaded —
    /// a reload restarts a running widget from zero for a colour change.
    static let styleElementID = "mlx-stage"

    /// The stage's stylesheet, as the page sees it.
    static func stageCSS(theme: Theme) -> String {
        let scheme = theme.dark ? "dark light" : "light dark"
        return """
        :root {
          color-scheme: \(scheme);
          --mlx-fg: \(theme.foreground);
          --mlx-bg: \(theme.background);
          --mlx-accent: \(theme.accent);
          --mlx-font: -apple-system, BlinkMacSystemFont, system-ui, "Helvetica Neue", Arial, sans-serif;
          --mlx-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
        }
        html { background: var(--mlx-bg); }
        body {
          margin: 0;
          padding: 14px;
          background: transparent;
          color: var(--mlx-fg);
          font: 14px/1.55 var(--mlx-font);
          -webkit-font-smoothing: antialiased;
          -webkit-text-size-adjust: 100%;
        }
        h1, h2, h3, h4 { line-height: 1.2; }
        a { color: var(--mlx-accent); }
        code, pre, kbd, samp { font-family: var(--mlx-mono); }
        img, svg, canvas, video, table, pre, iframe { max-width: 100%; }
        button, input, select, textarea { font-family: inherit; font-size: inherit; }
        input, select, textarea, progress, meter { accent-color: var(--mlx-accent); }
        ::selection { background: var(--mlx-accent); color: var(--mlx-bg); }
        ::-webkit-scrollbar { width: 9px; height: 9px; }
        ::-webkit-scrollbar-thumb { background: color-mix(in srgb, var(--mlx-fg) 28%, transparent); border-radius: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        \(clampMarker)
        html, body {
          height: auto !important;
          min-height: 0 !important;
          max-height: none !important;
        }
        """
    }

    /// Installs (or replaces) the stage's stylesheet.
    ///
    /// One builder for both the injected copy and the live appearance swap, so
    /// a running page and a fresh one cannot end up with different palettes.
    static func styleInstallScript(theme: Theme) -> String {
        """
        (function () {
          var id = \(jsString(styleElementID));
          var style = document.getElementById(id);
          if (!style) {
            style = document.createElement('style');
            style.id = id;
            (document.head || document.documentElement).appendChild(style);
          }
          style.textContent = \(jsString(stageCSS(theme: theme)));
        })();
        """
    }

    /// Injected at document START, before a single line of the model's page
    /// runs.
    ///
    /// Document start matters twice. The stylesheet lands ahead of everything
    /// the page declares, which is what makes it a floor rather than an
    /// override. And the error listeners are installed before the page's own
    /// inline scripts execute, which is the only way to see one of them throw —
    /// a listener added at document end has already missed it.
    static func stageScript(theme: Theme) -> String {
        styleInstallScript(theme: theme) + """

        (function () {
          var state = window.__mlxArtifact = window.__mlxArtifact || {};
          state.blocked = 0;
          state.error = null;
          state.collapsed = false;
          state.setCollapsed = function (on) {
            state.collapsed = !!on;
            if (document.documentElement) {
              document.documentElement.style.overflow = on ? 'hidden' : '';
            }
          };

          // Capture phase, on window: a subresource that fails fires its error
          // event on the ELEMENT and never bubbles, so this is the only place
          // that sees a blocked <script src> at all.
          window.addEventListener('error', function (e) {
            var node = e.target;
            if (node && node !== window && (node.src || node.href)) {
              var url = String(node.src || node.href);
              if (/^[a-z][a-z0-9+.-]*:\\/\\//i.test(url) && !/^(data|blob):/i.test(url)) {
                state.blocked += 1;
              }
              if (state.report) { state.report(); }
              return;
            }
            state.error = state.error || String((e && e.message) || 'Script error');
            if (state.report) { state.report(); }
          }, true);

          window.addEventListener('unhandledrejection', function (e) {
            var reason = e && e.reason;
            state.error = state.error || String((reason && reason.message) || reason || 'Unhandled rejection');
            if (state.report) { state.report(); }
          });
        })();
        """
    }

    /// Injected at document END: measures, un-sticks viewport-locked boxes, and
    /// reports the page's own colours.
    ///
    /// Height comes from `body`, never `documentElement`, which never reports
    /// less than the viewport and so yields a block that can grow and never
    /// shrink. The re-measure hooks are the difference between a static
    /// document and a widget: a slider that reveals a row, a canvas drawn after
    /// load, a font swapping in and a CSS transition settling all change the
    /// height well after the page has "finished".
    static let probeScript: String = #"""
    (function () {
      var state = window.__mlxArtifact = window.__mlxArtifact || {};
      var last = null;
      var pending = false;

      // A page written for a browser window floors itself at the viewport
      // height. Inside a transcript that measurement is circular — the viewport
      // IS the frame we guessed — so the floor has to go, and it has to go on
      // descendants too: `.wrap { min-height: 100vh }` locks the page just as
      // hard as `body` does. Only the floor is removed; nothing is made
      // smaller than its own content.
      function unstick() {
        var viewport = window.innerHeight;
        if (!viewport) { return; }
        var nodes = document.body ? document.body.querySelectorAll('*') : [];
        var limit = Math.min(nodes.length, 500);
        for (var i = 0; i < limit; i++) {
          var node = nodes[i];
          if (node.dataset && node.dataset.mlxUnstuck) { continue; }
          var min = parseFloat(window.getComputedStyle(node).minHeight);
          if (min && min >= viewport * 0.9) {
            node.style.minHeight = '0';
            if (node.dataset) { node.dataset.mlxUnstuck = '1'; }
          }
        }
      }

      // The colour the stage itself paints, resolved the way the page sees it.
      //
      // The page is given the transcript card's own colour as its default
      // background rather than `transparent`, because WebKit paints an opaque
      // backdrop of its own under a transparent page and the only switch for
      // that on macOS is a private one (`underPageBackgroundColor = .clear` was
      // measured NOT to composite; this app also ships to the App Store, so KVC
      // into `drawsBackground` is not an option). Painting the card colour
      // makes the seam impossible instead of fixing it.
      //
      // The cost is that "unpainted" is no longer "transparent", so it has to
      // be RECOGNISED: a scratch element resolves `--mlx-bg` through the same
      // engine that computed the page's colours, and anything equal to it is
      // ours. Self-calibrating, so a live appearance change needs no second
      // copy of the value on the Swift side.
      function baseBackground() {
        var probe = document.createElement('div');
        probe.style.cssText = 'position:absolute;left:-9999px;top:0;width:1px;height:1px;background:var(--mlx-bg)';
        document.body.appendChild(probe);
        var value = window.getComputedStyle(probe).backgroundColor;
        probe.parentNode.removeChild(probe);
        return value;
      }

      function isClear(color) {
        return !color || /,\s*0\)$/.test(color) || color === 'transparent';
      }

      function colors() {
        var body = document.body;
        if (!body) { return {}; }
        var base = baseBackground();
        var bodyStyle = window.getComputedStyle(body);
        var rootStyle = window.getComputedStyle(document.documentElement);
        var background = bodyStyle.backgroundColor;
        var image = bodyStyle.backgroundImage;
        // A complete document often paints on <html> and leaves <body> clear.
        if (isClear(background) || background === base) {
          background = rootStyle.backgroundColor;
          if (image === 'none') { image = rootStyle.backgroundImage; }
        }
        // Ours, not theirs: the card is already this colour.
        if (background === base) { background = null; }
        return { bg: background, fg: bodyStyle.color, img: image !== 'none' };
      }

      function measure() {
        var body = document.body;
        if (!body) { return 0; }
        var style = window.getComputedStyle(body);
        var margins = (parseFloat(style.marginTop) || 0) + (parseFloat(style.marginBottom) || 0);
        var height = Math.max(body.scrollHeight + margins,
                              body.getBoundingClientRect().height + margins);
        if (!isFinite(height)) { return 0; }
        return Math.ceil(Math.min(height, 100000));
      }

      function report() {
        if (!document.body) { return; }
        unstick();
        var payload = colors();
        payload.h = measure();
        payload.blocked = state.blocked || 0;
        payload.err = state.error;
        var key = payload.h + '|' + payload.bg + '|' + payload.fg + '|' + payload.blocked + '|' + payload.err;
        if (key === last) { return; }
        last = key;
        try { window.webkit.messageHandlers.mlxArtifact.postMessage(payload); } catch (e) {}
      }

      // Coalesced to one report per frame: a page animating its own layout
      // would otherwise re-lay out the whole transcript per tick.
      function schedule() {
        if (pending) { return; }
        pending = true;
        window.requestAnimationFrame(function () { pending = false; report(); });
      }
      state.report = schedule;

      report();
      window.addEventListener('load', schedule);
      window.addEventListener('resize', schedule);
      window.addEventListener('transitionend', schedule, true);
      window.addEventListener('animationend', schedule, true);
      // Interaction is the whole point of an artifact: a slider that reveals a
      // row has to grow the block that holds it.
      ['input', 'change', 'click', 'keyup'].forEach(function (name) {
        window.addEventListener(name, schedule, true);
      });
      if (window.ResizeObserver) { new ResizeObserver(schedule).observe(document.body); }
      if (window.MutationObserver) {
        new MutationObserver(schedule).observe(document.body,
          { childList: true, subtree: true, attributes: true, characterData: true });
      }
      if (document.fonts && document.fonts.ready) { document.fonts.ready.then(schedule); }
      [16, 120, 400, 1200, 3000].forEach(function (delay) { window.setTimeout(report, delay); });
    })();
    """#

    /// A JS string literal for arbitrary text.
    ///
    /// Hand-rolled rather than `JSONSerialization`, which escapes a forward
    /// slash as `\\/`. That is valid JSON and valid JS, and it silently
    /// rewrote every CSS comment in the stylesheet — including `clampMarker`,
    /// which is how the halves of the stage are told apart. Escaping exactly
    /// what has to be escaped is the whole job.
    static func jsString(_ text: String) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\\": out += "\\\\"
            case "\"": out += "\\\""
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            // `</script` inside a JS string literal ends the enclosing element
            // in an HTML parser. Nothing here writes one today; a stylesheet
            // that grows one later must not be able to break out of the page.
            case "<": out += "\\u003c"
            case ">": out += "\\u003e"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }
}
