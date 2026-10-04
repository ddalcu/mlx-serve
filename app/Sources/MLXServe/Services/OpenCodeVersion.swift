import Foundation

/// OpenCode version detection: the executable's name no longer says the
/// generation (Homebrew ships v2 as `opencode`). Twin of Zig `launch.zig`'s
/// `parseOpencodeVersion` / `resolveOpencode2Bin` — same rule, same vectors.

enum OpenCodeGeneration: Equatable {
    case v1
    /// The newest integration profile we ship; a later major is expected to
    /// stay call-compatible, so major >= 2 routes here (a 3.x is not an error).
    case v2
}

struct OpenCodeVersion: Equatable {
    let generation: OpenCodeGeneration
    /// The detected token verbatim — a 3.0.0 is never displayed as a 2.x.
    /// The parser only ever emits digits and dots, so it is safe to embed.
    let version: String
}

enum OpenCodeProbe: Equatable {
    case missing
    /// `opencode --version` exited non-zero; the value is its captured output.
    case versionFailed(output: String)
    /// Versioned output that names no supported generation — never silently
    /// rounded to one.
    case unparsed(output: String)
    /// The login shell itself could not start — NOT the same as "not installed".
    case shellUnrunnable
    case ok(OpenCodeVersion)
}

/// Everything `launch opencode` / `launch opencode2` need to route: the
/// version probe plus legacy availability (nil means it was not checked).
struct OpenCodeDetection: Equatable {
    let probe: OpenCodeProbe
    let legacyOpencode2Installed: Bool?
}

/// Reads the first version token (optional `v` prefix, `digits.digits…`) out
/// of `opencode --version` output: major 1 → v1, major >= 2 → the newest
/// profile. No token, major 0, or junk like `dev` is undecided (nil).
func parseOpencodeVersion(_ output: String) -> OpenCodeVersion? {
    let bytes = Array(output.utf8)
    let dot = UInt8(ascii: ".")
    func isDigit(_ b: UInt8) -> Bool { b >= UInt8(ascii: "0") && b <= UInt8(ascii: "9") }
    var i = 0
    while i < bytes.count {
        var start = i
        if bytes[i] == UInt8(ascii: "v") || bytes[i] == UInt8(ascii: "V") {
            guard i + 1 < bytes.count, isDigit(bytes[i + 1]) else { i += 1; continue }
            start = i + 1
        } else if !isDigit(bytes[i]) {
            i += 1
            continue
        }
        var j = start
        while j < bytes.count, isDigit(bytes[j]) { j += 1 }
        guard j < bytes.count, bytes[j] == dot else { i += 1; continue }
        guard let major = UInt32(String(decoding: bytes[start..<j], as: UTF8.self)) else {
            i += 1
            continue
        }
        if major < 1 { return nil }
        var k = j
        while k < bytes.count, isDigit(bytes[k]) || bytes[k] == dot { k += 1 }
        let raw = String(decoding: bytes[start..<k], as: UTF8.self)
        return OpenCodeVersion(
            generation: major == 1 ? .v1 : .v2,
            version: raw.trimmingCharacters(in: CharacterSet(charactersIn: ".")))
    }
    return nil
}

/// The `launch opencode2` compatibility alias forces the v2 profile: the
/// canonical `opencode` name when it resolves to major >= 2, else the legacy
/// standalone binary — a v1 install never starts under the v2 config.
func resolveOpencode2Bin(detected: OpenCodeVersion?, legacyInstalled: Bool?) -> String? {
    if let d = detected, d.generation == .v2 { return "opencode" }
    if legacyInstalled == true { return "opencode2" }
    return nil
}

/// What a launch does after probing — the Zig `cmdLaunch` routing decision:
/// which generation starts under which binary name, or why it refuses
/// instead of falling back to a generation.
enum OpenCodeLaunchDecision: Equatable {
    case v1(OpenCodeVersion)
    /// `version` is nil when the alias resolved through the legacy binary
    /// alone — then there is no detection notice to echo.
    case v2(version: OpenCodeVersion?, binary: String)
    case notInstalled
    case undetermined(output: String)
    case shellFailed
    case noV2Binary
}

func decideOpenCodeLaunch(forcedV2: Bool, detection: OpenCodeDetection) -> OpenCodeLaunchDecision {
    if forcedV2 {
        var detected: OpenCodeVersion?
        if case .ok(let v) = detection.probe, v.generation == .v2 { detected = v }
        if detected == nil, detection.legacyOpencode2Installed == nil { return .shellFailed }
        guard let bin = resolveOpencode2Bin(detected: detected,
                                            legacyInstalled: detection.legacyOpencode2Installed) else {
            return .noV2Binary
        }
        return .v2(version: detected, binary: bin)
    }
    switch detection.probe {
    case .missing: return .notInstalled
    case .versionFailed(let out), .unparsed(let out): return .undetermined(output: out)
    case .shellUnrunnable: return .shellFailed
    case .ok(let v):
        return v.generation == .v1 ? .v1(v) : .v2(version: v, binary: "opencode")
    }
}

/// Rc files print banners to stdout, so the version runs in a marked subshell
/// and only the marker's payload parses (keyed output, like `detectInstalled`).
/// Twin of the Zig `launch.zig` marker constants — keep in sync.
let versionMarker = "MLXOCV="
let versionProbeCmd = #"if ! command -v opencode >/dev/null 2>&1; then printf 'MLXOCV=missing\n'; else out=$(opencode --version 2>&1); rc=$?; printf 'MLXOCV=%s %s\n' $rc "$out"; fi"#

/// The version subshell's exit code and output from a login-shell capture:
/// everything after the LAST `MLXOCV=<rc> ` token, so rc banners sitting
/// before it can never pose as the version. Nil = it never answered.
func extractMarkedVersion(_ captured: String) -> (rc: Int, out: String)? {
    guard let pos = captured.range(of: versionMarker, options: .backwards) else { return nil }
    let after = captured[pos.upperBound...]
    guard let sp = after.firstIndex(of: " "), let rc = Int(after[..<sp]) else { return nil }
    return (rc, String(after[after.index(after: sp)...]))
}

func classifyOpenCodeProbe(_ captured: String, shellOK: Bool) -> OpenCodeProbe {
    guard shellOK, let pos = captured.range(of: versionMarker, options: .backwards) else {
        return .versionFailed(output: captured)
    }
    if captured[pos.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines) == "missing" {
        return .missing
    }
    guard let marked = extractMarkedVersion(captured) else { return .versionFailed(output: captured) }
    guard marked.rc == 0 else { return .versionFailed(output: marked.out) }
    guard let version = parseOpencodeVersion(marked.out) else { return .unparsed(output: marked.out) }
    return .ok(version)
}

/// Resolve the v2 binary INSIDE the launch shell (the MAS tab is copy-paste):
/// `opencode` when its `--version` names major >= 2, else the legacy binary.
/// The grep matches any such token; the parsers judge only the first.
let opencodeV2BinResolver = """
oc_bin=opencode2
if opencode --version 2>/dev/null | grep -qE '(^|[[:space:]])[Vv]?([2-9]|[1-9][0-9]+)\\.[0-9]'; then oc_bin=opencode; fi
if ! command -v "$oc_bin" >/dev/null 2>&1; then echo 'no OpenCode v2 binary found: need an `opencode` 2.x+ or the legacy `opencode2` on PATH'; exit 127; fi
"$oc_bin" --standalone
"""
