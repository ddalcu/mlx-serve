import Foundation

/// Which build we are.
///
/// Four subsystems cannot exist in it under MAS App
/// Review guideline 2.5.2 — *"may not download, install, or execute code which
/// introduces or changes features or functionality of the app, including other
/// apps"*:
///
///  * `UpdateChecker` downloads a DMG and swaps the installed bundle.
///  * `CLIInstaller` symlinks into `/usr/local/bin` behind an admin prompt.
///  * `CLILauncher` detects and launches `claude` / `pi` / `opencode`.
///  * `HostMCPSpawner` runs `npx @modelcontextprotocol/…`, downloading code at
///    runtime that adds tools to the agent.
///
/// `HostMCPSpawner` is `#if MAS_BUILD`-compiled-out entirely. The other three
/// are gated at their entry points on these flags — which are decided by the
/// same `#if MAS_BUILD`, so the gate is a compile-time constant, not a runtime
/// mode a user could flip. (Their UI is hidden behind the same flags.)
///
/// Set by `swift build -Xswiftc -DMAS_BUILD` (see `app/build.sh`).
struct BuildFeatures: Equatable {

    /// Self-update by downloading and replacing the .app.
    let selfUpdate: Bool
    /// Symlink `mlx-serve` onto the user's PATH.
    let cliInstaller: Bool
    /// Detect + launch third-party agent CLIs.
    let cliLauncher: Bool
    /// Run the agent's `shell` tool directly on macOS.
    let hostShell: Bool
    /// Pull a guest rootfs from a container registry at runtime.
    let ociPull: Bool
    /// AppleScript / AppleEvents automation.
    let appleEvents: Bool
    /// Point the model library at folders outside the app's own storage.
    ///
    /// Not an App Review rule — a sandbox one. Under MAS the server helper is
    /// signed `com.apple.security.inherit`, which inherits the app's CONTAINER
    /// but NOT its security-scoped grants, so the app could pick a folder that
    /// the process which actually reads the weights cannot open. (Today's path
    /// works precisely because inside the sandbox `~` IS the container, so both
    /// processes resolve `~/.mlx-serve/models` to the same place.) A setting
    /// that silently breaks model loading is worse than one that isn't there.
    let customModelFolders: Bool

    let guest: GuestCapability

    var isMAS: Bool { self == .mas }

    /// What the Linux guest is allowed to do.
    ///
    /// The `packageManagers` knob is what makes shipping live `apt`/`npm` an
    /// experiment rather than a bet. Guideline 2.5.2 forbids downloading code
    /// that changes "features or functionality of the app" — an MCP server
    /// fetched by `npx` plainly does, which is why they are pre-baked into the
    /// rootfs. `npm install lodash` inside the user's own project does not, and
    /// that is the position we ship. If review disagrees, flipping this to
    /// `.stripped` is one rebuild, not a re-architecture.
    struct GuestCapability: Equatable {
        enum PackageManagers: Equatable {
            /// `apt` / `npm` / `pip` reach their upstreams. iSH ships this on iOS.
            case live
            /// Pre-baked toolbox only; the system prompt stops advertising them.
            case stripped
        }
        enum MCPServers: Equatable {
            /// Baked into the rootfs at build time, pinned by version.
            case prebaked
            /// Fetched by `npx` on each launch. Host-only; never on the store.
            case fetched
        }
        var packageManagers: PackageManagers
        var mcpServers: MCPServers
    }

    static let developerID = BuildFeatures(
        selfUpdate: true, cliInstaller: true, cliLauncher: true,
        hostShell: true, ociPull: true, appleEvents: true, customModelFolders: true,
        guest: .init(packageManagers: .live, mcpServers: .fetched))

    static let mas = BuildFeatures(
        selfUpdate: false, cliInstaller: false, cliLauncher: false,
        hostShell: false, ociPull: false, appleEvents: false, customModelFolders: false,
        guest: .init(packageManagers: .live, mcpServers: .prebaked))

    static let current: BuildFeatures = {
        #if MAS_BUILD
        return .mas
        #else
        return .developerID
        #endif
    }()
}
