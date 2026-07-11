import Foundation
import Virtualization

// MARK: - Sentinel protocol (pure, guest-free — unit tested)

/// The agent sandbox boots a persistent `/bin/sh` inside a Linux guest and we
/// drive it as a raw byte stream (a dedicated virtio-console port), not a
/// structured exec — so we frame each command to make the stream parseable:
///
///  1. write `<command>\n`
///  2. write a `printf` that emits a UNIQUE, nonce+seq-tagged marker carrying the
///     command's exit code: `__CTN_<nonce>_EXIT<seq>=<code>__`
///
/// The reader accumulates console bytes and scans for that marker. Everything
/// before it (minus the one newline we inject) is the command's merged
/// stdout+stderr; the digits are its exit status. The nonce (random per guest)
/// + seq (incrementing per command) make a stray/stale/echoed marker unable to
/// match the wrong command.
enum ShellSentinel {
    static func readyMarker(_ nonce: String) -> String { "__CTN_\(nonce)_READY__" }
    static func exitPrefix(_ nonce: String, _ seq: Int) -> String { "__CTN_\(nonce)_EXIT\(seq)=" }

    /// Bytes that quiet terminal echo + the prompt and then emit the ready
    /// marker. Sent (with retries) after boot until `isReady` sees the marker —
    /// proof the shell is alive and configured for clean parsing.
    ///
    /// ECHO-PROOF: the marker is assembled by `printf %s` from the nonce as an
    /// ARGUMENT, so the literal bytes we send contain `__CTN_%s_READY__` (never
    /// the assembled `__CTN_<nonce>_READY__`). While the terminal still echoes
    /// input (before `stty -echo` takes effect), the echoed probe therefore
    /// cannot satisfy `isReady` — only the shell actually RUNNING the printf can.
    /// Without this, `boot()` matched its own echoed input and returned before
    /// the shell was live, desyncing the first commands.
    static func readyProbe(nonce: String) -> [UInt8] {
        Array("stty -echo 2>/dev/null; export PS1='' PS2='' 2>/dev/null; printf '\\n__CTN_%s_READY__\\n' '\(nonce)'\n".utf8)
    }

    static func isReady(_ text: String, nonce: String) -> Bool {
        text.contains(readyMarker(nonce))
    }

    /// The two writes (concatenated) that run `command` then print its tagged
    /// exit marker. The marker `printf` is on its OWN line so `$?` is the
    /// command's status, not the printf's.
    ///
    /// ECHO-PROOF (same reason as `readyProbe`): the marker is assembled from
    /// `printf %s` ARGS, so the bytes we send contain `__CTN_%s_EXIT%s=` — never
    /// the assembled `__CTN_<nonce>_EXIT<seq>=` that `scan` matches. If the guest
    /// tty has echo on (some images' `stty -echo` doesn't take), the echoed
    /// `printf` line would otherwise contain a matching prefix followed by the
    /// literal `%d` (not digits), and scan would lock onto it and never find the
    /// real marker → every command times out.
    static func frame(command: String, nonce: String, seq: Int) -> [UInt8] {
        let s = command + "\n"
            + "printf '\\n__CTN_%s_EXIT%s=%d__\\n' '\(nonce)' '\(seq)' \"$?\"\n"
        return Array(s.utf8)
    }

    /// Parse the accumulated console `text` for THIS command's marker. Returns
    /// the command output + exit code once the full marker has arrived, else nil
    /// (keep reading). Interior newlines in the output are preserved; the single
    /// newline we inject just before the marker is stripped.
    ///
    /// Scans ALL occurrences of the prefix and returns the first whose value
    /// parses as an integer — so a stray/echoed `…EXIT<seq>=%d__` (non-numeric)
    /// is skipped rather than aborting the search.
    static func scan(_ text: String, nonce: String, seq: Int) -> (output: String, code: Int32)? {
        let prefix = exitPrefix(nonce, seq)
        var from = text.startIndex
        while let r = text.range(of: prefix, range: from..<text.endIndex) {
            let after = text[r.upperBound...]
            if let close = after.range(of: "__"), let code = Int32(after[..<close.lowerBound]) {
                var out = String(text[..<r.lowerBound])
                if out.hasSuffix("\n") { out.removeLast() } // drop the newline we injected
                return (out, code)
            }
            from = r.upperBound // this occurrence didn't parse (e.g. echoed %d); try the next
        }
        return nil
    }
}

// MARK: - Terminal output sanitizing (pure — unit tested)

/// The sandbox shell runs on a real tty (hvc1), so CLI tools detect an
/// interactive terminal and emit ANSI color codes + cursor-move progress
/// animations (npm's braille spinner, `\e[1G\e[0K` line rewrites, curl's
/// `\e[1m…\e[0m` bold headers). The host path uses a pipe, so tools stay
/// quiet — this brings the guest to the same clean text for BOTH the Sandbox
/// Terminal display and the text handed back to the agent.
enum TerminalOutput {
    /// Strip ANSI escape sequences and resolve carriage-return / cursor-to-
    /// column-1 overwrites so only the final content of each line remains.
    static func sanitize(_ raw: String) -> String {
        let scalars = Array(raw.unicodeScalars)
        var out: [Unicode.Scalar] = []
        out.reserveCapacity(scalars.count)
        let esc: Unicode.Scalar = "\u{1B}"
        let bel: Unicode.Scalar = "\u{07}"
        var i = 0
        while i < scalars.count {
            let c = scalars[i]
            if c == esc {
                i += 1
                guard i < scalars.count else { break }
                let kind = scalars[i]
                if kind == "[" { // CSI: ESC [ params … final (0x40–0x7E)
                    i += 1
                    var params = ""
                    while i < scalars.count, !(scalars[i].value >= 0x40 && scalars[i].value <= 0x7E) {
                        params.unicodeScalars.append(scalars[i]); i += 1
                    }
                    let final: Unicode.Scalar? = i < scalars.count ? scalars[i] : nil
                    i += 1
                    // Cursor-horizontal-absolute (…G) means "return to column N";
                    // treat it (and column-1 specifically) as a carriage return so
                    // the line-collapse below discards the overwritten prefix.
                    if final == "G" { out.append("\r") }
                    // Every other CSI (color `m`, erase `K`/`J`, cursor moves) is
                    // dropped — no textual content.
                } else if kind == "]" { // OSC: ESC ] … (BEL | ESC \)
                    i += 1
                    while i < scalars.count {
                        if scalars[i] == bel { i += 1; break }
                        if scalars[i] == esc, i + 1 < scalars.count, scalars[i + 1] == "\\" { i += 2; break }
                        i += 1
                    }
                } else {
                    // Other escapes: optional intermediate bytes (0x20–0x2F, e.g.
                    // the `(` in the charset-designator `ESC ( B`) then one final
                    // byte. Drop the whole run.
                    while i < scalars.count, scalars[i].value >= 0x20, scalars[i].value <= 0x2F { i += 1 }
                    if i < scalars.count { i += 1 }
                }
                continue
            }
            out.append(c); i += 1
        }
        // Normalize CRLF → LF FIRST: the guest tty is ONLCR, so every real
        // newline arrives as `\r\n`. Without this the per-line \r-collapse below
        // treats that trailing `\r` as an overwrite and wipes the line content
        // (live regression: `echo a; uname; pwd` came back as one run-on line).
        let normalized = String(String.UnicodeScalarView(out)).replacingOccurrences(of: "\r\n", with: "\n")
        // Resolve bare carriage-return overwrites per line: within each
        // \n-delimited line, only the text after the LAST \r survives (a real
        // terminal reprints from column 0). Also drop any other C0 control bytes.
        var result = ""
        for (idx, line) in normalized.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            if idx > 0 { result.append("\n") }
            let collapsed = line.split(separator: "\r", omittingEmptySubsequences: false).last.map(String.init) ?? ""
            for ch in collapsed.unicodeScalars where ch.value >= 0x20 || ch == "\t" {
                result.unicodeScalars.append(ch)
            }
        }
        return result
    }
}

// MARK: - Thread-safe console buffer

/// Accumulates guest console bytes delivered by pipe readability handlers (which
/// fire on a dispatch-io thread) and lets the exec loop read a suffix.
private final class ConsoleBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var buf: [UInt8] = []

    func append(_ d: Data) {
        guard !d.isEmpty else { return }
        lock.lock(); buf.append(contentsOf: d); lock.unlock()
    }

    /// Current byte length — used to mark where a command's output begins.
    func mark() -> Int { lock.lock(); defer { lock.unlock() }; return buf.count }

    /// Decode the console text produced since `offset`.
    func text(from offset: Int) -> String {
        lock.lock(); defer { lock.unlock() }
        guard offset < buf.count else { return "" }
        return String(decoding: buf[offset...], as: UTF8.self)
    }
}

// MARK: - VzGuest

/// A live Linux sandbox guest on Apple's Virtualization.framework with a
/// persistent shell, driven over a dedicated virtio-console port via
/// `ShellSentinel`. One guest is meant to serve a whole agent session (shell
/// state — cwd, env, installed packages — persists between `exec` calls).
///
/// Topology (proven by the VZ spike + SandboxSmoke):
///  - `serialPorts[0]` → guest `/dev/hvc0`: kernel console (printk / boot log)
///  - `serialPorts[1]` → guest `/dev/hvc1`: the shell channel — clean bytes,
///    never interleaved with kernel messages
///  - virtiofs tag `rootfs`: the unpacked OCI image dir, mounted as the guest
///    ROOT by the kernel itself (`rootfstype=virtiofs`). No initramfs — the
///    image is demand-paged from the host, so guest RAM stays workload-sized
///    (1 GiB default) instead of image+workload (the old 6 GiB).
///  - virtiofs tag `workspace`: the host working directory, mounted by our
///    generated `/.vz-init` (written into the rootfs dir before boot).
///
/// Thread-safety: `exec` is serialized through an internal lock (the channel is
/// a single stream, one command at a time). All VZVirtualMachine calls happen on
/// `vmQueue` (the queue the VM was bound to at init).
final class VzGuest {

    /// How the host talks to the guest.
    ///
    /// `.vsock` is the real transport: `vz-agent` (`src/vz_agent.zig`) listens
    /// on `AF_VSOCK`, one connection per process, so exit codes are real, stdout
    /// and stderr stay separated, and MCP servers get their own connections. It
    /// requires a kernel with `CONFIG_VSOCKETS` + `CONFIG_VIRTIO_VSOCKETS`
    /// (contain's `kernels-v3`) and the `vz-agent` binary to inject.
    ///
    /// `.legacyConsole` is the previous design — one persistent `/bin/sh` on the
    /// hvc1 tty, framed by `ShellSentinel`. Kept so a stale cached kernel still
    /// boots, and so `SandboxSmoke` can prove both arms before it is deleted.
    enum Transport: Equatable {
        case vsock
        case legacyConsole
    }

    struct Config {
        var kernelPath: String
        var rootfsDir: String
        /// Host directory shared into the guest at `guestWorkspacePath` (rw).
        var workspacePath: String? = nil
        var guestWorkspacePath: String = "/workspace"
        /// "KEY=VALUE" entries from the OCI image config, exported before the shell.
        var imageEnv: [String] = []
        /// Directory the shell starts in (best-effort `cd`).
        var workdir: String? = "/workspace"
        var ramBytes: UInt64 = 1 << 30 // rootfs is demand-paged over virtiofs — workload headroom only
        var cpuCount: Int = 4
        /// NAT networking + the live port-report stream. When false the guest
        /// gets NO network device and never DHCPs — fully isolated.
        var network: Bool = false

        var transport: Transport = .vsock
        /// Host path to the `vz-agent` ELF, copied into the rootfs before boot.
        /// Required by `.vsock`; ignored by `.legacyConsole`.
        var agentBinaryPath: String? = nil
    }

    struct ExecResult {
        /// stdout and stderr merged, in arrival order — what the agent reads.
        var output: String
        /// Separated streams. Empty on `.legacyConsole`, which cannot tell them
        /// apart: one tty carried both.
        var stdout: String = ""
        var stderr: String = ""
        var exitCode: Int32
        var timedOut: Bool
        /// The child's pid inside the guest. Only `.vsock` reports one.
        var pid: Int32? = nil
    }

    enum GuestError: Error, CustomStringConvertible {
        case bootFailed(String)
        case notReady
        case guestExited
        var description: String {
            switch self {
            case .bootFailed(let why):
                return "sandbox VM failed to start: \(why) (is the binary signed with the com.apple.security.virtualization entitlement? Dev builds must go through app/build.sh)"
            case .notReady:   return "sandbox guest shell did not become ready in time"
            case .guestExited: return "sandbox guest exited unexpectedly"
            }
        }
    }

    // Virtiofs tags + boot plumbing (shared with the unit tests).
    static let rootfsTag = "rootfs"
    static let workspaceTag = "workspace"
    static let initScriptGuestPath = "/.vz-init"
    /// Where the guest agent is injected, alongside `/.vz-init`. Host-injected
    /// rather than baked into the image, so ANY base image works.
    static let agentGuestPath = "/.vz-agent"

    /// The guest console device the monitor loop writes its report to.
    ///
    /// hvc numbering follows `serialPorts` ORDER, so removing the shell port
    /// renumbers everything after it. `.legacyConsole` has three ports
    /// (boot, shell, monitor) → hvc2; `.vsock` has two (boot, monitor) → hvc1.
    /// Getting this wrong is silent: the monitor writes into a device nobody
    /// reads, and the tray's RAM readout plus the live port map just stop.
    static func monitorDevice(transport: Transport) -> String {
        transport == .vsock ? "/dev/hvc1" : "/dev/hvc2"
    }

    /// Kernel command line. With `network`, `ip=dhcp` makes the KERNEL acquire
    /// address/gateway/DNS from VZ's NAT (CONFIG_IP_PNP — verified present in
    /// the prebuilt kernel) before init runs, so networking works with ANY
    /// image: no userspace DHCP client required. VZ's vmnet answers the DHCP
    /// immediately, so the boot-time cost is negligible.
    static func kernelCommandLine(network: Bool) -> String {
        "console=hvc0 root=\(rootfsTag) rootfstype=virtiofs rw init=\(initScriptGuestPath) panic=-1"
            + (network ? " ip=dhcp" : "")
    }

    // MARK: Pure builders (unit tested)

    /// POSIX single-quote escaping: the only character that needs handling inside
    /// single quotes is the single quote itself ('\'' = close, literal, reopen).
    static func shellQuote(_ s: String) -> String {
        "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    /// The `/.vz-init` PID-1 script written into the rootfs dir before boot.
    /// Mounts the kernel filesystems + the workspace share, applies the image's
    /// env/workdir, then hands control to the transport:
    ///
    ///  * `.vsock` — execs `/.vz-agent`, which serves one process per AF_VSOCK
    ///    connection. No tty, no persistent shell.
    ///  * `.legacyConsole` — hands a persistent `/bin/sh` the dedicated hvc1
    ///    channel as its controlling tty.
    ///
    /// When that process exits, the guest powers off (contain's proven poweroff
    /// sequence — slim images ship no poweroff binary, so SysRq 'o' is the
    /// fallback).
    ///
    /// The legacy arm deliberately runs /bin/sh (dash), NOT interactive bash: a
    /// host driver feeding commands over a byte stream + matching an exit-code
    /// sentinel needs a clean, predictable stream; readline escapes and job
    /// control fight that.
    static func buildInitScript(config: Config) -> String {
        var s = """
        #!/bin/sh
        export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root TERM=linux
        mkdir -p /proc /sys /dev\(config.workspacePath != nil ? " \(config.guestWorkspacePath)" : "") 2>/dev/null
        mount -t proc proc /proc 2>/dev/null
        mount -t sysfs sysfs /sys 2>/dev/null
        mount -t devtmpfs devtmpfs /dev 2>/dev/null

        """
        if config.workspacePath != nil {
            s += "mount -t virtiofs \(workspaceTag) \(config.guestWorkspacePath) 2>/dev/null\n"
        }
        if config.network {
            // The kernel already DHCPed eth0 (`ip=dhcp` on the cmdline) and wrote
            // its answer to /proc/net/pnp. Userspace clients are only a fallback
            // for a custom kernel without IP_PNP; DNS comes from the pnp file.
            s += """
            ip link set lo up 2>/dev/null
            [ -s /proc/net/pnp ] || dhclient -1 eth0 2>/dev/null || udhcpc -i eth0 -n -q 2>/dev/null || dhcpcd -1 eth0 2>/dev/null || true
            grep -E '^(nameserver|domain|search)' /proc/net/pnp > /etc/resolv.conf 2>/dev/null || true

            """
        }
        // Guest monitor: stream a once-a-second report to the host over the LAST
        // console port — RAM (/proc/meminfo, feeds the tray readout) always,
        // plus the guest IP + /proc/net/tcp(6) when networked (feeds the live
        // port map, SandboxPortForwarder). Framed with =EOS= so the host can
        // split complete snapshots. Runs detached so it never blocks anything.
        //
        // The device number depends on the transport — see `monitorDevice`.
        let monitor = monitorDevice(transport: config.transport)
        s += "i=0; while [ ! -e \(monitor) ] && [ $i -lt 100 ]; do sleep 0.1; i=$((i+1)); done\n"
        s += "( while true; do\n"
        s += "    grep -E '^(MemTotal|MemAvailable)' /proc/meminfo 2>/dev/null\n"
        if config.network {
            s += "    ip4=$(grep -B1 -F '32 host LOCAL' /proc/net/fib_trie 2>/dev/null | grep -oE '([0-9]{1,3}\\.){3}[0-9]{1,3}' | grep -v '^127\\.' | head -1)\n"
            s += "    printf '=IP=%s\\n' \"$ip4\"\n"
            s += "    cat /proc/net/tcp 2>/dev/null\n"
            s += "    cat /proc/net/tcp6 2>/dev/null\n"
        }
        s += "    printf '=EOS=\\n'\n"
        s += "    sleep 1\n"
        s += "  done ) >\(monitor) 2>/dev/null &\n\n"
        for entry in config.imageEnv {
            guard let eq = entry.firstIndex(of: "="), eq != entry.startIndex else { continue }
            let key = String(entry[..<eq])
            let value = String(entry[entry.index(after: eq)...])
            s += "export \(key)=\(shellQuote(value))\n"
        }
        if let wd = config.workdir {
            s += "cd \(shellQuote(wd)) 2>/dev/null\n"
        }

        switch config.transport {
        case .vsock:
            // The agent never returns; if it does, the guest is broken, so fall
            // through to poweroff rather than leaving a VM spinning.
            s += "\(agentGuestPath)\n"
        case .legacyConsole:
            s += "i=0; while [ ! -e /dev/hvc1 ] && [ $i -lt 100 ]; do sleep 0.1; i=$((i+1)); done\n"
            s += "setsid -c /bin/sh </dev/hvc1 >/dev/hvc1 2>&1\n"
        }

        s += """
        sync
        poweroff -f 2>/dev/null
        halt -f 2>/dev/null
        reboot -f 2>/dev/null
        echo o > /proc/sysrq-trigger 2>/dev/null

        """
        return s
    }

    // MARK: State

    private let nonce: String
    private let bootConsole = ConsoleBuffer()
    private let shellConsole = ConsoleBuffer()
    private let vmQueue = DispatchQueue(label: "mlxserve.vzguest")
    private var vm: VZVirtualMachine?
    /// nil until `boot`; nil forever on `.legacyConsole`.
    private var socketDevice: VZVirtioSocketDevice?
    /// The config `boot` ran with. `.legacyConsole` before boot so a stray
    /// `exec` can't try to open a vsock stream that was never configured.
    private var config = Config(kernelPath: "", rootfsDir: "", transport: .legacyConsole)
    private var delegateBox: DelegateBox?
    private let execLock = NSLock()
    private var seq = 0
    private let stopped = NSLock() // guards `stoppedFlag`
    private var stoppedFlag = false

    // Pipes held for the guest's lifetime (the attachments borrow their fds).
    private var bootIn = Pipe(), bootOut = Pipe()
    private var shellIn = Pipe(), shellOut = Pipe()
    private var netIn = Pipe(), netOut = Pipe()

    /// Complete net-report snapshots from the guest's hvc2 monitor loop (raw
    /// text between =EOS= sentinels; parse with `GuestNetParser`). Set BEFORE
    /// `boot`; called on a dispatch-io thread.
    var onNetSnapshot: ((String) -> Void)?
    private let netSplitter = GuestNetSnapshotSplitter()

    init(nonce: String = String(UUID().uuidString.prefix(8))) {
        self.nonce = nonce
    }

    /// Marks the guest dead when VZ reports a stop (delegate runs on vmQueue).
    private final class DelegateBox: NSObject, VZVirtualMachineDelegate {
        let onStop: () -> Void
        init(onStop: @escaping () -> Void) { self.onStop = onStop }
        func guestDidStop(_ virtualMachine: VZVirtualMachine) { onStop() }
        func virtualMachine(_ virtualMachine: VZVirtualMachine, didStopWithError error: Error) { onStop() }
    }

    var isFinished: Bool {
        stopped.lock(); let dead = stoppedFlag; stopped.unlock()
        if dead { return true }
        guard let vm else { return true }
        let state = vmQueue.sync { vm.state }
        switch state {
        case .running, .starting, .pausing, .resuming, .paused: return false
        default: return true
        }
    }

    /// All guest BOOT console bytes seen so far (kernel log + init banner), plus
    /// the shell channel tail. For diagnostics — the exec path reads suffixes of
    /// the shell channel, never this.
    func consoleSnapshot() -> String {
        let boot = bootConsole.text(from: 0)
        let shell = shellConsole.text(from: 0)
        return shell.isEmpty ? boot : boot + "\n--- shell channel ---\n" + shell.suffix(2000)
    }

    // MARK: Boot

    /// Boot the guest and wait for its shell to be ready. Blocking — call off the
    /// main thread. `readyTimeout` covers kernel boot + shell spawn (virtiofs
    /// root boots in well under a second; the margin is for cold page-ins).
    func boot(_ cfg: Config, readyTimeout: TimeInterval = 60) throws {
        config = cfg

        // 1. Write the init script into the rootfs the kernel will mount as /.
        let initHostPath = cfg.rootfsDir + Self.initScriptGuestPath
        do {
            try Self.buildInitScript(config: cfg).write(toFile: initHostPath, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: initHostPath)
        } catch {
            throw GuestError.bootFailed("could not write \(initHostPath): \(error)")
        }

        // 1b. Inject the guest agent beside it. Host-injected rather than baked
        // into the image, so a user-chosen base image needs no cooperation.
        if cfg.transport == .vsock {
            guard let source = cfg.agentBinaryPath, FileManager.default.isReadableFile(atPath: source) else {
                throw GuestError.bootFailed("vz-agent binary missing (build it with `zig build vz-agent`)")
            }
            let destination = cfg.rootfsDir + Self.agentGuestPath
            do {
                try? FileManager.default.removeItem(atPath: destination)
                try FileManager.default.copyItem(atPath: source, toPath: destination)
                try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: destination)
            } catch {
                throw GuestError.bootFailed("could not install \(destination): \(error)")
            }
        }

        // 2. Assemble the VM.
        let vmConfig = VZVirtualMachineConfiguration()
        vmConfig.cpuCount = max(VZVirtualMachineConfiguration.minimumAllowedCPUCount,
                                min(cfg.cpuCount, ProcessInfo.processInfo.activeProcessorCount))
        vmConfig.memorySize = max(VZVirtualMachineConfiguration.minimumAllowedMemorySize,
                                  min(cfg.ramBytes, VZVirtualMachineConfiguration.maximumAllowedMemorySize))
        vmConfig.platform = VZGenericPlatformConfiguration()

        let bootLoader = VZLinuxBootLoader(kernelURL: URL(fileURLWithPath: cfg.kernelPath))
        bootLoader.commandLine = Self.kernelCommandLine(network: cfg.network)
        vmConfig.bootLoader = bootLoader

        func serialPort(_ inPipe: Pipe, _ outPipe: Pipe) -> VZVirtioConsoleDeviceSerialPortConfiguration {
            let p = VZVirtioConsoleDeviceSerialPortConfiguration()
            p.attachment = VZFileHandleSerialPortAttachment(
                fileHandleForReading: inPipe.fileHandleForReading,
                fileHandleForWriting: outPipe.fileHandleForWriting)
            return p
        }
        // Index order is guest hvc order (spike-verified). `.legacyConsole`:
        // 0 = boot console, 1 = shell, 2 = net-report. `.vsock` drops the shell
        // port entirely, so the net-report becomes hvc1 — `monitorDevice` and
        // `buildInitScript` must agree with this list, or the monitor writes to
        // a device nobody reads.
        switch cfg.transport {
        case .vsock:
            vmConfig.serialPorts = [serialPort(bootIn, bootOut), serialPort(netIn, netOut)]
            vmConfig.socketDevices = [VZVirtioSocketDeviceConfiguration()]
        case .legacyConsole:
            vmConfig.serialPorts = [serialPort(bootIn, bootOut), serialPort(shellIn, shellOut), serialPort(netIn, netOut)]
        }

        let rootfsDev = VZVirtioFileSystemDeviceConfiguration(tag: Self.rootfsTag)
        rootfsDev.share = VZSingleDirectoryShare(
            directory: VZSharedDirectory(url: URL(fileURLWithPath: cfg.rootfsDir), readOnly: false))
        var shares = [rootfsDev]
        if let ws = cfg.workspacePath {
            let wsDev = VZVirtioFileSystemDeviceConfiguration(tag: Self.workspaceTag)
            wsDev.share = VZSingleDirectoryShare(
                directory: VZSharedDirectory(url: URL(fileURLWithPath: ws), readOnly: false))
            shares.append(wsDev)
        }
        vmConfig.directorySharingDevices = shares

        vmConfig.entropyDevices = [VZVirtioEntropyDeviceConfiguration()]
        vmConfig.memoryBalloonDevices = [VZVirtioTraditionalMemoryBalloonDeviceConfiguration()]
        // NAT network — only when the user left sandbox networking on. With it
        // off there is NO network device at all: the guest is fully isolated.
        if cfg.network {
            let net = VZVirtioNetworkDeviceConfiguration()
            net.attachment = VZNATNetworkDeviceAttachment()
            vmConfig.networkDevices = [net]
        }

        do { try vmConfig.validate() } catch {
            throw GuestError.bootFailed("invalid VM configuration: \(error.localizedDescription)")
        }

        // 3. Console capture + start.
        bootOut.fileHandleForReading.readabilityHandler = { [bootConsole] h in bootConsole.append(h.availableData) }
        if cfg.transport == .legacyConsole {
            shellOut.fileHandleForReading.readabilityHandler = { [shellConsole] h in shellConsole.append(h.availableData) }
        }
        netOut.fileHandleForReading.readabilityHandler = { [netSplitter, weak self] h in
            for snapshot in netSplitter.feed(h.availableData) {
                self?.onNetSnapshot?(snapshot)
            }
        }

        let machine = vmQueue.sync { VZVirtualMachine(configuration: vmConfig, queue: vmQueue) }
        socketDevice = vmQueue.sync { machine.socketDevices.first as? VZVirtioSocketDevice }
        let box = DelegateBox { [weak self] in
            guard let self else { return }
            self.stopped.lock(); self.stoppedFlag = true; self.stopped.unlock()
        }
        vmQueue.sync { machine.delegate = box }
        delegateBox = box
        vm = machine

        var startError: Error?
        let started = DispatchSemaphore(value: 0)
        vmQueue.async {
            machine.start { result in
                if case .failure(let err) = result { startError = err }
                started.signal()
            }
        }
        if started.wait(timeout: .now() + 30) == .timedOut {
            throw GuestError.bootFailed("start timed out")
        }
        if let startError {
            throw GuestError.bootFailed(startError.localizedDescription)
        }

        // 4. Handshake.
        if cfg.transport == .vsock {
            try waitForAgent(deadline: Date().addingTimeInterval(readyTimeout))
            return
        }

        // Legacy: repeatedly nudge the shell until the (echo-proof) ready marker
        // appears — proof it's actually executing, not just echoing. Probes sent
        // before /.vz-init attaches the shell are simply dropped by the closed
        // port; the retry loop is what makes this robust.
        let deadline = Date().addingTimeInterval(readyTimeout)
        let probe = ShellSentinel.readyProbe(nonce: nonce)
        var lastProbe = Date.distantPast
        var ready = false
        while Date() < deadline {
            if isFinished { throw GuestError.guestExited }
            if ShellSentinel.isReady(shellConsole.text(from: 0), nonce: nonce) { ready = true; break }
            if Date().timeIntervalSince(lastProbe) > 0.4 {
                shellWrite(probe); lastProbe = Date()
            }
            Thread.sleep(forTimeInterval: 0.02)
        }
        guard ready else { throw GuestError.notReady }

        // Readiness can fire from a partially-received probe (only the trailing
        // `printf` ran, not the leading `stty -echo`), leaving the tty echoing.
        // Re-send the quieting config now that the shell is definitely reading.
        shellWrite(Array("stty -echo 2>/dev/null; export PS1='' PS2='' 2>/dev/null\n".utf8))

        // Drain to quiet: several ready probes may have queued during boot; let
        // them + their echoes flush so the first exec starts from an idle shell.
        var last = shellConsole.mark()
        var quietSince = Date()
        let drainDeadline = Date().addingTimeInterval(5)
        while Date() < drainDeadline {
            Thread.sleep(forTimeInterval: 0.05)
            let n = shellConsole.mark()
            if n != last { last = n; quietSince = Date() }
            else if Date().timeIntervalSince(quietSince) > 0.3 { break }
        }
    }

    // MARK: Exec

    /// Open one connection to the guest agent.
    ///
    /// `VZVirtioSocketDevice` calls must happen on the queue the VM was created
    /// with, so the connect is dispatched there and the caller blocks on a
    /// semaphore. Never call this FROM `vmQueue` — the completion handler runs
    /// there too, and you would deadlock.
    func openStream(port: UInt32 = GuestProtocol.execPort, timeout: TimeInterval = 10) throws -> GuestByteStream {
        guard let device = socketDevice else {
            throw GuestError.bootFailed("guest has no vsock device (transport is \(config.transport))")
        }
        if isFinished { throw GuestError.guestExited }

        var outcome: Result<VZVirtioSocketConnection, Error>?
        let done = DispatchSemaphore(value: 0)
        vmQueue.async {
            // NS_REFINED_FOR_SWIFT: the handler takes a Result, not (conn, err).
            device.connect(toPort: port) { result in
                outcome = result
                done.signal()
            }
        }
        guard done.wait(timeout: .now() + timeout) == .success else {
            throw GuestError.bootFailed("vsock connect to port \(port) timed out")
        }
        return makeGuestStream(from: try outcome!.get())
    }

    /// Poll the agent's port until it accepts. Replaces the legacy ready-probe
    /// handshake: a successful connect IS the proof the agent is serving.
    private func waitForAgent(deadline: Date) throws {
        var lastError: Error?
        while Date() < deadline {
            if isFinished { throw GuestError.guestExited }
            do {
                let stream = try openStream(timeout: 2)
                stream.close()
                return
            } catch {
                lastError = error
                Thread.sleep(forTimeInterval: 0.05)
            }
        }
        throw GuestError.bootFailed("guest agent never accepted on vsock port \(GuestProtocol.execPort)"
            + (lastError.map { ": \($0)" } ?? ""))
    }

    /// Run one command in the guest and return its output + exit code. Blocking.
    ///
    /// `.vsock` runs each command as its own `sh -c`, so shell-local state
    /// (`export FOO=1`, `cd`) does NOT persist between calls — exactly like the
    /// host path, which spawns a fresh `zsh -l -c` per command. Filesystem state
    /// (installed packages, files) persists as before. Callers that need a
    /// sticky cwd pass it in.
    func exec(_ command: String, cwd: String? = nil, timeout: TimeInterval = 120) throws -> ExecResult {
        guard vm != nil, !isFinished else { throw GuestError.guestExited }

        if config.transport == .vsock {
            let stream = try openStream()
            defer { stream.close() }
            // `output` is stdout+stderr in ARRIVAL order (as well as two pipes
            // can express it) — built from the streaming callbacks, not by
            // concatenating the finished buffers, which would shove all of a
            // compiler's stderr after all of its stdout.
            var merged = Data()
            let result = try GuestExec.run(
                stream: stream,
                request: .init(command: command, cwd: cwd ?? ""),
                timeout: timeout,
                streaming: .init(onStdout: { merged.append($0) }, onStderr: { merged.append($0) }))
            return ExecResult(
                output: String(decoding: merged, as: UTF8.self),
                stdout: result.stdout,
                stderr: result.stderr,
                exitCode: result.exitCode,
                timedOut: result.timedOut,
                pid: result.pid)
        }
        return try execLegacy(command, timeout: timeout)
    }

    /// Start a process that outlives its connection, logging to `logPath`.
    /// Returns the guest pid so the host can `kill` it later.
    func execDetached(command: String, cwd: String?, logPath: String, timeout: TimeInterval = 30) throws -> Int32? {
        guard config.transport == .vsock else { throw GuestError.bootFailed("detached exec needs the vsock transport") }
        let stream = try openStream()
        defer { stream.close() }
        let result = try GuestExec.run(
            stream: stream,
            request: .init(command: command, cwd: cwd ?? "", logPath: logPath, detach: true),
            timeout: timeout)
        return result.pid
    }

    /// Legacy hvc1 shell. Merged output only — one tty carried stdout and
    /// stderr, so they cannot be told apart. On timeout, sends Ctrl-C (SIGINT
    /// via the tty line discipline).
    private func execLegacy(_ command: String, timeout: TimeInterval) throws -> ExecResult {
        execLock.lock(); defer { execLock.unlock() }
        guard vm != nil else { throw GuestError.guestExited }
        if isFinished { throw GuestError.guestExited }

        seq += 1
        let mySeq = seq
        let start = shellConsole.mark()
        shellWrite(ShellSentinel.frame(command: command, nonce: nonce, seq: mySeq))

        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if let r = ShellSentinel.scan(shellConsole.text(from: start), nonce: nonce, seq: mySeq) {
                return ExecResult(output: r.output, exitCode: r.code, timedOut: false)
            }
            if isFinished {
                // Guest died mid-command: return partial output rather than hang.
                return ExecResult(output: shellConsole.text(from: start), exitCode: -1, timedOut: false)
            }
            Thread.sleep(forTimeInterval: 0.02)
        }
        // Timed out — interrupt the running command so the shell is usable again.
        shellWrite([0x03]) // Ctrl-C
        return ExecResult(output: shellConsole.text(from: start), exitCode: 124, timedOut: true)
    }

    /// Feed raw bytes to the shell channel. The pipe's read end is owned by the
    /// VM config (and by us), so writes never hit a closed pipe.
    private func shellWrite(_ bytes: [UInt8]) {
        shellIn.fileHandleForWriting.write(Data(bytes))
    }

    // MARK: Shutdown

    /// Stop the guest and release the console handlers. Idempotent, blocking
    /// (bounded). Safe to call from any thread except `vmQueue`.
    func shutdown() {
        if let machine = vm {
            let done = DispatchSemaphore(value: 0)
            vmQueue.async {
                if machine.state == .running || machine.state == .paused {
                    machine.stop { _ in done.signal() }
                } else {
                    done.signal()
                }
            }
            _ = done.wait(timeout: .now() + 10)
            vm = nil
        }
        stopped.lock(); stoppedFlag = true; stopped.unlock()
        bootOut.fileHandleForReading.readabilityHandler = nil
        shellOut.fileHandleForReading.readabilityHandler = nil
        netOut.fileHandleForReading.readabilityHandler = nil
        onNetSnapshot = nil
        delegateBox = nil
    }

    deinit { shutdown() }
}
