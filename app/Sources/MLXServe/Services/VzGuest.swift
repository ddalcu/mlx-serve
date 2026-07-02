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
    }

    struct ExecResult {
        var output: String
        var exitCode: Int32
        var timedOut: Bool
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
    static let kernelCommandLine =
        "console=hvc0 root=\(rootfsTag) rootfstype=virtiofs rw init=\(initScriptGuestPath) panic=-1"

    // MARK: Pure builders (unit tested)

    /// POSIX single-quote escaping: the only character that needs handling inside
    /// single quotes is the single quote itself ('\'' = close, literal, reopen).
    static func shellQuote(_ s: String) -> String {
        "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    /// The `/.vz-init` PID-1 script written into the rootfs dir before boot.
    /// Mounts the kernel filesystems + the workspace share, applies the image's
    /// env/workdir, then hands a persistent `/bin/sh` the DEDICATED hvc1 channel
    /// as its controlling tty. When the shell exits, the guest powers off
    /// (contain's proven poweroff sequence — slim images ship no poweroff binary,
    /// so SysRq 'o' is the fallback).
    ///
    /// We deliberately run /bin/sh (dash), NOT interactive bash: a host driver
    /// feeding commands over a byte stream + matching an exit-code sentinel needs
    /// a clean, predictable stream; readline escapes and job control fight that.
    /// A command that genuinely needs bash can invoke `bash -c '…'` explicitly.
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
        // Best-effort networking: VZ's NAT device hands out addresses over DHCP,
        // so this only works when the image ships a DHCP client (the default
        // agent-shell image currently does not — the guest is network-isolated,
        // which is the safe default for a sandbox anyway).
        s += """
        ip link set lo up 2>/dev/null
        ip link set eth0 up 2>/dev/null
        dhclient -1 eth0 2>/dev/null || udhcpc -i eth0 -n -q 2>/dev/null || dhcpcd -1 eth0 2>/dev/null || true

        """
        for entry in config.imageEnv {
            guard let eq = entry.firstIndex(of: "="), eq != entry.startIndex else { continue }
            let key = String(entry[..<eq])
            let value = String(entry[entry.index(after: eq)...])
            s += "export \(key)=\(shellQuote(value))\n"
        }
        if let wd = config.workdir {
            s += "cd \(shellQuote(wd)) 2>/dev/null\n"
        }
        s += """
        i=0; while [ ! -e /dev/hvc1 ] && [ $i -lt 100 ]; do sleep 0.1; i=$((i+1)); done
        setsid -c /bin/sh </dev/hvc1 >/dev/hvc1 2>&1
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
    private var delegateBox: DelegateBox?
    private let execLock = NSLock()
    private var seq = 0
    private let stopped = NSLock() // guards `stoppedFlag`
    private var stoppedFlag = false

    // Pipes held for the guest's lifetime (the attachments borrow their fds).
    private var bootIn = Pipe(), bootOut = Pipe()
    private var shellIn = Pipe(), shellOut = Pipe()

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
        // 1. Write the init script into the rootfs the kernel will mount as /.
        let initHostPath = cfg.rootfsDir + Self.initScriptGuestPath
        do {
            try Self.buildInitScript(config: cfg).write(toFile: initHostPath, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: initHostPath)
        } catch {
            throw GuestError.bootFailed("could not write \(initHostPath): \(error)")
        }

        // 2. Assemble the VM.
        let vmConfig = VZVirtualMachineConfiguration()
        vmConfig.cpuCount = max(VZVirtualMachineConfiguration.minimumAllowedCPUCount,
                                min(cfg.cpuCount, ProcessInfo.processInfo.activeProcessorCount))
        vmConfig.memorySize = max(VZVirtualMachineConfiguration.minimumAllowedMemorySize,
                                  min(cfg.ramBytes, VZVirtualMachineConfiguration.maximumAllowedMemorySize))
        vmConfig.platform = VZGenericPlatformConfiguration()

        let bootLoader = VZLinuxBootLoader(kernelURL: URL(fileURLWithPath: cfg.kernelPath))
        bootLoader.commandLine = Self.kernelCommandLine
        vmConfig.bootLoader = bootLoader

        func serialPort(_ inPipe: Pipe, _ outPipe: Pipe) -> VZVirtioConsoleDeviceSerialPortConfiguration {
            let p = VZVirtioConsoleDeviceSerialPortConfiguration()
            p.attachment = VZFileHandleSerialPortAttachment(
                fileHandleForReading: inPipe.fileHandleForReading,
                fileHandleForWriting: outPipe.fileHandleForWriting)
            return p
        }
        // Index order is guest hvc order (spike-verified): 0 = boot console, 1 = shell.
        vmConfig.serialPorts = [serialPort(bootIn, bootOut), serialPort(shellIn, shellOut)]

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
        // NAT network: only usable by images that ship a DHCP client (see init
        // script). Harmless otherwise — the guest simply has no address.
        let net = VZVirtioNetworkDeviceConfiguration()
        net.attachment = VZNATNetworkDeviceAttachment()
        vmConfig.networkDevices = [net]

        do { try vmConfig.validate() } catch {
            throw GuestError.bootFailed("invalid VM configuration: \(error.localizedDescription)")
        }

        // 3. Console capture + start.
        bootOut.fileHandleForReading.readabilityHandler = { [bootConsole] h in bootConsole.append(h.availableData) }
        shellOut.fileHandleForReading.readabilityHandler = { [shellConsole] h in shellConsole.append(h.availableData) }

        let machine = vmQueue.sync { VZVirtualMachine(configuration: vmConfig, queue: vmQueue) }
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

        // 4. Handshake: repeatedly nudge the shell until the (echo-proof) ready
        // marker appears — proof it's actually executing, not just echoing.
        // Probes sent before /.vz-init attaches the shell are simply dropped by
        // the closed port; the retry loop is what makes this robust.
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

    /// Run one command in the guest shell and return its merged output + exit
    /// code. Blocking. On timeout, sends Ctrl-C (SIGINT via the hvc tty line
    /// discipline) and returns `timedOut = true` with whatever output arrived.
    func exec(_ command: String, timeout: TimeInterval = 120) throws -> ExecResult {
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
        delegateBox = nil
    }

    deinit { shutdown() }
}
