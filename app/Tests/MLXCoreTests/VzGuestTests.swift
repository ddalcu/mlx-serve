import XCTest
@testable import MLXCore

/// Pure-logic tests for the Virtualization.framework guest: the generated
/// `/.vz-init` script, shell quoting, and the kernel command line. No VM, no
/// entitlement — safe in CI. The live boot path is covered by SandboxSmoke
/// (run in the signed app: SANDBOX_SMOKE=1).
final class VzGuestTests: XCTestCase {

    /// The legacy hvc1-shell layout. Explicit, because the DEFAULT transport is
    /// now `.vsock` — these assertions are about the arm that is on its way out.
    private func baseConfig() -> VzGuest.Config {
        var c = VzGuest.Config(kernelPath: "/k", rootfsDir: "/r")
        c.workspacePath = "/Users/d/proj"
        c.workdir = "/workspace"
        c.transport = .legacyConsole
        return c
    }

    private func vsockConfig() -> VzGuest.Config {
        var c = baseConfig()
        c.transport = .vsock
        c.agentBinaryPath = "/tmp/vz-agent"
        return c
    }

    // MARK: kernel command line

    func testKernelCommandLineBootsVirtiofsRootWithOurInit() {
        let cl = VzGuest.kernelCommandLine(network: false)
        XCTAssertTrue(cl.contains("console=hvc0"), "boot console must be the first hvc port")
        XCTAssertTrue(cl.contains("root=\(VzGuest.rootfsTag)"))
        XCTAssertTrue(cl.contains("rootfstype=virtiofs"), "rootfs is served over virtio-fs — no initramfs, no RAM-resident image")
        XCTAssertTrue(cl.contains("init=\(VzGuest.initScriptGuestPath)"))
        XCTAssertFalse(cl.contains("rootwait"), "rootwait is invalid for a non-block root and trips a kernel warning")
    }

    func testKernelCommandLineNetworkTogglesInKernelDhcp() {
        // The prebuilt kernel has CONFIG_IP_PNP — `ip=dhcp` makes the KERNEL
        // acquire address/route/DNS from VZ's NAT before init runs, so the guest
        // gets networking with any image (no userspace DHCP client needed).
        XCTAssertTrue(VzGuest.kernelCommandLine(network: true).contains(" ip=dhcp"))
        XCTAssertFalse(VzGuest.kernelCommandLine(network: false).contains("ip=dhcp"),
                       "network off must not DHCP — the isolated guest stays address-less")
        // The base boot plumbing must be identical in both variants.
        XCTAssertTrue(VzGuest.kernelCommandLine(network: true)
            .hasPrefix(VzGuest.kernelCommandLine(network: false)))
    }

    // MARK: init script

    func testInitScriptMountsKernelFilesystemsAndWorkspace() {
        let s = VzGuest.buildInitScript(config: baseConfig())
        XCTAssertTrue(s.hasPrefix("#!/bin/sh\n"))
        XCTAssertTrue(s.contains("mount -t proc proc /proc"))
        XCTAssertTrue(s.contains("mount -t sysfs sysfs /sys"))
        XCTAssertTrue(s.contains("mount -t devtmpfs devtmpfs /dev"))
        XCTAssertTrue(s.contains("mount -t virtiofs \(VzGuest.workspaceTag) /workspace"),
                      "the host share must be mounted from the workspace virtiofs tag")
    }

    func testInitScriptWithoutWorkspaceSkipsShareMount() {
        var c = baseConfig()
        c.workspacePath = nil
        let s = VzGuest.buildInitScript(config: c)
        XCTAssertFalse(s.contains("mount -t virtiofs \(VzGuest.workspaceTag)"))
    }

    func testLegacyInitScriptRunsShellOnSecondConsolePort() {
        let s = VzGuest.buildInitScript(config: baseConfig())
        // The shell lives on /dev/hvc1 — a dedicated clean channel. hvc0 keeps
        // kernel printk so boot failures stay diagnosable without polluting the
        // sentinel stream.
        XCTAssertTrue(s.contains("/dev/hvc1"))
        XCTAssertTrue(s.contains("setsid -c /bin/sh </dev/hvc1 >/dev/hvc1 2>&1"),
                      "shell needs hvc1 as its controlling tty (Ctrl-C interrupt, stty)")
        // PID 1 must power the guest off when the shell exits (contain's
        // poweroff_seq: slim images ship no poweroff binary → sysrq fallback).
        XCTAssertTrue(s.contains("poweroff -f"))
        XCTAssertTrue(s.contains("echo o > /proc/sysrq-trigger"))
    }

    // MARK: vsock transport

    func testVsockInitScriptExecsTheAgentAndHasNoShell() {
        let s = VzGuest.buildInitScript(config: vsockConfig())
        XCTAssertTrue(s.contains(VzGuest.agentGuestPath),
                      "PID 1 must hand off to the guest agent")
        XCTAssertFalse(s.contains("setsid -c /bin/sh"),
                       "the persistent hvc1 shell is the legacy transport")
        // If the agent ever returns, the guest must power off, not spin.
        XCTAssertTrue(s.contains("poweroff -f"))
    }

    /// hvc numbering follows `serialPorts` order. `.vsock` drops the shell port,
    /// so the monitor slides from hvc2 to hvc1. Getting this wrong is SILENT:
    /// the monitor writes to a device nobody reads, and the tray's RAM readout
    /// plus the live port map simply stop, with no error anywhere.
    func testMonitorDeviceFollowsTheSerialPortCount() {
        XCTAssertEqual(VzGuest.monitorDevice(transport: .legacyConsole), "/dev/hvc2")
        XCTAssertEqual(VzGuest.monitorDevice(transport: .vsock), "/dev/hvc1")
    }

    func testVsockInitScriptWritesTheMonitorToHvc1NotHvc2() {
        var c = vsockConfig()
        c.network = true
        let s = VzGuest.buildInitScript(config: c)

        XCTAssertTrue(s.contains(">/dev/hvc1 2>/dev/null &"),
                      "the monitor loop must target hvc1 when the shell port is gone")
        XCTAssertFalse(s.contains("/dev/hvc2"),
                       "hvc2 does not exist with only two serial ports")
        // Everything the host parser needs must still be there.
        XCTAssertTrue(s.contains("=EOS="))
        XCTAssertTrue(s.contains("=IP="))
        XCTAssertTrue(s.contains("/proc/net/tcp"))
    }

    func testLegacyInitScriptStillWritesTheMonitorToHvc2() {
        var c = baseConfig()
        c.network = true
        let s = VzGuest.buildInitScript(config: c)
        XCTAssertTrue(s.contains(">/dev/hvc2 2>/dev/null &"))
    }

    func testInitScriptExportsImageEnvShellQuoted() {
        var c = baseConfig()
        c.imageEnv = ["PIP_BREAK_SYSTEM_PACKAGES=1",
                      "PATH=/custom/bin:/usr/bin",
                      "MOTD=it's a trap \"quoted\""]
        let s = VzGuest.buildInitScript(config: c)
        XCTAssertTrue(s.contains("export PIP_BREAK_SYSTEM_PACKAGES='1'"))
        // The image's own PATH must be exported AFTER the bootstrap PATH so it wins.
        let bootstrapIdx = s.range(of: "export PATH=/usr/local/sbin")!.lowerBound
        let imageIdx = s.range(of: "export PATH='/custom/bin:/usr/bin'")!.lowerBound
        XCTAssertTrue(bootstrapIdx < imageIdx, "image PATH must override the bootstrap PATH")
        // A value containing a single quote must survive quoting.
        XCTAssertTrue(s.contains("export MOTD='it'\\''s a trap \"quoted\"'"))
    }

    func testInitScriptSkipsMalformedEnvEntries() {
        var c = baseConfig()
        c.imageEnv = ["NOEQUALS", "OK=yes"]
        let s = VzGuest.buildInitScript(config: c)
        XCTAssertFalse(s.contains("NOEQUALS"))
        XCTAssertTrue(s.contains("export OK='yes'"))
    }

    func testInitScriptCdsIntoWorkdir() {
        var c = baseConfig()
        c.workdir = "/workspace/sub dir"
        let s = VzGuest.buildInitScript(config: c)
        XCTAssertTrue(s.contains("cd '/workspace/sub dir' 2>/dev/null"))
    }

    // MARK: networking in the init script

    func testLegacyInitScriptNetworkOnWiresDnsAndPortMonitor() {
        var c = baseConfig()
        c.network = true
        let s = VzGuest.buildInitScript(config: c)
        // DNS: the kernel's DHCP answer lands in /proc/net/pnp — copy the
        // resolver lines into /etc/resolv.conf so libc can resolve.
        XCTAssertTrue(s.contains("/proc/net/pnp"))
        XCTAssertTrue(s.contains("/etc/resolv.conf"))
        // Live port map source: a background loop streams the guest IP +
        // /proc/net/tcp(6) snapshots to the host over the third console port.
        XCTAssertTrue(s.contains("/dev/hvc2"))
        XCTAssertTrue(s.contains("/proc/net/tcp"))
        XCTAssertTrue(s.contains("=EOS="), "snapshots must be framed so the host can split them")
        XCTAssertTrue(s.contains("=IP="), "each snapshot must carry the guest address")
    }

    func testInitScriptNetworkOffStaysIsolated() {
        var c = baseConfig()
        c.network = false
        let s = VzGuest.buildInitScript(config: c)
        XCTAssertFalse(s.contains("dhclient"), "network off must not even try a DHCP client")
        XCTAssertFalse(s.contains("/etc/resolv.conf"))
        // The hvc2 monitor still runs (it carries the RAM readout for the tray)
        // but must not report addresses when the guest is isolated.
        XCTAssertTrue(s.contains("/dev/hvc2"))
        XCTAssertTrue(s.contains("/proc/meminfo"))
        XCTAssertFalse(s.contains("fib_trie"))
    }

    func testInitScriptMonitorReportsMemoryRegardlessOfNetwork() {
        for network in [true, false] {
            var c = baseConfig()
            c.network = network
            let s = VzGuest.buildInitScript(config: c)
            XCTAssertTrue(s.contains("/proc/meminfo"), "network=\(network)")
            XCTAssertTrue(s.contains("=EOS="), "network=\(network)")
        }
    }

    // MARK: shell quoting

    func testShellQuoteWrapsAndEscapesSingleQuotes() {
        XCTAssertEqual(VzGuest.shellQuote("plain"), "'plain'")
        XCTAssertEqual(VzGuest.shellQuote("it's"), "'it'\\''s'")
        XCTAssertEqual(VzGuest.shellQuote(""), "''")
        XCTAssertEqual(VzGuest.shellQuote("a\"b $HOME `x`"), "'a\"b $HOME `x`'",
                       "double quotes / $ / backticks are inert inside single quotes")
    }
}
