import XCTest
@testable import MLXCore

/// Pure-logic tests for the Virtualization.framework guest: the generated
/// `/.vz-init` script, shell quoting, and the kernel command line. No VM, no
/// entitlement — safe in CI. The live boot path is covered by SandboxSmoke
/// (run in the signed app: SANDBOX_SMOKE=1).
final class VzGuestTests: XCTestCase {

    private func baseConfig() -> VzGuest.Config {
        var c = VzGuest.Config(kernelPath: "/k", rootfsDir: "/r")
        c.workspacePath = "/Users/d/proj"
        c.workdir = "/workspace"
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

    func testInitScriptRunsShellOnSecondConsolePort() {
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

    func testInitScriptNetworkOnWiresDnsAndPortMonitor() {
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
