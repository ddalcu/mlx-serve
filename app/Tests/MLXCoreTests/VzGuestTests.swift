import XCTest
import Virtualization
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

    func testInitScriptAlwaysMountsProjectsShare() {
        // The `projects` device is always configured (empty until a chat folder
        // is hot-mounted), so its mount point must be established at boot — even
        // with no workspace share.
        XCTAssertTrue(VzGuest.buildInitScript(config: baseConfig())
            .contains("mount -t virtiofs \(VzGuest.projectsTag) \(VzGuest.guestProjectsPath)"),
            "projects tag must be mounted at boot for hot-mounts to surface")
        var noWs = baseConfig(); noWs.workspacePath = nil
        XCTAssertTrue(VzGuest.buildInitScript(config: noWs)
            .contains("mount -t virtiofs \(VzGuest.projectsTag) \(VzGuest.guestProjectsPath)"))
        XCTAssertTrue(VzGuest.buildInitScript(config: baseConfig())
            .contains("mkdir -p /proc /sys /dev \(VzGuest.guestProjectsPath)"),
            "the /projects mount point must be created before the mount")
    }

    func testMultiDirectoryShareMapsSlugsToHostDirs() {
        let share = VzGuest.multiDirectoryShare(["proj-abc": "/Users/d/proj", "lib-def": "/Users/d/lib"])
        XCTAssertEqual(Set(share.directories.keys), ["proj-abc", "lib-def"])
        XCTAssertEqual(share.directories["proj-abc"]?.url.path, "/Users/d/proj")
        XCTAssertEqual(VzGuest.multiDirectoryShare([:]).directories.count, 0,
                       "an empty set is valid — mounts as an empty /projects")
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

    // MARK: ssh (dropbear) arm

    func testInitScriptSshArmStartsDropbearKeyOnlyWithPtys() {
        var c = vsockConfig()
        c.network = true
        c.sshEnabled = true
        let s = VzGuest.buildInitScript(config: c)
        // Key-only auth (-s), stable host keys generated into the persistent
        // rootfs (-R), on the standard port the dedicated forwarder targets.
        XCTAssertTrue(s.contains("dropbear -R -s -p 22"), s)
        // dropbear is baked into the image, not host-injected — a stale cached
        // image simply lacks it, so the start must be gated, never a hard fail.
        XCTAssertTrue(s.contains("command -v dropbear"), "missing dropbear must not break boot")
        // ssh sessions allocate ptys; the base init only mounts devtmpfs, which
        // does NOT auto-mount /dev/pts — without it every session dies with
        // "PTY allocation request failed".
        XCTAssertTrue(s.contains("mount -t devpts devpts /dev/pts"), s)
    }

    func testInitScriptWithoutSshHasNoDropbear() {
        var c = vsockConfig()
        c.network = true
        let s = VzGuest.buildInitScript(config: c)
        XCTAssertFalse(s.contains("dropbear"), "ssh off → no sshd, no devpts requirement")
        XCTAssertFalse(s.contains("devpts"))
    }

    /// Root guest, but models write `sudo`: a pass-through shim, only where
    /// no real sudo exists.
    /// Debian installs games (chocolate-doom, …) into /usr/games, which is
    /// on a login shell's PATH but not on ours; `which chocolate-doom` failed
    /// right after a successful install (live 2026-09-05).
    func testInitScriptPathIncludesUsrGames() {
        XCTAssertTrue(VzGuest.buildInitScript(config: vsockConfig()).contains(":/usr/games HOME="))
    }

    func testInitScriptShimsSudoForModelsThatTypeIt() {
        let s = VzGuest.buildInitScript(config: vsockConfig())
        XCTAssertTrue(s.contains("[ -x /usr/bin/sudo ] ||"), s)
        XCTAssertTrue(s.contains("/usr/local/bin/sudo"), s)
        XCTAssertTrue(s.contains(#"exec "$@""#), "the shim runs the command as-is: \(s)")
    }

    // MARK: desktop (computer use)

    /// Headless stays byte-for-byte what it was: no display → no graphics, no
    /// keyboard, no pointing device, and no desktop arm in the init script.
    func testNoDisplayMeansNoDisplayDevicesAndNoDesktopArm() {
        let c = vsockConfig()
        XCTAssertNil(c.display)
        let vm = VZVirtualMachineConfiguration()
        VzGuest.applyDisplay(c.display, to: vm)
        XCTAssertTrue(vm.graphicsDevices.isEmpty)
        XCTAssertTrue(vm.keyboards.isEmpty)
        XCTAssertTrue(vm.pointingDevices.isEmpty)
        XCTAssertEqual(VzGuest.displayDeviceSummary(for: c), [])
        let s = VzGuest.buildInitScript(config: c)
        XCTAssertFalse(s.contains(VzGuest.desktopStartGuestPath), s)
        XCTAssertFalse(s.contains("/dev/dri"), s)
    }

    /// With a display: ONE virtio-gpu scanout at the requested size, a USB
    /// keyboard and a USB screen-coordinate pointing device (what the Linux
    /// guest's xHCI + HID drivers serve; kernels-v5). The summary the tests
    /// read is derived from the same spec the VM is built from.
    func testDisplayAddsOneScanoutPlusUsbKeyboardAndPointer() {
        var c = vsockConfig()
        c.display = VzGuest.DisplaySpec(width: 1280, height: 800, pixelsPerInch: 80)
        let vm = VZVirtualMachineConfiguration()
        VzGuest.applyDisplay(c.display, to: vm)
        XCTAssertEqual(vm.graphicsDevices.count, 1)
        let gpu = vm.graphicsDevices.first as? VZVirtioGraphicsDeviceConfiguration
        XCTAssertNotNil(gpu, "the graphics device is virtio-gpu (DRM_VIRTIO_GPU in the guest)")
        XCTAssertEqual(gpu?.scanouts.count, 1)
        XCTAssertEqual(gpu?.scanouts.first?.widthInPixels, 1280)
        XCTAssertEqual(gpu?.scanouts.first?.heightInPixels, 800)
        XCTAssertEqual(vm.keyboards.count, 1)
        XCTAssertTrue(vm.keyboards.first is VZUSBKeyboardConfiguration)
        XCTAssertEqual(vm.pointingDevices.count, 1)
        XCTAssertTrue(vm.pointingDevices.first is VZUSBScreenCoordinatePointingDeviceConfiguration)
        XCTAssertEqual(VzGuest.displayDeviceSummary(for: c),
                       ["graphics virtio-gpu 1280x800", "keyboard usb", "pointing usb-screen-coordinate"])
    }

    func testDefaultDisplaySpecIs1280x800() {
        let d = VzGuest.DisplaySpec()
        XCTAssertEqual(d.width, 1280)
        XCTAssertEqual(d.height, 800)
        XCTAssertEqual(d.pixelsPerInch, 80)
    }

    /// The init script starts the desktop ONLY when the VM has a display, and
    /// only when the kernel exposed a DRM node AND the start script was
    /// injected — a headless kernel or an unprovisioned rootfs boots exactly
    /// like before. Backgrounded: the agent transport must not wait on Xorg.
    func testDisplayInitScriptStartsTheDesktopGatedOnDrmAndTheStartScript() {
        var c = vsockConfig()
        c.display = VzGuest.DisplaySpec()
        let s = VzGuest.buildInitScript(config: c)
        XCTAssertTrue(s.contains("mkdir -p /tmp/.X11-unix"), s)
        XCTAssertTrue(s.contains("mount -t tmpfs -o mode=755 tmpfs /run"), "dbus/at-spi sockets cannot bind on virtiofs")
        XCTAssertTrue(s.contains("mount -t tmpfs -o mode=1777 tmpfs /tmp"),
                      "LibreOffice's single-instance socket lives in /tmp; a stale one on virtiofs kills every later launch")
        let tmpMount = s.range(of: "tmpfs /tmp")!.lowerBound
        let x11 = s.range(of: "mkdir -p /tmp/.X11-unix")!.lowerBound
        XCTAssertLessThan(tmpMount, x11, "the X socket dir is created INSIDE the tmpfs, after the mount")
        XCTAssertTrue(s.contains("[ -e /dev/dri/card0 ]"), s)
        XCTAssertTrue(s.contains("[ -x \(VzGuest.desktopStartGuestPath) ]"), s)
        XCTAssertTrue(s.contains("\(VzGuest.desktopStartGuestPath) >"), "desktop output goes to a log, never the console")
        XCTAssertTrue(s.contains("2>&1 &"), "the desktop must be backgrounded")
        // The arm sits before the agent exec (which never returns).
        XCTAssertLessThan(s.range(of: VzGuest.desktopStartGuestPath)!.lowerBound,
                          s.range(of: VzGuest.agentGuestPath + "\n")!.lowerBound)
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
