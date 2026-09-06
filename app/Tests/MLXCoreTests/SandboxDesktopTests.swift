import XCTest
@testable import MLXCore

/// The sandbox desktop (computer use): the guest-side scripts are generated
/// text, so their contracts are pinned here without a VM. The live path is
/// `SANDBOX_DESKTOP_SMOKE=1` in the signed app.
final class SandboxDesktopTests: XCTestCase {

    // MARK: provisioning script

    /// Idempotent by a VERSION marker in the writable rootfs: a second enable
    /// (or a boot after one) must not re-run apt, and a bumped package list
    /// must. The marker lives in the rootfs dir a base-image re-pull wipes, so
    /// a re-pull re-installs by construction.
    func testProvisionScriptIsGatedOnAVersionedMarker() {
        let s = SandboxDesktop.provisionScript
        XCTAssertTrue(s.contains(SandboxDesktop.markerPath), s)
        XCTAssertTrue(s.contains("provisioned=\(SandboxDesktop.provisionVersion)"), "the marker carries the version")
        XCTAssertTrue(s.contains(SandboxDesktop.alreadyProvisionedLine), "a fast exit names itself so the host can tell it from an install")
        // The marker is written LAST, after a successful install.
        let install = s.range(of: "apt-get install")!.lowerBound
        let marker = s.range(of: "> \(SandboxDesktop.markerPath)")!.lowerBound
        XCTAssertLessThan(install, marker)
    }

    /// The package set is the whole contract: an X server on the virtio-gpu
    /// DRM node, XFCE, and the three things `mlx-computer` shells out to
    /// (xdotool, scrot, AT-SPI via pyatspi). No recommends: a recommends pull
    /// on xfce4 is a gigabyte of apps nobody asked for.
    func testProvisionScriptInstallsTheDesktopAndTheComputerToolDeps() {
        let s = SandboxDesktop.provisionScript
        for pkg in ["xserver-xorg-core", "xinit", "xfce4-session", "xfwm4", "xfce4-panel",
                    "xfce4-terminal", "xdotool", "scrot", "at-spi2-core", "python3-pyatspi", "dbus-x11"] {
            XCTAssertTrue(SandboxDesktop.packages.contains(pkg), "missing \(pkg)")
        }
        XCTAssertTrue(s.contains("--no-install-recommends"), s)
        XCTAssertTrue(s.contains("DEBIAN_FRONTEND=noninteractive"), "apt must never prompt inside the guest")
        // The install exit code lands in the done file so the host can poll
        // it (there is no streaming exec — the host tails the log).
        XCTAssertTrue(s.contains(SandboxDesktop.doneFilePath), s)
        XCTAssertTrue(SandboxDesktop.launchProvisionCommand.contains(SandboxDesktop.installLogPath))
        XCTAssertTrue(SandboxDesktop.launchProvisionCommand.hasSuffix("&"), "detached: the exec must return at once")
        XCTAssertTrue(SandboxDesktop.pollCommand.contains(SandboxDesktop.doneMarker))
    }

    // MARK: start script

    /// The launcher is the boot-time arm AND the after-provision arm, so it
    /// must be safe on an unprovisioned rootfs (no Xorg → one line, exit 0),
    /// and must not start a second X on a live one.
    func testStartScriptNoOpsWithoutXorgAndRefusesADoubleStart() {
        let s = SandboxDesktop.startScript
        XCTAssertTrue(s.hasPrefix("#!/bin/sh"))
        XCTAssertTrue(s.contains("command -v Xorg") || s.contains("command -v xinit"), s)
        XCTAssertTrue(s.contains("pidof Xorg"), "a live X means a second start is a no-op")
        XCTAssertTrue(s.contains("rm -f /tmp/.X0-lock"), "the rootfs /tmp persists across boots; a stale lock blocks X")
        XCTAssertTrue(s.contains("Driver \"evdev\""), "libinput needs udev, which the guest has not")
        XCTAssertTrue(s.contains("/dev/input/event*"), "input devices are enumerated explicitly")
        XCTAssertTrue(s.contains("startxfce4"), s)
        XCTAssertTrue(s.contains("-nolisten tcp"), "X never listens on the network")
        XCTAssertTrue(s.contains("DISPLAY=:0"), s)
        XCTAssertTrue(s.contains("at-spi-bus-launcher"), "the accessibility bus feeds observe")
        // No udev in the guest: explicit input devices, hotplug off; and an
        // explicit layout must name its Screen (live: "Screen 0 deleted").
        XCTAssertTrue(s.contains("AutoAddDevices\" \"off"), s)
        XCTAssertTrue(s.contains("/dev/input/event*"), s)
        XCTAssertTrue(s.contains("Driver \"modesetting\""), s)
        XCTAssertTrue(s.contains("Screen \"screen\""), "the ServerLayout names the Screen")
        XCTAssertTrue(s.contains("chmod 755 /root/.xinitrc"), "xinit execs the client script")
        // The Mac's mouse: VZ's pointer is a USB digitizer that enumerates
        // AFTER the keyboard, so the wait keys on it (a node COUNT wrote the
        // config without it, live 2026-09-06: keyboard worked, clicks did
        // not); its cursor is drawn in software (the hardware cursor plane
        // never shows in the pane).
        XCTAssertTrue(s.contains("grep -q Digitizer /proc/bus/input/devices"), "wait for the pointer, not a node count")
        XCTAssertTrue(s.contains("Option \"SWcursor\" \"on\""), "visible pointer")
        let wait = s.range(of: "grep -q Digitizer")!.lowerBound
        let config = s.range(of: "10-mlx-input.conf")!.lowerBound
        XCTAssertLessThan(wait, config, "the wait precedes the config write")
    }

    // MARK: provisioning state machine (pure)

    func testProvisionPollingReadsTheDoneFileAndTheLogTail() {
        // Still running: the done file is absent, the last log line is progress.
        let running = SandboxDesktop.pollOutcome(pollOutput: "Get:12 http://deb.debian.org trixie/main arm64 xfwm4 arm64 4.20.0-1 [500 kB]\n")
        XCTAssertEqual(running, .installing("Get:12 http://deb.debian.org trixie/main arm64 xfwm4 arm64 4.20.0-1 [500 kB]"))
        // Done, success.
        XCTAssertEqual(SandboxDesktop.pollOutcome(pollOutput: "Setting up xfce4-panel\n\(SandboxDesktop.doneMarker)0\n"), .done(exitCode: 0))
        // Done, failure: the exit code rides along.
        XCTAssertEqual(SandboxDesktop.pollOutcome(pollOutput: "E: Unable to locate package nope\n\(SandboxDesktop.doneMarker)100\n"), .done(exitCode: 100))
        // Nothing yet.
        XCTAssertEqual(SandboxDesktop.pollOutcome(pollOutput: ""), .installing(""))
    }

    /// Network off + no marker = a named refusal, never an apt that hangs on
    /// DNS for ten minutes.
    func testNetworkOffRefusesTheInstallByName() {
        let reason = SandboxDesktop.installRefusal(networkEnabled: false, kernelRefusal: nil)
        XCTAssertNotNil(reason)
        XCTAssertTrue(reason!.contains("network"), reason!)
        XCTAssertNil(SandboxDesktop.installRefusal(networkEnabled: true, kernelRefusal: nil))
        // The kernel refusal wins (it names the fix).
        XCTAssertEqual(SandboxDesktop.installRefusal(networkEnabled: true, kernelRefusal: "old kernel"), "old kernel")
    }

    // MARK: settings

    func testDesktopSettingDefaultsOffAndDecodesTolerantly() throws {
        XCTAssertFalse(ServerOptions.SandboxConfig().desktop)
        let legacy = try JSONDecoder().decode(ServerOptions.SandboxConfig.self,
                                              from: Data(#"{"enabled":true,"network":false}"#.utf8))
        XCTAssertFalse(legacy.desktop)
        XCTAssertTrue(legacy.enabled)
        let on = try JSONDecoder().decode(ServerOptions.SandboxConfig.self,
                                          from: Data(#"{"enabled":true,"desktop":true}"#.utf8))
        XCTAssertTrue(on.desktop)
    }

    /// Enabling the desktop implies the sandbox (the desktop IS the guest),
    /// and networking, since the first enable installs it with apt.
    func testEnablingTheDesktopForcesTheSandboxAndNetworkOn() {
        var c = ServerOptions.SandboxConfig()
        c = SandboxDesktop.withDesktop(true, c)
        XCTAssertTrue(c.enabled)
        XCTAssertTrue(c.network)
        XCTAssertTrue(c.desktop)
        var off = SandboxDesktop.withDesktop(false, c)
        XCTAssertFalse(off.desktop)
        XCTAssertTrue(off.enabled, "turning the desktop off leaves the sandbox as it was")
        off.enabled = false
        XCTAssertFalse(SandboxDesktop.withDesktop(false, off).enabled)
    }
}
