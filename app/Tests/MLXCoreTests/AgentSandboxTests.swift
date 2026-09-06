import XCTest
import Combine
@testable import MLXCore

/// Pure-logic tests for the sandbox manager's host↔guest mapping and provisioning
/// decisions. No guest, no VM — safe in CI.
final class AgentSandboxTests: XCTestCase {

    // MARK: image ref -> cache dir name

    func testImageDirNameSanitizesRefs() {
        XCTAssertEqual(AgentSandbox.imageDirName("nikolaik/python-nodejs"), "nikolaik_python-nodejs")
        XCTAssertEqual(AgentSandbox.imageDirName("alpine:3.20"), "alpine_3.20")
        XCTAssertEqual(AgentSandbox.imageDirName("ghcr.io/acme/img:tag"), "ghcr.io_acme_img_tag")
    }

    func testGuestArchIsArm64() {
        // The HVF guest on Apple Silicon is arm64; pulling amd64 boots to an
        // ENOEXEC kernel panic ("No working init"). Regression guard for that.
        XCTAssertEqual(AgentSandbox.guestArch, "arm64")
        XCTAssertTrue(AgentSandbox.archMarkerName().contains("arm64"),
                      "the cache marker must encode the arch so a wrong-arch cache is invalidated")
    }

    func testImageDirNameNeverEscapesWithDotsOrSlashes() {
        // Must not produce path traversal into a parent dir.
        let name = AgentSandbox.imageDirName("../../etc/passwd")
        XCTAssertFalse(name.contains("/"))
        XCTAssertFalse(name.contains(".."))
    }

    // MARK: guest background-process control (kill / log tail)

    func testGuestKillCommandTargetsPidWithGraceKill() {
        let cmd = AgentSandbox.guestKillCommand(pid: 4242)
        XCTAssertTrue(cmd.contains("kill -TERM 4242"), cmd)
        XCTAssertTrue(cmd.contains("kill -KILL 4242"), "must escalate to SIGKILL after a grace: \(cmd)")
        XCTAssertTrue(cmd.contains("&"), "the grace kill must be backgrounded so the exec returns: \(cmd)")
    }

    func testGuestReadLogCommandTailsQuotedPath() {
        let cmd = AgentSandbox.guestReadLogCommand(logPath: "/tmp/mlx-bg-1.log")
        XCTAssertTrue(cmd.contains("tail -c"), cmd)
        XCTAssertTrue(cmd.contains("'/tmp/mlx-bg-1.log'"), "path must be shell-quoted: \(cmd)")
    }

    /// With no live guest booted, the guest-kill / log-tail helpers are safe
    /// no-ops (they must never boot a guest just to kill/read).
    func testGuestControlHelpersAreNoopWithoutLiveGuest() async {
        AgentSandbox.shared.killGuestProcess(pid: 12345) // must not crash / boot
        let log = await AgentSandbox.shared.tailGuestLog(logPath: "/tmp/nope.log")
        XCTAssertNil(log, "no live guest → nil log (never boots one)")
    }

    // MARK: fallback shared root (no working directory supplied)

    func testFallbackSharedRootIsTheSessionWorkspaceNeverHomeOrCwd() {
        // When no working directory is supplied (e.g. the Sandbox Terminal
        // booting the guest before any agent command), the guest must share the
        // app's default session workspace — NOT the user's home folder (the old
        // libcontain-era fallback) and NOT the process cwd (`/` for a
        // Finder-launched app). Mounting either rw into the guest defeats the
        // point of the sandbox.
        let root = AgentSandbox.fallbackSharedRoot()
        XCTAssertEqual(root, ChatSession.defaultWorkingDirectory)
        XCTAssertTrue(root.hasSuffix(".mlx-serve/workspace"))
        XCTAssertNotEqual(root, FileManager.default.homeDirectoryForCurrentUser.path)
        XCTAssertNotEqual(root, "/")
    }

    // MARK: host cwd -> guest /workspace path

    func testGuestPathAtRoot() {
        let r = AgentSandbox.guestPath(hostPath: "/Users/d/proj", sharedRoot: "/Users/d/proj")
        XCTAssertEqual(r.path, "/workspace")
        XCTAssertTrue(r.mapped)
    }

    func testGuestPathForSubdir() {
        let r = AgentSandbox.guestPath(hostPath: "/Users/d/proj/src/app", sharedRoot: "/Users/d/proj")
        XCTAssertEqual(r.path, "/workspace/src/app")
        XCTAssertTrue(r.mapped)
    }

    func testGuestPathOutsideRootFallsBackToWorkspaceRoot() {
        let r = AgentSandbox.guestPath(hostPath: "/tmp/elsewhere", sharedRoot: "/Users/d/proj")
        XCTAssertEqual(r.path, "/workspace")
        XCTAssertFalse(r.mapped, "a cwd outside the shared root isn't visible in the guest")
    }

    func testGuestPathNilHostFallsBackToWorkspaceRoot() {
        let r = AgentSandbox.guestPath(hostPath: nil, sharedRoot: "/Users/d/proj")
        XCTAssertEqual(r.path, "/workspace")
    }

    func testGuestPathDoesNotPartialSegmentMatch() {
        // "/Users/d/project2" must NOT be treated as under "/Users/d/proj".
        let r = AgentSandbox.guestPath(hostPath: "/Users/d/project2", sharedRoot: "/Users/d/proj")
        XCTAssertFalse(r.mapped)
        XCTAssertEqual(r.path, "/workspace")
    }

    // MARK: workspace remount when the session working folder moves

    func testNeedsRemountOnlyWhenCwdLeavesTheSharedRoot() {
        // Same root / subdir / no cwd → the live guest keeps its share.
        XCTAssertFalse(AgentSandbox.needsRemount(sharedRoot: "/a/proj", requestedCwd: nil),
                       "no cwd supplied → keep the current share")
        XCTAssertFalse(AgentSandbox.needsRemount(sharedRoot: "/a/proj", requestedCwd: "/a/proj"))
        XCTAssertFalse(AgentSandbox.needsRemount(sharedRoot: "/a/proj", requestedCwd: "/a/proj/src/deep"))
        // The user switched the session's working folder → /workspace must
        // follow it (guest reboots with the new share; boot is sub-second).
        XCTAssertTrue(AgentSandbox.needsRemount(sharedRoot: "/a/proj", requestedCwd: "/a/other"))
        XCTAssertTrue(AgentSandbox.needsRemount(sharedRoot: "/a/proj", requestedCwd: "/a/project2"),
                      "sibling sharing a path prefix is NOT under the root")
    }

    // MARK: per-chat project folders → /projects/<slug> (hot-mount on first use)

    func testProjectSlugIsDeterministicAndFilesystemSafe() {
        let a = AgentSandbox.projectSlug(hostPath: "/Users/d/My Project")
        XCTAssertEqual(a, AgentSandbox.projectSlug(hostPath: "/Users/d/My Project"),
                       "same path → same slug (stable across commands)")
        XCTAssertFalse(a.contains("/"), "slug must be a single path segment: \(a)")
        XCTAssertFalse(a.contains(" "), "slug must be filesystem-safe: \(a)")
        XCTAssertTrue(a.hasPrefix("My-Project-"), "keeps a readable basename: \(a)")
    }

    func testProjectSlugDisambiguatesSameBasename() {
        // Two different "/src" folders must not collide on one guest subdir.
        let a = AgentSandbox.projectSlug(hostPath: "/Users/d/alpha/src")
        let b = AgentSandbox.projectSlug(hostPath: "/Users/d/beta/src")
        XCTAssertNotEqual(a, b, "distinct paths sharing a basename need distinct slugs")
    }

    func testResolveDefaultFolderMapsToWorkspace() {
        let r = AgentSandbox.resolveGuestCwd(hostPath: "/Users/d/ws/sub",
                                             defaultRoot: "/Users/d/ws", mounted: [:])
        XCTAssertEqual(r.path, "/workspace/sub")
        XCTAssertNil(r.mountSlug, "the default workspace never needs a project mount")
        XCTAssertTrue(r.mapped)
    }

    func testResolveNewFolderRequestsAHotMount() {
        let r = AgentSandbox.resolveGuestCwd(hostPath: "/Users/d/other/proj",
                                             defaultRoot: "/Users/d/ws", mounted: [:])
        XCTAssertEqual(r.mountHost, "/Users/d/other/proj", "the folder must be scheduled for mounting")
        let slug = AgentSandbox.projectSlug(hostPath: "/Users/d/other/proj")
        XCTAssertEqual(r.mountSlug, slug)
        XCTAssertEqual(r.path, "/projects/\(slug)")
        XCTAssertTrue(r.mapped)
    }

    func testResolveAlreadyMountedFolderNeedsNoRemount() {
        let slug = AgentSandbox.projectSlug(hostPath: "/Users/d/other/proj")
        let mounted = [slug: "/Users/d/other/proj"]
        // Exact folder.
        let r1 = AgentSandbox.resolveGuestCwd(hostPath: "/Users/d/other/proj",
                                              defaultRoot: "/Users/d/ws", mounted: mounted)
        XCTAssertEqual(r1.path, "/projects/\(slug)")
        XCTAssertNil(r1.mountSlug, "an already-mounted folder must NOT trigger another mount")
        // Descendant of a mounted folder.
        let r2 = AgentSandbox.resolveGuestCwd(hostPath: "/Users/d/other/proj/src",
                                              defaultRoot: "/Users/d/ws", mounted: mounted)
        XCTAssertEqual(r2.path, "/projects/\(slug)/src")
        XCTAssertNil(r2.mountSlug)
    }

    // MARK: command wrapping (cd into mapped dir, then run)

    func testWrapCommandCdsIntoGuestPath() {
        let wrapped = AgentSandbox.wrap(command: "ls -la", guestCwd: "/workspace/src")
        XCTAssertTrue(wrapped.hasPrefix("cd '/workspace/src'"), "must cd into the mapped dir first")
        XCTAssertTrue(wrapped.contains("ls -la"))
    }

    func testWrapEscapesApostropheInGuestCwd() {
        // An apostrophe in the working-directory name must not unbalance the
        // single-quoting — an unescaped `'` desyncs the ShellSentinel framing
        // and every subsequent command times out.
        let wrapped = AgentSandbox.wrap(command: "ls", guestCwd: "/workspace/it's here")
        XCTAssertTrue(wrapped.contains("cd '/workspace/it'\\''s here'"),
                      "guest cwd must be POSIX single-quote escaped: \(wrapped)")
        // Balanced: quote count net-zero — dash must parse it as one word.
        // '/workspace/it'  \'  's here'  → 6 quotes + 1 escaped = odd raw count
        // is fine; the load-bearing check is the '\'' escape sequence above and
        // that no bare `'s ` (unescaped apostrophe run) survives.
        XCTAssertFalse(wrapped.contains("cd '/workspace/it's here'"),
                       "raw unescaped interpolation must be gone: \(wrapped)")
    }

    // MARK: teardown must not block the caller (main-thread call sites)

    func testTeardownDetachesGuestSynchronouslyAndShutsDownOffThread() {
        // configure()/teardown() run on the main thread (Settings didSet fires
        // per keystroke; Sandbox Terminal Stop button). VzGuest.shutdown blocks
        // up to 10s on a semaphore — teardown must detach the guest reference
        // synchronously and push the blocking stop to a background queue.
        let sandbox = AgentSandbox.shared
        sandbox._testInstallGuest(VzGuest())
        XCTAssertTrue(sandbox._testHasGuest)

        let shutdownRan = expectation(description: "blocking shutdown ran off-thread")
        let start = Date()
        sandbox.teardown(shutdownBlocking: { _ in
            Thread.sleep(forTimeInterval: 2) // simulate the VZ stop wait
            shutdownRan.fulfill()
        })
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertLessThan(elapsed, 1.0, "teardown blocked the caller for \(elapsed)s")
        XCTAssertFalse(sandbox._testHasGuest, "guest must be detached synchronously so a fresh boot is safe")
        wait(for: [shutdownRan], timeout: 5)
    }

    // MARK: re-pull must not delete the rootfs under a still-stopping VM

    func testRepullStopsTheGuestBeforeDeletingTheImageCache() {
        // repullBaseImage tears down a guest whose virtiofs root IS the cached
        // image dir. The delete must be sequenced strictly AFTER the blocking
        // stop completes — removing the rootfs under a still-stopping VM risks
        // a partial delete that provisionRootfs later mistakes for a valid
        // cache (marker + bin/sh + sidecar surviving = cache hit on a gutted
        // tree). Same hazard resetAllData documents; re-pull is triggered
        // exactly when a guest is likely live (the stale-image alert).
        let sandbox = AgentSandbox.shared
        sandbox._testInstallGuest(VzGuest())
        let lock = NSLock()
        var events: [String] = []
        let stopped = expectation(description: "blocking stop completed")
        let returned = expectation(description: "repull returned")
        DispatchQueue.global().async {
            sandbox.repullBaseImage(
                shutdownBlocking: { _ in
                    Thread.sleep(forTimeInterval: 0.3) // simulate the VZ stop wait
                    lock.lock(); events.append("stopped"); lock.unlock()
                    stopped.fulfill()
                },
                deleteImageDir: { _ in
                    lock.lock(); events.append("deleted"); lock.unlock()
                })
            returned.fulfill()
        }
        wait(for: [returned, stopped], timeout: 5)
        XCTAssertEqual(events, ["stopped", "deleted"],
                       "the image-dir delete must wait for the guest's blocking stop")
        XCTAssertFalse(sandbox._testHasGuest, "guest must be detached so the next boot re-pulls fresh")
    }

    // MARK: reset must detach the rootfs volume before deleting it

    func testResetStopsTheGuestThenDetachesTheVolumeThenDeletes() {
        // The images directory is a mounted sparse bundle: deleting the bundle
        // while mounted leaves a dangling mount the next boot mistakes for an
        // attached (empty) volume. Stop → detach → delete, strictly.
        let sandbox = AgentSandbox.shared
        sandbox._testInstallGuest(VzGuest())
        let lock = NSLock()
        var events: [String] = []
        let done = expectation(description: "reset completed")
        sandbox.resetAllData(
            shutdownBlocking: { _ in
                Thread.sleep(forTimeInterval: 0.2)
                lock.lock(); events.append("stopped"); lock.unlock()
            },
            detachVolume: { _ in lock.lock(); events.append("detached"); lock.unlock() },
            deleteData: { _ in lock.lock(); events.append("deleted"); lock.unlock() },
            completion: { done.fulfill() })
        wait(for: [done], timeout: 5)
        XCTAssertEqual(events, ["stopped", "detached", "deleted"])
        XCTAssertFalse(sandbox._testHasGuest)
    }

    // MARK: CLI-session pinning (issue #89 follow-up: agent CLIs in the guest)

    func testRemountBlockMessageOnlyWhenPinnedAndRemountNeeded() {
        // A live CLI session pins the shared guest: a workspace switch that
        // would remount (= reboot) the VM is DECLINED with a clear message,
        // never a silent kill mid-session.
        XCTAssertNil(AgentSandbox.remountBlockMessage(pinnedLabels: [], needsRemount: true),
                     "no session → remount proceeds as before")
        XCTAssertNil(AgentSandbox.remountBlockMessage(pinnedLabels: ["pi"], needsRemount: false),
                     "pinned but no remount needed → nothing to block")
        let msg = AgentSandbox.remountBlockMessage(pinnedLabels: ["pi"], needsRemount: true)
        XCTAssertNotNil(msg)
        XCTAssertTrue(msg!.contains("pi"), "the message must NAME the session holding the pin: \(msg!)")
        XCTAssertTrue(msg!.lowercased().contains("close"), "must tell the user the way out: \(msg!)")
    }

    func testCliSessionPinRefcounts() {
        let sandbox = AgentSandbox.shared
        XCTAssertNil(sandbox.pinnedCliSessionLabel)
        sandbox.pinCliSession(label: "pi")
        sandbox.pinCliSession(label: "hermes")
        XCTAssertNotNil(sandbox.pinnedCliSessionLabel)
        sandbox.unpinCliSession(label: "pi")
        XCTAssertEqual(sandbox.pinnedCliSessionLabel, "hermes",
                       "two sessions → dropping one keeps the other's pin")
        sandbox.unpinCliSession(label: "hermes")
        XCTAssertNil(sandbox.pinnedCliSessionLabel, "last session gone → guest unpinned")
        // Unbalanced unpin must not underflow.
        sandbox.unpinCliSession(label: "hermes")
        XCTAssertNil(sandbox.pinnedCliSessionLabel)
    }

    // MARK: workspace change → VM remount (Settings default / chat folder pick)

    func testWorkspaceChangeActionRules() {
        // A workspace pick (Settings default or the chat toolbar) must leave
        // the VM sharing the RIGHT folder: live guest whose share no longer
        // covers the new folder tears down so the next command reboots with
        // the new share; a pinned guest declines exactly like shell does.
        typealias Action = AgentSandbox.WorkspaceChangeAction
        XCTAssertEqual(AgentSandbox.workspaceChangeAction(
            guestAlive: false, sharedRoot: nil, newWorkspace: "/w", pinnedLabels: []),
            Action.none, "no live guest → the next boot shares the right folder anyway")
        XCTAssertEqual(AgentSandbox.workspaceChangeAction(
            guestAlive: true, sharedRoot: "/w", newWorkspace: "/w/sub", pinnedLabels: []),
            Action.none, "share already covers the new folder → keep the guest")
        XCTAssertEqual(AgentSandbox.workspaceChangeAction(
            guestAlive: true, sharedRoot: "/w", newWorkspace: nil, pinnedLabels: []),
            Action.none, "nil workspace → nothing to remount to")
        XCTAssertEqual(AgentSandbox.workspaceChangeAction(
            guestAlive: true, sharedRoot: "/w", newWorkspace: "/elsewhere", pinnedLabels: []),
            Action.teardown, "stale share, unpinned → tear down for a fresh remount")
        guard case .blocked(let msg) = AgentSandbox.workspaceChangeAction(
            guestAlive: true, sharedRoot: "/w", newWorkspace: "/elsewhere", pinnedLabels: ["pi"])
        else { return XCTFail("pinned guest must decline the remount, not die under the session") }
        XCTAssertEqual(msg, AgentSandbox.remountBlockMessage(pinnedLabels: ["pi"], needsRemount: true),
                       "same decline message as shell — one workspace-pin story everywhere")
    }

    func testWorkspaceChangeActionForceRestartsPinnedSessions() {
        // A Settings workspace pick is EXPLICIT intent to re-anchor the
        // sandbox (live 2026-07-20: `ls /workspace` kept showing the old
        // folder until an app restart, because a live terminal session pinned
        // the guest and the change was quietly declined). With
        // restartPinnedSessions the pick tears the guest down anyway and the
        // terminal UI respawns its sessions in the new share — the pin only
        // keeps declining IMPLICIT switches (a chat command from a
        // different-workspace tab).
        typealias Action = AgentSandbox.WorkspaceChangeAction
        XCTAssertEqual(AgentSandbox.workspaceChangeAction(
            guestAlive: true, sharedRoot: "/w", newWorkspace: "/elsewhere",
            pinnedLabels: ["pi"], restartPinnedSessions: true),
            Action.teardownRestartingSessions,
            "an explicit pick must remount even under a live session — restarting it")
        XCTAssertEqual(AgentSandbox.workspaceChangeAction(
            guestAlive: true, sharedRoot: "/w", newWorkspace: "/elsewhere",
            pinnedLabels: [], restartPinnedSessions: true),
            Action.teardown, "no sessions → plain teardown, nothing to restart")
        XCTAssertEqual(AgentSandbox.workspaceChangeAction(
            guestAlive: true, sharedRoot: "/w", newWorkspace: "/w/sub",
            pinnedLabels: ["pi"], restartPinnedSessions: true),
            Action.none, "share already covers the new folder → sessions keep running")
        guard case .blocked = AgentSandbox.workspaceChangeAction(
            guestAlive: true, sharedRoot: "/w", newWorkspace: "/elsewhere",
            pinnedLabels: ["pi"], restartPinnedSessions: false)
        else { return XCTFail("implicit switches must still decline under a pin") }
    }

    func testNoteWorkspaceChangedIsSafeWithNoGuest() {
        // The Settings row calls this on every pick; with no live guest it must
        // be a quiet no-op (nil = proceed, nothing torn down, nothing thrown).
        let sandbox = AgentSandbox.shared
        sandbox._testInstallGuest(nil)
        XCTAssertNil(sandbox.noteWorkspaceChanged("/anywhere"))
        XCTAssertFalse(sandbox._testHasGuest)
    }

    // MARK: stale image (dropbear preflight)

    func testStaleImageMessageNamesTheImageAndTheFix() {
        let msg = AgentSandbox.staleImageMessage(image: "ddalcu/agent-shell-mlxserve")
        XCTAssertTrue(msg.contains("ddalcu/agent-shell-mlxserve"), msg)
        XCTAssertTrue(msg.lowercased().contains("ssh"), "must say WHAT the cached image lacks: \(msg)")
        XCTAssertTrue(msg.lowercased().contains("re-pull") || msg.lowercased().contains("update"),
                      "must name the fix: \(msg)")
    }

    // MARK: factory reset (Settings → Reset Sandbox)

    func testResetScopeIsTheSandboxDirAndCoversTheSshIdentity() {
        // The reset deletes EXACTLY this directory: wide enough that the ssh
        // identity + known hosts go with the images (a keypair that outlives
        // the guest's authorized_keys would just break the next boot), narrow
        // enough that it can never touch models/logs/workspace next door.
        let root = AgentSandbox.shared.dataDirectory
        XCTAssertEqual(root.lastPathComponent, "sandbox",
                       "reset scope must be ~/.mlx-serve/sandbox — NEVER ~/.mlx-serve itself")
        XCTAssertTrue(root.path.hasSuffix(".mlx-serve/sandbox"), root.path)
        XCTAssertTrue(SandboxSSH.sshDir.path.hasPrefix(root.path + "/"),
                      "the ssh identity must live INSIDE the reset scope: \(SandboxSSH.sshDir.path)")
    }

    func testTranscriptResetEmptiesTheStore() {
        let store = SandboxTranscript()
        store.append(AgentSandbox.Entry(source: .user, command: "ls", output: "", exitCode: 0, at: Date()))
        store.append(AgentSandbox.Entry(source: .system, command: "", output: "x", exitCode: 0, at: Date()))
        store.reset()
        XCTAssertTrue(store.entries.isEmpty)
    }

    // MARK: transcript store (tray redraw churn)

    func testTranscriptStoreCapsAt400Entries() {
        let store = SandboxTranscript()
        for i in 0..<405 {
            store.append(AgentSandbox.Entry(source: .agent, command: "cmd\(i)", output: "",
                                            exitCode: 0, at: Date()))
        }
        XCTAssertEqual(store.entries.count, 400)
        XCTAssertEqual(store.entries.first?.command, "cmd5", "cap must drop the OLDEST entries")
        XCTAssertEqual(store.entries.last?.command, "cmd404")
    }

    func testTranscriptAppendsDoNotPublishThroughAgentSandbox() {
        // The tray menu observes AgentSandbox.shared; per-command transcript
        // appends must publish ONLY through the nested transcript store, or
        // every append re-renders the whole tray (MenuBarExtra churn class).
        let sandbox = AgentSandbox.shared
        var sandboxFired = 0
        var storeFired = 0
        var subs = Set<AnyCancellable>()
        sandbox.objectWillChange.sink { _ in sandboxFired += 1 }.store(in: &subs)
        sandbox.transcriptStore.objectWillChange.sink { _ in storeFired += 1 }.store(in: &subs)

        sandbox.transcriptStore.append(AgentSandbox.Entry(source: .user, command: "echo hi",
                                                          output: "hi", exitCode: 0, at: Date()))
        XCTAssertEqual(storeFired, 1, "the terminal's store must publish the append")
        XCTAssertEqual(sandboxFired, 0, "AgentSandbox itself must NOT publish transcript appends")
    }
}
