import Foundation

/// Opt-in end-to-end diagnostic for the agent sandbox: boots a real Linux guest
/// on Virtualization.framework, runs a handful of commands through the
/// persistent shell, checks output + exit codes, and exits the process. Invoked
/// from `MLXCoreEntryPoint` when `SANDBOX_SMOKE=1` (legacy alias CONTAIN_SMOKE=1).
/// Requires the running binary to be signed with the
/// `com.apple.security.virtualization` entitlement (app/build.sh does this;
/// plain `swift build` binaries can't create VMs).
///
/// Config via env (with dev defaults pointing at the sibling ../contain repo's
/// artifacts, so the default run needs no network):
///   SANDBOX_SMOKE_KERNEL — arm64 kernel Image (virtiofs-capable)
///   SANDBOX_SMOKE_ROOTFS — unpacked OCI rootfs dir (must contain /bin/sh)
///   SANDBOX_SMOKE_SHARE  — optional host dir to share at /workspace
///   SANDBOX_SMOKE_REAL_PROVISION=1 — exercise the REAL path (kernel fetch +
///                          anonymous OCI pull) instead of the local artifacts
enum SandboxSmoke {
    static func run() -> Never {
        let env = ProcessInfo.processInfo.environment
        let repo = "/Users/david/projects/agents/contain"
        let kernel = env["SANDBOX_SMOKE_KERNEL"] ?? env["CONTAIN_SMOKE_KERNEL"]
            ?? "\(repo)/zig-out/bin/artifacts/kernel-contain-arm64"
        let rootfs = env["SANDBOX_SMOKE_ROOTFS"] ?? env["CONTAIN_SMOKE_ROOTFS"]
            ?? "\(repo)/zig-out/bin/oci-rootfs"
        let share = env["SANDBOX_SMOKE_SHARE"] ?? env["CONTAIN_SMOKE_SHARE"]
        // The rootfs is demand-paged over virtiofs — RAM is workload headroom only.
        let ramGB = UInt64(env["SANDBOX_SMOKE_RAM_GB"].flatMap { UInt64($0) } ?? 1)

        func log(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

        log("[smoke] kernel=\(kernel)")
        log("[smoke] rootfs=\(rootfs)")
        if let share { log("[smoke] share=\(share)") }

        let guest = VzGuest()
        var cfg = VzGuest.Config(kernelPath: kernel, rootfsDir: rootfs)
        cfg.workspacePath = share
        cfg.guestWorkspacePath = "/workspace"
        cfg.workdir = share != nil ? "/workspace" : "/"
        cfg.ramBytes = ramGB * 1024 * 1024 * 1024

        do {
            log("[smoke] booting guest…")
            try guest.boot(cfg, readyTimeout: 45)
            log("[smoke] shell ready ✅")

            let checkTimeout = Double(env["SANDBOX_SMOKE_TIMEOUT"].flatMap { Double($0) } ?? 20)
            let dumpOnStall = env["SANDBOX_SMOKE_DUMP"] == "1"
            var ok = true
            func check(_ cmd: String, expectContains: String? = nil, expectExit: Int32 = 0) {
                do {
                    let r = try guest.exec(cmd, timeout: checkTimeout)
                    let out = r.output.trimmingCharacters(in: .whitespacesAndNewlines)
                    log("[smoke] $ \(cmd)\n\(out)\n  → exit=\(r.exitCode) timedOut=\(r.timedOut)")
                    if dumpOnStall && r.timedOut {
                        log("[smoke] ===== CONSOLE DUMP (first stall) =====")
                        log(guest.consoleSnapshot())
                        log("[smoke] ===== END CONSOLE DUMP =====")
                        guest.shutdown(); exit(2)
                    }
                    if let e = expectContains, !r.output.contains(e) {
                        log("[smoke]   ✗ expected output to contain \"\(e)\""); ok = false
                    }
                    if r.exitCode != expectExit {
                        log("[smoke]   ✗ expected exit \(expectExit), got \(r.exitCode)"); ok = false
                    }
                } catch {
                    log("[smoke]   ✗ exec threw: \(error)"); ok = false
                }
            }

            check("echo HELLO_FROM_SWIFT_$((6*7))", expectContains: "HELLO_FROM_SWIFT_42")
            check("uname -sm", expectContains: "Linux")
            check("sh -c 'exit 7'", expectExit: 7)                     // exit code plumbed
            check("X=hi; echo state_$X", expectContains: "state_hi")   // shell state persists
            check("echo persisted_$X", expectContains: "persisted_hi") // …across exec calls
            if share != nil {
                check("ls /workspace", expectExit: 0)                  // virtiofs share works
            }

            guest.shutdown()

            // Phase 2: prove the real integration path — ShellHandler routes to
            // AgentSandbox when enabled, which boots + drives its own guest.
            if ok {
                log("[smoke] phase 2: ShellHandler → AgentSandbox routing…")
                // By default point the provisioner at the same local artifacts
                // (fast, no pull). Set SANDBOX_SMOKE_REAL_PROVISION=1 to exercise
                // the REAL path (kernel fetch + anonymous arm64 OCI pull).
                let realProvision = env["SANDBOX_SMOKE_REAL_PROVISION"] == "1"
                    || env["CONTAIN_SMOKE_REAL_PROVISION"] == "1"
                if !realProvision {
                    setenv("SANDBOX_KERNEL", kernel, 1)
                    setenv("SANDBOX_ROOTFS", rootfs, 1)
                } else {
                    log("[smoke] (real provisioning: fetch kernel + pull \(Self.guestArchNote))")
                }
                let image = env["SANDBOX_SMOKE_IMAGE"] ?? env["CONTAIN_SMOKE_IMAGE"] ?? "ddalcu/agent-shell"
                AgentSandbox.shared.configure(enabled: true, baseImage: image)
                let handler = ShellHandler(timeoutSeconds: 40)
                let routed = syncAwait {
                    (try? await handler.execute(
                        parameters: ["command": "echo ROUTED_$((3*4)); uname -s; pwd"],
                        workingDirectory: share)) ?? "<threw>"
                }
                log("[smoke] routed output:\n\(routed)")
                // Clean output (no timeout / escape garbage) proves the sentinel
                // round-trips: exact tokens on their own, no "[timed out]".
                if !routed.contains("ROUTED_12") || !routed.contains("Linux")
                    || routed.contains("timed out") {
                    log("[smoke]   ✗ routed command did not cleanly run in the sandbox"); ok = false
                } else {
                    log("[smoke]   ✓ routed command clean (sentinel round-trips)")
                }
                // Under real provisioning, verify the freshly-pulled image + config:
                // image ENV reaches the guest, and the agent toolchain is present.
                if realProvision {
                    func probe(_ label: String, _ cmd: String, _ needle: String) {
                        let out = syncAwait {
                            (try? await handler.execute(parameters: ["command": cmd], workingDirectory: share)) ?? "<threw>"
                        }
                        // Reject timed-out output — a needle can otherwise match in
                        // the partial garbage of a stalled exec (false positive).
                        if out.contains(needle) && !out.contains("timed out") { log("[smoke]   ✓ \(label)") }
                        else { log("[smoke]   ✗ \(label) (got: \(out.trimmingCharacters(in: .whitespacesAndNewlines).prefix(80)))"); ok = false }
                    }
                    // Dockerfile ENV must reach the guest (image-config sidecar → init script).
                    probe("image ENV applied (PIP_BREAK_SYSTEM_PACKAGES)", "echo PBSP=${PIP_BREAK_SYSTEM_PACKAGES:-unset}", "PBSP=1")
                    // The agentic toolchain the image ships.
                    probe("node present", "node --version", "v")
                    probe("python3 present", "python3 --version", "Python 3")
                    probe("git present", "git --version", "git version")
                    probe("curl present", "curl --version", "curl")
                    // pip is unblocked (EXTERNALLY-MANAGED removed).
                    probe("pip unblocked (no EXTERNALLY-MANAGED)",
                          "ls /usr/lib/python3.*/EXTERNALLY-MANAGED 2>/dev/null || echo PIP_UNBLOCKED", "PIP_UNBLOCKED")
                }

                // Phase 3: the Sandbox Terminal path — runUserCommand + transcript
                // (exactly what SandboxTerminalView calls).
                log("[smoke] phase 3: Sandbox Terminal (runUserCommand + transcript)…")
                syncAwait { await AgentSandbox.shared.runUserCommand("echo USER_TERMINAL_$((8*8))") }
                // The transcript is published on the main queue; this smoke has no
                // running runloop, so pump it until the entry lands (the real app's
                // runloop delivers these live to SandboxTerminalView).
                var userHit = false, agentHit = false
                let deadline = Date().addingTimeInterval(3)
                while Date() < deadline {
                    RunLoop.main.run(until: Date().addingTimeInterval(0.1))
                    let entries = AgentSandbox.shared.transcript
                    userHit = entries.contains { $0.source == .user && $0.output.contains("USER_TERMINAL_64") }
                    agentHit = entries.contains { $0.source == .agent }
                    if userHit && agentHit { break }
                }
                log("[smoke] transcript: \(AgentSandbox.shared.transcript.count) entries; user-cmd recorded=\(userHit); agent-cmd recorded=\(agentHit)")
                for e in AgentSandbox.shared.transcript.suffix(4) {
                    let out = e.output.prefix(50).replacingOccurrences(of: "\n", with: "⏎")
                    log("[smoke]   entry src=\(e.source.rawValue) exit=\(e.exitCode.map(String.init) ?? "nil") timedOut=\(e.timedOut) out=[\(out)]")
                }
                if !userHit { log("[smoke]   ✗ user command not executed/recorded"); ok = false }
                if !agentHit { log("[smoke]   ✗ agent command not recorded in transcript"); ok = false }

                // Phase 4: networking + live port map (default-on; skip with
                // SANDBOX_SMOKE_NO_NETWORK=1 for isolated-mode runs).
                if env["SANDBOX_SMOKE_NO_NETWORK"] != "1" {
                    log("[smoke] phase 4: guest networking + port map…")
                    // Outbound: kernel-DHCP'd eth0 + resolv.conf → HTTPS works.
                    let curl = syncAwait {
                        (try? await handler.execute(
                            parameters: ["command": "curl -sS -m 15 -o /dev/null -w NET_%{http_code} https://example.com || echo NET_FAIL"],
                            workingDirectory: share)) ?? "<threw>"
                    }
                    if curl.contains("NET_200") { log("[smoke]   ✓ outbound HTTPS from the guest") }
                    else { log("[smoke]   ✗ outbound network failed: \(curl.trimmingCharacters(in: .whitespacesAndNewlines).prefix(120))"); ok = false }

                    // Port map: start a server on a guest port, then fetch it from
                    // the HOST at localhost:<same port> (the forwarder mirrors it
                    // from the guest's /proc/net/tcp within ~1s snapshots).
                    let port = 8123
                    _ = syncAwait {
                        (try? await handler.execute(
                            parameters: ["command": "mkdir -p /tmp/smokeweb && echo PORTMAP_OK > /tmp/smokeweb/index.html && (cd /tmp/smokeweb && (python3 -m http.server \(port) >/tmp/smokeweb.log 2>&1 &)) && echo started"],
                            workingDirectory: share)) ?? "<threw>"
                    }
                    var mapped = false
                    let mapDeadline = Date().addingTimeInterval(15)
                    while Date() < mapDeadline && !mapped {
                        Thread.sleep(forTimeInterval: 0.5)
                        guard let url = URL(string: "http://127.0.0.1:\(port)/") else { break }
                        let body: String? = syncAwait {
                            guard let (data, _) = try? await URLSession.shared.data(from: url) else { return nil }
                            return String(decoding: data, as: UTF8.self)
                        }
                        if body?.contains("PORTMAP_OK") == true { mapped = true }
                    }
                    if mapped { log("[smoke]   ✓ guest :\(port) reachable at localhost:\(port)") }
                    else { log("[smoke]   ✗ port map: localhost:\(port) did not reach the guest server"); ok = false }

                    // `localhost` resolves to ::1 first in modern clients — the
                    // map must answer on BOTH loopback families, or IPv6-first
                    // resolvers see a refused connection (live 2026-07-02).
                    if mapped {
                        let v6Body: String? = syncAwait {
                            guard let url = URL(string: "http://[::1]:\(port)/"),
                                  let (data, _) = try? await URLSession.shared.data(from: url) else { return nil }
                            return String(decoding: data, as: UTF8.self)
                        }
                        if v6Body?.contains("PORTMAP_OK") == true { log("[smoke]   ✓ guest :\(port) reachable at [::1]:\(port)") }
                        else { log("[smoke]   ✗ port map: [::1]:\(port) did not reach the guest server"); ok = false }
                    }
                }

                // Phase 4b: ANSI sanitize — a real tty makes tools emit color +
                // cursor escapes; the recorded transcript must be clean text.
                log("[smoke] phase 4b: terminal control-sequence sanitize…")
                syncAwait { await AgentSandbox.shared.runUserCommand("printf '\\033[31mRED\\033[0m \\033[1mBOLD\\033[0m done\\n'") }
                var sane: Bool?
                let saneDeadline = Date().addingTimeInterval(3)
                while Date() < saneDeadline && sane == nil {
                    RunLoop.main.run(until: Date().addingTimeInterval(0.1))
                    if let e = AgentSandbox.shared.transcript.last(where: { $0.command.contains("printf") }) {
                        sane = e.output.contains("RED BOLD done") && !e.output.contains("\u{1B}") && !e.output.contains("[31m")
                    }
                }
                if sane == true { log("[smoke]   ✓ ANSI escapes stripped (clean transcript text)") }
                else { log("[smoke]   ✗ ANSI sanitize failed (escapes leaked into the transcript)"); ok = false }

                // Tray RAM readout: the guest's /proc/meminfo report must have
                // produced a published display string by now (@Published lands
                // on the main queue — pump it like the transcript check above).
                var memText: String?
                let memDeadline = Date().addingTimeInterval(5)
                while Date() < memDeadline && memText == nil {
                    RunLoop.main.run(until: Date().addingTimeInterval(0.1))
                    memText = AgentSandbox.shared.guestMemoryText
                }
                if let memText { log("[smoke]   ✓ guest RAM readout: \(memText)") }
                else { log("[smoke]   ✗ guest RAM readout never arrived"); ok = false }

                AgentSandbox.shared.teardown()
            }

            if ok { log("[smoke] PASS ✅"); exit(0) }
            log("[smoke] ----- guest console dump -----")
            log(guest.consoleSnapshot())
            log("[smoke] ----- end console dump -----")
            guest.shutdown()
            log("[smoke] FAIL ❌"); exit(1)
        } catch {
            log("[smoke] ERROR: \(error)")
            log("[smoke] ----- guest console dump -----")
            log(guest.consoleSnapshot())
            log("[smoke] ----- end console dump -----")
            guest.shutdown()
            exit(1)
        }
    }

    static var guestArchNote: String {
        let img = ProcessInfo.processInfo.environment["SANDBOX_SMOKE_IMAGE"] ?? "ddalcu/agent-shell"
        return "\(img) (\(AgentSandbox.guestArch))"
    }

    /// Run an async op to completion from this synchronous entry point. The op
    /// resolves on a background thread (AgentSandbox dispatches off-main), so
    /// blocking this thread on the semaphore can't deadlock it.
    private static func syncAwait<T>(_ op: @escaping () async -> T) -> T {
        let sem = DispatchSemaphore(value: 0)
        var result: T!
        Task { result = await op(); sem.signal() }
        sem.wait()
        return result
    }
}
