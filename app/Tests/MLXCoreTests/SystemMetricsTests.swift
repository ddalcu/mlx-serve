import Darwin
import XCTest
@testable import MLXCore

/// `SystemMetrics` replaces three spawned binaries — `lsof`, `ps`, `vm_stat` —
/// none of which is reachable inside the App Sandbox container, and each of
/// which is a host escape from the Agent Sandbox.
///
/// The tests are differential: they run the tool being removed and require the
/// in-process implementation to agree. If they ever disagree, the replacement
/// is wrong, not the tool.
final class SystemMetricsTests: XCTestCase {

    private func shell(_ path: String, _ args: [String]) throws -> String {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: path)
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = FileHandle.nullDevice
        try p.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        return String(decoding: data, as: UTF8.self)
    }

    // MARK: - vm_stat

    func testAvailableBytesMatchesVmStat() throws {
        let text = try shell("/usr/bin/vm_stat", [])

        var pageSize = 4096
        if let m = text.range(of: #"page size of (\d+)"#, options: .regularExpression) {
            pageSize = Int(String(text[m]).filter(\.isNumber)) ?? 4096
        }
        func pages(_ key: String) -> Int {
            guard let line = text.split(separator: "\n").first(where: { $0.contains(key) }) else { return 0 }
            return Int(line.filter(\.isNumber)) ?? 0
        }
        let expected = UInt64((pages("Pages free") + pages("Pages inactive")) * pageSize)
        let actual = SystemMetrics.availableBytes()

        XCTAssertGreaterThan(actual, 0, "availableBytes returned 0")
        // Memory moves between the two samples; allow 5% or 256 MB, whichever is larger.
        let tolerance = max(UInt64(Double(expected) * 0.05), 256 * 1024 * 1024)
        let delta = actual > expected ? actual - expected : expected - actual
        XCTAssertLessThan(delta, tolerance,
                          "vm_stat says \(expected / (1 << 20)) MB, host_statistics64 says \(actual / (1 << 20)) MB")
    }

    /// `vm_stat`'s `Pages free` is `free_count - speculative_count`. Forgetting
    /// the subtraction over-reports free memory by however many pages the
    /// kernel is speculatively holding — often gigabytes.
    func testAvailableBytesSubtractsSpeculativePages() throws {
        let text = try shell("/usr/bin/vm_stat", [])
        func pages(_ key: String) -> Int {
            guard let line = text.split(separator: "\n").first(where: { $0.contains(key) }) else { return 0 }
            return Int(line.filter(\.isNumber)) ?? 0
        }
        // This machine must actually be holding speculative pages, else the
        // assertion below proves nothing.
        try XCTSkipIf(pages("Pages speculative") < 1000, "no speculative pages to distinguish the two formulas")

        let pageSize = Int(vm_kernel_page_size)
        let naive = UInt64((pages("Pages free") + pages("Pages speculative") + pages("Pages inactive")) * pageSize)
        let actual = SystemMetrics.availableBytes()
        let delta = actual > naive ? actual - naive : naive - actual
        XCTAssertGreaterThan(delta, UInt64(500 * pages("Pages speculative") * pageSize) / 1000,
                             "availableBytes looks like it included speculative pages")
    }

    // MARK: - ps

    /// The only contract the app relies on is the *name* (`killOrphanedServers`
    /// does `processName(pid:).hasPrefix("mlx-serve")`).
    ///
    /// The full paths can legitimately differ: `ps -o comm=` reports the path as
    /// the process was launched, while `proc_pidpath` resolves symlinks to the
    /// real executable. Observed under xctest, which is launched via
    /// `Developer/usr/bin/xctest` but really lives in `Xcode/Agents/xctest`.
    func testProcessNameMatchesPsForOurselves() throws {
        let me = ProcessInfo.processInfo.processIdentifier
        let ps = try shell("/bin/ps", ["-p", "\(me)", "-o", "comm="])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertFalse(ps.isEmpty, "ps produced nothing")

        XCTAssertEqual(SystemMetrics.processName(pid: me), (ps as NSString).lastPathComponent)

        let path = try XCTUnwrap(SystemMetrics.processPath(pid: me))
        XCTAssertTrue(path.hasPrefix("/"), path)
        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: path), path)
        XCTAssertEqual((path as NSString).lastPathComponent, (ps as NSString).lastPathComponent)
    }

    func testProcessNameOfDeadPidIsEmpty() {
        // PID 0 is the kernel; proc_pidpath cannot read it.
        XCTAssertEqual(SystemMetrics.processName(pid: 0), "")
        XCTAssertNil(SystemMetrics.processPath(pid: 0))
    }

    // MARK: - ps -axo pid=,ppid= (process tree)

    func testParentPidMatchesGetppid() {
        let me = ProcessInfo.processInfo.processIdentifier
        XCTAssertEqual(SystemMetrics.parentPid(of: me), getppid())
    }

    /// The parent map drives `ProcessRegistry.descendantPids`, which kills a
    /// server's whole subtree. A missing edge leaks a process.
    func testProcessParentMapAgreesWithPs() throws {
        let map = SystemMetrics.processParentMap()
        XCTAssertGreaterThan(map.count, 20, "walked only \(map.count) processes")

        var fromPs: [pid_t: pid_t] = [:]
        for line in try shell("/bin/ps", ["-axo", "pid=,ppid="]).split(whereSeparator: \.isNewline) {
            let cols = line.split(whereSeparator: { $0 == " " || $0 == "\t" }).compactMap { pid_t($0) }
            if cols.count == 2 { fromPs[cols[0]] = cols[1] }
        }
        XCTAssertGreaterThan(fromPs.count, 20, "ps produced only \(fromPs.count) rows")

        // Processes come and go between the two snapshots, so compare the
        // intersection — every pid both saw must have the same parent.
        var disagreements: [String] = []
        for (pid, ppid) in map {
            if let theirs = fromPs[pid], theirs != ppid {
                disagreements.append("pid \(pid): libproc says \(ppid), ps says \(theirs)")
            }
        }
        XCTAssertTrue(disagreements.isEmpty, disagreements.prefix(10).joined(separator: "\n"))

        // And the overlap must be substantial, or the comparison proved nothing.
        let overlap = map.keys.filter { fromPs[$0] != nil }.count
        XCTAssertGreaterThan(overlap, min(map.count, fromPs.count) / 2,
                             "only \(overlap) pids in common")
    }

    func testDescendantPidsFindsARealChild() throws {
        let child = Process()
        child.executableURL = URL(fileURLWithPath: "/bin/sh")
        child.arguments = ["-c", "sleep 5"]
        try child.run()
        defer { child.terminate() }

        let me = ProcessInfo.processInfo.processIdentifier
        let descendants = ProcessRegistry.descendantPids(of: me)
        XCTAssertTrue(descendants.contains(child.processIdentifier),
                      "child \(child.processIdentifier) missing from \(descendants)")
    }

    // MARK: - lsof

    /// Bind a listener, then require both lsof and libproc to name this process.
    func testPidsListeningMatchesLsof() throws {
        let (fd, port) = try boundListener()
        defer { close(fd) }

        let mine = SystemMetrics.pidsListening(onTCPPort: port)
        XCTAssertEqual(mine, [ProcessInfo.processInfo.processIdentifier])

        let lsof = try shell("/usr/sbin/lsof", ["-nP", "-iTCP:\(port)", "-sTCP:LISTEN", "-t"])
            .split(whereSeparator: { $0.isNewline || $0.isWhitespace })
            .compactMap { pid_t($0) }
        XCTAssertEqual(mine.sorted(), lsof.sorted(), "libproc and lsof disagree on port \(port)")
    }

    func testPidsListeningIsEmptyOnceTheListenerCloses() throws {
        let (fd, port) = try boundListener()
        XCTAssertFalse(SystemMetrics.pidsListening(onTCPPort: port).isEmpty)
        close(fd)
        XCTAssertTrue(SystemMetrics.pidsListening(onTCPPort: port).isEmpty,
                      "closed listener still reported on port \(port)")
    }

    func testIsTCPPortInUseTracksTheListener() throws {
        let (fd, port) = try boundListener(loopbackOnly: true)
        XCTAssertTrue(SystemMetrics.isTCPPortInUse(port))
        close(fd)
        XCTAssertFalse(SystemMetrics.isTCPPortInUse(port))
    }

    // MARK: -

    /// Bind an ephemeral port and listen; returns the fd and the chosen port.
    private func boundListener(loopbackOnly: Bool = false) throws -> (Int32, UInt16) {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw XCTSkip("socket() failed") }

        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = 0 // kernel picks a free port
        addr.sin_addr.s_addr = loopbackOnly ? inet_addr("127.0.0.1") : INADDR_ANY

        let ok = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                // `bind` here would resolve to UnsafePointer.bind(to:capacity:).
                Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard ok == 0, listen(fd, 1) == 0 else {
            close(fd)
            throw XCTSkip("could not bind an ephemeral listener")
        }

        var bound = sockaddr_in()
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &bound) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &len) }
        }
        return (fd, UInt16(bigEndian: bound.sin_port))
    }
}
