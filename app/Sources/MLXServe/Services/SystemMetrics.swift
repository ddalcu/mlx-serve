import Darwin
import Foundation
import IOKit

/// Host telemetry read straight from the kernel — no subprocesses.
///
/// Two groups:
///  * GPU utilization + memory pressure (IOKit/Mach, the same APIs as
///    `status.zig`), moved here out of `ChatView`.
///  * In-process replacements for the three binaries the app used to spawn:
///    `/usr/sbin/lsof`, `/bin/ps` and `/usr/bin/vm_stat`. None is reachable
///    from inside the App Sandbox container, and each spawn is a host escape
///    from the Agent Sandbox. These call the same kernel interfaces the tools
///    themselves use — `libproc` (what lsof reads) and `host_statistics64`
///    (what vm_stat prints) — so the results match.
enum SystemMetrics {

    // MARK: - GPU / memory pressure

    /// GPU utilization percentage (0–100) via IOKit AGXAccelerator.
    static func gpuUtilization() -> UInt32 {
        var iter: io_iterator_t = 0
        guard let matching = IOServiceMatching("AGXAccelerator") else { return 0 }
        guard IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iter) == KERN_SUCCESS else { return 0 }
        defer { IOObjectRelease(iter) }

        let entry = IOIteratorNext(iter)
        guard entry != 0 else { return 0 }
        defer { IOObjectRelease(entry) }

        var propsRef: Unmanaged<CFMutableDictionary>?
        guard IORegistryEntryCreateCFProperties(entry, &propsRef, kCFAllocatorDefault, 0) == KERN_SUCCESS,
              let props = propsRef?.takeRetainedValue() as? [String: Any],
              let perf = props["PerformanceStatistics"] as? [String: Any],
              let util = perf["Device Utilization %"] as? Int else { return 0 }
        return UInt32(min(max(util, 0), 100))
    }

    /// System memory pressure as percentage (0–100) via Mach host_statistics64.
    static func memoryPressure() -> UInt32 {
        var totalMem: UInt64 = 0
        var len = MemoryLayout<UInt64>.size
        guard sysctlbyname("hw.memsize", &totalMem, &len, nil, 0) == 0, totalMem > 0 else { return 0 }

        var pageSize: vm_size_t = 0
        guard host_page_size(mach_host_self(), &pageSize) == KERN_SUCCESS, pageSize > 0 else { return 0 }

        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<Int32>.size)
        let result = withUnsafeMutablePointer(to: &stats) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, intPtr, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }

        let used = (UInt64(stats.active_count) + UInt64(stats.wire_count) + UInt64(stats.compressor_page_count)) * UInt64(pageSize)
        return UInt32(used * 100 / totalMem)
    }

    // MARK: - Memory (was: /usr/bin/vm_stat)

    /// Free + inactive bytes: pages reclaimable without paging out.
    ///
    /// `vm_stat` prints `Pages free` as `free_count - speculative_count`, not
    /// `free_count` — speculative pages are listed on their own line. Matching
    /// that keeps the number identical to what the tool reported.
    static func availableBytes() -> UInt64 {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)

        let result = withUnsafeMutablePointer(to: &stats) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }

        let free = UInt64(stats.free_count) &- UInt64(stats.speculative_count)
        let inactive = UInt64(stats.inactive_count)
        return (free &+ inactive) &* UInt64(vm_kernel_page_size)
    }

    // MARK: - Process identity (was: /bin/ps -o comm=)

    /// Absolute executable path, or nil if the process is gone or unreadable
    /// (another user's process under the sandbox).
    static func processPath(pid: pid_t) -> String? {
        var buffer = [CChar](repeating: 0, count: Int(MAXPATHLEN))
        let written = proc_pidpath(pid, &buffer, UInt32(MAXPATHLEN))
        guard written > 0 else { return nil }
        return String(cString: buffer)
    }

    /// Last path component of the executable — what `ps -o comm=` yields on
    /// macOS once the caller takes `lastPathComponent`.
    static func processName(pid: pid_t) -> String {
        guard let path = processPath(pid: pid) else { return "" }
        return (path as NSString).lastPathComponent
    }

    // MARK: - Process tree (was: /bin/ps -axo pid=,ppid=)

    /// Parent pid, or nil for processes we can't introspect (pid 0, other users
    /// under the sandbox).
    static func parentPid(of pid: pid_t) -> pid_t? {
        var info = proc_bsdinfo()
        let size = Int32(MemoryLayout<proc_bsdinfo>.size)
        guard proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, size) == size else { return nil }
        return pid_t(info.pbi_ppid)
    }

    /// `pid → ppid` for every process we can read. One pass, no subprocess.
    static func processParentMap() -> [pid_t: pid_t] {
        var map: [pid_t: pid_t] = [:]
        for pid in allPids() where pid > 0 {
            if let ppid = parentPid(of: pid) { map[pid] = ppid }
        }
        return map
    }

    // MARK: - Listening sockets (was: /usr/sbin/lsof -nP -iTCP:<port> -sTCP:LISTEN -t)

    /// PIDs with a TCP socket in LISTEN state bound to `port`.
    ///
    /// Processes we can't introspect (other users, or restricted by the
    /// sandbox) return no fd list and are skipped — the same practical result
    /// as unprivileged `lsof`.
    static func pidsListening(onTCPPort port: UInt16) -> [pid_t] {
        var found: [pid_t] = []
        for pid in allPids() where pid > 0 {
            if processListensOn(pid: pid, port: port) { found.append(pid) }
        }
        return found
    }

    /// Cheap "is anything serving here" probe — a loopback `connect()`, no
    /// process enumeration. Use when the PID doesn't matter.
    static func isTCPPortInUse(_ port: UInt16) -> Bool {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { close(fd) }

        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")

        let connected = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return connected == 0
    }

    // MARK: - libproc plumbing

    private static func allPids() -> [pid_t] {
        let capacity = proc_listpids(UInt32(PROC_ALL_PIDS), 0, nil, 0)
        guard capacity > 0 else { return [] }
        // Pad: the table can grow between the sizing call and the fetch.
        var pids = [pid_t](repeating: 0, count: Int(capacity) / MemoryLayout<pid_t>.size + 64)
        let written = proc_listpids(UInt32(PROC_ALL_PIDS), 0,
                                    &pids, Int32(pids.count * MemoryLayout<pid_t>.size))
        guard written > 0 else { return [] }
        return Array(pids.prefix(Int(written) / MemoryLayout<pid_t>.size))
    }

    private static func processListensOn(pid: pid_t, port: UInt16) -> Bool {
        let size = proc_pidinfo(pid, PROC_PIDLISTFDS, 0, nil, 0)
        guard size > 0 else { return false } // gone, or not ours to inspect

        var fds = [proc_fdinfo](repeating: proc_fdinfo(),
                                count: Int(size) / MemoryLayout<proc_fdinfo>.size)
        let written = proc_pidinfo(pid, PROC_PIDLISTFDS, 0, &fds, size)
        guard written > 0 else { return false }

        for fd in fds.prefix(Int(written) / MemoryLayout<proc_fdinfo>.size)
        where fd.proc_fdtype == UInt32(PROX_FDTYPE_SOCKET) {
            var info = socket_fdinfo()
            let got = proc_pidfdinfo(pid, fd.proc_fd, PROC_PIDFDSOCKETINFO,
                                     &info, Int32(MemoryLayout<socket_fdinfo>.size))
            guard got == Int32(MemoryLayout<socket_fdinfo>.size),
                  info.psi.soi_kind == SOCKINFO_TCP else { continue }

            let tcp = info.psi.soi_proto.pri_tcp
            guard tcp.tcpsi_state == TSI_S_LISTEN else { continue }
            // `insi_lport` is stored in network byte order.
            let localPort = UInt16(bigEndian: UInt16(truncatingIfNeeded: tcp.tcpsi_ini.insi_lport))
            if localPort == port { return true }
        }
        return false
    }
}
