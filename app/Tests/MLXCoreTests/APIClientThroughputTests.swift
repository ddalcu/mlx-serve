import XCTest
@testable import MLXCore

/// The usage poll starts when the server process is spawned, before the server listens.
final class APIClientThroughputTests: XCTestCase {
    /// A loopback port nothing listens on: bound to learn a free number, then closed.
    private static func closedPort() -> UInt16 {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        defer { close(fd) }
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        addr.sin_port = 0
        _ = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &length) }
        }
        return UInt16(bigEndian: addr.sin_port)
    }

    func testAServerThatIsNotListeningYetAnswersAtOnceInsteadOfWaiting() async {
        let port = Self.closedPort()
        let answered = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                _ = try? await APIClient().fetchThroughput(port: port)
                return true
            }
            group.addTask {
                try? await Task.sleep(for: .seconds(3))
                return false
            }
            let first = await group.next() ?? false
            group.cancelAll()
            return first
        }
        XCTAssertTrue(answered, "a request to a closed port has to fail; waiting for the server to appear never ends")
    }
}
