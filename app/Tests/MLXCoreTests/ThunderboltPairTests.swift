import XCTest
@testable import MLXCore

/// The Cluster page's one-button Thunderbolt state, from the ports, the role and the server.
final class ThunderboltPairTests: XCTestCase {
    private let up = [RdmaPort(device: "rdma_en13", active: true)]
    private let down = [RdmaPort(device: "rdma_en13", active: false)]

    func testWithoutRdmaPortsPairingIsOff() {
        XCTAssertEqual(ThunderboltPair.of(ports: [], role: .off, hasAddress: false, cluster: nil, leader: nil), .off)
    }

    func testAPortWithNoCableAsksForTheCable() {
        XCTAssertEqual(ThunderboltPair.of(ports: down, role: .off, hasAddress: false, cluster: nil, leader: nil), .noCable)
    }

    /// A Mac on the network that leads a pair is offered as the one to join.
    func testACableInOffersToPairOrJoinTheLeader() {
        XCTAssertEqual(ThunderboltPair.of(ports: up, role: .off, hasAddress: false, cluster: nil, leader: nil), .ready(leader: nil))
        XCTAssertEqual(ThunderboltPair.of(ports: up, role: .off, hasAddress: false, cluster: nil, leader: "Studio"), .ready(leader: "Studio"))
    }

    /// The port's address goes with a restart; a paired Mac then needs one click to reconnect.
    func testAPairedMacWithoutItsAddressReconnects() {
        XCTAssertEqual(ThunderboltPair.of(ports: up, role: .host, hasAddress: false, cluster: nil, leader: nil), .reconnect(.host))
        XCTAssertEqual(ThunderboltPair.of(ports: up, role: .worker, hasAddress: false, cluster: nil, leader: nil), .reconnect(.worker))
    }

    func testTheLeaderNamesItsPartnerOnceItJoins() {
        var c = ClusterInfo(mode: "rows", model: "H3", joined: false, planned: false, workerReady: false, layers: 0, here: 0)
        XCTAssertEqual(ThunderboltPair.of(ports: up, role: .host, hasAddress: true, cluster: c, leader: nil), .leading(partner: nil))
        c.joined = true
        c.workerName = "M4Max.local"
        XCTAssertEqual(ThunderboltPair.of(ports: up, role: .host, hasAddress: true, cluster: c, leader: nil), .leading(partner: "M4Max"))
        XCTAssertEqual(ThunderboltPair.of(ports: up, role: .worker, hasAddress: true, cluster: nil, leader: nil), .helping)
    }

    func testOnlyALeaderStillWaitingIsOffered() {
        let waiting = Data(#"{"status":"ok","thunderbolt":{"leads":"MiniMax-H3","joined":false}}"#.utf8)
        let taken = Data(#"{"status":"ok","thunderbolt":{"leads":"MiniMax-H3","joined":true}}"#.utf8)
        XCTAssertEqual(ThunderboltPair.waitingLeader(health: waiting), "MiniMax-H3")
        XCTAssertNil(ThunderboltPair.waitingLeader(health: taken))
        XCTAssertNil(ThunderboltPair.waitingLeader(health: Data(#"{"status":"ok"}"#.utf8)))
    }

    /// A Mac that led and now joins drops the leader's address before taking its own.
    func testTheAddressCommandSwapsTheOtherRolesAddress() {
        let cmd = ClusterSetup.addressCommand(interface: "en13", role: .worker)
        XCTAssertTrue(cmd.hasPrefix("/sbin/ifconfig en13 inet 10.254.77.1 -alias"))
        XCTAssertTrue(cmd.hasSuffix("/sbin/ifconfig en13 inet 10.254.77.2 netmask 255.255.255.0 alias"))
    }
}
