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
        let body = #"{"data":[],"lan_peers":[{"name":"Studio","os":"macos","model":"Mac17,15","leads":"MiniMax-H3","joined":false},{"name":"Taken","leads":"MiniMax-H3","joined":true},{"name":"Plain","leads":""}]}"#
        let peers = APIClient.parseModelsBody(Data(body.utf8)).lanPeers
        XCTAssertEqual(peers.filter(\.value.waitsForPartner).keys.sorted(), ["Studio"])
    }

    /// The address job is a shell script launchd runs as root: it must parse, read the role
    /// file, and know both of the pair's addresses.
    func testTheAddressScriptParsesAndKnowsBothRoles() throws {
        let script = ThunderboltAddressJob.script
        XCTAssertTrue(script.contains(ThunderboltAddressJob.roleFile))
        XCTAssertTrue(script.contains("host) mine=10.254.77.1"))
        XCTAssertTrue(script.contains("worker) mine=10.254.77.2"))
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("tb-address-\(UUID()).sh")
        try script.write(to: file, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: file) }
        let sh = Process()
        sh.executableURL = URL(fileURLWithPath: "/bin/sh")
        sh.arguments = ["-n", file.path]
        try sh.run()
        sh.waitUntilExit()
        XCTAssertEqual(sh.terminationStatus, 0, "the script does not parse")
    }

    /// launchd runs it at boot, whenever the role file changes, and every 30 s (a cable moved).
    func testTheJobRunsAtBootOnARoleChangeAndPeriodically() throws {
        let plist = try XCTUnwrap(PropertyListSerialization.propertyList(from: Data(ThunderboltAddressJob.plist.utf8), format: nil) as? [String: Any])
        XCTAssertEqual(plist["Label"] as? String, ThunderboltAddressJob.label)
        XCTAssertEqual(plist["RunAtLoad"] as? Bool, true)
        XCTAssertEqual(plist["StartInterval"] as? Int, 30)
        XCTAssertEqual(plist["WatchPaths"] as? [String], [ThunderboltAddressJob.roleFile])
        XCTAssertEqual(plist["ProgramArguments"] as? [String], ["/bin/sh", ThunderboltAddressJob.scriptPath])
    }

    /// The install line rides inside an AppleScript string: no double quote or backslash may break it.
    func testTheInstallCommandIsSafeInsideAppleScript() {
        let cmd = ThunderboltAddressJob.installCommand(role: .host)
        XCTAssertFalse(cmd.contains("\""))
        XCTAssertFalse(cmd.contains("\\"))
        XCTAssertTrue(cmd.contains("launchctl bootstrap system"))
        XCTAssertTrue(cmd.contains("printf host"))
    }

    /// One dialog per leader while it waits; a leader that stops waiting and waits again asks again.
    func testEachWaitingLeaderIsAskedAboutOnce() {
        var asked: Set<String> = []
        XCTAssertEqual(PairInvites.next(waiting: ["Studio"], asked: &asked), "Studio")
        asked.insert("Studio")
        XCTAssertNil(PairInvites.next(waiting: ["Studio"], asked: &asked))
        XCTAssertNil(PairInvites.next(waiting: [], asked: &asked))
        XCTAssertEqual(PairInvites.next(waiting: ["Studio"], asked: &asked), "Studio")
    }

    /// Layers follow free memory, experts split in half, a video model sits whole on both.
    func testThePairPlanSharesTheModelTheWayTheServerSplitsIt() {
        let gb: Int64 = 1 << 30
        let layers = PairPlan.share(bytes: 300 * gb, video: false, mode: .pipeline, freeHere: 200 * gb, freeThere: 100 * gb)
        XCTAssertEqual(layers.here, 200 * gb)
        XCTAssertEqual(layers.there, 100 * gb)
        XCTAssertEqual(PairPlan.share(bytes: 80 * gb, video: false, mode: .tensor, freeHere: 200 * gb, freeThere: 100 * gb).there, 40 * gb)
        XCTAssertEqual(PairPlan.share(bytes: 30 * gb, video: true, mode: .pipeline, freeHere: 200 * gb, freeThere: 100 * gb).there, 30 * gb)
        XCTAssertEqual(PairPlan.fit(bytes: 80 * gb, freeHere: 200 * gb, freeThere: nil), .alone)
        XCTAssertEqual(PairPlan.fit(bytes: 250 * gb, freeHere: 200 * gb, freeThere: 100 * gb), .needsBoth)
        XCTAssertEqual(PairPlan.fit(bytes: 290 * gb, freeHere: 200 * gb, freeThere: 50 * gb), .tooBig)
    }

    /// The leader reports whether the worker has the pair's model.
    func testTheLeaderReportsWhetherTheWorkerHasTheModel() {
        let c = ClusterInfo.parse(["cluster": ["mode": "pipeline", "model": "Flash", "joined": true, "worker_model": "missing"]])
        XCTAssertEqual(c?.workerModel, "missing")
        XCTAssertEqual(ClusterInfo.parse(["cluster": ["joined": false, "worker_gone": true]])?.workerGone, true)
    }
}
