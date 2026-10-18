import XCTest
@testable import MLXCore

final class ClusterSetupTests: XCTestCase {

    private let devinfo = """
    hca_id:\trdma_en2
    \t\tport:\t1
    \t\t\tstate:\t\t\tPORT_DOWN (1)
    hca_id:\trdma_en13
    \t\tport:\t1
    \t\t\tstate:\t\t\tPORT_ACTIVE (4)
    """

    /// Each `hca_id` block becomes a port with its state and interface.
    func testPortsParseFromIbvDevinfo() {
        let ports = ClusterSetup.parsePorts(devinfo)
        XCTAssertEqual(ports, [RdmaPort(device: "rdma_en2", active: false), RdmaPort(device: "rdma_en13", active: true)])
        XCTAssertEqual(ports[1].interface, "en13")
    }

    /// A host with a model launches with the active port and the link address; anything less launches plain.
    func testHostArgsNeedAModelAndAPort() {
        var o = ServerOptions()
        let ports = { ClusterSetup.parsePorts(self.devinfo) }
        XCTAssertEqual(ClusterSetup.hostArgs(o, ports: ports), [])
        o.clusterRole = .host
        XCTAssertEqual(ClusterSetup.hostArgs(o, ports: ports), [], "no model chosen")
        o.clusterModelPath = "/m/Flash"
        o.clusterMode = .tensor
        XCTAssertEqual(ClusterSetup.hostArgs(o, ports: ports, address: { $0 == "en13" ? "10.254.77.1" : nil }),
                       ["--dist", "tensor", "--dist-coordinator", "10.254.77.1:32323",
                        "--dist-device", "rdma_en13", "--dist-model", "/m/Flash"])
        XCTAssertEqual(ClusterSetup.hostArgs(o, ports: ports, address: { _ in nil })[3], "10.254.77.1:32323",
                       "a port with no address yet still names the cable's host end")
        XCTAssertEqual(ClusterSetup.hostArgs(o, ports: { [] }), [], "no port up")
    }

    /// A worker on x.y.z.2 dials the host at x.y.z.1 on the same cable.
    func testWorkerDialsTheHostEndOfItsCable() {
        var o = ServerOptions()
        o.clusterRole = .worker
        let args = ClusterSetup.workerArgs(o, modelDirs: [], ports: { ClusterSetup.parsePorts(self.devinfo) },
                                           address: { _ in "10.254.77.2" })
        XCTAssertEqual(args?[2], "10.254.77.1:32323")
    }

    /// A self-assigned 169.254 address is the Thunderbolt Bridge's, never the pair's: the
    /// worker dials the pair's host address until the cable is set up.
    func testWorkerIgnoresASelfAssignedAddress() {
        var o = ServerOptions()
        o.clusterRole = .worker
        let args = ClusterSetup.workerArgs(o, modelDirs: [], ports: { ClusterSetup.parsePorts(self.devinfo) },
                                           address: { _ in "169.254.0.2" })
        XCTAssertEqual(args?[2], "10.254.77.1:32323")
    }

    /// The worker dials the typed host address, adding the port when it is missing.
    func testWorkerArgsDialTheHost() {
        var o = ServerOptions()
        o.clusterRole = .worker
        o.clusterCoordinator = "192.168.2.238"
        XCTAssertEqual(ClusterSetup.workerArgs(o, modelDirs: ["/a", "/b"], ports: { ClusterSetup.parsePorts(self.devinfo) }),
                       ["worker", "--coordinator", "192.168.2.238:32323", "--device", "rdma_en13",
                        "--model-dir", "/a", "--model-dir", "/b"])
        XCTAssertNil(ClusterSetup.workerArgs(o, modelDirs: [], ports: { [] }))
    }

    /// Joining or leaving as the worker changes no server flag, so it asks for no restart;
    /// becoming or changing the host does.
    func testOnlyTheHostsClusterFieldsNeedARestart() {
        let off = ServerOptions()
        var worker = off
        worker.clusterRole = .worker
        worker.clusterCoordinator = "10.254.77.1"
        XCTAssertTrue(off.serverLaunchEquals(worker))
        var host = off
        host.clusterRole = .host
        XCTAssertFalse(off.serverLaunchEquals(host))
        var other = host
        other.clusterModelPath = "/m/Other"
        XCTAssertFalse(host.serverLaunchEquals(other))
    }

    /// The tray names the model the worker holds its part of, and which part.
    func testTheWorkersPartNamesItsModel() {
        let pipeline = ClusterWorker.part(of: .ready("Qwen3.8-Flash-Next-FP4-Local layers 30..60 of 60 (pipeline)"))
        XCTAssertEqual(pipeline?.model, "Qwen3.8-Flash-Next-FP4-Local")
        XCTAssertEqual(pipeline?.detail, "layers 30..60 of 60")
        XCTAssertEqual(pipeline?.ready, true)
        let tensor = ClusterWorker.part(of: .loading("Qwen3.8-Flash-Next-FP4-Local, half of every expert (tensor)"))
        XCTAssertEqual(tensor?.model, "Qwen3.8-Flash-Next-FP4-Local")
        XCTAssertEqual(tensor?.detail, "half of every expert")
        XCTAssertEqual(tensor?.ready, false)
        XCTAssertNil(ClusterWorker.part(of: .waiting))
        XCTAssertNil(ClusterWorker.part(of: .ready("")))
    }

    /// The tray counts the worker's part from its ready line.
    func testTheWorkersResidentMemoryComesFromItsReadyLine() {
        XCTAssertEqual(ClusterWorker.residentMiB(in: "[dist] worker ready (70179 MiB resident)"), 70179)
        XCTAssertEqual(ClusterWorker.residentMiB(in: "[dist] worker ready (812 MiB resident, 9000 us per 64-row pass)"), 812)
        XCTAssertNil(ClusterWorker.residentMiB(in: "[dist] worker ready: layers 2..14 resident, 70179 MiB"))
    }

    /// The default launch carries no cluster flag.
    func testDefaultLaunchHasNoClusterFlags() {
        XCTAssertFalse(ServerOptions().toCLIArgs(modelDirs: []).contains("--dist"))
    }

    /// The worker's log lines move its state: waiting, loading, ready, failed.
    func testWorkerStateFollowsItsLog() {
        var s = ClusterWorker.State.stopped
        s = ClusterWorker.state(after: "[dist] worker: the host at 169.254.0.1:32323 is not listening yet; retrying", current: s)
        XCTAssertEqual(s, .waiting)
        s = ClusterWorker.state(after: "[dist] worker: Flash-FP4 layers 16..48 of 48 (pipeline)", current: s)
        XCTAssertEqual(s, .loading("Flash-FP4 layers 16..48 of 48 (pipeline)"))
        s = ClusterWorker.state(after: "[dist] worker ready (49152 MiB resident)", current: s)
        XCTAssertEqual(s, .ready("Flash-FP4 layers 16..48 of 48 (pipeline)"))
        s = ClusterWorker.state(after: "[dist] worker: no model named Flash-FP4 in /Users/x/.mlx-serve/models", current: s)
        XCTAssertEqual(s, .failed("worker: no model named Flash-FP4 in /Users/x/.mlx-serve/models"))
    }

    /// `/props` "cluster" before and after the worker joins.
    func testClusterInfoParses() {
        XCTAssertNil(ClusterInfo.parse([:]))
        let waiting = ClusterInfo.parse(["cluster": ["mode": "pipeline", "model": "Flash", "joined": false, "planned": false]])
        XCTAssertEqual(waiting?.joined, false)
        let live = ClusterInfo.parse(["cluster": ["mode": "pipeline", "model": "Flash", "joined": true, "planned": true,
                                                  "worker_ready": true, "layers": 48, "here": [0, 16]]])
        XCTAssertEqual(live?.here, 16)
        XCTAssertEqual(live?.layers, 48)
        XCTAssertEqual(live?.workerReady, true)
    }

    /// `/props` "cluster" for video steps: the worker's name, speed and the last split.
    func testVideoClusterInfoParses() {
        let c = ClusterInfo.parse(["cluster": ["mode": "rows", "model": "H3-4bit", "joined": true, "planned": true,
                                               "worker_ready": true, "worker_name": "Davids-MBP.local", "worker_speed": 0.42,
                                               "link_gbps": 8.9, "last_job": "split 4000/3000 rows", "split_rows": [4000, 3000]]])
        XCTAssertEqual(c?.workerName, "Davids-MBP.local")
        XCTAssertEqual(c?.lastSplit, true)
        XCTAssertEqual(c?.serves("/Users/x/.mlx-serve/models/ddalcu/H3-4bit"), true)
        XCTAssertEqual(c?.serves("/m/H3-8bit"), false)
    }

    /// The worker's measured speed decides the advice; nothing measured yet says so.
    func testVideoAdviceFollowsTheWorkersSpeed() {
        var c = ClusterInfo(mode: "rows", model: "H3", joined: true, planned: true, workerReady: true, layers: 0, here: 0)
        XCTAssertEqual(ClusterVideoAdvice.of(c), .unmeasured)
        XCTAssertEqual(ClusterVideoAdvice.stepSpeedup(c), 1)
        c.workerSpeed = 1.0
        XCTAssertEqual(ClusterVideoAdvice.of(c), .shares(speed: 1.0))
        XCTAssertEqual(ClusterVideoAdvice.stepSpeedup(c), 1.8, accuracy: 1e-9)
        c.workerSpeed = 0.4
        XCTAssertEqual(ClusterVideoAdvice.of(c), .largeOnly(speed: 0.4))
        c.workerSpeed = 0.2
        XCTAssertEqual(ClusterVideoAdvice.of(c), .tooSlow(speed: 0.2))
        XCTAssertEqual(ClusterVideoAdvice.stepSpeedup(c), 1)
        XCTAssertTrue(ClusterVideoAdvice.tooSlow(speed: 0.2).text(worker: "Davids-MBP.local").hasPrefix("Davids-MBP runs at 0.2×"))
    }

    /// A LAN peer named after the worker's host is the same Mac.
    func testLanPeerMatchesTheWorkersHost() {
        XCTAssertTrue(StoryboardCluster.sameMachine("Davids-MBP", "Davids-MBP.local"))
        XCTAssertTrue(StoryboardCluster.sameMachine("davids mbp", "Davids-MBP"))
        XCTAssertFalse(StoryboardCluster.sameMachine("5060-linux", "Davids-MBP.local"))
        XCTAssertFalse(StoryboardCluster.sameMachine(nil, "Davids-MBP"))
    }
}
