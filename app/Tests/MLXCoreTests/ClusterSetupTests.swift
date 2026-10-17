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
}
