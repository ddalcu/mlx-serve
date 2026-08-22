import XCTest
@testable import MLXCore

/// The Settings picker shows what "Automatic" resolved to rather than
/// offering a second "Probe" entry — a user cannot choose between them
/// without benchmarking. These pin the pure decode + label.
final class SpecCostInfoTests: XCTestCase {
    func testParsesTheMeasuredLadderAndResolvedWidth() {
        let json: [String: Any] = [
            "memory": ["active_bytes": 1],
            "spec_cost": [
                "widths": [1, 2, 4],
                "ms": [38.0, 44.6, 59.2],
                "kv_ms_per_token": 0.00125,
                "mtp_depth_cap": 6,
            ],
        ]
        let info = SpecCostInfo.parse(json)
        XCTAssertEqual(info?.mtpDepthCap, 6)
        XCTAssertEqual(info?.widths, [1, 2, 4])
        XCTAssertEqual(info?.msPerWidth.last, 59.2)
        XCTAssertEqual(info?.kvMsPerToken ?? 0, 0.00125, accuracy: 1e-9)
    }

    func testAServerWithNoMeasuredCurveHasNothingToShow() {
        // MLX_SERVE_SPEC_COST_PROBE=0, or a declined probe: the per-silicon
        // tables applied and the picker falls back to a bare "Automatic".
        XCTAssertNil(SpecCostInfo.parse(["memory": ["active_bytes": 1]]))
        // A cap of 0 is not a resolved width either.
        XCTAssertNil(SpecCostInfo.parse(["spec_cost": ["mtp_depth_cap": 0]]))
    }

    func testLabelNamesTheWidthAndThatItWasMeasured() {
        // A bare "Automatic (6)" reads the same as a hardcoded cap.
        let six = SpecCostInfo(mtpDepthCap: 6, widths: [], msPerWidth: [], kvMsPerToken: 0)
        XCTAssertEqual(six.automaticLabel, "Automatic (measured: 6 tokens)")
        let one = SpecCostInfo(mtpDepthCap: 1, widths: [], msPerWidth: [], kvMsPerToken: 0)
        XCTAssertEqual(one.automaticLabel, "Automatic (measured: 1 token)")
    }
}
