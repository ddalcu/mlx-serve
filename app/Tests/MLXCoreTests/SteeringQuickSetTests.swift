import XCTest
@testable import MLXCore

final class SteeringQuickSetTests: XCTestCase {
    func testParseAcceptsScalesTheServerWouldAccept() {
        XCTAssertEqual(SteeringQuickSet.parse("1"), 1)
        XCTAssertEqual(SteeringQuickSet.parse("1.5"), 1.5)
        XCTAssertEqual(SteeringQuickSet.parse("-0.1"), -0.1)   // negative amplifies
        XCTAssertEqual(SteeringQuickSet.parse("  2.0  "), 2.0)
        XCTAssertEqual(SteeringQuickSet.parse("0"), 0)          // 0 is how you turn it off
    }

    func testParseRejectsWhatTheServerWould400() {
        XCTAssertNil(SteeringQuickSet.parse(""))
        XCTAssertNil(SteeringQuickSet.parse("   "))
        XCTAssertNil(SteeringQuickSet.parse("abc"))
        XCTAssertNil(SteeringQuickSet.parse("1.2.3"))
        XCTAssertNil(SteeringQuickSet.parse("nan"))
        XCTAssertNil(SteeringQuickSet.parse("inf"))
        XCTAssertNil(SteeringQuickSet.parse("101"))             // past SCALE_LIMIT
        XCTAssertNil(SteeringQuickSet.parse("-101"))
    }

    func testSteppingStaysOnOneDecimalAndInsideTheBound() {
        XCTAssertEqual(SteeringQuickSet.stepped(1.0, by: 1), 1.1)
        XCTAssertEqual(SteeringQuickSet.stepped(1.0, by: -1), 0.9)
        // Repeated stepping must not accumulate float dust: 0.1 * 3 is not 0.3 in binary.
        var v = 0.0
        for _ in 0..<3 { v = SteeringQuickSet.stepped(v, by: 1) }
        XCTAssertEqual(v, 0.3)
        XCTAssertEqual(SteeringQuickSet.stepped(100, by: 5), 100)
        XCTAssertEqual(SteeringQuickSet.stepped(-100, by: -5), -100)
    }

    func testZeroOnBothArmsIsOffNotANoOpEdit() {
        XCTAssertEqual(SteeringQuickSet.override(name: "terse", ffn: 0, attn: 0), .off)
        XCTAssertEqual(SteeringQuickSet.override(name: "terse", ffn: 1, attn: 0),
                       .configured(name: "terse", ffn: 1, attn: 0))
        // attn alone still steers, so it is not off.
        XCTAssertEqual(SteeringQuickSet.override(name: "terse", ffn: 0, attn: 0.5),
                       .configured(name: "terse", ffn: 0, attn: 0.5))
    }

    func testTurningItOffDoesNotDeleteTheControl() {
        // Bar: at 0/0 the server reports nothing and the remembered bank keeps the box up.
        XCTAssertEqual(SteeringQuickSet.visibleBank(live: "terse", remembered: nil), "terse")
        XCTAssertEqual(SteeringQuickSet.visibleBank(live: nil, remembered: "terse"), "terse")
        // The live value wins: a Model Settings save can move the bank under us.
        XCTAssertEqual(SteeringQuickSet.visibleBank(live: "verbosity", remembered: "terse"), "verbosity")
        // Never configured in this session: nothing to draw.
        XCTAssertNil(SteeringQuickSet.visibleBank(live: nil, remembered: nil))
        XCTAssertNil(SteeringQuickSet.visibleBank(live: "", remembered: ""))
    }

    func testAConfiguredBankIsVisibleBeforeTheModelLoads() {
        // Bar: a saved, non-zero bank is announced from the settings file alone.
        var file = ModelSettingsFile()
        file.set(ModelOverride(steering: .configured(name: "terse", ffn: 1, attn: 0)), for: "/models/qwen")
        XCTAssertEqual(SteeringQuickSet.configuredBank(path: "/models/qwen", file: file), "terse")
        // Off, unset, and an unknown model each have nothing to announce.
        var offFile = ModelSettingsFile()
        offFile.set(ModelOverride(steering: .off), for: "/models/qwen")
        XCTAssertNil(SteeringQuickSet.configuredBank(path: "/models/qwen", file: offFile))
        XCTAssertNil(SteeringQuickSet.configuredBank(path: "/models/other", file: file))
        XCTAssertNil(SteeringQuickSet.configuredBank(path: "", file: file))
        // Configured at 0/0 arms nothing on load, so there is nothing to announce.
        var zeroFile = ModelSettingsFile()
        zeroFile.set(ModelOverride(steering: .configured(name: "terse", ffn: 0, attn: 0)), for: "/models/qwen")
        XCTAssertNil(SteeringQuickSet.configuredBank(path: "/models/qwen", file: zeroFile))
    }

    func testTheArchGateMirrorsTheServer() {
        // Bar: only qwen4_exp (the server's steering.archSupported) offers the picker.
        XCTAssertTrue(SteeringQuickSet.archSupportsSteering("qwen4_exp"))
        XCTAssertTrue(SteeringQuickSet.archSupportsSteering("qwen4_exp_text"))
        XCTAssertFalse(SteeringQuickSet.archSupportsSteering("gemma4"))
        XCTAssertFalse(SteeringQuickSet.archSupportsSteering("qwen3_5_moe"))
        XCTAssertFalse(SteeringQuickSet.archSupportsSteering(""))
    }

    func testTheBoxOnlyDrawsOnTheRowItDescribes() {
        // Bar: the box draws only on the row /props described.
        XCTAssertTrue(SteeringQuickSet.ownsLiveSteering(row: "A", liveChatModel: "A"))
        XCTAssertFalse(SteeringQuickSet.ownsLiveSteering(row: "B", liveChatModel: "A"))
        XCTAssertFalse(SteeringQuickSet.ownsLiveSteering(row: "A", liveChatModel: nil))
        XCTAssertFalse(SteeringQuickSet.ownsLiveSteering(row: "", liveChatModel: ""))
    }

    func testScalesAreClampedToWhatTheServerAccepts() {
        // Bar: every saved scale is one the server's validateScale accepts.
        XCTAssertEqual(SteeringQuickSet.clamped(500), 100)
        XCTAssertEqual(SteeringQuickSet.clamped(-500), -100)
        XCTAssertEqual(SteeringQuickSet.clamped(1.5), 1.5)
        XCTAssertEqual(SteeringQuickSet.clamped(.nan), 0)
        XCTAssertEqual(SteeringQuickSet.clamped(.infinity), 0)
    }

    func testAnAbsolutePathBankRoundTripsAsAPath() {
        // Bar: a path-armed bank posts back as `file:`, a registry bank as `name:`.
        let ov = SteeringQuickSet.override(name: "terse", ffn: 1, attn: 0)
        let byPath = APIClient.steeringBody(model: "m", override: ov, file: "/Volumes/Ext/banks/terse.f32")
        XCTAssertEqual(byPath["file"] as? String, "/Volumes/Ext/banks/terse.f32")
        XCTAssertNil(byPath["name"], "a path-armed bank must not also send a name")
        // A registry bank still travels by name; the server resolves it in its own dir.
        let byName = APIClient.steeringBody(model: "m", override: ov, file: nil)
        XCTAssertEqual(byName["name"] as? String, "terse")
        XCTAssertNil(byName["file"])
        // An empty string is not a path.
        XCTAssertEqual(APIClient.steeringBody(model: "m", override: ov, file: "")["name"] as? String, "terse")
    }

    func testTheLoadBadgeNeverDrawsBesideTheBox() {
        // Bar: the badge and the box never draw together for one model.
        XCTAssertEqual(SteeringQuickSet.loadBadge(configured: "terse", boxDrawn: false), "terse")
        XCTAssertNil(SteeringQuickSet.loadBadge(configured: "terse", boxDrawn: true))
        XCTAssertNil(SteeringQuickSet.loadBadge(configured: nil, boxDrawn: false))
    }
}
