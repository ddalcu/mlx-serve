import XCTest
@testable import MLXCore

final class PrefillDecodeShareSettingsTests: XCTestCase {
    func testDefaultAndLegacyDecodeLeaveShareAtZero() throws {
        XCTAssertEqual(ServerOptions().prefillDecodeShare, 0)
        for json in ["{}", #"{"port":8080,"maxConcurrent":4}"#] {
            let options = try JSONDecoder().decode(ServerOptions.self, from: Data(json.utf8))
            XCTAssertEqual(options.prefillDecodeShare, 0)
            XCTAssertFalse(options.toCLIArgs().contains("--prefill-decode-share"))
        }
        let legacy = try JSONDecoder().decode(
            ServerOptions.self, from: Data(#"{"port":8080,"maxConcurrent":4}"#.utf8))
        XCTAssertEqual(legacy.port, 8080)
        XCTAssertEqual(legacy.maxConcurrent, 4)
    }

    func testShareRoundTripsWithoutChangingOtherOptions() throws {
        var options = ServerOptions()
        options.prefillDecodeShare = 0.35
        options.port = 8080
        let data = try JSONEncoder().encode(options)
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: data)
        XCTAssertEqual(decoded, options)
        XCTAssertEqual(decoded.prefillDecodeShare, 0.35)
    }

    func testEnabledShareEmitsExactlyOneFlagAndDefaultOmitsIt() throws {
        for share in [0.1, 0.35, 0.9] {
            var options = ServerOptions()
            options.prefillDecodeShare = share
            let args = options.toCLIArgs()
            let index = try XCTUnwrap(args.firstIndex(of: "--prefill-decode-share"))
            XCTAssertEqual(args.filter { $0 == "--prefill-decode-share" }.count, 1)
            XCTAssertEqual(Double(args[index + 1]), share)
            options.prefillDecodeShare = 0
            XCTAssertFalse(options.toCLIArgs().contains("--prefill-decode-share"))
        }
    }

    func testSliderPresetsAreFiniteAndBounded() {
        let percentages = ServerOptions.prefillDecodeSharePresets
        XCTAssertEqual(percentages.first, 0)
        XCTAssertEqual(percentages.last, 90)
        XCTAssertEqual(percentages, Array(Set(percentages)).sorted())
        for percent in percentages {
            let share = Double(percent) / 100
            XCTAssertTrue(share.isFinite)
            XCTAssertTrue((0...0.9).contains(share))
            XCTAssertEqual(ServerOptions.normalizedPrefillDecodeShare(share), share)
        }
    }

    func testNormalizationKeepsUIAndLaunchValuesFiniteAndBounded() throws {
        for (input, expected) in [(Double.nan, 0.0), (.infinity, 0), (-.infinity, 0),
                                  (-0.2, 0), (0, 0), (0.35, 0.35), (0.9, 0.9), (1, 0.9)] {
            XCTAssertEqual(ServerOptions.normalizedPrefillDecodeShare(input), expected)
            var options = ServerOptions()
            options.prefillDecodeShare = input
            let args = options.toCLIArgs()
            if expected == 0 {
                XCTAssertFalse(args.contains("--prefill-decode-share"))
            } else {
                let index = try XCTUnwrap(args.firstIndex(of: "--prefill-decode-share"))
                XCTAssertEqual(Double(args[index + 1]), expected)
            }
        }
    }

    func testOutOfRangeStoredSharesAreBoundedOnDecode() throws {
        for (input, expected) in [(-0.2, 0.0), (1.0, 0.9)] {
            let data = Data("{\"prefillDecodeShare\":\(input),\"port\":8080}".utf8)
            let options = try JSONDecoder().decode(ServerOptions.self, from: data)
            XCTAssertEqual(options.prefillDecodeShare, expected)
            XCTAssertEqual(options.port, 8080)
        }
    }

    func testShareRequiresRestartAndHasRestartMetadata() throws {
        let defaults = ServerOptions()
        var options = defaults
        options.prefillDecodeShare = 0.3
        XCTAssertFalse(options.serverLaunchEquals(defaults))
        XCTAssertFalse(defaults.serverLaunchEquals(options))
        XCTAssertTrue(options.serverLaunchEquals(options))
        let metadata = try XCTUnwrap(ServerOptions.serverFlagFields["prefillDecodeShare"])
        XCTAssertTrue(metadata.needsRestart)
        XCTAssertFalse(metadata.title.isEmpty)
        XCTAssertFalse(metadata.explainer.isEmpty)
    }

    func testOnlyPerformanceAndGlobalResetRestoreShare() {
        var options = ServerOptions()
        options.prefillDecodeShare = 0.4
        options.port = 8080
        let owners = SettingsCategory.allCases.filter { category in
            SettingsReset.fields(for: category).contains { $0.name == "prefillDecodeShare" }
        }
        XCTAssertEqual(owners, [.performance])
        let reset = SettingsReset.apply(.performance, to: options)
        XCTAssertEqual(reset.prefillDecodeShare, 0)
        XCTAssertEqual(reset.port, 8080)
        XCTAssertFalse(reset.toCLIArgs().contains("--prefill-decode-share"))
        XCTAssertEqual(SettingsReset.apply(.server, to: options).prefillDecodeShare, 0.4)
        XCTAssertEqual(SettingsReset.applyAll(to: options).prefillDecodeShare, 0)
    }
}
