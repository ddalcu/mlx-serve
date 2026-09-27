import XCTest
@testable import MLXCore

/// The Image pane's switcher reports what the steps ARE: a tier while they
/// equal one tier's schedule, Custom once the Advanced slider moved them.
final class ImageQualityMatchTests: XCTestCase {

    /// Applying a tier leaves the switcher reading that tier, on every model.
    func testApplyingATierLeavesTheSwitcherOnIt() {
        for model in ImageModelPreset.all {
            for tier in QualityPreset.allCases {
                let steps = model.settings(tier).steps
                XCTAssertEqual(ImageQualityMatch.match(steps: steps, model: model, preferring: tier),
                               tier, "\(model.id) \(tier)")
            }
        }
    }

    func testStepsMatchingNoTierReadCustom() {
        let model = ImageModelPreset.flux2Klein4B_Q4
        let taken = Set(QualityPreset.allCases.map { model.settings($0).steps })
        let odd = (1...200).first { !taken.contains($0) }!
        XCTAssertNil(ImageQualityMatch.match(steps: odd, model: model, preferring: .good))
    }

    /// A tier the steps do not match never wins on preference alone.
    func testPreferenceNeverOverridesTheSteps() {
        let model = ImageModelPreset.flux2Klein4B_Q4
        let good = model.settings(.good).steps
        XCTAssertEqual(ImageQualityMatch.match(steps: good, model: model, preferring: .superQuality), .good)
    }

    /// Mage-Flow Turbo's four tiers all run 4 steps: the user's last click
    /// settles the tie, and with no preference the first tier reads.
    func testPreferenceSettlesATie() {
        let model = ImageModelPreset.mageFlowTurbo
        let steps = model.settings(.quality).steps
        XCTAssertEqual(ImageQualityMatch.match(steps: steps, model: model, preferring: .quality), .quality)
        XCTAssertEqual(ImageQualityMatch.match(steps: steps, model: model, preferring: nil),
                       QualityPreset.allCases.first)
    }
}
