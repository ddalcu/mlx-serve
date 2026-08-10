import XCTest
@testable import MLXCore

/// Which media models a Create pane puts in front of you.
///
/// The picker used to be a flat menu of every preset — seven for images —
/// ordered by download size, with the RAM figure only visible once you'd chosen.
/// So the question it asked ("which of these seven?") was one only somebody who
/// already knew the answer could answer. It now leads with the best model per
/// CAPABILITY that this Mac can actually run, says why each is there, and puts
/// the rest behind "Other Models".
final class MediaModelPicksTests: XCTestCase {

    private let gb: UInt64 = 1_073_741_824

    // MARK: - Best that fits, per capability

    /// Two models that do DIFFERENT things are both featured — a generator and
    /// an editor are not competing entries, and hiding one behind "Other" makes
    /// the pane look like it can't do that thing at all.
    func testOneFeaturedPickPerCapability() {
        let picks = MediaModelPicks.featured(ImageModelPreset.all,
                                             physicalMemoryBytes: 64 * gb,
                                             capabilityOf: \.capabilityLabel)
        let capabilities = picks.map(\.capability)
        XCTAssertEqual(Set(capabilities).count, capabilities.count,
                       "one pick per capability — never two entries for the same job")
        XCTAssertTrue(capabilities.contains(ImageModelPreset.krea2Turbo.capabilityLabel))
        XCTAssertTrue(picks.contains { $0.preset.supportsReferenceEdit },
                      "an image editor must be offered alongside the generator")
    }

    /// "Best this Mac supports" = the largest that FITS, not the largest.
    func testTheFeaturedPickIsTheLargestThatFitsThisMac() {
        let small = MediaModelPicks.featured(ImageModelPreset.all,
                                             physicalMemoryBytes: 16 * gb,
                                             capabilityOf: \.capabilityLabel)
        for pick in small where pick.fits {
            XCTAssertLessThanOrEqual(pick.preset.approxRAMGB, 16,
                                     "\(pick.preset.name) does not fit 16 GB")
        }
        let big = MediaModelPicks.featured(ImageModelPreset.all,
                                           physicalMemoryBytes: 128 * gb,
                                           capabilityOf: \.capabilityLabel)
        // More memory can only ever move a pick UP.
        for capability in Set(small.map(\.capability)) {
            let a = small.first { $0.capability == capability }!.preset.approxRAMGB
            let b = big.first { $0.capability == capability }!.preset.approxRAMGB
            XCTAssertGreaterThanOrEqual(b, a, "more RAM must not downgrade \(capability)")
        }
    }

    /// Warn, don't block — the app's policy everywhere else too. A Mac too small
    /// for anything in a capability still gets the smallest option, flagged, not
    /// an empty picker that reads as "unsupported".
    func testATinyMacStillGetsAPickFlaggedAsTooBig() {
        let picks = MediaModelPicks.featured(ImageModelPreset.all,
                                             physicalMemoryBytes: 4 * gb,
                                             capabilityOf: \.capabilityLabel)
        XCTAssertFalse(picks.isEmpty, "never an empty picker")
        for pick in picks {
            XCTAssertFalse(pick.fits)
            let smallestInCapability = ImageModelPreset.all
                .filter { $0.capabilityLabel == pick.capability }
                .map(\.approxRAMGB).min()
            XCTAssertEqual(pick.preset.approxRAMGB, smallestInCapability,
                           "with nothing that fits, offer the cheapest one")
        }
    }

    // MARK: - Everything else

    /// "Other Models" is the REST — never a second copy of a featured entry, and
    /// nothing may vanish from the catalogue entirely.
    func testOthersAreEverythingNotFeaturedAndNothingIsLost() {
        for ram in [UInt64(8), 16, 24, 64, 128] {
            let picks = MediaModelPicks.featured(ImageModelPreset.all,
                                                 physicalMemoryBytes: ram * gb,
                                                 capabilityOf: \.capabilityLabel)
            let others = MediaModelPicks.others(ImageModelPreset.all, featured: picks)
            let featuredIds = Set(picks.map(\.preset.id))
            XCTAssertTrue(others.allSatisfy { !featuredIds.contains($0.id) },
                          "a featured model must not also be listed under Other")
            XCTAssertEqual(Set(others.map(\.id)).union(featuredIds),
                           Set(ImageModelPreset.all.map(\.id)),
                           "every model stays reachable at \(ram) GB")
        }
    }

    /// The catalogue's order is the pane's order — `others` must not resort.
    func testOthersKeepCatalogueOrder() {
        let picks = MediaModelPicks.featured(ImageModelPreset.all,
                                             physicalMemoryBytes: 64 * gb,
                                             capabilityOf: \.capabilityLabel)
        let others = MediaModelPicks.others(ImageModelPreset.all, featured: picks)
        let expected = ImageModelPreset.all
            .filter { p in !picks.contains { $0.preset.id == p.id } }
            .map(\.id)
        XCTAssertEqual(others.map(\.id), expected)
    }

    // MARK: - The capability labels themselves

    /// The label is the REASON shown under the model, so it has to say what the
    /// model is for — not what it is.
    func testImageCapabilitiesSplitEditingFromGenerating() {
        XCTAssertNotEqual(ImageModelPreset.krea2Turbo.capabilityLabel,
                          ImageModelPreset.mageFlowEditTurbo8bit.capabilityLabel)
        XCTAssertTrue(ImageModelPreset.mageFlowEditTurbo8bit.capabilityLabel
            .localizedCaseInsensitiveContains("edit"))
    }

    /// Video's two capabilities are the ones a user would actually choose
    /// between: one makes a silent clip, the other brings its own soundtrack.
    func testVideoCapabilitiesSplitOnSound() {
        let withSound = VideoModelPreset.all.filter(\.generatesAudio)
        let silent = VideoModelPreset.all.filter { !$0.generatesAudio }
        XCTAssertFalse(withSound.isEmpty)
        XCTAssertFalse(silent.isEmpty)
        XCTAssertNotEqual(withSound[0].capabilityLabel, silent[0].capabilityLabel)
        XCTAssertTrue(withSound[0].capabilityLabel.localizedCaseInsensitiveContains("sound"))
    }

    /// A catalogue with a single job features exactly one model and puts the
    /// rest under Other — no capability heading that splits nothing.
    func testSingleCapabilityCataloguesFeatureExactlyOne() {
        // Speech: every entry in `.all` does the one job.
        XCTAssertEqual(
            MediaModelPicks.featured(AudioModelPreset.all,
                                     physicalMemoryBytes: 64 * gb,
                                     capabilityOf: \.capabilityLabel).count, 1)
        // 3D: one family, and it rides `MediaModelSizing` alone — conforming it
        // to the fuller `MediaModelPreset` would pull it into the Model
        // Browser's Media tab, which deliberately covers four modalities.
        XCTAssertEqual(
            MediaModelPicks.featured(Model3DModelPreset.all,
                                     physicalMemoryBytes: 64 * gb,
                                     capabilityOf: \.capabilityLabel).count, 1)
    }
}
