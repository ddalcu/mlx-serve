import XCTest
@testable import MLXCore

/// Important-usage capacity is APFS-only: exFAT/NTFS/SMB report 0 (#474).
final class VolumeCapacityTests: XCTestCase {

    func testZeroImportantFallsBackToPlain() {
        XCTAssertEqual(VolumeCapacity.resolve(important: 0, plain: 384_000_000_000, total: 1_000_000_000_000),
                       384_000_000_000)
    }

    func testMissingImportantFallsBackToPlain() {
        XCTAssertEqual(VolumeCapacity.resolve(important: nil, plain: 50, total: 100), 50)
    }

    func testApfsPrefersImportantIncludingPurgeable() {
        XCTAssertEqual(VolumeCapacity.resolve(important: 153, plain: 134, total: 1000), 153)
    }

    func testImportantAboveVolumeSizeIsNotTrusted() {
        XCTAssertEqual(VolumeCapacity.resolve(important: 2000, plain: 134, total: 1000), 134)
    }
}
