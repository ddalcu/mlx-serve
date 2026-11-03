import XCTest
@testable import MLXCore

/// The pipe reader behind `ProcessRegistry`: it appends what the child wrote and
/// lets go of the handle at EOF, so an exited process costs no CPU.
final class ProcessCaptureTests: XCTestCase {

    func testCaptureAppendsOutputAndDetachesAtEOF() throws {
        let buffer = ProcessOutputBuffer()
        let pipe = Pipe()
        let capture = ProcessRegistry.capture(into: buffer)
        let detached = expectation(description: "handler cleared at EOF")
        detached.assertForOverFulfill = false
        pipe.fileHandleForReading.readabilityHandler = { h in
            capture(h)
            if h.readabilityHandler == nil { detached.fulfill() }
        }

        pipe.fileHandleForWriting.write(Data("hello\n".utf8))
        try pipe.fileHandleForWriting.close()
        wait(for: [detached], timeout: 5)

        XCTAssertNil(pipe.fileHandleForReading.readabilityHandler)
        XCTAssertEqual(buffer.snapshot(), "hello\n")
    }
}
