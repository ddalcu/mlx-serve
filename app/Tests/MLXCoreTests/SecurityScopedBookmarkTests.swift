import XCTest
@testable import MLXCore

/// The App Sandbox forgets a user's file pick on relaunch unless a
/// security-scoped bookmark is stored. These tests pin the persistence logic;
/// the AppKit scope start/stop is a thin no-op outside the sandbox.
final class SecurityScopedBookmarkTests: XCTestCase {

    private var defaults: UserDefaults!
    private let suite = "bookmark-tests-\(UUID().uuidString)"

    override func setUp() {
        defaults = UserDefaults(suiteName: suite)
    }
    override func tearDown() {
        defaults.removePersistentDomain(forName: suite)
    }

    func testKeyIsNamespaced() {
        XCTAssertEqual(SecurityScopedBookmark.defaultsKey("voiceClone"),
                       "securityBookmark.voiceClone")
        XCTAssertNotEqual(SecurityScopedBookmark.defaultsKey("a"),
                          SecurityScopedBookmark.defaultsKey("b"))
    }

    /// Store → resolve round-trips a real path (bookmarks resolve fine outside
    /// the sandbox too, which is what the tests run under).
    func testStoreThenResolveRoundTripsARealPath() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ssb-\(UUID().uuidString)", isDirectory: true)
            .resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertTrue(SecurityScopedBookmark.store(dir, name: "folder", defaults: defaults))
        XCTAssertTrue(SecurityScopedBookmark.has(name: "folder", defaults: defaults))

        let resolved = try XCTUnwrap(SecurityScopedBookmark.resolve(name: "folder", defaults: defaults))
        defer { if resolved.started { resolved.url.stopAccessingSecurityScopedResource() } }
        XCTAssertEqual(resolved.url.resolvingSymlinksInPath().path, dir.path)
    }

    func testResolveMissingIsNil() {
        XCTAssertNil(SecurityScopedBookmark.resolve(name: "never-stored", defaults: defaults))
        XCTAssertFalse(SecurityScopedBookmark.has(name: "never-stored", defaults: defaults))
    }

    func testClearRemovesTheBookmark() throws {
        let file = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ssb-\(UUID().uuidString).txt")
        try Data("x".utf8).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }

        SecurityScopedBookmark.store(file, name: "clip", defaults: defaults)
        XCTAssertTrue(SecurityScopedBookmark.has(name: "clip", defaults: defaults))

        SecurityScopedBookmark.clear(name: "clip", defaults: defaults)
        XCTAssertFalse(SecurityScopedBookmark.has(name: "clip", defaults: defaults))
        XCTAssertNil(SecurityScopedBookmark.resolve(name: "clip", defaults: defaults))
    }

    func testWithResolvedBalancesTheScopeAndReturnsTheBody() throws {
        let file = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ssb-\(UUID().uuidString).txt")
        try Data("hello".utf8).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }

        SecurityScopedBookmark.store(file, name: "clip", defaults: defaults)
        let contents = SecurityScopedBookmark.withResolved(name: "clip", defaults: defaults) { url in
            try? String(contentsOf: url, encoding: .utf8)
        }
        XCTAssertEqual(contents, "hello")
    }

    func testWithResolvedReturnsNilWhenAbsent() {
        let ran = SecurityScopedBookmark.withResolved(name: "absent", defaults: defaults) { _ in true }
        XCTAssertNil(ran)
    }

    // MARK: - Per-session slots (working folder + attached document folder)

    /// One bookmark per session per purpose: sessions must not share grants, and
    /// a session's working folder must not collide with its attached folder.
    func testPerSessionSlotNamesAreNamespacedAndDistinct() {
        let a = UUID(), b = UUID()
        XCTAssertNotEqual(SecurityScopedBookmark.workingFolderName(a),
                          SecurityScopedBookmark.workingFolderName(b))
        XCTAssertNotEqual(SecurityScopedBookmark.workingFolderName(a),
                          SecurityScopedBookmark.attachedFolderName(a))
        XCTAssertTrue(SecurityScopedBookmark.workingFolderName(a).contains(a.uuidString))
    }

    /// The agent-turn seam calls this before tools touch the session's cwd: it
    /// resolves the stored bookmark and holds the scope for the rest of the
    /// launch (a turn runs for minutes; a scope balanced per-call would drop
    /// access mid-turn). Repeat calls are idempotent — one started scope per
    /// name per launch, so the kernel resource can't leak per turn.
    func testStartAccessOnceResolvesAndIsIdempotent() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ssb-once-\(UUID().uuidString)", isDirectory: true)
            .resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let name = "once-\(UUID().uuidString)"
        SecurityScopedBookmark.store(dir, name: name, defaults: defaults)

        let first = SecurityScopedBookmark.startAccessOnce(name: name, defaults: defaults)
        XCTAssertEqual(first?.resolvingSymlinksInPath().path, dir.path)
        let second = SecurityScopedBookmark.startAccessOnce(name: name, defaults: defaults)
        XCTAssertEqual(second?.path, first?.path, "repeat calls return the same live grant")
    }

    func testStartAccessOnceIsNilWithoutABookmark() {
        XCTAssertNil(SecurityScopedBookmark.startAccessOnce(name: "never-\(UUID().uuidString)",
                                                            defaults: defaults))
    }

    /// Distinct names must not collide — the voice clip and the attached folder
    /// are stored side by side.
    func testDistinctNamesAreIndependent() throws {
        let a = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("a-\(UUID().uuidString)")
        let b = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("b-\(UUID().uuidString)")
        try Data().write(to: a); try Data().write(to: b)
        defer { try? FileManager.default.removeItem(at: a); try? FileManager.default.removeItem(at: b) }

        SecurityScopedBookmark.store(a, name: "clip", defaults: defaults)
        SecurityScopedBookmark.store(b, name: "folder", defaults: defaults)
        SecurityScopedBookmark.clear(name: "clip", defaults: defaults)

        XCTAssertFalse(SecurityScopedBookmark.has(name: "clip", defaults: defaults))
        XCTAssertTrue(SecurityScopedBookmark.has(name: "folder", defaults: defaults))
    }
}
