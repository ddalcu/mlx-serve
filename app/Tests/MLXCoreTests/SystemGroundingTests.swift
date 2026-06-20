import XCTest
@testable import MLXCore

/// Unit tests for the pure layer of `SystemGrounding` — the LAN-IP selection
/// policy and the grounding sentence. The `getifaddrs` enumeration itself is the
/// untestable I/O shell; everything that decides *which* address to surface and
/// *how* to phrase it is pure and pinned here.
final class SystemGroundingTests: XCTestCase {

    func testDateLineStatesTheClock() {
        // 2026-06-19 — the line must contain the formatted moment, not a guess.
        let line = SystemGrounding.dateTimeLine()
        XCTAssertTrue(line.contains("current date and time"))
    }

    func testAgentDateLineIsDateOnlyAndStableWithinADay() {
        // The agent grounding must be byte-identical for two different TIMES on
        // the same day, so the system-prompt prefix stays cacheable on the
        // server (the old per-minute timestamp defeated the KV prefix cache and
        // made every agent turn cold-re-prefill → slow TTFB).
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone.current
        func at(_ day: Int, _ hour: Int, _ minute: Int) -> Date {
            cal.date(from: DateComponents(year: 2026, month: 6, day: day, hour: hour, minute: minute))!
        }
        let morning = SystemGrounding.dateLine(now: at(20, 8, 3))
        let evening = SystemGrounding.dateLine(now: at(20, 22, 47))
        XCTAssertEqual(morning, evening, "same day, different time → identical (cacheable)")

        // A different day differs, so date questions stay correct.
        XCTAssertNotEqual(morning, SystemGrounding.dateLine(now: at(21, 8, 3)))

        // No clock time leaked in (that's what changed per request).
        XCTAssertFalse(morning.contains("AM"))
        XCTAssertFalse(morning.contains("PM"))
        XCTAssertFalse(morning.contains(":"))
        XCTAssertTrue(morning.contains("2026"), "still states the date")
    }

    func testPickPrefersWiFiThenEthernetThenOther() {
        let picked = SystemGrounding.pickLanIPv4(from: [
            ("en1", "10.0.0.5"),
            ("en0", "192.168.1.42"),
            ("utun3", "172.16.0.9"),
        ])
        XCTAssertEqual(picked, "192.168.1.42", "en0 (Wi-Fi) wins when present")
    }

    func testPickFallsBackToEthernetThenAny() {
        XCTAssertEqual(
            SystemGrounding.pickLanIPv4(from: [("utun3", "172.16.0.9"), ("en1", "10.0.0.5")]),
            "10.0.0.5", "en1 (Ethernet) when no en0")
        XCTAssertEqual(
            SystemGrounding.pickLanIPv4(from: [("bridge0", "192.168.64.1")]),
            "192.168.64.1", "any remaining interface when no en0/en1")
    }

    func testPickSkipsLoopbackAndLinkLocal() {
        let picked = SystemGrounding.pickLanIPv4(from: [
            ("lo0", "127.0.0.1"),
            ("en0", "169.254.10.10"),   // self-assigned link-local — not reachable
            ("en1", "192.168.0.7"),
        ])
        XCTAssertEqual(picked, "192.168.0.7", "loopback + link-local are filtered out")
    }

    func testPickReturnsNilWhenNothingUsable() {
        XCTAssertNil(SystemGrounding.pickLanIPv4(from: []))
        XCTAssertNil(SystemGrounding.pickLanIPv4(from: [("lo0", "127.0.0.1")]))
    }

    func testIPLineIncludesAddressAndReachabilityGuidance() {
        let line = SystemGrounding.localIPLine(ip: "192.168.1.42")
        XCTAssertTrue(line.contains("192.168.1.42"))
        XCTAssertTrue(line.contains("0.0.0.0"), "must steer the agent to bind 0.0.0.0")
        XCTAssertTrue(line.contains("http://192.168.1.42:"), "must show the reachable URL shape")
    }

    func testIPLineEmptyWhenOffline() {
        XCTAssertEqual(SystemGrounding.localIPLine(ip: nil), "")
        XCTAssertEqual(SystemGrounding.localIPLine(ip: ""), "")
    }

    // MARK: - System-prompt composition (cache stability)

    func testComposeSystemPromptKeepsStableCoreAsSharedPrefix() {
        // Two agent turns that differ ONLY in volatile content (a new file in
        // the working-dir listing, an updated recent-commands snippet, a freshly
        // matched skill) must share a byte-identical prefix covering the ENTIRE
        // stable core — so the server re-prefills only the short tail instead of
        // the whole tool+instruction block. This is what makes mid-session agent
        // TTFB fast.
        let stable = "BASE AGENT INSTRUCTIONS …\n\n# MCP Tools\nconnected: serverX"
        let g = "Today's date is Friday, June 20, 2026."
        let turnA = ChatTurnEngine.composeSystemPrompt(
            stable: stable,
            volatileTail: "[skill:deploy]\n[Files]\na.swift b.swift\n[Memory]\ncmds: ls, grep",
            grounding: g)
        let turnB = ChatTurnEngine.composeSystemPrompt(
            stable: stable,
            volatileTail: "[Files]\na.swift b.swift c.swift\n[Memory]\ncmds: cat, ls, grep",
            grounding: g)

        XCTAssertTrue(turnA.hasPrefix(stable), "stable core stays at the front")
        XCTAssertTrue(turnB.hasPrefix(stable))
        XCTAssertGreaterThanOrEqual(commonPrefixLength(turnA, turnB), stable.count,
            "the whole stable core is a shared prefix → server caches it across turns")
        XCTAssertNotEqual(turnA, turnB, "volatile content still differs (it's not dropped)")
        XCTAssertTrue(turnA.contains("skill:deploy"))
        XCTAssertTrue(turnB.contains("c.swift"))
        XCTAssertTrue(turnA.hasSuffix(g), "grounding is last")
    }

    private func commonPrefixLength(_ a: String, _ b: String) -> Int {
        let ca = Array(a), cb = Array(b)
        var i = 0
        while i < ca.count, i < cb.count, ca[i] == cb[i] { i += 1 }
        return i
    }
}
