// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MLXCore",
    // 26.2 floor matches the bundled libmlx's deployment target (NAX-enabled
    // self-built mlx — scripts/build-mlx.sh) and Info.plist LSMinimumSystemVersion.
    platforms: [.macOS("26.2")],
    dependencies: [
        // 0.12.x needs Swift 6.2+ (uses 6.2-only `withThrowingTaskGroup { }`
        // syntax) — fine everywhere since CI moved to the macos-26 runner
        // (Xcode 26 / Swift 6.3); the old 0.10.x pin existed only for the
        // retired macos-14 / Swift 6.1 runner. 0.10.x is UNBUILDABLE on Swift
        // 6.3 in its own Swift-6 language mode ([#SendingRisksDataRace] in
        // NetworkTransport.swift), which the SwiftPM path papered over with a
        // global `-swift-version 5` but the Mac App Store Xcode project
        // (project.yml) cannot — packages there compile with their own
        // settings. Keep this pin >= 0.12.1 or the Xcode build breaks.
        .package(url: "https://github.com/modelcontextprotocol/swift-sdk.git", "0.12.1" ..< "0.13.0"),
        // Already pulled transitively by swift-sdk; declared here so we can use OrderedDictionary
        // directly to preserve user-edited key order in mcp.json.
        .package(url: "https://github.com/apple/swift-collections.git", from: "1.0.0"),
        // libopus + libogg wrapper (Element/Matrix team) — decodes Telegram's
        // Ogg/Opus voice notes, which AVFoundation can't read. Used only by
        // `VoicePreprocessor`; statically linked, so no dylib bundling/signing.
        .package(url: "https://github.com/element-hq/swift-ogg.git", from: "0.0.4"),
        // Terminal emulator for the Sandbox window's embedded terminal
        // (agent CLIs over ssh into the guest). Imported by exactly ONE file —
        // EmbeddedTerminalView — so a libghostty-backed view can replace it
        // wholesale later. Pinned to the 1.14 line.
        .package(url: "https://github.com/migueldeicaza/SwiftTerm.git", "1.14.0" ..< "2.0.0"),
        // Native TeX parser/layout + bundled KaTeX fonts for assistant chat.
        // Keeping this in Swift avoids a WebView/JavaScript renderer and the
        // security, selection, and streaming seams that would come with one.
        .package(url: "https://github.com/PhraseHQ/SwaTex", "0.5.0" ..< "0.6.0"),
    ],
    targets: [
        .executableTarget(
            name: "MLXCore",
            dependencies: [
                .product(name: "MCP", package: "swift-sdk"),
                .product(name: "OrderedCollections", package: "swift-collections"),
                .product(name: "SwiftOGG", package: "swift-ogg"),
                .product(name: "SwiftTerm", package: "SwiftTerm"),
                .product(name: "SwaTexRender", package: "SwaTex"),
            ],
            path: "Sources/MLXServe",
            exclude: ["Resources"]
        ),
        .testTarget(
            name: "MLXCoreTests",
            dependencies: [
                "MLXCore",
                // Used only to synthesise an Ogg/Opus fixture for the
                // VoicePreprocessor round-trip test.
                .product(name: "SwiftOGG", package: "swift-ogg"),
            ],
            path: "Tests/MLXCoreTests",
            // The GLB test fixture is loaded by source-relative path (#filePath),
            // not as a bundled resource — exclude it so SwiftPM doesn't flag it
            // as an unhandled resource.
            exclude: ["Fixtures"]
        ),
    ]
)
