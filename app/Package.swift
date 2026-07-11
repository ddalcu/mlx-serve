// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MLXCore",
    platforms: [.macOS(.v14)],
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
    ],
    targets: [
        .executableTarget(
            name: "MLXCore",
            dependencies: [
                .product(name: "MCP", package: "swift-sdk"),
                .product(name: "OrderedCollections", package: "swift-collections"),
                .product(name: "SwiftOGG", package: "swift-ogg"),
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
