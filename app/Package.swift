// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MLXCore",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "MLXCore",
            path: "Sources/MLXServe",
            exclude: ["Resources"]
        ),
        .testTarget(
            name: "MLXCoreTests",
            path: "Tests/MLXCoreTests"
        ),
    ]
)
