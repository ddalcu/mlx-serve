import Foundation

/// Where the guest's kernel and root filesystem come from.
///
/// The Developer ID build fetches them on first use: the kernel from a GitHub
/// release, the rootfs from a container registry. MAS build cannot
/// So: same `VzGuest`, same `vz-agent`, two provisioners.
protocol SandboxProvisioner {
    /// Absolute path to an uncompressed arm64 kernel image.
    func kernelPath() throws -> String
    /// Absolute path to an unpacked rootfs directory.
    func rootfsDir(image: String) throws -> String
    /// Human-readable name for the guest image, for the transcript and UI.
    func imageDescription(image: String) -> String
}

// MARK: - Bundled (Mac App Store)

/// Unpacks `Contents/Resources/guest/` into the container on first use.
///
/// The rootfs ships as ONE `rootfs.tar.gz`, not a loose directory tree: a
/// Debian userland is ~30k files, and code-signing hashes every one of them
/// into the bundle seal. One archive is one signed resource, and it is
/// unambiguously data rather than code.
///
/// Unpacking into the container is "installing resources into your own
/// container", which is allowed. Installing into a shared location is not.
struct BundledProvisioner: SandboxProvisioner {

    /// Bumped whenever the bundled guest changes, so a stale unpack is replaced.
    static let version = "1"
    static let marker = ".bundled-guest-version"

    let resourcesURL: URL
    let containerRoot: URL

    enum ProvisionError: LocalizedError {
        case missing(String)
        var errorDescription: String? {
            switch self {
            case .missing(let what):
                return "the bundled guest is incomplete: \(what) is missing from the app bundle"
            }
        }
    }

    func kernelPath() throws -> String {
        let kernel = resourcesURL.appendingPathComponent("guest/kernel")
        guard FileManager.default.isReadableFile(atPath: kernel.path) else {
            throw ProvisionError.missing("guest/kernel")
        }
        return kernel.path
    }

    func rootfsDir(image: String) throws -> String {
        let archive = resourcesURL.appendingPathComponent("guest/rootfs.tar.gz")
        guard FileManager.default.isReadableFile(atPath: archive.path) else {
            throw ProvisionError.missing("guest/rootfs.tar.gz")
        }

        let destination = containerRoot.appendingPathComponent("images/bundled", isDirectory: true)
        if Self.isUnpacked(at: destination) { return destination.path }

        // A partial unpack from a previous crash must not be trusted.
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)

        // In-process: no `/usr/bin/tar`, no network. `TarReader` refuses entries
        // that would escape the destination.
        let compressed = try Data(contentsOf: archive, options: .mappedIfSafe)
        try TarReader.extract(try OCIClient.decompressLayer(compressed), into: destination)

        try Data(Self.version.utf8).write(to: destination.appendingPathComponent(Self.marker))
        return destination.path
    }

    func imageDescription(image: String) -> String { "bundled" }

    /// The version marker is written LAST, so its presence means the unpack
    /// finished. `/bin/sh` alone would also be true half-way through.
    static func isUnpacked(at destination: URL) -> Bool {
        guard let data = try? Data(contentsOf: destination.appendingPathComponent(marker)),
              String(decoding: data, as: UTF8.self) == version else { return false }
        return FileManager.default.isExecutableFile(atPath: destination.appendingPathComponent("bin/sh").path)
    }
}

// MARK: - Downloading (Developer ID)

/// Today's behavior, unchanged: fetch the kernel from a GitHub release and pull
/// the rootfs from a container registry. Backed by `AgentSandbox`'s existing
/// `provisionKernel` / `provisionRootfs`, which own the caching.
struct DownloadingProvisioner: SandboxProvisioner {
    let fetchKernel: () throws -> String
    let pullRootfs: (String) throws -> String

    func kernelPath() throws -> String { try fetchKernel() }
    func rootfsDir(image: String) throws -> String { try pullRootfs(image) }
    func imageDescription(image: String) -> String { image }
}

// MARK: - Selection

enum SandboxProvisionerFactory {
    /// Pure decision, so the choice is testable without a bundle.
    ///
    /// The App Store build has no downloading arm at all — `ociPull` is false
    /// there, and `OCIClient`'s network methods are compiled out.
    static func kind(ociPullAllowed: Bool = BuildFeatures.current.ociPull) -> Kind {
        ociPullAllowed ? .downloading : .bundled
    }

    enum Kind: Equatable { case downloading, bundled }
}