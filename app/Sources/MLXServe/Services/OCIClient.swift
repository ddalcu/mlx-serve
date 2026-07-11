import Foundation
import Compression

/// Minimal, dependency-free OCI/Docker registry client: pulls an image's arm64
/// rootfs (anonymous auth token dance → manifest → layers → in-process
/// `TarReader` unpack with whiteout handling) and extracts the image config
/// (Env/WorkingDir) the guest init script needs. Replaces libcontain's
/// `contain_pull_image`. No subprocesses — App Sandbox safe.
///
/// Scope: anonymous pulls (public images), tag or digest refs, Docker Hub +
/// generic v2 registries (ghcr.io etc). Pure parsing helpers are unit-tested;
/// the network path is covered by SandboxSmoke REAL_PROVISION.
enum OCIClient {

    struct PullError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    // MARK: Image reference parsing (docker conventions)

    struct ImageRef: Equatable {
        var registry: String
        var repository: String
        var tag: String

        /// Docker Hub's API host differs from its name; auth realm comes from the
        /// WWW-Authenticate challenge, so only the host matters here.
        var apiBase: String { "https://\(registry)/v2/\(repository)" }
    }

    /// Parse `[registry/]repo[:tag|@digest]` with Docker's conventions: a first
    /// path component containing `.`/`:` (or "localhost") is a registry; bare
    /// official images get the `library/` prefix; default tag is `latest`.
    static func parseImageRef(_ raw: String) -> ImageRef {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        var registry = "registry-1.docker.io"

        if let slash = s.firstIndex(of: "/") {
            let first = String(s[..<slash])
            if first.contains(".") || first.contains(":") || first == "localhost" {
                registry = first
                s = String(s[s.index(after: slash)...])
            }
        }

        var tag = "latest"
        if let at = s.firstIndex(of: "@") {
            // Digest ref: everything after @ is the manifest reference.
            tag = String(s[s.index(after: at)...])
            s = String(s[..<at])
        } else if let colon = s.lastIndex(of: ":"), !s[colon...].contains("/") {
            tag = String(s[s.index(after: colon)...])
            s = String(s[..<colon])
        }

        var repository = s
        if registry == "registry-1.docker.io" && !repository.contains("/") {
            repository = "library/" + repository
        }
        return ImageRef(registry: registry, repository: repository, tag: tag)
    }

    // MARK: WWW-Authenticate (anonymous bearer-token dance)

    struct AuthChallenge {
        var realm: String
        var params: [String: String]
    }

    /// Parse `Bearer realm="…",service="…",scope="…"`. Returns nil for non-Bearer.
    static func parseWWWAuthenticate(_ header: String) -> AuthChallenge? {
        let trimmed = header.trimmingCharacters(in: .whitespaces)
        guard trimmed.lowercased().hasPrefix("bearer ") else { return nil }
        var params: [String: String] = [:]
        // key="value" pairs, comma-separated; values never contain commas in
        // practice (scope uses colons) so a simple split is sufficient.
        for pair in trimmed.dropFirst("bearer ".count).components(separatedBy: ",") {
            let kv = pair.split(separator: "=", maxSplits: 1)
            guard kv.count == 2 else { continue }
            let key = kv[0].trimmingCharacters(in: .whitespaces)
            let value = kv[1].trimmingCharacters(in: .whitespaces)
                .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
            params[key] = value
        }
        guard let realm = params["realm"] else { return nil }
        return AuthChallenge(realm: realm, params: params)
    }

    // MARK: Manifest index → platform digest

    /// From a manifest INDEX / list, pick the digest for `os`/`arch`. Returns nil
    /// when the JSON isn't an index (direct manifest — use it as-is) or no
    /// platform matches. Attestation entries (`architecture: "unknown"`) never match.
    static func selectManifestDigest(indexJSON: Data, arch: String, os: String = "linux") -> String? {
        guard let root = try? JSONSerialization.jsonObject(with: indexJSON) as? [String: Any],
              let manifests = root["manifests"] as? [[String: Any]] else { return nil }
        for m in manifests {
            guard let platform = m["platform"] as? [String: Any],
                  platform["architecture"] as? String == arch,
                  platform["os"] as? String == os,
                  let digest = m["digest"] as? String else { continue }
            return digest
        }
        return nil
    }

    // MARK: Image config

    struct ImageConfig {
        var env: [String] = []
        var workingDir: String? = nil
    }

    static func parseImageConfig(_ json: Data) -> ImageConfig {
        var out = ImageConfig()
        guard let root = try? JSONSerialization.jsonObject(with: json) as? [String: Any],
              let cfg = root["config"] as? [String: Any] else { return out }
        out.env = (cfg["Env"] as? [String]) ?? []
        if let wd = cfg["WorkingDir"] as? String, !wd.isEmpty { out.workingDir = wd }
        return out
    }

    // MARK: Whiteouts

    /// Apply OCI layer whiteouts after extracting a layer: `.wh.<name>` markers
    /// delete `<name>` (from a lower layer) and the marker itself. Opaque-dir
    /// markers (`.wh..wh..opq`) are removed without clearing the directory — a
    /// deliberate simplification (sequential extraction can't distinguish lower-
    /// layer content from this layer's); acceptable for the simple tool images
    /// the sandbox targets.
    @discardableResult
    static func applyWhiteouts(in dir: URL) throws -> Int {
        let fm = FileManager.default
        var markers: [URL] = []
        if let walker = fm.enumerator(at: dir, includingPropertiesForKeys: nil,
                                      options: [.producesRelativePathURLs]) {
            for case let url as URL in walker where url.lastPathComponent.hasPrefix(".wh.") {
                markers.append(url)
            }
        }
        var applied = 0
        for marker in markers {
            let name = marker.lastPathComponent
            if name == ".wh..wh..opq" {
                try? fm.removeItem(at: marker)
                continue
            }
            let target = marker.deletingLastPathComponent()
                .appendingPathComponent(String(name.dropFirst(".wh.".count)))
            try? fm.removeItem(at: target)
            try? fm.removeItem(at: marker)
            applied += 1
        }
        return applied
    }

    // MARK: gunzip (kernel release asset + gzip image layers)

    /// Decompress a gzip container (RFC 1952 header + raw DEFLATE payload) using
    /// the Compression framework — no Process, sandbox-safe, works for the ~12 MB
    /// kernel asset in one call.
    static func gunzip(_ data: Data) throws -> Data {
        let payload = try deflatePayload(of: data)
        return try inflateRaw(payload)
    }

    /// Strip the gzip header (handling FEXTRA/FNAME/FCOMMENT/FHCRC) + the 8-byte
    /// CRC/size trailer, returning the raw DEFLATE stream.
    private static func deflatePayload(of data: Data) throws -> Data {
        let bytes = [UInt8](data.prefix(64 * 1024)) // header is tiny; avoid copying the body
        guard data.count > 18, bytes[0] == 0x1f, bytes[1] == 0x8b, bytes[2] == 8 else {
            throw PullError(message: "not a gzip stream")
        }
        let flags = bytes[3]
        var idx = 10
        if flags & 0x04 != 0 { // FEXTRA
            guard idx + 2 <= bytes.count else { throw PullError(message: "truncated gzip header") }
            let xlen = Int(bytes[idx]) | (Int(bytes[idx + 1]) << 8)
            idx += 2 + xlen
        }
        for flag: UInt8 in [0x08, 0x10] where flags & flag != 0 { // FNAME, FCOMMENT
            while idx < bytes.count && bytes[idx] != 0 { idx += 1 }
            idx += 1
        }
        if flags & 0x02 != 0 { idx += 2 } // FHCRC
        guard idx < data.count - 8 else { throw PullError(message: "truncated gzip header") }
        return data.subdata(in: idx ..< data.count - 8)
    }

    /// Raw-DEFLATE inflate via compression_stream (COMPRESSION_ZLIB is raw
    /// deflate — exactly gzip's payload encoding).
    private static func inflateRaw(_ input: Data) throws -> Data {
        var stream = compression_stream(dst_ptr: UnsafeMutablePointer<UInt8>(bitPattern: 1)!,
                                        dst_size: 0, src_ptr: UnsafePointer<UInt8>(bitPattern: 1)!,
                                        src_size: 0, state: nil)
        guard compression_stream_init(&stream, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB) == COMPRESSION_STATUS_OK else {
            throw PullError(message: "inflate init failed")
        }
        defer { compression_stream_destroy(&stream) }

        let dstCapacity = 1 << 20
        let dstBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: dstCapacity)
        defer { dstBuffer.deallocate() }

        var out = Data()
        try input.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let base = src.bindMemory(to: UInt8.self).baseAddress else {
                throw PullError(message: "empty deflate payload")
            }
            stream.src_ptr = base
            stream.src_size = src.count
            while true {
                stream.dst_ptr = dstBuffer
                stream.dst_size = dstCapacity
                let status = compression_stream_process(&stream, Int32(COMPRESSION_STREAM_FINALIZE.rawValue))
                switch status {
                case COMPRESSION_STATUS_OK, COMPRESSION_STATUS_END:
                    out.append(dstBuffer, count: dstCapacity - stream.dst_size)
                    if status == COMPRESSION_STATUS_END { return }
                    if stream.src_size == 0 && dstCapacity == stream.dst_size {
                        throw PullError(message: "truncated deflate stream")
                    }
                default:
                    throw PullError(message: "corrupt gzip data")
                }
            }
        }
        return out
    }

    // MARK: HTTP plumbing

    /// URLSession delegate that drops the Authorization header on redirects —
    /// registry blob GETs redirect to pre-signed CDN URLs (S3/Cloudflare) that
    /// reject requests carrying a second auth mechanism.
    private final class RedirectSanitizer: NSObject, URLSessionTaskDelegate {
        func urlSession(_ session: URLSession, task: URLSessionTask,
                        willPerformHTTPRedirection response: HTTPURLResponse,
                        newRequest request: URLRequest,
                        completionHandler: @escaping (URLRequest?) -> Void) {
            var sanitized = request
            sanitized.setValue(nil, forHTTPHeaderField: "Authorization")
            completionHandler(sanitized)
        }
    }

    private static let session: URLSession = {
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 60
        cfg.timeoutIntervalForResource = 1800 // large layers on slow links
        return URLSession(configuration: cfg, delegate: RedirectSanitizer(), delegateQueue: nil)
    }()

    /// Blocking GET returning body + response. Used from the sandbox's
    /// provisioning thread (never the main thread).
    static func httpGet(_ url: URL, headers: [String: String] = [:]) throws -> (Data, HTTPURLResponse) {
        var req = URLRequest(url: url)
        for (k, v) in headers { req.setValue(v, forHTTPHeaderField: k) }
        let sem = DispatchSemaphore(value: 0)
        var out: (Data?, URLResponse?, Error?) = (nil, nil, nil)
        session.dataTask(with: req) { d, r, e in out = (d, r, e); sem.signal() }.resume()
        sem.wait()
        if let e = out.2 { throw PullError(message: "\(url.host ?? ""): \(e.localizedDescription)") }
        guard let data = out.0, let resp = out.1 as? HTTPURLResponse else {
            throw PullError(message: "no response from \(url.host ?? "")")
        }
        return (data, resp)
    }

    /// Blocking download-to-file GET (layers can be hundreds of MB).
    private static func httpDownload(_ url: URL, headers: [String: String]) throws -> URL {
        var req = URLRequest(url: url)
        for (k, v) in headers { req.setValue(v, forHTTPHeaderField: k) }
        let sem = DispatchSemaphore(value: 0)
        var out: (URL?, URLResponse?, Error?) = (nil, nil, nil)
        session.downloadTask(with: req) { tmp, r, e in
            // Persist before the completion handler returns — URLSession deletes
            // the temp file after it.
            if let tmp {
                let kept = FileManager.default.temporaryDirectory
                    .appendingPathComponent("oci-layer-\(UUID().uuidString)")
                try? FileManager.default.moveItem(at: tmp, to: kept)
                out = (kept, r, e)
            } else {
                out = (nil, r, e)
            }
            sem.signal()
        }.resume()
        sem.wait()
        if let e = out.2 { throw PullError(message: "layer download: \(e.localizedDescription)") }
        guard let file = out.0, let resp = out.1 as? HTTPURLResponse, resp.statusCode == 200 else {
            throw PullError(message: "layer download failed (HTTP \((out.1 as? HTTPURLResponse)?.statusCode ?? -1))")
        }
        return file
    }

    // MARK: Pull

    private static let manifestAccept = [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ].joined(separator: ", ")

    /// Anonymous bearer token for `ref`, obtained by following the registry's
    /// WWW-Authenticate challenge. Returns nil when the registry needs no auth.
    private static func anonymousToken(for ref: ImageRef) throws -> String? {
        guard let probeURL = URL(string: "\(ref.apiBase)/manifests/\(ref.tag)") else {
            throw PullError(message: "bad image reference")
        }
        let (_, probe) = try httpGet(probeURL, headers: ["Accept": manifestAccept])
        if probe.statusCode != 401 { return nil }
        guard let challengeHeader = probe.value(forHTTPHeaderField: "WWW-Authenticate"),
              let challenge = parseWWWAuthenticate(challengeHeader),
              var comps = URLComponents(string: challenge.realm) else {
            throw PullError(message: "registry auth challenge not understood")
        }
        var items: [URLQueryItem] = []
        if let service = challenge.params["service"] { items.append(.init(name: "service", value: service)) }
        items.append(.init(name: "scope", value: challenge.params["scope"] ?? "repository:\(ref.repository):pull"))
        comps.queryItems = (comps.queryItems ?? []) + items
        guard let tokenURL = comps.url else { throw PullError(message: "bad auth realm") }
        let (body, resp) = try httpGet(tokenURL)
        guard resp.statusCode == 200,
              let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let token = (json["token"] as? String) ?? (json["access_token"] as? String) else {
            throw PullError(message: "anonymous token request failed (HTTP \(resp.statusCode)) — is the image public?")
        }
        return token
    }

    /// The image-config sidecar written next to the unpacked rootfs, so the boot
    /// path can export the image's Env/WorkingDir without re-fetching.
    static let configSidecarName = ".vz-image-config.json"

    /// Pull `image` (public, anonymous) for `arch` and unpack its layers into
    /// `dir` (created if needed). Writes the raw image config JSON to
    /// `configSidecarName` inside `dir`. Blocking; call from a worker thread.
    static func pull(image: String, into dir: URL, arch: String,
                     log: (String) -> Void = { _ in }) throws -> ImageConfig {
        let ref = parseImageRef(image)
        log("pulling \(ref.repository):\(ref.tag) (\(arch)) from \(ref.registry)")
        let token = try anonymousToken(for: ref)
        var auth: [String: String] = ["Accept": manifestAccept]
        if let token { auth["Authorization"] = "Bearer \(token)" }

        func getManifest(_ reference: String) throws -> [String: Any] {
            guard let url = URL(string: "\(ref.apiBase)/manifests/\(reference)") else {
                throw PullError(message: "bad manifest reference")
            }
            let (data, resp) = try httpGet(url, headers: auth)
            guard resp.statusCode == 200 else {
                throw PullError(message: "manifest fetch failed (HTTP \(resp.statusCode)) for \(image)")
            }
            if let digest = selectManifestDigest(indexJSON: data, arch: arch) {
                return try getManifest(digest) // index → platform manifest
            }
            guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw PullError(message: "manifest is not JSON")
            }
            if json["layers"] == nil, json["manifests"] != nil {
                throw PullError(message: "image has no \(arch) build — the sandbox guest is \(arch)-only")
            }
            return json
        }

        let manifest = try getManifest(ref.tag)
        guard let layers = manifest["layers"] as? [[String: Any]], !layers.isEmpty,
              let configDesc = manifest["config"] as? [String: Any],
              let configDigest = configDesc["digest"] as? String else {
            throw PullError(message: "manifest missing layers/config")
        }

        // Image config: env + workdir for the guest init script.
        guard let cfgURL = URL(string: "\(ref.apiBase)/blobs/\(configDigest)") else {
            throw PullError(message: "bad config digest")
        }
        let (cfgData, cfgResp) = try httpGet(cfgURL, headers: auth)
        guard cfgResp.statusCode == 200 else {
            throw PullError(message: "config blob fetch failed (HTTP \(cfgResp.statusCode))")
        }

        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        for (i, layer) in layers.enumerated() {
            guard let digest = layer["digest"] as? String,
                  let blobURL = URL(string: "\(ref.apiBase)/blobs/\(digest)") else {
                throw PullError(message: "layer \(i) missing digest")
            }
            let size = (layer["size"] as? Int).map { " (\($0 / (1024 * 1024)) MB)" } ?? ""
            log("layer \(i + 1)/\(layers.count)\(size)…")
            let file = try httpDownload(blobURL, headers: auth)
            defer { try? FileManager.default.removeItem(at: file) }
            try extractLayer(file, into: dir)
            try applyWhiteouts(in: dir)
        }

        try cfgData.write(to: dir.appendingPathComponent(configSidecarName))
        log("pull complete")
        return parseImageConfig(cfgData)
    }

    /// Unpack one layer tarball in-process (`TarReader`). This used to shell out
    /// to `/usr/bin/tar`, which is unreachable from inside the App Sandbox
    /// container and is a host escape from the Agent Sandbox.
    ///
    /// Compression is sniffed from the magic bytes rather than trusted from the
    /// manifest's `mediaType`. Zstd layers (`application/vnd.oci.image.layer.v1
    /// .tar+zstd`) are rejected loudly: the Compression framework has no zstd
    /// decoder, and silently mis-unpacking a rootfs is far worse than failing.
    static func extractLayer(_ tarball: URL, into dir: URL) throws {
        let raw = try Data(contentsOf: tarball, options: .mappedIfSafe)
        try TarReader.extract(try decompressLayer(raw), into: dir)
    }

    /// gzip (`1f 8b`) → inflate. zstd (`28 b5 2f fd`) → unsupported. Otherwise
    /// assume a plain tar stream.
    static func decompressLayer(_ raw: Data) throws -> Data {
        let magic = [UInt8](raw.prefix(4))
        if magic.count >= 2, magic[0] == 0x1f, magic[1] == 0x8b {
            return try gunzip(raw)
        }
        if magic.count >= 4, magic[0] == 0x28, magic[1] == 0xb5, magic[2] == 0x2f, magic[3] == 0xfd {
            throw PullError(message: "layer is zstd-compressed, which this client cannot decode — use a gzip-layer image")
        }
        return raw
    }
}
