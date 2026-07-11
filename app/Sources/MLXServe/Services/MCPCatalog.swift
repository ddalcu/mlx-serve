import Foundation

/// A required input for a server (env var or positional arg). Surfaced in the marketplace UI as a field.
struct MCPCatalogInput: Identifiable, Hashable {
    enum Kind: Hashable {
        case env(key: String)                                  // value goes into env[key] verbatim
        case arg(placeholder: String)                          // value replaces the placeholder in args
        case envEncoded(key: String, encoding: Encoding)       // value is transformed before being stored in env[key]
    }
    /// How to transform a raw user input before stuffing it into an env var. Used by servers that
    /// require non-trivial encodings (e.g. Azure DevOps PAT auth wants base64("dummy:<pat>")).
    enum Encoding: Hashable {
        /// base64(leftSide + ":" + value). Used by ADO PAT auth where the email portion is unused.
        case base64Pair(leftSide: String)
    }
    let id: String          // stable per-entry id (used by SwiftUI ForEach)
    let kind: Kind
    let label: String       // shown above the field
    let helpText: String?   // optional caption
    let isSecret: Bool      // SecureField vs TextField
    let required: Bool
    let placeholder: String?
    /// Args appended to the server command only when this input has a non-empty value.
    /// Lets a single input flip multiple things at once — e.g. ADO's PAT field both sets PERSONAL_ACCESS_TOKEN
    /// AND adds `--authentication pat` to args. When the field is blank, neither is added (default auth runs).
    let argsWhenPresent: [String]

    init(id: String, kind: Kind, label: String, helpText: String? = nil,
         isSecret: Bool, required: Bool, placeholder: String? = nil,
         argsWhenPresent: [String] = []) {
        self.id = id
        self.kind = kind
        self.label = label
        self.helpText = helpText
        self.isSecret = isSecret
        self.required = required
        self.placeholder = placeholder
        self.argsWhenPresent = argsWhenPresent
    }
}

/// One curated MCP server. The marketplace shows these by default; users can also add ad-hoc entries
/// by editing mcp.json directly.
struct MCPCatalogEntry: Identifiable, Hashable {
    let id: String           // stable id used as the key in mcp.json's "mcpServers" map
    let name: String         // display name
    let description: String  // 1-line caption in the marketplace
    let icon: String         // SF Symbol
    let command: String      // executable (typically "npx")
    let args: [String]       // base args; placeholders from `inputs` are spliced in by `materialize(...)`
    let inputs: [MCPCatalogInput]
    let notes: String?       // requirement note (e.g. "Requires Docker daemon"), shown under the row

    static func == (lhs: MCPCatalogEntry, rhs: MCPCatalogEntry) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }

    /// Build an MCPServerEntry from user-provided values. Empty values for non-required inputs are skipped.
    /// `values` keys are MCPCatalogInput.id.
    func materialize(values: [String: String]) -> MCPServerEntry {
        var env: [String: String] = [:]
        var finalArgs = args
        for input in inputs {
            let raw = values[input.id]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard !raw.isEmpty else { continue }
            switch input.kind {
            case .env(let key):
                env[key] = raw
            case .envEncoded(let key, let encoding):
                env[key] = Self.encode(raw, with: encoding)
            case .arg(let placeholder):
                finalArgs = finalArgs.map { $0 == placeholder ? raw : $0 }
                if !finalArgs.contains(raw) {
                    finalArgs.append(raw)
                }
            }
            // Append any conditional args declared by this input.
            finalArgs.append(contentsOf: input.argsWhenPresent)
        }
        return MCPServerEntry(
            command: command,
            args: finalArgs,
            env: env.isEmpty ? nil : env,
            disabled: false
        )
    }

    /// Reverse of materialize: read existing values out of an MCPServerEntry to pre-fill the marketplace UI.
    /// Encoded values (e.g. base64 PATs) are NOT reversed — the user re-enters them on edit so we don't
    /// risk surfacing a partially-decoded secret.
    func extractValues(from entry: MCPServerEntry) -> [String: String] {
        var out: [String: String] = [:]
        for input in inputs {
            switch input.kind {
            case .env(let key):
                if let v = entry.env?[key] { out[input.id] = v }
            case .envEncoded:
                continue   // Don't reverse-engineer encoded secrets back into the field.
            case .arg(let placeholder):
                // Find the placeholder slot in the catalog `args` and read the same index from `entry.args`.
                let entryArgs = entry.args ?? []
                if let idx = args.firstIndex(of: placeholder), entryArgs.indices.contains(idx) {
                    let v = entryArgs[idx]
                    if v != placeholder { out[input.id] = v }
                } else {
                    // Placeholder was appended (extra arg): try the last extra arg.
                    if entryArgs.count > args.count {
                        out[input.id] = entryArgs.last
                    }
                }
            }
        }
        return out
    }

    static func encode(_ raw: String, with encoding: MCPCatalogInput.Encoding) -> String {
        switch encoding {
        case .base64Pair(let left):
            return Data("\(left):\(raw)".utf8).base64EncodedString()
        }
    }
}

/// MCP servers whose executables are PRE-BAKED into the bundled guest image.
///
/// Why this exists (proven live): `npx -y <package>` inside the guest contacts
/// registry.npmjs.org even when the package is already installed — npx ignores
/// `npm -g` globals — so every MCP start DOWNLOADED CODE at runtime, which App
/// Review guideline 2.5.2 forbids. The baked bins run fine when invoked
/// directly (and with guest networking OFF). This table translates the npx form
/// the catalog/mcp.json authors into the direct-bin form the guest executes.
enum MCPGuestPrebaked {

    /// npm package → executable name inside the guest rootfs.
    ///
    /// DOCUMENTED DUPLICATION: must match what the `ddalcu/agent-shell-mlxserve`
    /// image bakes in (same convention as `isMediaModelType` ↔ `modalityFromType`).
    /// Adding a server = bake the bin into the image AND add a row here.
    static let prebakedGuestBins: [String: String] = [
        "@modelcontextprotocol/server-filesystem": "mcp-server-filesystem",
        "@modelcontextprotocol/server-github": "mcp-server-github",
    ]

    /// Pure translator: the direct guest invocation for (command, args), or nil
    /// when the server is not pre-baked. Handles three author spellings:
    /// the catalog's `npx -y <package> [args…]`, a version-pinned
    /// `npx <package>@<version>`, and the direct bin name itself.
    static func translate(command: String, args: [String]) -> (command: String, args: [String])? {
        if prebakedGuestBins.values.contains(command) { return (command, args) }
        guard command == "npx" || command.hasSuffix("/npx") else { return nil }
        // npx [-y|--yes|…] <package>[@version] [server args…]
        var rest = args[...]
        while let first = rest.first, first.hasPrefix("-") { rest = rest.dropFirst() }
        guard let token = rest.first, let bin = prebakedGuestBins[packageBaseName(token)] else {
            return nil
        }
        return (bin, Array(rest.dropFirst()))
    }

    /// "@scope/name@1.2.3" → "@scope/name"; "name@latest" → "name". The leading
    /// "@" of a scoped package is not a version separator.
    static func packageBaseName(_ token: String) -> String {
        guard let at = token[token.index(after: token.startIndex)...].lastIndex(of: "@") else {
            return token
        }
        return String(token[..<at])
    }

    /// What actually runs in the guest, by build policy.
    ///
    /// `prebakedOnly` (the Mac App Store build): a server outside the table is
    /// an explicit, honest error — never a silent npx download. Otherwise (the
    /// DMG build): pre-baked servers still translate (they then work with guest
    /// networking off), everything else runs as authored, as it always did.
    static func invocation(command: String, args: [String],
                           prebakedOnly: Bool) throws -> (command: String, args: [String]) {
        if let direct = translate(command: command, args: args) { return direct }
        guard !prebakedOnly else {
            throw MCPSpawnError.notPrebaked(command: ([command] + args).joined(separator: " "))
        }
        return (command, args)
    }
}

enum MCPCatalog {
    static let entries: [MCPCatalogEntry] = [
        .init(
            id: "github",
            name: "GitHub",
            description: "Repos, issues, PRs, code search via the GitHub API",
            icon: "chevron.left.forwardslash.chevron.right",
            command: "npx",
            args: ["-y", "@modelcontextprotocol/server-github"],
            inputs: [
                .init(id: "github_token",
                      kind: .env(key: "GITHUB_PERSONAL_ACCESS_TOKEN"),
                      label: "GitHub Personal Access Token",
                      helpText: "Create at github.com/settings/tokens (needs repo + read:org)",
                      isSecret: true,
                      required: true,
                      placeholder: "ghp_...")
            ],
            notes: nil
        ),
        .init(
            id: "azure-devops",
            name: "Azure DevOps",
            description: "Work items, pull requests, builds, pipelines, wiki, test plans",
            icon: "chart.bar.doc.horizontal",
            command: "npx",
            // Default args = interactive browser auth (no --authentication flag).
            // The PAT input below conditionally appends --authentication pat when filled.
            args: ["-y", "@azure-devops/mcp", "<ORG>"],
            inputs: [
                .init(id: "ado_org",
                      kind: .arg(placeholder: "<ORG>"),
                      label: "Organization Name",
                      helpText: "e.g. \"contoso\" (the part after dev.azure.com/)",
                      isSecret: false,
                      required: true,
                      placeholder: "contoso"),
                .init(id: "ado_pat",
                      // Optional. When filled: base64("x:<pat>") goes into PERSONAL_ACCESS_TOKEN AND
                      // --authentication pat is appended to args (via argsWhenPresent). When blank: server
                      // uses default interactive auth (browser login on first tool call).
                      kind: .envEncoded(key: "PERSONAL_ACCESS_TOKEN", encoding: .base64Pair(leftSide: "x")),
                      label: "Personal Access Token (PAT) — optional",
                      helpText: "Leave blank for browser login (recommended). Fill in only for headless / CI use. Create at dev.azure.com/<org>/_usersSettings/tokens.",
                      isSecret: true,
                      required: false,
                      placeholder: "Paste raw PAT for headless auth — we base64-encode it",
                      argsWhenPresent: ["--authentication", "pat"])
            ],
            notes: "Default: opens your browser on the first ADO tool call so you can log in with Microsoft. The token is cached for later runs. Fill in the PAT field instead if you can't use a browser (CI, sandboxed env)."
        ),
        .init(
            id: "dbhub",
            name: "DBHub (Universal SQL)",
            description: "Postgres, MySQL, SQLite, SQL Server — one MCP for all major SQL DBs (dbhub.ai)",
            icon: "cylinder.split.1x2",
            command: "npx",
            args: ["-y", "@bytebase/dbhub@latest", "--transport", "stdio", "--dsn", "<DSN>"],
            inputs: [
                .init(id: "dsn",
                      kind: .arg(placeholder: "<DSN>"),
                      label: "Connection String (DSN)",
                      helpText: "e.g. postgres://user:pass@host:5432/db, mysql://..., sqlite:///path/to/db.sqlite",
                      isSecret: true,
                      required: true,
                      placeholder: "postgres://user:pass@localhost:5432/mydb")
            ],
            notes: nil
        ),
        .init(
            id: "docker",
            name: "Docker",
            description: "Manage containers, images, networks, volumes",
            icon: "shippingbox",
            command: "npx",
            args: ["-y", "docker-mcp"],
            inputs: [],
            notes: "Requires the Docker daemon to be running locally."
        ),
        .init(
            id: "kubernetes",
            name: "Kubernetes",
            description: "Pods, deployments, services — uses your local kubeconfig",
            icon: "circle.hexagongrid",
            command: "npx",
            args: ["-y", "mcp-server-kubernetes"],
            inputs: [],
            notes: "Uses ~/.kube/config or KUBECONFIG. Set KUBECONFIG via mcp.json for non-default paths."
        ),
        .init(
            id: "playwright",
            name: "Playwright",
            description: "Browser automation: navigate, click, screenshot, scrape",
            icon: "safari",
            command: "npx",
            args: ["-y", "@playwright/mcp@latest"],
            inputs: [],
            notes: "First run downloads browser binaries (~few hundred MB)."
        ),
        .init(
            id: "slack",
            name: "Slack",
            description: "Read channels, post messages, search (Zencoder maintained fork)",
            icon: "bubble.left.and.bubble.right",
            command: "npx",
            args: ["-y", "@zencoderai/slack-mcp-server"],
            inputs: [
                .init(id: "slack_bot_token",
                      kind: .env(key: "SLACK_BOT_TOKEN"),
                      label: "Slack Bot Token",
                      helpText: "Bot user OAuth token (xoxb-...)",
                      isSecret: true,
                      required: true,
                      placeholder: "xoxb-..."),
                .init(id: "slack_team_id",
                      kind: .env(key: "SLACK_TEAM_ID"),
                      label: "Slack Team ID",
                      helpText: "Your workspace ID (T...)",
                      isSecret: false,
                      required: true,
                      placeholder: "T012ABCDEF")
            ],
            notes: nil
        ),
        .init(
            id: "notion",
            name: "Notion",
            description: "Read & write pages, databases, search",
            icon: "doc.text",
            command: "npx",
            args: ["-y", "@notionhq/notion-mcp-server"],
            inputs: [
                .init(id: "notion_token",
                      kind: .env(key: "OPENAPI_MCP_HEADERS"),
                      label: "Authorization Header",
                      helpText: #"Paste: {"Authorization": "Bearer secret_...", "Notion-Version": "2022-06-28"}"#,
                      isSecret: true,
                      required: true,
                      placeholder: #"{"Authorization": "Bearer secret_xxx", "Notion-Version": "2022-06-28"}"#)
            ],
            notes: "Create an integration at notion.so/my-integrations and share pages with it."
        ),
        .init(
            id: "filesystem",
            name: "Filesystem",
            description: "Read, write, list files within a chosen root directory",
            icon: "folder",
            command: "npx",
            args: ["-y", "@modelcontextprotocol/server-filesystem", "<ROOT>"],
            inputs: [
                .init(id: "root_path",
                      kind: .arg(placeholder: "<ROOT>"),
                      label: "Root Directory",
                      helpText: "Tools are confined to this path. Use an absolute path. Suggestion: ~/.mlx-serve/workspace (the default cwd for all MCP servers).",
                      isSecret: false,
                      required: true,
                      placeholder: NSString(string: "~/.mlx-serve/workspace").expandingTildeInPath)
            ],
            notes: nil
        ),
        .init(
            id: "shell",
            name: "Shell",
            description: "Run arbitrary shell commands (use with caution)",
            icon: "terminal",
            command: "npx",
            args: ["-y", "@mkusaka/mcp-shell-server"],
            inputs: [],
            notes: "Grants the model arbitrary command execution. Only enable if you trust the model."
        ),
    ]

    /// Look up a catalog entry by ID. Returns nil if the server is custom (not in the curated list).
    static func entry(for id: String) -> MCPCatalogEntry? {
        entries.first { $0.id == id }
    }

    /// The entries the marketplace may OFFER. Under the pre-baked-only policy
    /// (Mac App Store build) only servers whose bins ride the bundled guest
    /// image are shown — offering one that would error at spawn with
    /// "not in the bundled image" is a dead toggle.
    static func visibleEntries(
        prebakedOnly: Bool = BuildFeatures.current.guest.mcpServers == .prebaked
    ) -> [MCPCatalogEntry] {
        guard prebakedOnly else { return entries }
        return entries.filter { MCPGuestPrebaked.translate(command: $0.command, args: $0.args) != nil }
    }

    /// `entry(for:)` restricted to the visible set. A configured catalog server
    /// that is HIDDEN in this build (e.g. playwright carried over from a DMG
    /// install) then renders in the marketplace's custom section instead of
    /// disappearing while still erroring at spawn.
    static func visibleEntry(
        for id: String,
        prebakedOnly: Bool = BuildFeatures.current.guest.mcpServers == .prebaked
    ) -> MCPCatalogEntry? {
        visibleEntries(prebakedOnly: prebakedOnly).first { $0.id == id }
    }
}
