const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const File = Io.File;
const Dir = Io.Dir;

pub const separator: u32 = 1 << 30;
pub const max_order: u32 = 64;
pub const max_vote_nodes: usize = 128;
pub const max_depth: usize = 32;
pub const max_candidates: usize = 64;
pub const persistent_capacity: usize = 262144;
pub const corpus_capacity: usize = 16 * 1024 * 1024;
const magic = "MLXLOOK1";
const header_size = 40;

// Follow the host's private-file convention: POSIX owner-only modes; Windows
// inherits the containing directory ACL, which must already protect user data.
const store_file_permissions: File.Permissions = if (builtin.os.tag == .windows) .default_file else .fromMode(0o600);
const store_dir_permissions: File.Permissions = if (builtin.os.tag == .windows) .default_dir else .fromMode(0o700);

const VoteNode = struct {
    token: u32,
    parent: ?u32,
};
pub const Look = struct {
    tokens: []const u32,
    order: u32,
};

/// Owned bounded-prefix suffix index. Tokens include document separators.
/// Borrow only through *const Corpus; the owner must outlive every borrower.
pub const Corpus = struct {
    allocator: Allocator,
    tokens: []const u32,
    positions: []const u32,
    max_order: u32,

    pub fn build(allocator: Allocator, ids: []const u32, order: u32) !Corpus {
        try validateOrder(order);
        if (ids.len > corpus_capacity) return error.CorpusTooLarge;
        for (ids) |id| if (id > separator) return error.InvalidToken;
        const tokens = try allocator.dupe(u32, ids);
        errdefer allocator.free(tokens);
        return fromOwnedTokens(allocator, tokens, order);
    }

    fn fromOwnedTokens(allocator: Allocator, tokens: []u32, order: u32) !Corpus {
        const positions = try allocator.alloc(u32, tokens.len);
        errdefer allocator.free(positions);
        for (positions, 0..) |*position, i| position.* = @intCast(i);
        const context = SortContext{ .tokens = tokens, .order = order };
        std.mem.sort(u32, positions, context, SortContext.less);
        return .{ .allocator = allocator, .tokens = tokens, .positions = positions, .max_order = order };
    }

    pub fn deinit(self: *Corpus) void {
        self.allocator.free(self.tokens);
        self.allocator.free(self.positions);
        self.* = undefined;
    }

    fn lowerBound(self: *const Corpus, query: []const u32, upper: bool) usize {
        var lo: usize = 0;
        var hi = self.positions.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const cmp = compareQuery(self.tokens, self.positions[mid], query);
            if (cmp == .lt or (upper and cmp == .eq)) lo = mid + 1 else hi = mid;
        }
        return lo;
    }
};

const SortContext = struct {
    tokens: []const u32,
    order: u32,
    fn less(self: SortContext, a: u32, b: u32) bool {
        const left = self.tokens[a..@min(self.tokens.len, @as(usize, a) + self.order)];
        const right = self.tokens[b..@min(self.tokens.len, @as(usize, b) + self.order)];
        return std.mem.order(u32, left, right) == .lt;
    }
};

fn validateOrder(order: u32) !void {
    if (order == 0 or order > max_order) return error.InvalidOrder;
}

fn compareQuery(tokens: []const u32, start: usize, query: []const u32) std.math.Order {
    const available = @min(query.len, tokens.len - start);
    const cmp = std.mem.order(u32, tokens[start..][0..available], query[0..available]);
    if (cmp != .eq) return cmp;
    return if (available < query.len) .lt else .eq;
}

/// Request-owned proposal scratch. corpus/persistent are borrowed
/// immutable indices; their cache leases must outlive this object.
pub const Drafter = struct {
    allocator: Allocator,
    vocab_limit: u32 = separator,
    corpus: ?*const Corpus = null,
    persistent: ?*const Corpus = null,
    vote_nodes: [max_vote_nodes]VoteNode = undefined,
    counts: [max_vote_nodes]u32 = undefined,
    node_count: usize = 0,
    observations: u32 = 0,

    pub fn init(allocator: Allocator) Drafter {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Drafter) void {
        self.* = undefined;
    }

    fn addContinuation(self: *Drafter, ids: []const u32) void {
        if (ids.len == 0) return;
        self.observations += 1;
        var parent: ?u32 = null;
        for (ids) |token| {
            var found: ?usize = null;
            for (self.vote_nodes[0..self.node_count], 0..) |node, i| {
                if (node.parent == parent and node.token == token) {
                    found = i;
                    break;
                }
            }
            const index = found orelse blk: {
                if (self.node_count == max_vote_nodes) return;
                const added = self.node_count;
                self.node_count += 1;
                self.vote_nodes[added] = .{ .token = token, .parent = parent };
                self.counts[added] = 0;
                break :blk added;
            };
            self.counts[index] += 1;
            parent = @intCast(index);
        }
    }

    fn collectCorpus(self: *Drafter, corpus: *const Corpus, query: []const u32, depth: usize) void {
        if (query.len > corpus.max_order) return;
        const first = corpus.lowerBound(query, false);
        const end = corpus.lowerBound(query, true);
        const examined = @min(end - first, max_candidates);
        // Evenly sample an oversized interval rather than biasing its first page.
        for (0..examined) |sample| {
            const index = first + sample * (end - first) / examined;
            const start = @as(usize, corpus.positions[index]) + query.len;
            var len: usize = 0;
            while (len < depth and start + len < corpus.tokens.len) : (len += 1) {
                if (corpus.tokens[start + len] >= @min(self.vocab_limit, separator)) break;
            }
            self.addContinuation(corpus.tokens[start..][0..len]);
        }
    }

    pub fn look(self: *Drafter, context: []const u32, depth: usize, out: *std.ArrayList(u32)) !Look {
        out.clearRetainingCapacity();
        self.node_count = 0;
        self.observations = 0;
        var order: usize = @min(context.len, max_order);
        // Query suffixes never include a separator or an invalid vocabulary ID.
        var valid_suffix: usize = 0;
        while (valid_suffix < order and context[context.len - valid_suffix - 1] < @min(self.vocab_limit, separator)) : (valid_suffix += 1) {}
        order = valid_suffix;
        const bounded_depth = @min(depth, max_depth);
        if (bounded_depth == 0) order = 0;
        while (order != 0) : (order -= 1) {
            const query = context[context.len - order..];
            if (self.corpus) |corpus| self.collectCorpus(corpus, query, bounded_depth);
            if (self.persistent) |persistent| self.collectCorpus(persistent, query, bounded_depth);
            // An occurrence without a usable continuation is not evidence.
            if (self.observations != 0) break;
        }
        if (self.node_count != 0) try out.ensureTotalCapacity(self.allocator, bounded_depth);
        var parent: ?u32 = null;
        while (out.items.len < bounded_depth) {
            var best: ?usize = null;
            for (self.vote_nodes[0..self.node_count], 0..) |node, i| {
                if (node.parent != parent) continue;
                if (best == null or self.counts[i] > self.counts[best.?] or
                    (self.counts[i] == self.counts[best.?] and node.token < self.vote_nodes[best.?].token)) best = i;
            }
            const index = best orelse break;
            const node = self.vote_nodes[index];
            out.appendAssumeCapacity(node.token);
            parent = @intCast(index);
        }
        return .{ .tokens = out.items, .order = @intCast(order) };
    }
};

// NPY header grammar: the three documented dictionary fields, in any order,
// either quote style, a one-element shape tuple, no Python evaluation.
// https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
const NpyHeader = struct {
    text: []const u8,
    cursor: usize = 0,
    fn spaces(self: *NpyHeader) void {
        while (self.cursor < self.text.len and std.ascii.isWhitespace(self.text[self.cursor])) self.cursor += 1;
    }
    fn take(self: *NpyHeader, byte: u8) !void {
        self.spaces();
        if (self.cursor == self.text.len or self.text[self.cursor] != byte) return error.InvalidNpy;
        self.cursor += 1;
    }
    fn quoted(self: *NpyHeader) ![]const u8 {
        self.spaces();
        if (self.cursor == self.text.len) return error.InvalidNpy;
        const quote = self.text[self.cursor];
        if (quote != '\'' and quote != '"') return error.InvalidNpy;
        self.cursor += 1;
        const start = self.cursor;
        while (self.cursor < self.text.len and self.text[self.cursor] != quote) : (self.cursor += 1) {
            if (self.text[self.cursor] == '\\' or self.text[self.cursor] < 32) return error.InvalidNpy;
        }
        if (self.cursor == self.text.len) return error.InvalidNpy;
        const result = self.text[start..self.cursor];
        self.cursor += 1;
        return result;
    }
    fn literal(self: *NpyHeader, value: []const u8) bool {
        self.spaces();
        if (!std.mem.startsWith(u8, self.text[self.cursor..], value)) return false;
        self.cursor += value.len;
        return true;
    }
    fn parse(self: *NpyHeader) !usize {
        try self.take('{');
        var seen: u3 = 0;
        var count: usize = 0;
        while (true) {
            const key = try self.quoted();
            try self.take(':');
            const bit: u3 = if (std.mem.eql(u8, key, "descr")) 1 else if (std.mem.eql(u8, key, "fortran_order")) 2 else if (std.mem.eql(u8, key, "shape")) 4 else return error.InvalidNpy;
            if (seen & bit != 0) return error.InvalidNpy;
            seen |= bit;
            switch (bit) {
                1 => if (!std.mem.eql(u8, try self.quoted(), "<i4")) return error.UnsupportedNpyType,
                2 => if (!self.literal("False") and !self.literal("True")) return error.InvalidNpy,
                4 => {
                    try self.take('(');
                    self.spaces();
                    const start = self.cursor;
                    while (self.cursor < self.text.len and std.ascii.isDigit(self.text[self.cursor])) self.cursor += 1;
                    count = std.fmt.parseInt(usize, self.text[start..self.cursor], 10) catch return error.InvalidNpy;
                    try self.take(',');
                    try self.take(')');
                },
                else => unreachable,
            }
            self.spaces();
            if (self.cursor < self.text.len and self.text[self.cursor] == ',') {
                self.cursor += 1;
                self.spaces();
                if (self.cursor < self.text.len and self.text[self.cursor] != '}') continue;
            }
            try self.take('}');
            self.spaces();
            if (seen != 7 or self.cursor != self.text.len) return error.InvalidNpy;
            return count;
        }
    }
};

const NpyLayout = struct {
    data_offset: usize,
    count: usize,

    fn offset(prefix: []const u8) !usize {
        if (prefix.len < 10 or !std.mem.eql(u8, prefix[0..6], "\x93NUMPY") or prefix[7] != 0) return error.InvalidNpy;
        const prefix_len: usize = switch (prefix[6]) { 1 => 10, 2, 3 => 12, else => return error.UnsupportedNpyVersion };
        if (prefix.len < prefix_len) return error.InvalidNpy;
        const length: usize = if (prefix_len == 10) std.mem.readInt(u16, prefix[8..10], .little) else std.mem.readInt(u32, prefix[8..12], .little);
        if (length == 0 or length > 65536) return error.InvalidNpy;
        return prefix_len + length;
    }

    fn parse(header: []const u8, file_size: u64) !NpyLayout {
        const end = try offset(header);
        const prefix_len: usize = if (header[6] == 1) 10 else 12;
        // Older writers aligned to 16 bytes; current writers align to 64.
        if (end > header.len or end > file_size or end % 16 != 0 or header[end - 1] != '\n') return error.InvalidNpy;
        var parser = NpyHeader{ .text = header[prefix_len..end] };
        const count = try parser.parse();
        if (count > corpus_capacity or count != (file_size - end) / 4 or (file_size - end) % 4 != 0) return error.InvalidNpy;
        return .{ .data_offset = end, .count = count };
    }
};

fn littleToNative(tokens: []u32) void {
    if (@import("builtin").cpu.arch.endian() == .big) {
        for (tokens) |*token| token.* = @byteSwap(token.*);
    }
}

/// Reads directly into the retained aligned array, with no whole-file byte
/// staging allocation. Hashing, when requested, covers header plus raw payload.
fn readNpy(dir: Dir, io: Io, allocator: Allocator, name: []const u8, expected: ?[]const u8) ![]u32 {
    const file = try dir.openFile(io, name, .{ .follow_symlinks = false, .allow_directory = false });
    defer file.close(io);
    const before = try file.stat(io);
    if (before.kind != .file or before.nlink != 1) return error.UnsafeFile;
    if (before.size > corpus_capacity * 4 + 65548) return error.FileTooLarge;
    var header: [65548]u8 = undefined;
    if (try file.readPositionalAll(io, header[0..12], 0) != 12) return error.InvalidNpy;
    const end = try NpyLayout.offset(header[0..12]);
    if (end < 12 or end > before.size) return error.InvalidNpy;
    if (try file.readPositionalAll(io, header[12..end], 12) != end - 12) return error.TruncatedFile;
    const layout = try NpyLayout.parse(header[0..end], before.size);
    const values = try allocator.alloc(u32, layout.count);
    errdefer allocator.free(values);
    const bytes = std.mem.sliceAsBytes(values);
    if (try file.readPositionalAll(io, bytes, end) != bytes.len) return error.TruncatedFile;
    if (expected) |text| {
        const wanted = try decodeDigest(text);
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update(header[0..end]);
        hash.update(bytes);
        var actual: [32]u8 = undefined;
        hash.final(&actual);
        if (!std.mem.eql(u8, &wanted, &actual)) return error.FingerprintMismatch;
    }
    littleToNative(values);
    for (values) |value| if (value > std.math.maxInt(i32)) return error.InvalidNpy;
    if (!Stamp.eql(Stamp.from(before), Stamp.from(try file.stat(io))) or
        !Stamp.eql(Stamp.from(before), try fileStamp(dir, io, name))) return error.ChangedDuringRead;
    return values;
}

const Stamp = struct {
    inode: File.INode,
    size: u64,
    modified: i96,
    changed: i96,
    fn from(stat: File.Stat) Stamp {
        return .{ .inode = stat.inode, .size = stat.size, .modified = stat.mtime.nanoseconds, .changed = stat.ctime.nanoseconds };
    }
    fn eql(a: Stamp, b: Stamp) bool {
        return a.inode == b.inode and a.size == b.size and a.modified == b.modified and a.changed == b.changed;
    }
};

fn fileStamp(dir: Dir, io: Io, name: []const u8) !Stamp {
    const stat = try dir.statFile(io, name, .{ .follow_symlinks = false });
    if (stat.kind != .file or stat.nlink != 1) return error.UnsafeFile;
    return Stamp.from(stat);
}

fn readBounded(dir: Dir, io: Io, allocator: Allocator, name: []const u8, limit: usize) ![]u8 {
    const file = try dir.openFile(io, name, .{ .follow_symlinks = false, .allow_directory = false });
    defer file.close(io);
    const before = try file.stat(io);
    if (before.kind != .file or before.nlink != 1) return error.UnsafeFile;
    if (before.size > limit) return error.FileTooLarge;
    const bytes = try allocator.alloc(u8, @intCast(before.size));
    errdefer allocator.free(bytes);
    if (try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.TruncatedFile;
    if (!Stamp.eql(Stamp.from(before), Stamp.from(try file.stat(io))) or
        !Stamp.eql(Stamp.from(before), try fileStamp(dir, io, name))) return error.ChangedDuringRead;
    return bytes;
}

const CorpusStamp = struct {
    directory: File.INode,
    metadata: Stamp,
    tokens: Stamp,
    positions: Stamp,
    fn read(io: Io, path: []const u8) !CorpusStamp {
        const dir = try Dir.cwd().openDir(io, path, .{});
        defer dir.close(io);
        return .{
            .directory = (try dir.stat(io)).inode,
            .metadata = try fileStamp(dir, io, "meta.json"),
            .tokens = try fileStamp(dir, io, "tokens.npy"),
            .positions = try fileStamp(dir, io, "sa.npy"),
        };
    }
    fn eql(a: CorpusStamp, b: CorpusStamp) bool {
        return a.directory == b.directory and Stamp.eql(a.metadata, b.metadata) and Stamp.eql(a.tokens, b.tokens) and Stamp.eql(a.positions, b.positions);
    }
};

const Metadata = struct {
    tokenizer_sha256: []const u8,
    max_order: u32,
    doc_sep: u32,
    n_tokens: ?usize = null,
    documents: ?usize = null,
    tokens_sha256: ?[]const u8 = null,
    sa_sha256: ?[]const u8 = null,
};

fn decodeDigest(text: []const u8) ![32]u8 {
    if (text.len != 64) return error.InvalidFingerprint;
    var digest: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&digest, text) catch return error.InvalidFingerprint;
    return digest;
}


fn validateDocuments(tokens: []const u32, require_complete: bool) !usize {
    var documents: usize = 0;
    var length: usize = 0;
    for (tokens) |token| {
        if (token > separator) return error.InvalidToken;
        if (token == separator) {
            if (length == 0) return error.EmptyDocument;
            documents += 1;
            length = 0;
        } else length += 1;
    }
    if (require_complete and length != 0) return error.IncompleteDocument;
    return documents;
}

/// Missing corpus directory is optional. A present but malformed corpus errors;
/// missing tokenizer identity refuses the tier rather than trusting token IDs.
pub fn openCorpus(io: Io, allocator: Allocator, path: []const u8, tokenizer_sha: ?[32]u8) !?Corpus {
    const wanted = tokenizer_sha orelse return error.TokenizerRequired;
    const dir = Dir.cwd().openDir(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer dir.close(io);
    const before = try CorpusStamp.read(io, path);
    const metadata_bytes = try readBounded(dir, io, allocator, "meta.json", 65536);
    defer allocator.free(metadata_bytes);
    const parsed = try std.json.parseFromSlice(Metadata, allocator, metadata_bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const metadata = parsed.value;
    if (metadata.doc_sep != separator) return error.InvalidSeparator;
    try validateOrder(metadata.max_order);
    const identity = try decodeDigest(metadata.tokenizer_sha256);
    if (!std.mem.eql(u8, &wanted, &identity)) return error.TokenizerMismatch;
    const tokens = try readNpy(dir, io, allocator, "tokens.npy", metadata.tokens_sha256);
    errdefer allocator.free(tokens);
    const documents = try validateDocuments(tokens, true);
    if (metadata.n_tokens) |count| if (tokens.len != count) return error.InvalidMetadata;
    if (metadata.documents) |count| if (documents != count) return error.InvalidMetadata;
    const positions = try readNpy(dir, io, allocator, "sa.npy", metadata.sa_sha256);
    errdefer allocator.free(positions);
    if (positions.len != tokens.len) return error.InvalidSuffixIndex;
    const seen = try allocator.alloc(bool, tokens.len);
    defer allocator.free(seen);
    @memset(seen, false);
    const sort = SortContext{ .tokens = tokens, .order = metadata.max_order };
    for (positions, 0..) |position, i| {
        if (position >= tokens.len or seen[position]) return error.InvalidSuffixIndex;
        seen[position] = true;
        if (i != 0 and sort.less(position, positions[i - 1])) return error.InvalidSuffixIndex;
    }
    if (!CorpusStamp.eql(before, try CorpusStamp.read(io, path))) return error.ChangedDuringRead;
    return .{ .allocator = allocator, .tokens = tokens, .positions = positions, .max_order = metadata.max_order };
}

const StoreGuard = struct {
    allocator: Allocator,
    io: Io,
    path: []u8,
    dir: Dir,
    lock: File,
    directory_inode: File.INode,
    lock_stamp: Stamp,

    fn open(io: Io, allocator: Allocator, path: []const u8, read_only: bool) !StoreGuard {
        const dir = if (read_only)
            try Dir.cwd().openDir(io, path, .{})
        else
            try Dir.cwd().createDirPathOpen(io, path, .{
                .permissions = store_dir_permissions, .open_options = .{ .iterate = true },
            });
        errdefer dir.close(io);
        if (!read_only and builtin.os.tag != .windows and
            (try dir.stat(io)).permissions.toMode() & 0o7777 != 0o700)
            try dir.setPermissions(io, store_dir_permissions);
        const lock = dir.openFile(io, "store.lock", .{
            .mode = if (read_only) .read_only else .read_write,
            .follow_symlinks = false, .allow_directory = false,
        }) catch |err| blk: {
            if (err != error.FileNotFound or read_only) return err;
            break :blk dir.createFile(io, "store.lock", .{
                .read = true, .exclusive = true, .permissions = store_file_permissions,
            }) catch |create_err| switch (create_err) {
                error.PathAlreadyExists => try dir.openFile(io, "store.lock", .{
                    .mode = .read_write, .follow_symlinks = false, .allow_directory = false,
                }),
                else => return create_err,
            };
        };
        errdefer lock.close(io);
        if (!try lock.tryLock(io, if (read_only) .shared else .exclusive)) return error.StoreBusy;
        errdefer lock.unlock(io);
        const original_stat = try lock.stat(io);
        if (original_stat.kind != .file or original_stat.nlink != 1 or original_stat.size != 0) return error.UnsafeLockFile;
        if (!read_only and builtin.os.tag != .windows and original_stat.permissions.toMode() & 0o7777 != 0o600)
            try lock.setPermissions(io, store_file_permissions);
        const stat = try lock.stat(io);
        const owned_path = try allocator.dupe(u8, path);
        errdefer allocator.free(owned_path);
        var guard = StoreGuard{
            .allocator = allocator, .io = io, .path = owned_path, .dir = dir,
            .lock = lock, .directory_inode = (try dir.stat(io)).inode,
            .lock_stamp = Stamp.from(stat),
        };
        try guard.validate();
        return guard;
    }

    fn validate(self: *const StoreGuard) !void {
        const named_directory = try Dir.cwd().statFile(self.io, self.path, .{});
        if (named_directory.kind != .directory or named_directory.inode != self.directory_inode or
            !Stamp.eql(self.lock_stamp, try fileStamp(self.dir, self.io, "store.lock")))
            return error.StoreChanged;
    }

    fn deinit(self: *StoreGuard) void {
        self.lock.unlock(self.io);
        self.lock.close(self.io);
        self.dir.close(self.io);
        self.allocator.free(self.path);
        self.* = undefined;
    }
};

fn makeHeader(identity: [32]u8) [header_size]u8 {
    var header: [header_size]u8 = undefined;
    @memcpy(header[0..8], magic);
    @memcpy(header[8..], &identity);
    return header;
}

fn checkHeader(bytes: []const u8, identity: [32]u8) !void {
    if (bytes.len < header_size or !std.mem.eql(u8, bytes[0..8], magic)) return error.InvalidStoreHeader;
    if (!std.mem.eql(u8, bytes[8..header_size], &identity)) return error.TokenizerMismatch;
}


fn readPersistentTokens(guard: *const StoreGuard, file: File, allocator: Allocator, identity: [32]u8) ![]u32 {
    const before = try file.stat(guard.io);
    if (before.kind != .file or before.nlink != 1) return error.UnsafeFile;
    if (before.size < header_size) return error.InvalidStoreHeader;
    if (before.size > header_size + persistent_capacity * 4) return error.FileTooLarge;
    var header: [header_size]u8 = undefined;
    if (try file.readPositionalAll(guard.io, &header, 0) != header.len) return error.InvalidStoreHeader;
    try checkHeader(&header, identity);
    if ((before.size - header_size) % 4 != 0) return error.TruncatedFile;
    const tokens = try allocator.alloc(u32, @intCast((before.size - header_size) / 4));
    errdefer allocator.free(tokens);
    const bytes = std.mem.sliceAsBytes(tokens);
    if (try file.readPositionalAll(guard.io, bytes, header_size) != bytes.len) return error.TruncatedFile;
    littleToNative(tokens);
    _ = try validateDocuments(tokens, true);
    if (!Stamp.eql(Stamp.from(before), Stamp.from(try file.stat(guard.io))) or
        !Stamp.eql(Stamp.from(before), try fileStamp(guard.dir, guard.io, "tokens.bin"))) return error.StoreChanged;
    try guard.validate();
    return tokens;
}

fn writeTokens(file: File, io: Io, ids: []const u32, start: u64) !void {
    if (@import("builtin").cpu.arch.endian() == .little) {
        return file.writePositionalAll(io, std.mem.sliceAsBytes(ids), start);
    }
    var buffer: [4096]u8 = undefined;
    var cursor: usize = 0;
    while (cursor < ids.len) {
        const count: usize = @min(buffer.len / 4, ids.len - cursor);
        for (ids[cursor..][0..count], 0..) |token, i| std.mem.writeInt(u32, buffer[i * 4..][0..4], token, .little);
        try file.writePositionalAll(io, buffer[0 .. count * 4], start + cursor * 4);
        cursor += count;
    }
}

fn syncDirectory(dir: Dir, io: Io) !void {
    // A borrowed directory descriptor: never close it through the File view.
    const file = File{ .handle = dir.handle, .flags = .{ .nonblocking = false } };
    try file.sync(io);
}

const pending_log = ".tokens.pending";

fn recoverPending(guard: *const StoreGuard) !void {
    try guard.validate();
    guard.dir.deleteFile(guard.io, pending_log) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    try syncDirectory(guard.dir, guard.io);
}

/// New inode becomes visible only after its complete contents are durable.
fn replaceLog(guard: *const StoreGuard, identity: [32]u8, parts: []const []const u32) !File {
    var committed = false;
    try recoverPending(guard);
    const name = pending_log;
    const file = try guard.dir.createFile(guard.io, name, .{
        .read = true, .exclusive = true, .permissions = store_file_permissions,
    });
    errdefer file.close(guard.io);
    defer if (!committed) guard.dir.deleteFile(guard.io, name) catch {};
    if (builtin.os.tag != .windows) try file.setPermissions(guard.io, store_file_permissions);
    try file.writePositionalAll(guard.io, &makeHeader(identity), 0);
    var offset: u64 = header_size;
    for (parts) |tokens| {
        try writeTokens(file, guard.io, tokens, offset);
        offset += tokens.len * 4;
    }
    try file.sync(guard.io);
    try guard.validate();
    try guard.dir.rename(name, guard.dir, "tokens.bin", guard.io);
    committed = true;
    try syncDirectory(guard.dir, guard.io);
    return file;
}

/// Scoped current-log transaction, never a request-owned writer or suffix index.
const AppendTransaction = struct {
    allocator: Allocator,
    guard: StoreGuard,
    file: File,
    stamp: Stamp,
    tokens: []u32,
    identity: [32]u8,

    fn open(io: Io, allocator: Allocator, path: []const u8, identity: [32]u8) !AppendTransaction {
        var guard = try StoreGuard.open(io, allocator, path, false);
        errdefer guard.deinit();
        const file = guard.dir.openFile(io, "tokens.bin", .{
            .mode = .read_write, .follow_symlinks = false, .allow_directory = false,
        }) catch |err| blk: {
            if (err != error.FileNotFound) return err;
            break :blk try replaceLog(&guard, identity, &.{});
        };
        errdefer file.close(io);
        const tokens = try readPersistentTokens(&guard, file, allocator, identity);
        errdefer allocator.free(tokens);
        if (builtin.os.tag != .windows and (try file.stat(io)).permissions.toMode() & 0o7777 != 0o600)
            try file.setPermissions(io, store_file_permissions);
        const stamp = Stamp.from(try file.stat(io));
        if (!Stamp.eql(stamp, try fileStamp(guard.dir, io, "tokens.bin"))) return error.StoreChanged;
        try recoverPending(&guard);
        return .{ .allocator = allocator, .guard = guard, .file = file,
            .stamp = stamp, .tokens = tokens, .identity = identity };
    }

    fn deinit(self: *AppendTransaction) void {
        self.allocator.free(self.tokens);
        self.file.close(self.guard.io);
        self.guard.deinit();
        self.* = undefined;
    }

    fn append(self: *AppendTransaction, prompt: []const u32, output: []const u32) !void {
        try self.guard.validate();
        const io = self.guard.io;
        if (!Stamp.eql(self.stamp, try fileStamp(self.guard.dir, io, "tokens.bin")) or
            !Stamp.eql(self.stamp, Stamp.from(try self.file.stat(io)))) return error.StoreChanged;
        const document_size = prompt.len + output.len + 1;
        if (document_size == 1) return;
        var discard: usize = 0;
        while (self.tokens.len - discard + document_size > persistent_capacity) {
            while (self.tokens[discard] != separator) discard += 1;
            discard += 1;
        }
        if (discard != 0) {
            const replacement = try replaceLog(&self.guard, self.identity,
                &.{ self.tokens[discard..], prompt, output, &.{separator} });
            replacement.close(io);
        } else {
            self.appendBytes(prompt, output) catch |err| {
                self.file.setLength(io, self.stamp.size) catch return error.RollbackFailed;
                self.file.sync(io) catch return error.RollbackFailed;
                return err;
            };
        }
    }

    fn appendBytes(self: *AppendTransaction, prompt: []const u32, output: []const u32) !void {
        const io = self.guard.io;
        try writeTokens(self.file, io, prompt, self.stamp.size);
        try writeTokens(self.file, io, output, self.stamp.size + prompt.len * 4);
        try writeTokens(self.file, io, &.{separator}, self.stamp.size + (prompt.len + output.len) * 4);
        try self.file.sync(io);
    }
};

/// Creates an empty private store, or validates an existing compatible one.
/// Releases the stable lock before returning; builds no suffix index.
pub fn initializePersistent(io: Io, allocator: Allocator, path: []const u8, identity: [32]u8) !void {
    var transaction = try AppendTransaction.open(io, allocator, path, identity);
    defer transaction.deinit();
}

/// Only the exact final published output belongs here. Both slices form one
/// document. StoreBusy and durability errors must reach the caller, not be ignored.
pub fn appendFinalTurn(io: Io, allocator: Allocator, path: []const u8, identity: [32]u8, prompt: []const u32, output: []const u32) !void {
    try appendTransaction(io, allocator, path, identity, prompt, output);
}

fn appendTransaction(io: Io, allocator: Allocator, path: []const u8, identity: [32]u8, prompt: []const u32, output: []const u32) !void {
    if (prompt.len >= persistent_capacity or output.len >= persistent_capacity - prompt.len) return error.DocumentTooLarge;
    for (prompt) |token| if (token >= separator) return error.InvalidToken;
    for (output) |token| if (token >= separator) return error.InvalidToken;
    var transaction = try AppendTransaction.open(io, allocator, path, identity);
    defer transaction.deinit();
    try transaction.append(prompt, output);
}

const CachedCorpus = struct {
    index: Corpus,
    path: []u8,
    identity: [32]u8,
    stamp: CorpusStamp,
    references: usize = 1,

    fn release(self: *CachedCorpus) void {
        std.debug.assert(self.references != 0);
        self.references -= 1;
        if (self.references != 0) return;
        const allocator = self.index.allocator;
        allocator.free(self.path);
        self.index.deinit();
        allocator.destroy(self);
    }
};

const CachedPersistent = struct {
    index: Corpus,
    path: []u8,
    identity: [32]u8,
    stamp: Stamp,
    lock_stamp: Stamp,
    directory_inode: File.INode,
    references: usize = 1,

    fn release(self: *CachedPersistent) void {
        std.debug.assert(self.references != 0);
        self.references -= 1;
        if (self.references != 0) return;
        const allocator = self.index.allocator;
        allocator.free(self.path);
        self.index.deinit();
        allocator.destroy(self);
    }
};

/// Single-inference-thread ownership: one current snapshot per tier, plus only
/// retired snapshots still leased. Leases are move-only and hold no OS handles.
/// Configured paths must belong to one shared trust boundary; identity is not ACL.
pub const HistoryCache = struct {
    allocator: Allocator,
    corpus_entry: ?*CachedCorpus = null,
    persistent_entry: ?*CachedPersistent = null,

    const Snapshot = struct {
        tokens: []u32,
        stamp: Stamp,
        lock_stamp: Stamp,
        directory_inode: File.INode,
    };

    pub fn init(allocator: Allocator) HistoryCache {
        return .{ .allocator = allocator };
    }

    /// Borrowers retain their immutable snapshots, even after cache destruction.
    pub fn clear(self: *HistoryCache) !void {
        if (self.corpus_entry) |entry| entry.release();
        if (self.persistent_entry) |entry| entry.release();
        self.corpus_entry = null;
        self.persistent_entry = null;
    }

    pub fn deinit(self: *HistoryCache) !void {
        try self.clear();
        self.* = undefined;
    }

    pub fn acquireCorpus(self: *HistoryCache, io: Io, path: []const u8, identity: ?[32]u8) !?CorpusLease {
        const wanted = identity orelse return error.TokenizerRequired;
        const stamp = CorpusStamp.read(io, path) catch |err| switch (err) {
            error.FileNotFound => {
                // Only an absent directory is optional, not missing corpus files.
                const dir = Dir.cwd().openDir(io, path, .{}) catch |dir_err| switch (dir_err) {
                    error.FileNotFound => return null,
                    else => return dir_err,
                };
                dir.close(io);
                return err;
            },
            else => return err,
        };
        if (self.corpus_entry) |entry| {
            if (std.mem.eql(u8, entry.path, path) and std.mem.eql(u8, &entry.identity, &wanted) and CorpusStamp.eql(entry.stamp, stamp)) {
                entry.references += 1;
                return .{ .entry = entry, .cache_hit = true };
            }
        }
        var index = (try openCorpus(io, self.allocator, path, wanted)) orelse return null;
        errdefer index.deinit();
        const owned_path = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(owned_path);
        if (!CorpusStamp.eql(stamp, try CorpusStamp.read(io, path))) return error.ChangedDuringRead;
        const entry = try self.allocator.create(CachedCorpus);
        entry.* = .{ .index = index, .path = owned_path, .identity = wanted, .stamp = stamp, .references = 2 };
        if (self.corpus_entry) |old| old.release();
        self.corpus_entry = entry;
        return .{ .entry = entry, .cache_hit = false };
    }

    /// Writable validation/recovery and acquisition share one owned token read.
    pub fn acquireWritablePersistent(self: *HistoryCache, io: Io, path: []const u8, identity: [32]u8) !PersistentLease {
        const loaded: Snapshot = blk: {
            var transaction = try AppendTransaction.open(io, self.allocator, path, identity);
            defer transaction.deinit();
            try transaction.guard.validate();
            if (!Stamp.eql(transaction.stamp, try fileStamp(transaction.guard.dir, io, "tokens.bin")) or
                !Stamp.eql(transaction.stamp, Stamp.from(try transaction.file.stat(io))))
                return error.StoreChanged;
            if (self.persistent_entry) |entry| {
                if (std.mem.eql(u8, entry.path, path) and std.mem.eql(u8, &entry.identity, &identity) and
                    Stamp.eql(entry.stamp, transaction.stamp) and Stamp.eql(entry.lock_stamp, transaction.guard.lock_stamp) and
                    entry.directory_inode == transaction.guard.directory_inode)
                {
                    entry.references += 1;
                    return .{ .entry = entry, .cache_hit = true };
                }
            }
            const snapshot: Snapshot = .{
                .tokens = transaction.tokens,
                .stamp = transaction.stamp,
                .lock_stamp = transaction.guard.lock_stamp,
                .directory_inode = transaction.guard.directory_inode,
            };
            transaction.tokens = &.{};
            break :blk snapshot;
        };
        return self.adoptPersistent(path, identity, loaded);
    }

    pub fn acquirePersistent(self: *HistoryCache, io: Io, path: []const u8, identity: [32]u8) !PersistentLease {
        const loaded: Snapshot = blk: {
            var guard = try StoreGuard.open(io, self.allocator, path, true);
            defer guard.deinit();
            const file = try guard.dir.openFile(io, "tokens.bin", .{ .follow_symlinks = false, .allow_directory = false });
            defer file.close(io);
            const stamp = try fileStamp(guard.dir, io, "tokens.bin");
            var header: [header_size]u8 = undefined;
            if (try file.readPositionalAll(io, &header, 0) != header.len) return error.InvalidStoreHeader;
            try checkHeader(&header, identity);
            if (!Stamp.eql(stamp, Stamp.from(try file.stat(io)))) return error.StoreChanged;
            try guard.validate();
            if (!Stamp.eql(stamp, try fileStamp(guard.dir, io, "tokens.bin"))) return error.StoreChanged;
            if (self.persistent_entry) |entry| {
                if (std.mem.eql(u8, entry.path, path) and std.mem.eql(u8, &entry.identity, &identity) and
                    Stamp.eql(entry.stamp, stamp) and Stamp.eql(entry.lock_stamp, guard.lock_stamp) and
                    entry.directory_inode == guard.directory_inode)
                {
                    entry.references += 1;
                    return .{ .entry = entry, .cache_hit = true };
                }
            }
            const tokens = try readPersistentTokens(&guard, file, self.allocator, identity);
            errdefer self.allocator.free(tokens);
            if (!Stamp.eql(stamp, try fileStamp(guard.dir, io, "tokens.bin"))) return error.StoreChanged;
            break :blk .{ .tokens = tokens, .stamp = stamp, .lock_stamp = guard.lock_stamp, .directory_inode = guard.directory_inode };
        };
        return self.adoptPersistent(path, identity, loaded);
    }

    fn adoptPersistent(self: *HistoryCache, path: []const u8, identity: [32]u8, loaded: Snapshot) !PersistentLease {
        // Sorting can be expensive; the validated token copy needs no file lock.
        const index = Corpus.fromOwnedTokens(self.allocator, loaded.tokens, max_order) catch |err| {
            self.allocator.free(loaded.tokens);
            return err;
        };
        var owned_index = index;
        errdefer owned_index.deinit();
        const owned_path = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(owned_path);
        const entry = try self.allocator.create(CachedPersistent);
        entry.* = .{ .index = index, .path = owned_path, .identity = identity, .stamp = loaded.stamp,
            .lock_stamp = loaded.lock_stamp, .directory_inode = loaded.directory_inode, .references = 2 };
        if (self.persistent_entry) |old| old.release();
        self.persistent_entry = entry;
        return .{ .entry = entry, .cache_hit = false };
    }

    pub fn appendFinalTurn(self: *HistoryCache, io: Io, path: []const u8, identity: [32]u8, prompt: []const u32, output: []const u32) !void {
        // An error after a rename or failed rollback makes disk state uncertain.
        // Drop only current ownership; existing request snapshots stay valid.
        defer if (self.persistent_entry) |entry| {
            if (std.mem.eql(u8, entry.path, path)) {
                self.persistent_entry = null;
                entry.release();
            }
        };
        try appendTransaction(io, self.allocator, path, identity, prompt, output);
    }
};

pub const CorpusLease = struct {
    entry: *CachedCorpus,
    cache_hit: bool,

    pub fn index(self: *const CorpusLease) *const Corpus {
        return &self.entry.index;
    }

    pub fn deinit(self: *CorpusLease) void {
        self.entry.release();
        self.* = undefined;
    }
};

pub const PersistentLease = struct {
    entry: *CachedPersistent,
    cache_hit: bool,

    pub fn index(self: *const PersistentLease) *const Corpus {
        return &self.entry.index;
    }

    pub fn deinit(self: *PersistentLease) void {
        self.entry.release();
        self.* = undefined;
    }
};

test "longest usable match wins without crossing a document" {
    const a = std.testing.allocator;
    var corpus = try Corpus.build(a, &.{ 41, 42, separator, 42, 96, separator }, 5);
    defer corpus.deinit();
    var drafter = Drafter.init(a);
    defer drafter.deinit();
    drafter.corpus = &corpus;
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    const found = try drafter.look(&.{ 41, 42 }, 6, &out);
    try std.testing.expectEqual(@as(u32, 1), found.order);
    try std.testing.expectEqualSlices(u32, &.{96}, found.tokens);
    const boundary = try drafter.look(&.{ 96, 41 }, 6, &out);
    try std.testing.expectEqual(@as(u32, 1), boundary.order);
    try std.testing.expectEqualSlices(u32, &.{42}, boundary.tokens);
    const miss = try drafter.look(&.{98765}, 6, &out);
    try std.testing.expectEqual(@as(usize, 0), miss.tokens.len);
}

test "corpus and persistent votes select a coherent continuation" {
    const a = std.testing.allocator;
    var corpus = try Corpus.build(a, &.{ 53, 54, 61, 62, separator, 53, 54, 71, 72, separator }, 4);
    defer corpus.deinit();
    var persistent = try Corpus.build(a, &.{ 53, 54, 71, 73, separator }, 4);
    defer persistent.deinit();
    var drafter = Drafter.init(a);
    defer drafter.deinit();
    drafter.corpus = &corpus;
    drafter.persistent = &persistent;
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    const result = try drafter.look(&.{ 53, 54 }, 8, &out);
    try std.testing.expectEqual(@as(u32, 2), result.order);
    try std.testing.expectEqualSlices(u32, &.{ 71, 72 }, result.tokens);
    for (drafter.vote_nodes[0..drafter.node_count], 0..) |node, i| {
        if (node.parent) |parent| try std.testing.expect(parent < i);
    }
}

test "proposal capacity keeps parent closure and bounded depth" {
    const a = std.testing.allocator;
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(a);
    for (0..max_candidates + 9) |i| {
        try ids.appendSlice(a, &.{ 8, @as(u32, @intCast(i)) + 100, 400, 500, 600, separator });
    }
    var corpus = try Corpus.build(a, ids.items, 5);
    defer corpus.deinit();
    var drafter = Drafter.init(a);
    defer drafter.deinit();
    drafter.corpus = &corpus;
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    const found = try drafter.look(&.{8}, max_depth + 10, &out);
    try std.testing.expect(found.tokens.len <= max_depth);
    try std.testing.expectEqualSlices(u32, &.{ 100, 400, 500, 600 }, found.tokens);
    try std.testing.expect(drafter.node_count <= max_vote_nodes);
    for (drafter.vote_nodes[0..drafter.node_count], 0..) |node, i| {
        if (node.parent) |parent| try std.testing.expect(parent < i);
    }
}

test "independent index agrees with exhaustive small-document occurrences" {
    const a = std.testing.allocator;
    const ids = [_]u32{ 6, 9, 13, separator, 9, 15, separator, 6, 9, 17, separator };
    var corpus = try Corpus.build(a, &ids, 3);
    defer corpus.deinit();
    var seen: [ids.len]bool = @splat(false);
    const sort = SortContext{ .tokens = &ids, .order = 3 };
    for (corpus.positions, 0..) |position, i| {
        try std.testing.expect(!seen[position]);
        seen[position] = true;
        if (i > 0) try std.testing.expect(!sort.less(position, corpus.positions[i - 1]));
    }
    for ([_][]const u32{ &.{6}, &.{9}, &.{ 6, 9 }, &.{ 9, 13 }, &.{77} }) |query| {
        var expected: usize = 0;
        for (0..ids.len) |position| {
            if (compareQuery(&ids, position, query) == .eq) expected += 1;
        }
        try std.testing.expectEqual(expected, corpus.lowerBound(query, true) - corpus.lowerBound(query, false));
    }
    try std.testing.expectError(error.InvalidOrder, Corpus.build(a, &ids, 0));
    try std.testing.expectError(error.InvalidToken, Corpus.build(a, &.{separator + 1}, 3));
}

fn testWrite(dir: Dir, name: []const u8, bytes: []const u8) !void {
    const file = try dir.createFile(std.testing.io, name, .{ .permissions = store_file_permissions });
    defer file.close(std.testing.io);
    try file.writePositionalAll(std.testing.io, bytes, 0);
}

fn testNpy(allocator: Allocator, values: []const u32, version: u8) ![]u8 {
    const dictionary = try std.fmt.allocPrint(allocator, "{{'shape': ({d},), \"fortran_order\": False, 'descr': '<i4', }}", .{values.len});
    defer allocator.free(dictionary);
    const prefix: usize = if (version == 1) 10 else 12;
    const data_offset = (prefix + dictionary.len + 1 + 63) / 64 * 64;
    const bytes = try allocator.alloc(u8, data_offset + values.len * 4);
    @memset(bytes[0..data_offset], ' ');
    @memcpy(bytes[0..6], "\x93NUMPY");
    bytes[6] = version;
    bytes[7] = 0;
    if (version == 1)
        std.mem.writeInt(u16, bytes[8..10], @intCast(data_offset - prefix), .little)
    else
        std.mem.writeInt(u32, bytes[8..12], @intCast(data_offset - prefix), .little);
    @memcpy(bytes[prefix..][0..dictionary.len], dictionary);
    bytes[data_offset - 1] = '\n';
    for (values, 0..) |value, i| std.mem.writeInt(u32, bytes[data_offset + i * 4..][0..4], value, .little);
    return bytes;
}

fn testCorpusFixture(dir: Dir, ids: []const u32, identity: [32]u8) !void {
    const a = std.testing.allocator;
    var corpus = try Corpus.build(a, ids, 4);
    defer corpus.deinit();
    const tokens = try testNpy(a, ids, 1);
    defer a.free(tokens);
    const positions = try testNpy(a, corpus.positions, 2);
    defer a.free(positions);
    try testWrite(dir, "tokens.npy", tokens);
    try testWrite(dir, "sa.npy", positions);
    const hex = std.fmt.bytesToHex(identity, .lower);
    const metadata = try std.fmt.allocPrint(a,
        "{{\"tokenizer_sha256\":\"{s}\",\"max_order\":4,\"doc_sep\":{d},\"n_tokens\":{d},\"documents\":{d}}}",
        .{ hex, separator, ids.len, try validateDocuments(ids, true) });
    defer a.free(metadata);
    try testWrite(dir, "meta.json", metadata);
}

test "NPY reads documented versions and rejects inconsistent layout" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_]u8{ 1, 2, 3 }) |version| {
        const bytes = try testNpy(a, &.{ 5, 27, 800 }, version);
        defer a.free(bytes);
        try testWrite(tmp.dir, "tokens.npy", bytes);
        const values = try readNpy(tmp.dir, std.testing.io, a, "tokens.npy", null);
        defer a.free(values);
        try std.testing.expectEqualSlices(u32, &.{ 5, 27, 800 }, values);
        try testWrite(tmp.dir, "tokens.npy", bytes[0 .. bytes.len - 1]);
        try std.testing.expectError(error.InvalidNpy, readNpy(tmp.dir, std.testing.io, a, "tokens.npy", null));
        const marker = std.mem.indexOf(u8, bytes, "<i4").?;
        bytes[marker] = '>';
        try testWrite(tmp.dir, "tokens.npy", bytes);
        try std.testing.expectError(error.UnsupportedNpyType, readNpy(tmp.dir, std.testing.io, a, "tokens.npy", null));
    }
    var parser = NpyHeader{ .text = "{'descr':'<i4','shape':(2,3),'fortran_order':False}\n" };
    try std.testing.expectError(error.InvalidNpy, parser.parse());
    var duplicate = NpyHeader{ .text = "{'descr':'<i4','descr':'<i4','shape':(1,),'fortran_order':False}\n" };
    try std.testing.expectError(error.InvalidNpy, duplicate.parse());
}


test "corpus snapshots share overlap and survive refresh clear and cache teardown" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [Dir.max_path_bytes]u8 = undefined;
    const path = path_buffer[0..try tmp.dir.realPath(io, &path_buffer)];
    const identity: [32]u8 = @splat(0x2d);
    try testCorpusFixture(tmp.dir, &.{ 51, 52, separator }, identity);
    var cache = HistoryCache.init(a);
    var cache_live = true;
    defer if (cache_live) cache.deinit() catch unreachable;
    var first = (try cache.acquireCorpus(io, path, identity)).?;
    defer first.deinit();
    var second = (try cache.acquireCorpus(io, path, identity)).?;
    defer second.deinit();
    try std.testing.expect(!first.cache_hit and second.cache_hit);
    try std.testing.expectEqual(first.index(), second.index());
    try testCorpusFixture(tmp.dir, &.{ 51, 58, separator }, identity);
    var changed = (try cache.acquireCorpus(io, path, identity)).?;
    defer changed.deinit();
    try std.testing.expect(!changed.cache_hit);
    try std.testing.expectEqualSlices(u32, &.{ 51, 52, separator }, first.index().tokens);
    try std.testing.expectEqual(@as(u32, 58), changed.index().tokens[1]);
    try std.testing.expectError(error.TokenizerRequired, cache.acquireCorpus(io, path, null));
    try std.testing.expectError(error.TokenizerMismatch, cache.acquireCorpus(io, path, @as([32]u8, @splat(0x2e))));
    var preserved = (try cache.acquireCorpus(io, path, identity)).?;
    defer preserved.deinit();
    try std.testing.expect(preserved.cache_hit);
    try std.testing.expectEqual(changed.index(), preserved.index());
    const malformed = try testNpy(a, &.{ 0, 0, 2 }, 1);
    defer a.free(malformed);
    try testWrite(tmp.dir, "sa.npy", malformed);
    try std.testing.expectError(error.InvalidSuffixIndex, cache.acquireCorpus(io, path, identity));
    try cache.clear();
    try std.testing.expect(cache.corpus_entry == null);
    try std.testing.expectEqual(@as(u32, 52), second.index().tokens[1]);
    try cache.deinit();
    cache_live = false;
    try std.testing.expectEqual(@as(u32, 58), preserved.index().tokens[1]);
}

test "read-only snapshots preserve bytes metadata and never create absent paths" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [Dir.max_path_bytes]u8 = undefined;
    const path = path_buffer[0..try tmp.dir.realPath(io, &path_buffer)];
    const identity: [32]u8 = @splat(0x39);
    var cache = HistoryCache.init(a);
    defer cache.deinit() catch unreachable;
    try std.testing.expectError(error.FileNotFound, cache.acquirePersistent(io, path, identity));
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "store.lock", .{}));
    try initializePersistent(io, a, path, identity);
    {
        var empty = try cache.acquirePersistent(io, path, identity);
        defer empty.deinit();
        try std.testing.expectEqual(@as(usize, 0), empty.index().tokens.len);
        try cache.appendFinalTurn(io, path, identity, &.{81}, &.{ 82, 83 });
        try std.testing.expectEqual(@as(usize, 0), empty.index().tokens.len);
    }
    const before = try fileStamp(tmp.dir, io, "tokens.bin");
    const lock_before = try fileStamp(tmp.dir, io, "store.lock");
    const bytes_before = try readBounded(tmp.dir, io, a, "tokens.bin", 1024);
    defer a.free(bytes_before);
    var reader = try cache.acquirePersistent(io, path, identity);
    defer reader.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 81, 82, 83, separator }, reader.index().tokens);
    try std.testing.expect(Stamp.eql(before, try fileStamp(tmp.dir, io, "tokens.bin")));
    try std.testing.expect(Stamp.eql(lock_before, try fileStamp(tmp.dir, io, "store.lock")));
    const bytes_after = try readBounded(tmp.dir, io, a, "tokens.bin", 1024);
    defer a.free(bytes_after);
    try std.testing.expectEqualSlices(u8, bytes_before, bytes_after);
    if (builtin.os.tag != .windows) {
        const permissions = (try tmp.dir.statFile(io, "tokens.bin", .{})).permissions.toMode();
        try std.testing.expectEqual(@as(@TypeOf(permissions), 0o600), permissions & 0o777);
        const directory_permissions = (try tmp.dir.stat(io)).permissions.toMode();
        try std.testing.expectEqual(@as(@TypeOf(directory_permissions), 0o700), directory_permissions & 0o777);
    }
    // A live read snapshot owns no lock, including against independent writers.
    var guard = try StoreGuard.open(io, a, path, false);
    defer guard.deinit();
    try std.testing.expectError(error.StoreBusy, cache.acquirePersistent(io, path, identity));
    try std.testing.expectError(error.StoreBusy, appendFinalTurn(io, a, path, identity, &.{84}, &.{85}));
}

test "overlapping persistent leases append B then A against current log and retain old queries" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [Dir.max_path_bytes]u8 = undefined;
    const path = path_buffer[0..try tmp.dir.realPath(io, &path_buffer)];
    const identity: [32]u8 = @splat(0x47);
    try appendFinalTurn(io, a, path, identity, &.{101}, &.{102});
    var cache = HistoryCache.init(a);
    var cache_live = true;
    defer if (cache_live) cache.deinit() catch unreachable;
    var request_a = try cache.acquirePersistent(io, path, identity);
    defer request_a.deinit();
    var request_b = try cache.acquirePersistent(io, path, identity);
    defer request_b.deinit();
    try std.testing.expect(!request_a.cache_hit and request_b.cache_hit);
    try std.testing.expectEqual(request_a.index(), request_b.index());
    try cache.appendFinalTurn(io, path, identity, &.{201}, &.{202});
    try std.testing.expect(cache.persistent_entry == null);
    var after_b = try cache.acquirePersistent(io, path, identity);
    defer after_b.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 101, 102, separator, 201, 202, separator }, after_b.index().tokens);
    // A's final turn does not serialize its pre-B snapshot back onto disk.
    try cache.appendFinalTurn(io, path, identity, &.{301}, &.{302});
    var after_a = try cache.acquirePersistent(io, path, identity);
    defer after_a.deinit();
    try std.testing.expect(!after_a.cache_hit);
    try std.testing.expectEqualSlices(u32, &.{ 101, 102, separator, 201, 202, separator, 301, 302, separator }, after_a.index().tokens);
    var drafter = Drafter.init(a);
    defer drafter.deinit();
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    drafter.persistent = request_b.index();
    try std.testing.expectEqualSlices(u32, &.{102}, (try drafter.look(&.{101}, 2, &out)).tokens);
    try std.testing.expectEqual(@as(usize, 0), (try drafter.look(&.{201}, 2, &out)).tokens.len);
    drafter.persistent = after_a.index();
    try std.testing.expectEqualSlices(u32, &.{202}, (try drafter.look(&.{201}, 2, &out)).tokens);
    try std.testing.expectEqualSlices(u32, &.{302}, (try drafter.look(&.{301}, 2, &out)).tokens);
    try std.testing.expectError(error.TokenizerMismatch, cache.acquirePersistent(io, path, @as([32]u8, @splat(0x48))));
    var preserved = try cache.acquirePersistent(io, path, identity);
    defer preserved.deinit();
    try std.testing.expect(preserved.cache_hit);
    try cache.deinit();
    cache_live = false;
    try std.testing.expectEqualSlices(u32, &.{ 101, 102, separator }, request_a.index().tokens);
    try std.testing.expectEqual(@as(u32, 302), preserved.index().tokens[7]);
}

test "terminal append uses only token storage and accepts borrowed prompt and output" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [Dir.max_path_bytes]u8 = undefined;
    const path = path_buffer[0..try tmp.dir.realPath(io, &path_buffer)];
    const identity: [32]u8 = @splat(0x61);
    try appendFinalTurn(io, a, path, identity, &.{121}, &.{122});
    var cache = HistoryCache.init(a);
    defer cache.deinit() catch unreachable;
    var lease = try cache.acquirePersistent(io, path, identity);
    defer lease.deinit();
    const borrowed = lease.index().tokens[0..2];
    // One path allocation and one token allocation; no suffix positions or join.
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 2 });
    try appendFinalTurn(io, failing.allocator(), path, identity, borrowed[0..1], borrowed[1..]);
    try std.testing.expectEqualSlices(u32, &.{ 121, 122 }, borrowed);
    var refreshed = try cache.acquirePersistent(io, path, identity);
    defer refreshed.deinit();
    try std.testing.expect(!refreshed.cache_hit);
    try std.testing.expectEqualSlices(u32, &.{ 121, 122, separator, 121, 122, separator }, refreshed.index().tokens);
}

test "invalid final document and tokenizer leave durable content intact" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [Dir.max_path_bytes]u8 = undefined;
    const path = path_buffer[0..try tmp.dir.realPath(io, &path_buffer)];
    const identity: [32]u8 = @splat(0x62);
    try appendFinalTurn(io, a, path, identity, &.{11}, &.{12});
    const before = try readBounded(tmp.dir, io, a, "tokens.bin", 1024);
    defer a.free(before);
    try std.testing.expectError(error.InvalidToken, appendFinalTurn(io, a, path, identity, &.{separator}, &.{}));
    try std.testing.expectError(error.InvalidToken, appendFinalTurn(io, a, path, identity, &.{}, &.{separator + 1}));
    const oversized = try a.alloc(u32, persistent_capacity);
    defer a.free(oversized);
    try std.testing.expectError(error.DocumentTooLarge, appendFinalTurn(io, a, path, identity, oversized, &.{}));
    try std.testing.expectError(error.TokenizerMismatch, appendFinalTurn(io, a, path, @splat(0x63), &.{13}, &.{14}));
    try std.testing.expectError(error.TokenizerMismatch, initializePersistent(io, a, path, @splat(0x63)));
    try appendFinalTurn(io, a, path, identity, &.{}, &.{});
    const after = try readBounded(tmp.dir, io, a, "tokens.bin", 1024);
    defer a.free(after);
    try std.testing.expectEqualSlices(u8, before, after);
}

test "host log rejects malformed documents headers truncation and wrong identity" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [Dir.max_path_bytes]u8 = undefined;
    const path = path_buffer[0..try tmp.dir.realPath(io, &path_buffer)];
    const identity: [32]u8 = @splat(0x55);
    var cache = HistoryCache.init(a);
    defer cache.deinit() catch unreachable;
    try testWrite(tmp.dir, "store.lock", "");
    try testWrite(tmp.dir, "tokens.bin", "not a log");
    try std.testing.expectError(error.InvalidStoreHeader, cache.acquirePersistent(io, path, identity));
    var bytes: [header_size + 8]u8 = @splat(0);
    @memcpy(bytes[0..header_size], &makeHeader(identity));
    try testWrite(tmp.dir, "tokens.bin", bytes[0 .. header_size + 1]);
    try std.testing.expectError(error.TruncatedFile, cache.acquirePersistent(io, path, identity));
    std.mem.writeInt(u32, bytes[header_size..][0..4], 117, .little);
    try testWrite(tmp.dir, "tokens.bin", bytes[0 .. header_size + 4]);
    try std.testing.expectError(error.IncompleteDocument, cache.acquirePersistent(io, path, identity));
    std.mem.writeInt(u32, bytes[header_size..][0..4], separator, .little);
    try testWrite(tmp.dir, "tokens.bin", bytes[0 .. header_size + 4]);
    try std.testing.expectError(error.EmptyDocument, cache.acquirePersistent(io, path, identity));
    std.mem.writeInt(u32, bytes[header_size..][0..4], separator + 7, .little);
    std.mem.writeInt(u32, bytes[header_size + 4..][0..4], separator, .little);
    try testWrite(tmp.dir, "tokens.bin", &bytes);
    try std.testing.expectError(error.InvalidToken, cache.acquirePersistent(io, path, identity));
    try testWrite(tmp.dir, "tokens.bin", &makeHeader(identity));
    try std.testing.expectError(error.TokenizerMismatch, cache.acquirePersistent(io, path, @as([32]u8, @splat(0x56))));
}

// Fault injection changes only one real filesystem operation. The first write
// really reaches the temporary test file; subsequent failure exercises rollback.
const TestPartialWrite = struct {
    var calls: usize = 0;
    fn write(userdata: ?*anyopaque, file: File, header: []const u8, data: []const []const u8, splat: usize, offset: u64) File.WritePositionalError!usize {
        calls += 1;
        if (calls > 1) return error.NoSpaceLeft;
        std.debug.assert(header.len == 0 and data.len == 1 and splat == 1);
        return std.testing.io.vtable.fileWritePositional(userdata, file, header, &.{data[0][0..@min(4, data[0].len)]}, 1, offset);
    }
};

test "partial append rolls back while overlapping snapshots remain queryable" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [Dir.max_path_bytes]u8 = undefined;
    const path = path_buffer[0..try tmp.dir.realPath(io, &path_buffer)];
    const identity: [32]u8 = @splat(0x63);
    try appendFinalTurn(io, a, path, identity, &.{131}, &.{132});
    var cache = HistoryCache.init(a);
    defer cache.deinit() catch unreachable;
    var first = try cache.acquirePersistent(io, path, identity);
    defer first.deinit();
    var second = try cache.acquirePersistent(io, path, identity);
    defer second.deinit();
    const old_bytes = try readBounded(tmp.dir, io, a, "tokens.bin", 1024);
    defer a.free(old_bytes);
    var vtable = io.vtable.*;
    vtable.fileWritePositional = TestPartialWrite.write;
    TestPartialWrite.calls = 0;
    const failing = Io{ .userdata = io.userdata, .vtable = &vtable };
    try std.testing.expectError(error.NoSpaceLeft, cache.appendFinalTurn(failing, path, identity, &.{141}, &.{ 142, 143 }));
    try std.testing.expect(cache.persistent_entry == null);
    try std.testing.expectEqual(first.index(), second.index());
    try std.testing.expectEqualSlices(u32, &.{ 131, 132, separator }, first.index().tokens);
    const actual = try readBounded(tmp.dir, io, a, "tokens.bin", 1024);
    defer a.free(actual);
    try std.testing.expectEqualSlices(u8, old_bytes, actual);
    var refreshed = try cache.acquirePersistent(io, path, identity);
    defer refreshed.deinit();
    try std.testing.expect(!refreshed.cache_hit);
    try cache.appendFinalTurn(io, path, identity, &.{151}, &.{152});
    var final = try cache.acquirePersistent(io, path, identity);
    defer final.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 131, 132, separator, 151, 152, separator }, final.index().tokens);
}

test "failed rollback drops current cache and closes the writer without freeing borrowers" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [Dir.max_path_bytes]u8 = undefined;
    const path = path_buffer[0..try tmp.dir.realPath(io, &path_buffer)];
    const identity: [32]u8 = @splat(0x71);
    try appendFinalTurn(io, a, path, identity, &.{159}, &.{160});
    var cache = HistoryCache.init(a);
    defer cache.deinit() catch unreachable;
    var lease = try cache.acquirePersistent(io, path, identity);
    defer lease.deinit();
    var vtable = io.vtable.*;
    vtable.fileWritePositional = TestPartialWrite.write;
    vtable.fileSetLength = Io.failingFileSetLength;
    TestPartialWrite.calls = 0;
    const failing = Io{ .userdata = io.userdata, .vtable = &vtable };
    try std.testing.expectError(error.RollbackFailed, cache.appendFinalTurn(failing, path, identity, &.{161}, &.{162}));
    try std.testing.expect(cache.persistent_entry == null);
    try std.testing.expectEqualSlices(u32, &.{ 159, 160, separator }, lease.index().tokens);
    try std.testing.expectError(error.IncompleteDocument, cache.acquirePersistent(io, path, identity));
    try std.testing.expectError(error.IncompleteDocument, appendFinalTurn(io, a, path, identity, &.{163}, &.{164}));
    var guard = try StoreGuard.open(io, a, path, false);
    defer guard.deinit();
}

test "whole-document eviction preserves leased snapshots and uses no replacement index" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [Dir.max_path_bytes]u8 = undefined;
    const path = path_buffer[0..try tmp.dir.realPath(io, &path_buffer)];
    const identity: [32]u8 = @splat(0x7f);
    const document = try a.alloc(u32, persistent_capacity / 2 - 1);
    defer a.free(document);
    for ([_]u32{ 100, 200000 }) |base| {
        for (document, 0..) |*token, i| token.* = base + @as(u32, @intCast(i));
        try appendFinalTurn(io, a, path, identity, document, &.{});
    }
    var cache = HistoryCache.init(a);
    defer cache.deinit() catch unreachable;
    var first = try cache.acquirePersistent(io, path, identity);
    defer first.deinit();
    var second = try cache.acquirePersistent(io, path, identity);
    defer second.deinit();
    const old_inode = (try fileStamp(tmp.dir, io, "tokens.bin")).inode;
    const lock_inode = (try fileStamp(tmp.dir, io, "store.lock")).inode;
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 1 });
    try std.testing.expectError(error.OutOfMemory, appendFinalTurn(io, failing.allocator(), path, identity, &.{181}, &.{ 182, 183 }));
    try std.testing.expectEqual(old_inode, (try fileStamp(tmp.dir, io, "tokens.bin")).inode);
    var vtable = io.vtable.*;
    vtable.fileSync = Io.failingFileSync;
    const failing_io = Io{ .userdata = io.userdata, .vtable = &vtable };
    try std.testing.expectError(error.NoSpaceLeft, cache.appendFinalTurn(failing_io, path, identity, &.{181}, &.{ 182, 183 }));
    try std.testing.expect(cache.persistent_entry == null);
    try std.testing.expectEqual(old_inode, (try fileStamp(tmp.dir, io, "tokens.bin")).inode);
    // Replacement writes retained tokens and both slices directly, with no join.
    var bounded = std.testing.FailingAllocator.init(a, .{ .fail_index = 2 });
    try appendFinalTurn(io, bounded.allocator(), path, identity, &.{181}, &.{ 182, 183 });
    var refreshed = try cache.acquirePersistent(io, path, identity);
    defer refreshed.deinit();
    try std.testing.expect(!refreshed.cache_hit);
    const stored = refreshed.index().tokens;
    try std.testing.expect(stored.len <= persistent_capacity);
    try std.testing.expectEqual(@as(u32, 200000), stored[0]);
    try std.testing.expectEqualSlices(u32, &.{ 181, 182, 183, separator }, stored[stored.len - 4..]);
    try std.testing.expectEqual(@as(usize, 2), try validateDocuments(stored, true));
    try std.testing.expect((try fileStamp(tmp.dir, io, "tokens.bin")).inode != old_inode);
    try std.testing.expectEqual(lock_inode, (try fileStamp(tmp.dir, io, "store.lock")).inode);
    try std.testing.expectEqual(first.index(), second.index());
    try std.testing.expectEqual(persistent_capacity, first.index().tokens.len);
    try std.testing.expectEqual(@as(u32, 100), second.index().tokens[0]);
    var drafter = Drafter.init(a);
    defer drafter.deinit();
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    drafter.persistent = first.index();
    try std.testing.expectEqualSlices(u32, &.{102}, (try drafter.look(&.{ 100, 101 }, 1, &out)).tokens);
    drafter.persistent = refreshed.index();
    try std.testing.expectEqual(@as(usize, 0), (try drafter.look(&.{ 100, 101 }, 1, &out)).tokens.len);
    // A failed directory sync may follow a committed rename, so refresh from disk.
    vtable.fileSync = TestDirectorySync.fail;
    try std.testing.expectError(error.InputOutput, cache.appendFinalTurn(failing_io, path, identity, document, &.{}));
    try std.testing.expect(cache.persistent_entry == null);
    try std.testing.expectEqual(@as(u32, 200000), refreshed.index().tokens[0]);
    var uncertain = try cache.acquirePersistent(io, path, identity);
    defer uncertain.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 181, 182, 183, separator }, uncertain.index().tokens[0..4]);
    try std.testing.expectEqual(@as(u32, 200000), uncertain.index().tokens[4]);
    try std.testing.expectEqual(@as(usize, 2), try validateDocuments(uncertain.index().tokens, true));
}

test "persistent snapshots retain full 8 16 and 64 token match orders" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [Dir.max_path_bytes]u8 = undefined;
    const path = path_buffer[0..try tmp.dir.realPath(io, &path_buffer)];
    const identity: [32]u8 = @splat(0x80);
    var ids: [max_order + 1]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @as(u32, @intCast(i)) + 100;
    try appendFinalTurn(io, a, path, identity, ids[0..max_order], ids[max_order..]);
    var cache = HistoryCache.init(a);
    defer cache.deinit() catch unreachable;
    var lease = try cache.acquirePersistent(io, path, identity);
    defer lease.deinit();
    try std.testing.expectEqual(max_order, lease.index().max_order);
    var drafter = Drafter.init(a);
    defer drafter.deinit();
    drafter.persistent = lease.index();
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    for ([_]u32{ 8, 16, 64 }) |order| {
        const found = try drafter.look(ids[0..order], 1, &out);
        try std.testing.expectEqual(order, found.order);
        try std.testing.expectEqualSlices(u32, ids[order..][0..1], found.tokens);
    }
}

test "same-size external edits refresh current history without mutating old snapshots" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [Dir.max_path_bytes]u8 = undefined;
    const path = path_buffer[0..try tmp.dir.realPath(io, &path_buffer)];
    const identity: [32]u8 = @splat(0x97);
    try appendFinalTurn(io, a, path, identity, &.{211}, &.{212});
    var cache = HistoryCache.init(a);
    defer cache.deinit() catch unreachable;
    var lease = try cache.acquirePersistent(io, path, identity);
    defer lease.deinit();
    const rogue = try tmp.dir.openFile(io, "tokens.bin", .{ .mode = .read_write });
    defer rogue.close(io);
    var replacement: [4]u8 = undefined;
    std.mem.writeInt(u32, &replacement, 219, .little);
    try rogue.writePositionalAll(io, &replacement, header_size);
    var changed = try cache.acquirePersistent(io, path, identity);
    defer changed.deinit();
    try std.testing.expect(!changed.cache_hit);
    try std.testing.expectEqual(@as(u32, 219), changed.index().tokens[0]);
    try std.testing.expectEqual(@as(u32, 211), lease.index().tokens[0]);
}

test "refresh read and allocation failures retain all live snapshots" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [Dir.max_path_bytes]u8 = undefined;
    const path = path_buffer[0..try tmp.dir.realPath(io, &path_buffer)];
    const identity: [32]u8 = @splat(0xa3);
    try appendFinalTurn(io, a, path, identity, &.{221}, &.{222});
    var cache = HistoryCache.init(a);
    defer cache.deinit() catch unreachable;
    var first = try cache.acquirePersistent(io, path, identity);
    defer first.deinit();
    var vtable = io.vtable.*;
    vtable.fileReadPositional = Io.failingFileReadPositional;
    const failing_io = Io{ .userdata = io.userdata, .vtable = &vtable };
    try std.testing.expectError(error.InputOutput, cache.acquirePersistent(failing_io, path, identity));
    var restored = try cache.acquirePersistent(io, path, identity);
    defer restored.deinit();
    try std.testing.expect(restored.cache_hit);
    try appendFinalTurn(io, a, path, identity, &.{223}, &.{224});
    for (0..5) |allocation| {
        var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = allocation });
        cache.allocator = failing.allocator();
        defer cache.allocator = a;
        try std.testing.expectError(error.OutOfMemory, cache.acquirePersistent(io, path, identity));
        try std.testing.expectEqual(first.entry, cache.persistent_entry.?);
        try std.testing.expectEqualSlices(u32, &.{ 221, 222, separator }, restored.index().tokens);
    }
    var refreshed = try cache.acquirePersistent(io, path, identity);
    defer refreshed.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 221, 222, separator, 223, 224, separator }, refreshed.index().tokens);
}

test "corpus optional counts and file fingerprints follow host metadata" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [Dir.max_path_bytes]u8 = undefined;
    const path = path_buffer[0..try tmp.dir.realPath(io, &path_buffer)];
    const identity: [32]u8 = @splat(0x89);
    try testCorpusFixture(tmp.dir, &.{ 201, 202, separator }, identity);
    const token_bytes = try readBounded(tmp.dir, io, a, "tokens.npy", 1024);
    defer a.free(token_bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(token_bytes, &digest, .{});
    const identity_hex = std.fmt.bytesToHex(identity, .lower);
    const digest_hex = std.fmt.bytesToHex(digest, .lower);
    const metadata = try std.fmt.allocPrint(a,
        "{{\"tokenizer_sha256\":\"{s}\",\"max_order\":4,\"doc_sep\":{d},\"tokens_sha256\":\"{s}\"}}",
        .{ identity_hex, separator, digest_hex });
    defer a.free(metadata);
    try testWrite(tmp.dir, "meta.json", metadata);
    var corpus = (try openCorpus(io, a, path, identity)).?;
    defer corpus.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 201, 202, separator }, corpus.tokens);
    try std.testing.expectError(error.TokenizerRequired, openCorpus(io, a, path, null));
    token_bytes[token_bytes.len - 8] ^= 1;
    try testWrite(tmp.dir, "tokens.npy", token_bytes);
    try std.testing.expectError(error.FingerprintMismatch, openCorpus(io, a, path, identity));
}

const TestDirectorySync = struct {
    fn fail(userdata: ?*anyopaque, file: File) File.SyncError!void {
        const stat = file.stat(std.testing.io) catch return error.InputOutput;
        if (stat.kind == .directory) return error.InputOutput;
        return std.testing.io.vtable.fileSync(userdata, file);
    }
};

test "atomic replacement distinguishes pre-commit and directory-sync failures" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [Dir.max_path_bytes]u8 = undefined;
    const path = path_buffer[0..try tmp.dir.realPath(io, &path_buffer)];
    const identity: [32]u8 = @splat(0xb1);
    var guard = try StoreGuard.open(io, a, path, false);
    defer guard.deinit();
    const initial = try replaceLog(&guard, identity, &.{&.{ 231, separator }});
    initial.close(io);
    const initial_stamp = try fileStamp(tmp.dir, io, "tokens.bin");
    var vtable = io.vtable.*;
    vtable.fileSync = Io.failingFileSync;
    guard.io = .{ .userdata = io.userdata, .vtable = &vtable };
    try std.testing.expectError(error.NoSpaceLeft, replaceLog(&guard, identity, &.{&.{ 232, separator }}));
    try std.testing.expect(Stamp.eql(initial_stamp, try fileStamp(tmp.dir, io, "tokens.bin")));
    vtable.fileSync = TestDirectorySync.fail;
    try std.testing.expectError(error.InputOutput, replaceLog(&guard, identity, &.{&.{ 233, separator }}));
    guard.io = io;
    const current = try guard.dir.openFile(io, "tokens.bin", .{});
    defer current.close(io);
    const complete = try readPersistentTokens(&guard, current, a, identity);
    defer a.free(complete);
    try std.testing.expectEqualSlices(u32, &.{ 233, separator }, complete);
}

test "releasing an obsolete lock handle does not unlock its replacement" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [Dir.max_path_bytes]u8 = undefined;
    const path = path_buffer[0..try tmp.dir.realPath(io, &path_buffer)];
    var first = try StoreGuard.open(io, a, path, false);
    // An out-of-protocol unlink creates a different lock domain. Validation
    // rejects it; closing the old handle must not affect the new domain.
    try tmp.dir.deleteFile(io, "store.lock");
    var replacement = try StoreGuard.open(io, a, path, false);
    defer replacement.deinit();
    try std.testing.expectError(error.StoreChanged, first.validate());
    first.deinit();
    try std.testing.expectError(error.StoreBusy, StoreGuard.open(io, a, path, true));
}

test "unsafe hard links are rejected without changing the other inode permissions" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    inline for (.{ "tokens.bin", "store.lock" }) |name| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
        defer a.free(path);
        try initializePersistent(io, a, path, @splat(3));
        const file = try tmp.dir.openFile(io, name, .{ .mode = .read_write });
        defer file.close(io);
        if (builtin.os.tag != .windows) try file.setPermissions(io, .fromMode(0o640));
        try Dir.hardLink(tmp.dir, name, tmp.dir, "unrelated", io, .{});
        const before = try tmp.dir.statFile(io, "unrelated", .{});
        const expected_error = if (std.mem.eql(u8, name, "store.lock")) error.UnsafeLockFile else error.UnsafeFile;
        try std.testing.expectError(expected_error, initializePersistent(io, a, path, @splat(3)));
        const after = try tmp.dir.statFile(io, "unrelated", .{});
        try std.testing.expectEqual(before.permissions, after.permissions);
    }
}

test "writable recovery removes abandoned replacement without changing committed history" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(path);
    const identity: [32]u8 = @splat(5);
    try appendFinalTurn(io, a, path, identity, &.{17}, &.{ 19, 23 });
    try testWrite(tmp.dir, pending_log, "interrupted replacement");
    var cache = HistoryCache.init(a);
    defer cache.deinit() catch unreachable;
    var reader = try cache.acquirePersistent(io, path, identity);
    defer reader.deinit();
    _ = try tmp.dir.statFile(io, pending_log, .{});
    var recovered = try cache.acquireWritablePersistent(io, path, identity);
    defer recovered.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 17, 19, 23, separator }, recovered.index().tokens);
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(io, pending_log, .{}));
    try std.testing.expectEqualSlices(u32, reader.index().tokens, recovered.index().tokens);
}

test "writable acquisition creates validates and retains independent snapshots" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(path);
    const identity: [32]u8 = @splat(0x71);
    var cache = HistoryCache.init(a);
    defer cache.deinit() catch unreachable;
    var initial = try cache.acquireWritablePersistent(io, path, identity);
    defer initial.deinit();
    try std.testing.expectEqualSlices(u32, &.{}, initial.index().tokens);
    try cache.appendFinalTurn(io, path, identity, &.{ 11, 12 }, &.{13});
    var first = try cache.acquireWritablePersistent(io, path, identity);
    defer first.deinit();
    var shared = try cache.acquireWritablePersistent(io, path, identity);
    defer shared.deinit();
    try std.testing.expect(shared.cache_hit);
    try std.testing.expectEqual(first.index(), shared.index());
    try std.testing.expectEqualSlices(u32, &.{ 11, 12, 13, separator }, first.index().tokens);
    try std.testing.expectEqualSlices(u32, &.{}, initial.index().tokens);
    try std.testing.expectError(error.TokenizerMismatch, cache.acquireWritablePersistent(io, path, @splat(0x72)));
    if (builtin.os.tag != .windows) {
        const file = try tmp.dir.openFile(io, "tokens.bin", .{ .mode = .read_write });
        defer file.close(io);
        try file.setPermissions(io, .fromMode(0o640));
        var repaired = try cache.acquireWritablePersistent(io, path, identity);
        defer repaired.deinit();
        try std.testing.expectEqual(@as(u32, 0o600), (try file.stat(io)).permissions.toMode() & 0o7777);
        try std.testing.expectEqualSlices(u32, first.index().tokens, repaired.index().tokens);
    }
    try cache.appendFinalTurn(io, path, identity, &.{21}, &.{22});
    var second = try cache.acquireWritablePersistent(io, path, identity);
    defer second.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 11, 12, 13, separator, 21, 22, separator }, second.index().tokens);
    try std.testing.expectEqualSlices(u32, &.{ 11, 12, 13, separator }, shared.index().tokens);
    var guard = try StoreGuard.open(io, a, path, false);
    defer guard.deinit();
    try std.testing.expectError(error.StoreBusy, cache.acquireWritablePersistent(io, path, identity));
}
