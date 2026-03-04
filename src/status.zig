const std = @import("std");

pub const State = enum { idle, prefill, decode };

pub const StatusBar = struct {
    rows: u16,
    cols: u16,
    enabled: bool,

    pub fn init() StatusBar {
        if (isatty(2) == 0)
            return .{ .rows = 0, .cols = 0, .enabled = false };
        const size = getTermSize();
        const self = StatusBar{ .rows = size.rows, .cols = size.cols, .enabled = true };
        var buf: [32]u8 = undefined;
        const seq = std.fmt.bufPrint(&buf, "\x1b[1;{d}r\x1b[1;1H", .{self.rows - 1}) catch return self;
        std.fs.File.stderr().writeAll(seq) catch {};
        self.update(.idle, null);
        return self;
    }

    pub fn update(self: StatusBar, state: State, detail: ?[]const u8) void {
        if (!self.enabled) return;
        var buf: [1024]u8 = undefined;
        const prefix = std.fmt.bufPrint(&buf, "\x1b7\x1b[{d};1H\x1b[2K", .{self.rows}) catch return;
        var pos: usize = prefix.len;

        const style: []const u8 = switch (state) {
            .idle => "\x1b[48;5;28m\x1b[97m",
            .prefill => "\x1b[48;5;172m\x1b[97m",
            .decode => "\x1b[48;5;25m\x1b[97m",
        };
        @memcpy(buf[pos..][0..style.len], style);
        pos += style.len;

        const label: []const u8 = switch (state) {
            .idle => "  IDLE  ",
            .prefill => "  PREFILL  ",
            .decode => "  DECODE  ",
        };
        @memcpy(buf[pos..][0..label.len], label);
        pos += label.len;

        var used: usize = label.len;
        if (detail) |d| {
            const n = @min(d.len, @as(usize, self.cols) -| used -| 1);
            @memcpy(buf[pos..][0..n], d[0..n]);
            pos += n;
            used += n;
        }

        while (used < self.cols and pos < buf.len - 16) : (used += 1) {
            buf[pos] = ' ';
            pos += 1;
        }

        const suffix = "\x1b[0m\x1b8";
        @memcpy(buf[pos..][0..suffix.len], suffix);
        pos += suffix.len;

        std.fs.File.stderr().writeAll(buf[0..pos]) catch {};
    }

    pub fn deinit(self: StatusBar) void {
        if (!self.enabled) return;
        var buf: [64]u8 = undefined;
        const seq = std.fmt.bufPrint(&buf, "\x1b[r\x1b[{d};1H\x1b[2K", .{self.rows}) catch return;
        std.fs.File.stderr().writeAll(seq) catch {};
    }
};

const Winsize = extern struct { ws_row: u16, ws_col: u16, ws_xpixel: u16, ws_ypixel: u16 };
const TIOCGWINSZ: c_ulong = 0x40087468;

extern "c" fn ioctl(fd: c_int, request: c_ulong, ...) c_int;
extern "c" fn isatty(fd: c_int) c_int;

fn getTermSize() struct { rows: u16, cols: u16 } {
    var ws = std.mem.zeroes(Winsize);
    if (ioctl(2, TIOCGWINSZ, &ws) == 0 and ws.ws_row > 0)
        return .{ .rows = ws.ws_row, .cols = ws.ws_col };
    return .{ .rows = 24, .cols = 80 };
}
