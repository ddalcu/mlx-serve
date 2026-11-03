//! A mutex for threads that carry no `Io` handle (log sinks, mlx-c's error handler):
//! pthread on POSIX, SRWLOCK on Windows. Zero-initialized on both.
const std = @import("std");
const builtin = @import("builtin");
const windows = std.os.windows;

pub const Mutex = if (builtin.os.tag == .windows) struct {
    srw: windows.SRWLOCK = .{},

    pub fn lock(m: *@This()) void {
        windows.ntdll.RtlAcquireSRWLockExclusive(&m.srw);
    }
    pub fn unlock(m: *@This()) void {
        windows.ntdll.RtlReleaseSRWLockExclusive(&m.srw);
    }
} else struct {
    m: std.c.pthread_mutex_t = .{},

    pub fn lock(m: *@This()) void {
        _ = std.c.pthread_mutex_lock(&m.m);
    }
    pub fn unlock(m: *@This()) void {
        _ = std.c.pthread_mutex_unlock(&m.m);
    }
};

test "os_mutex: lock and unlock" {
    var m: Mutex = .{};
    m.lock();
    m.unlock();
    m.lock();
    m.unlock();
}
