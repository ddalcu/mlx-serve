//! The machine probes live in the SDK (`sdk/util/system.zig`); the host reaches them here.

const system = @import("../sdk/util/system.zig");
pub const getAppRssMb = system.getAppRssMb;
pub const getAppMemFootprintMb = system.getAppMemFootprintMb;
pub const getProcAvailableMemBytes = system.getProcAvailableMemBytes;
pub const memoryPool = system.memoryPool;
pub const getTotalMemBytes = system.getTotalMemBytes;
pub const getAvailableMemBytes = system.getAvailableMemBytes;
pub const memoryPressureLevel = system.memoryPressureLevel;
pub const underMemoryPressure = system.underMemoryPressure;
pub const osReserveBytes = system.osReserveBytes;
pub const parseOsReserveGib = system.parseOsReserveGib;
pub const getSysMemPct = system.getSysMemPct;
pub const getCpuPct = system.getCpuPct;
pub const getGpuPct = system.getGpuPct;
pub const gpuCoreCount = system.gpuCoreCount;
