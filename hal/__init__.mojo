from .cpu import CPUCaps, CPUArch, CPUVendor, detect_cpu_caps
from .device import Device, DeviceType, DeviceCaps, DeviceBuffer

comptime DEFAULT_THREAD_POOL_SIZE: Int = 0

fn init() raises:
    detect_cpu_caps()