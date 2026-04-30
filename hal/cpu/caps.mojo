comptime CPU_ARCH_UNKNOWN: Int = 0
comptime CPU_ARCH_X86_64: Int = 1
comptime CPU_ARCH_ARM64: Int = 2
comptime CPU_ARCH_RISCV64: Int = 3
comptime CPU_ARCH_WASM32: Int = 4

comptime CPU_VENDOR_UNKNOWN: Int = 0
comptime CPU_VENDOR_INTEL: Int = 1
comptime CPU_VENDOR_AMD: Int = 2
comptime CPU_VENDOR_APPLE: Int = 3
comptime CPU_VENDOR_QUALCOMM: Int = 4

comptime CPUArch: Int = 0
comptime CPUVendor: Int = 0

struct CPUCaps:
    var arch: Int
    var vendor: Int
    var has_avx2: Bool
    var has_avx512f: Bool
    var has_avx512_vnni: Bool
    var has_amx_int8: Bool
    var has_neon: Bool
    var has_sve: Bool
    var has_sve2: Bool
    var has_dotprod: Bool
    var has_i8mm: Bool
    var cache_line_bytes: Int
    var l1d_size_kb: Int
    var l2_size_kb: Int
    var l3_size_kb: Int
    var physical_cores: Int
    var logical_cores: Int

    @staticmethod
    fn default() -> CPUCaps:
        return CPUCaps(
            arch=CPU_ARCH_UNKNOWN,
            vendor=CPU_VENDOR_UNKNOWN,
            has_avx2=False,
            has_avx512f=False,
            has_avx512_vnni=False,
            has_amx_int8=False,
            has_neon=False,
            has_sve=False,
            has_sve2=False,
            has_dotprod=False,
            has_i8mm=False,
            cache_line_bytes=64,
            l1d_size_kb=0,
            l2_size_kb=0,
            l3_size_kb=0,
            physical_cores=1,
            logical_cores=1,
        )

    fn is_x86(ref self) -> Bool:
        return self.arch == CPU_ARCH_X86_64

    fn is_arm(ref self) -> Bool:
        return self.arch == CPU_ARCH_ARM64

    fn simd_bytes(ref self) -> Int:
        if self.is_x86():
            if self.has_avx512f:
                return 64
            elif self.has_avx2:
                return 32
            return 16
        elif self.is_arm():
            if self.has_sve:
                return 16
            return 16
        return 16

fn detect_cpu_caps() raises -> CPUCaps:
    var caps = CPUCaps.default()
    caps.physical_cores = 1
    caps.logical_cores = 1
    caps.cache_line_bytes = 64
    return caps

fn get_global_cpu_caps() -> CPUCaps:
    return CPUCaps.default()