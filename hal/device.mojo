from enum import Enum, EnumElement

alias DEVICE_TYPE_CPU = 0
alias DEVICE_TYPE_CUDA = 1
alias DEVICE_TYPE_METAL = 2
alias DEVICE_TYPE_ROCM = 3
alias DEVICE_TYPE_VULKAN = 4

struct DeviceType(EnumElement):
    var value: Int

    @staticmethod
    fn cpu() -> DeviceType:
        return DeviceType(DEVICE_TYPE_CPU)

    @staticmethod
    fn cuda() -> DeviceType:
        return DeviceType(DEVICE_TYPE_CUDA)

    @staticmethod
    fn metal() -> DeviceType:
        return DeviceType(DEVICE_TYPE_METAL)

    @staticmethod
    fn rocm() -> DeviceType:
        return DeviceType(DEVICE_TYPE_ROCM)

    @staticmethod
    fn vulkan() -> DeviceType:
        return DeviceType(DEVICE_TYPE_VULKAN)

struct DeviceCaps:
    var device_type: DeviceType
    var total_memory_bytes: Int
    var compute_units: Int
    var max_shared_memory_bytes: Int
    var supports_fp16: Bool
    var supports_bf16: Bool
    var supports_int8: Bool
    var supports_int4: Bool
    var warp_size: Int

    @staticmethod
    fn cpu() -> DeviceCaps:
        return DeviceCaps(
            device_type=DeviceType.cpu(),
            total_memory_bytes=0,
            compute_units=1,
            max_shared_memory_bytes=0,
            supports_fp16=False,
            supports_bf16=False,
            supports_int8=False,
            supports_int4=False,
            warp_size=1,
        )

trait Device:
    fn name(self) -> String
    fn allocate(self, bytes: Int, alignment: Int) raises -> DeviceBuffer
    fn free(self, buf: DeviceBuffer)
    fn copy_to_device(self, src: UnsafePointer[UInt8], dst: DeviceBuffer, bytes: Int) raises
    fn copy_to_host(self, src: DeviceBuffer, dst: UnsafePointer[UInt8], bytes: Int) raises
    fn synchronize(self) raises
    fn capabilities(self) -> DeviceCaps

struct DeviceBuffer:
    var ptr: UnsafePointer[UInt8]
    var size: Int
    var device_type: DeviceType

    @staticmethod
    fn cpu(buffer: UnsafePointer[UInt8], size: Int) -> DeviceBuffer:
        return DeviceBuffer(ptr=buffer, size=size, device_type=DeviceType.cpu())

    fn is_cpu(ref self) -> Bool:
        return self.device_type.value == DEVICE_TYPE_CPU