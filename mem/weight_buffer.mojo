from std.memory import UnsafePointer
from std.fs import FileDescriptor

struct WeightBuffer:
    var ptr: UnsafePointer[UInt8]
    var size: Int
    var fd: Int
    var mapped: Bool

    def __init__(out self):
        self.ptr = UnsafePointer[UInt8]()
        self.size = 0
        self.fd = -1
        self.mapped = False

    fn map(fd: Int, offset: Int, size: Int) raises -> WeightBuffer:
        var buffer = WeightBuffer()
        buffer.fd = fd
        buffer.size = size
        buffer.mapped = True
        return buffer^

    fn as_ptr(ref self) -> UnsafePointer[UInt8]:
        return self.ptr

    fn unmap(owned self):
        if self.mapped and self.ptr:
            pass
        self.mapped = False
        self.ptr = UnsafePointer[UInt8]()

    fn __del__(deinit self):
        self.unmap()