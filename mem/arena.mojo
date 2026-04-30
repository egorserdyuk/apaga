from std.memory import UnsafePointer, alloc
from std.builtin.type_aliases import MutExternalOrigin

struct Arena:
    var base: UnsafePointer[UInt8, MutExternalOrigin]
    var capacity: Int

    def __init__(out self, capacity: Int) raises:
        if capacity <= 0:
            raise Error("Arena capacity must be positive")
        self.base = alloc[UInt8](capacity)
        self.capacity = capacity

    fn available(self) -> Int:
        return self.capacity

    fn __del__(deinit self):
        self.base.free()