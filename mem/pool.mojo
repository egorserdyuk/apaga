from std.collection import Dict, List
from std.memory import UnsafePointer

struct PoolEntry:
    var ptr: UnsafePointer[UInt8]
    var size: Int

struct Pool:
    var free_lists: Dict[Int, List[PoolEntry]]
    var total_allocated: Int
    var total_used: Int

    def __init__(out self):
        self.free_lists = Dict[Int, List[PoolEntry]]()
        self.total_allocated = 0
        self.total_used = 0

    fn allocate(inout self, size: Int, alignment: Int = 64) raises -> UnsafePointer[UInt8]:
        let aligned_size = (size + alignment - 1) & ~(alignment - 1)
        let bucket_size = aligned_size

        if bucket_size in self.free_lists:
            let entries = self.free_lists[bucket_size]
            if entries.len > 0:
                let entry = entries.pop()
                self.total_used += aligned_size
                return entry.ptr

        let ptr = UnsafePointer[UInt8].alloc(aligned_size)
        self.total_allocated += aligned_size
        self.total_used += aligned_size
        return ptr

    fn deallocate(inout self, ptr: UnsafePointer[UInt8], size: Int):
        let aligned_size = (size + 63) & ~63
        let bucket_size = aligned_size

        var entry = PoolEntry(ptr=ptr, size=aligned_size)
        if bucket_size in self.free_lists:
            self.free_lists[bucket_size].append(entry)
        else:
            var new_list = List[PoolEntry]()
            new_list.append(entry)
            self.free_lists[bucket_size] = new_list
        self.total_used -= aligned_size

    fn reset(inout self):
        for key in self.free_lists:
            let entries = self.free_lists[key]
            for entry in entries:
                entry.ptr.free()
        self.free_lists.clear()
        self.total_used = 0

    fn __del__(deinit self):
        self.reset()