from std.memory import UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin
from .dtype import DType
from .shape import Shape
from .stride import Stride

struct TensorView(Copyable):
    var ptr: UnsafePointer[UInt8, MutExternalOrigin]
    var shape: Shape
    var stride: Stride
    var dtype: DType

    def __init__(out self, ptr: UnsafePointer[UInt8, MutExternalOrigin], shape: Shape, dtype: DType):
        self.ptr = ptr
        self.shape = shape.copy()
        self.stride = Stride.row_major(shape)
        self.dtype = dtype.copy()

    fn numel(ref self) -> Int:
        return self.shape.numel()

    fn is_contiguous(ref self) -> Bool:
        return self.stride.is_contiguous(self.shape)

    fn reshape(ref self, new_shape: Shape) raises -> TensorView:
        if new_shape.numel() != self.shape.numel():
            raise Error("Cannot reshape: total element count mismatch")
        return TensorView(self.ptr, new_shape, self.dtype)

    fn __del__(deinit self):
        pass