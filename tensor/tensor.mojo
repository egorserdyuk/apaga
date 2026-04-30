from std.memory import UnsafePointer, alloc
from .dtype import DType
from .shape import Shape
from .view import TensorView

struct Tensor(Copyable):
    var dtype: DType
    var view: TensorView
    var owns_memory: Bool

    def __init__(out self, dtype: DType, shape: Shape, owned: Bool = True) raises:
        if shape.numel() == 0:
            raise Error("Cannot create tensor with zero elements")
        self.dtype = dtype.copy()
        var ptr = alloc[UInt8](shape.numel() * dtype.bytes_per_elem())
        self.view = TensorView(ptr, shape.copy(), dtype.copy())
        self.owns_memory = owned

    @staticmethod
    fn zeros(dtype: DType, shape: Shape) raises -> Tensor:
        var tensor = Tensor(dtype, shape)
        var n = tensor.view.shape.numel()
        var elem_size = dtype.bytes_per_elem()
        for i in range(n * elem_size):
            tensor.view.ptr[i] = UInt8(0)
        return tensor^

    @staticmethod
    fn ones(dtype: DType, shape: Shape) raises -> Tensor:
        var tensor = Tensor(dtype, shape)
        var n = tensor.view.shape.numel()
        var elem_size = dtype.bytes_per_elem()
        for i in range(n * elem_size):
            tensor.view.ptr[i] = UInt8(1)
        return tensor^

    fn view_tensor(ref self) -> TensorView:
        return self.view.copy()

    fn reshape(ref self, new_shape: Shape) raises -> Tensor:
        return Tensor(self.dtype, new_shape, self.owns_memory)

    fn __del__(deinit self):
        if self.owns_memory:
            self.view.ptr.free()