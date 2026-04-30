from std.collections import InlineArray
from .shape import MAX_DIMS, Shape

@fieldwise_init
struct Stride(Copyable):
    var data: InlineArray[Int, MAX_DIMS]
    var ndim: Int

    def __init__(out self):
        self.data = InlineArray[Int, MAX_DIMS](fill=0)
        self.ndim = 0

    @staticmethod
    fn row_major(shape: Shape) -> Stride:
        var stride = Stride()
        stride.ndim = shape.ndim
        if shape.ndim == 0:
            return stride^

        stride.data[shape.ndim - 1] = 1
        for i in range(shape.ndim - 2, -1, -1):
            stride.data[i] = stride.data[i + 1] * shape.dims[i + 1]
        return stride^

    @staticmethod
    fn col_major(shape: Shape) -> Stride:
        var stride = Stride()
        stride.ndim = shape.ndim
        if shape.ndim == 0:
            return stride^

        stride.data[0] = 1
        for i in range(1, shape.ndim):
            stride.data[i] = stride.data[i - 1] * shape.dims[i - 1]
        return stride^

    fn is_contiguous(ref self, shape: Shape) -> Bool:
        if shape.ndim == 0:
            return True

        var expected = 1
        for i in range(shape.ndim - 1, -1, -1):
            if self.data[i] != expected:
                return False
            expected = expected * shape.dims[i]
        return True

    fn __eq__(ref self, other: Stride) -> Bool:
        if self.ndim != other.ndim:
            return False
        for i in range(self.ndim):
            if self.data[i] != other.data[i]:
                return False
        return True