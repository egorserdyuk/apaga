from std.collections import InlineArray

comptime MAX_DIMS: Int = 8

@fieldwise_init
struct Shape(Copyable):
    var dims: InlineArray[Int, MAX_DIMS]
    var ndim: Int

    def __init__(out self):
        self.dims = InlineArray[Int, MAX_DIMS](fill=0)
        self.ndim = 0

    @staticmethod
    fn from_list(values: List[Int]) -> Shape:
        var shape = Shape()
        var n = values.len
        if n > MAX_DIMS:
            n = MAX_DIMS
        shape.ndim = n
        for i in range(n):
            shape.dims[i] = values[i]
        return shape^

    @staticmethod
    fn from_params(num_dims: Int, *dims: Int) -> Shape:
        var shape = Shape()
        var d = num_dims
        if d > MAX_DIMS:
            d = MAX_DIMS
        shape.ndim = d
        for i in range(d):
            shape.dims[i] = dims[i]
        return shape^

    fn numel(ref self) -> Int:
        if self.ndim == 0:
            return 0
        var total = 1
        for i in range(self.ndim):
            total = total * self.dims[i]
        return total

    fn __eq__(ref self, other: Shape) -> Bool:
        if self.ndim != other.ndim:
            return False
        for i in range(self.ndim):
            if self.dims[i] != other.dims[i]:
                return False
        return True

    fn __ne__(ref self, other: Shape) -> Bool:
        return not (self == other)

    fn __repr__(ref self) -> String:
        var result = "Shape("
        for i in range(self.ndim):
            if i > 0:
                result += ", "
            result += String(self.dims[i])
        result += ")"
        return result

fn broadcast_shape(a: Shape, b: Shape) raises -> Shape:
    var result = Shape()
    let max_ndim = a.ndim if a.ndim > b.ndim else b.ndim
    if max_ndim > MAX_DIMS:
        raise Error("Broadcast would exceed MAX_DIMS")
    result.ndim = max_ndim
    return result