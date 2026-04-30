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
    fn from_params(d0: Int, d1: Int) -> Shape:
        var shape = Shape()
        shape.ndim = 2
        shape.dims[0] = d0
        shape.dims[1] = d1
        return shape^

    @staticmethod
    fn from_params_3(d0: Int, d1: Int, d2: Int) -> Shape:
        var shape = Shape()
        shape.ndim = 3
        shape.dims[0] = d0
        shape.dims[1] = d1
        shape.dims[2] = d2
        return shape^

    @staticmethod
    fn from_params_4(d0: Int, d1: Int, d2: Int, d3: Int) -> Shape:
        var shape = Shape()
        shape.ndim = 4
        shape.dims[0] = d0
        shape.dims[1] = d1
        shape.dims[2] = d2
        shape.dims[3] = d3
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