from apaga import DType, Shape, Tensor, TensorView, Arena

fn test_shape_2d() raises:
    print("=== Shape 2D ===")
    var s = Shape.from_params(2, 3)
    print("Shape(2,3): numel:", s.numel(), "ndim:", s.ndim)

fn test_shape_3d() raises:
    print("=== Shape 3D ===")
    var s = Shape.from_params_3(2, 3, 4)
    print("Shape(2,3,4): numel:", s.numel(), "ndim:", s.ndim)

fn test_shape_4d() raises:
    print("=== Shape 4D ===")
    var s = Shape.from_params_4(1, 2, 3, 4)
    print("Shape(1,2,3,4): numel:", s.numel(), "ndim:", s.ndim)

fn test_multiple_shapes() raises:
    print("=== Multiple Shapes ===")
    var s1 = Shape.from_params(2, 2)
    var s2 = Shape.from_params(3, 3)
    var s3 = Shape.from_params_3(4, 4, 4)
    print("s1:", s1.numel(), "s2:", s2.numel(), "s3:", s3.numel())

fn test_tensor_creation() raises:
    print("=== Tensor Creation ===")
    var t = Tensor(DType.float32(), Shape.from_params(4, 4))
    print("Tensor(4,4):", t.view_tensor().numel())

fn test_multiple_tensors() raises:
    print("=== Multiple Tensors ===")
    var t1 = Tensor(DType.float32(), Shape.from_params(2, 2))
    var t2 = Tensor(DType.float32(), Shape.from_params(3, 3))
    var t3 = Tensor(DType.float32(), Shape.from_params(4, 4))
    print("t1:", t1.view_tensor().numel())
    print("t2:", t2.view_tensor().numel())
    print("t3:", t3.view_tensor().numel())

fn test_tensor_zeros() raises:
    print("=== Tensor.zeros ===")
    var t = Tensor.zeros(DType.float32(), Shape.from_params(2, 2))
    print("zeros:", t.view_tensor().numel())

fn test_tensor_ones() raises:
    print("=== Tensor.ones ===")
    var t = Tensor.ones(DType.float32(), Shape.from_params(2, 2))
    print("ones:", t.view_tensor().numel())

fn test_reshape() raises:
    print("=== Reshape ===")
    var t = Tensor(DType.float32(), Shape.from_params(2, 3))
    var new_shape = Shape()
    new_shape.ndim = 1
    new_shape.dims[0] = 6
    var r = t.reshape(new_shape)
    print("reshape:", r.view_tensor().numel())

fn test_ownership() raises:
    print("=== Ownership ===")
    var t = Tensor(DType.float32(), Shape.from_params(2, 2))
    print("owns:", t.owns_memory)
    var new_shape = Shape()
    new_shape.ndim = 1
    new_shape.dims[0] = 4
    var r = t.reshape(new_shape)
    print("reshaped:", r.owns_memory)

fn main() raises:
    test_shape_2d()
    test_shape_3d()
    test_shape_4d()
    test_multiple_shapes()
    print("---")
    test_tensor_creation()
    test_multiple_tensors()
    test_tensor_zeros()
    test_tensor_ones()
    test_reshape()
    test_ownership()
    print("\n=== All tests passed ===")