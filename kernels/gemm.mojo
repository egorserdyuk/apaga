from std.memory import UnsafePointer
from std.simd import SIMD
from tensor import DType, TensorView

fn gemm_cpu_f32(
    a: TensorView[DType.float32],
    b: TensorView[DType.float32],
    c: TensorView[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    let a_ptr = a.ptr
    let b_ptr = b.ptr
    let c_ptr = c.ptr

    for i in range(m):
        for j in range(n):
            var sum: Float32 = 0.0
            for p in range(k):
                let a_val = a_ptr[i * k + p]
                let b_val = b_ptr[p * n + j]
                sum = sum + a_val * b_val
            c_ptr[i * n + j] = sum

fn gemm[dtype: DType](
    a: TensorView[dtype],
    b: TensorView[dtype],
    mut c: TensorView[dtype],
) raises:
    if dtype != DType.float32():
        raise Error("GEMM only supports float32 for now")

    let a_shape = a.shape
    let b_shape = b.shape
    let c_shape = c.shape

    if a_shape.ndim != 2 or b_shape.ndim != 2 or c_shape.ndim != 2:
        raise Error("GEMM requires 2D tensors")

    let m = a_shape.dims[0]
    let k = a_shape.dims[1]
    let k_b = b_shape.dims[0]
    let n = b_shape.dims[1]

    if k != k_b:
        raise Error("GEMM inner dimension mismatch")

    gemm_cpu_f32(a, b, c, m, n, k)