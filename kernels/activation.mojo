from std.math import sigmoid
from tensor import DType, TensorView

fn silu[dtype: DType](x: TensorView[dtype], result: TensorView[dtype]) raises:
    let n = x.numel()
    if dtype == DType.float32():
        for i in range(n):
            let val = x.ptr[i]
            result.ptr[i] = val * sigmoid(val)
    elif dtype == DType.float16():
        for i in range(n):
            let val = x.ptr[i]
            result.ptr[i] = val * Float16(sigmoid(Float32(val)))
    else:
        raise Error("silu: unsupported dtype")

fn gelu[dtype: DType](x: TensorView[dtype], result: TensorView[dtype]) raises:
    let n = x.numel()
    if dtype == DType.float32():
        for i in range(n):
            let val = x.ptr[i]
            let cdf = 0.5 * (1.0 + sigmoid(1.702 * val))
            result.ptr[i] = val * cdf
    elif dtype == DType.float16():
        for i in range(n):
            let val = Float32(x.ptr[i])
            let cdf = 0.5 * (1.0 + sigmoid(1.702 * val))
            result.ptr[i] = Float16(val * cdf)
    else:
        raise Error("gelu: unsupported dtype")

fn relu[dtype: DType](x: TensorView[dtype], result: TensorView[dtype]) raises:
    let n = x.numel()
    for i in range(n):
        let val = x.ptr[i]
        if val < 0:
            result.ptr[i] = 0
        else:
            result.ptr[i] = val

fn geglu[dtype: DType](x: TensorView[dtype], result: TensorView[dtype]) raises:
    let n = x.numel()
    if n % 2 != 0:
        raise Error("geglu: input size must be even")

    let half = n // 2
    for i in range(half):
        let val0 = x.ptr[i]
        let val1 = x.ptr[i + half]
        result.ptr[i] = val0 * sigmoid(val1)
        result.ptr[i + half] = val1 * sigmoid(val0)