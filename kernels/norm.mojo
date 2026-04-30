from std.math import rsqrt
from tensor import DType, TensorView, Shape

fn rms_norm[dtype: DType](
    x: TensorView[dtype],
    weight: TensorView[dtype],
    eps: Float32,
    result: TensorView[dtype],
) raises:
    if x.shape.ndim != 1:
        raise Error("rms_norm: only 1D tensors supported for now")

    let n = x.shape.dims[0]
    var sum_sq: Float32 = 0.0

    if dtype == DType.float32():
        for i in range(n):
            let val = x.ptr[i]
            sum_sq = sum_sq + Float32(val) * Float32(val)
    else:
        raise Error("rms_norm: unsupported dtype")

    let rms = rsqrt(sum_sq / Float32(n) + eps)

    if dtype == DType.float32():
        for i in range(n):
            let val = x.ptr[i]
            result.ptr[i] = Float32(val) * Float32(weight.ptr[i]) * rms
    else:
        raise Error("rms_norm: unsupported dtype")

fn layer_norm[dtype: DType](
    x: TensorView[dtype],
    weight: TensorView[dtype],
    bias: TensorView[dtype],
    eps: Float32,
    result: TensorView[dtype],
) raises:
    if x.shape.ndim != 1:
        raise Error("layer_norm: only 1D tensors supported for now")

    let n = x.shape.dims[0]
    var mean: Float32 = 0.0
    var sum_sq: Float32 = 0.0

    if dtype == DType.float32():
        for i in range(n):
            mean = mean + Float32(x.ptr[i])
        mean = mean / Float32(n)

        for i in range(n):
            let diff = Float32(x.ptr[i]) - mean
            sum_sq = sum_sq + diff * diff

        let std = rsqrt(sum_sq / Float32(n) + eps)

        for i in range(n):
            let normed = (Float32(x.ptr[i]) - mean) * std
            result.ptr[i] = Float32(normed * Float32(weight.ptr[i]) + Float32(bias.ptr[i]))
    else:
        raise Error("layer_norm: unsupported dtype")