from tensor import DType, TensorView, Tensor
from .params import QuantParams, QuantScheme

struct QuantizationEngine:
    fn quantize(params: QuantParams, data: TensorView[DType]) raises -> Tensor[DType.int8]:
        raise Error("Quantization not implemented for this scheme")

    fn dequantize(params: QuantParams, data: TensorView[DType]) raises -> Tensor[DType.float32]:
        if params.scheme == QuantScheme.rtn():
            return dequantize_rtn(params, data)
        raise Error("Unsupported quantization scheme")

    fn select_scheme(bits: Int, group_size: Int) -> QuantScheme:
        if bits == 4:
            return QuantScheme.rtn()
        elif bits == 8:
            return QuantScheme.rtn()
        return QuantScheme.rtn()

fn dequantize_rtn(params: QuantParams, data: TensorView[DType]) raises -> Tensor[DType.float32]:
    let numel = data.numel()
    var result = Tensor[DType.float32].zeros(data.shape)
    let num_groups = params.num_groups(numel)

    for i in range(numel):
        let group_idx = i // params.group_size
        let scale = params.scales.ptr[group_idx]
        let zero = params.zeros.ptr[group_idx]
        let qval = data.ptr[i]
        result.view.ptr[i] = Float32(qval) * Float32(scale) + Float32(zero)

    return result^