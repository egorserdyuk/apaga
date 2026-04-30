from std.collection import Dict
from tensor import DType
from hal import DeviceType

alias MAX_OP_INPUTS = 16
alias MAX_OP_OUTPUTS = 16

struct OperatorKey:
    var op_name: String
    var dtype: DType
    var device: DeviceType

    def __init__(out self, op_name: String, dtype: DType, device: DeviceType):
        self.op_name = op_name
        self.dtype = dtype
        self.device = device

    fn __hash__(ref self) -> UInt64:
        var h: UInt64 = 0
        h = h ^ UInt64(self.op_name.__hash__)
        h = h ^ UInt64(self.dtype.value * 1000)
        h = h ^ UInt64(self.device.value * 1000000)
        return h

    fn __eq__(ref self, other: OperatorKey) -> Bool:
        return self.op_name == other.op_name and self.dtype == other.dtype and self.device == other.device

alias KernelFn = fn(
    inputs: TensorView[DType],
    outputs: TensorView[DType],
) raises -> None

struct OperatorRegistry:
    var table: Dict[OperatorKey, KernelFn]

    def __init__(out self):
        self.table = Dict[OperatorKey, KernelFn]()

    fn register(inout self, key: OperatorKey, fn: KernelFn):
        self.table[key] = fn

    fn dispatch(ref self, key: OperatorKey) raises -> KernelFn:
        if key in self.table:
            return self.table[key]
        raise Error("No kernel registered for: " + key.op_name)

    fn best_for(ref self, op_name: String, dtype: DType, device: DeviceType) raises -> KernelFn:
        let key = OperatorKey(op_name, dtype, device)
        if key in self.table:
            return self.table[key]

        raise Error("No kernel found for: " + op_name)

var _global_registry = OperatorRegistry()

fn get_global_registry() -> OperatorRegistry:
    return _global_registry