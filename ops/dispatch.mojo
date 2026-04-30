from tensor import DType
from hal import DeviceType
from .registry import OperatorRegistry, get_global_registry

fn dispatch_operator(
    op_name: String,
    dtype: DType,
    device: DeviceType,
) raises:
    let registry = get_global_registry()
    let kernel = registry.best_for(op_name, dtype, device)
    kernel