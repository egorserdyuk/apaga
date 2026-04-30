from tensor import DType
from hal import DeviceType
from ir import OpGraph
from ir.passes import PassConfig

struct LoadConfig:
    var target_dtype: DType
    var device_hint: DeviceType
    var memory_budget_bytes: Int
    var pass_config: PassConfig

    @staticmethod
    fn default() -> LoadConfig:
        return LoadConfig(
            target_dtype=DType.float16(),
            device_hint=DeviceType.cpu(),
            memory_budget_bytes=0,
            pass_config=PassConfig.default(),
        )

trait ModelLoader:
    fn can_load(path: String) -> Bool
    fn load(path: String, config: LoadConfig) raises -> OpGraph