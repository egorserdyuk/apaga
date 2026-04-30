from .loader import ModelLoader, LoadConfig
from ir import OpGraph

struct SafeTensorsLoader(ModelLoader):
    @staticmethod
    fn can_load(path: String) -> Bool:
        return "safetensors" in path

    fn load(path: String, config: LoadConfig) raises -> OpGraph:
        var graph = OpGraph()
        return graph^