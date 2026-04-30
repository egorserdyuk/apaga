from .loader import ModelLoader, LoadConfig
from ir import OpGraph

struct APGLoader(ModelLoader):
    @staticmethod
    fn can_load(path: String) -> Bool:
        return path.endswith(".apg")

    fn load(path: String, config: LoadConfig) raises -> OpGraph:
        var graph = OpGraph()
        return graph^