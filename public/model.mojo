from loaders import LoadConfig, GGUFLoader, SafeTensorsLoader, APGLoader
from ir import OpGraph
from runtime import SessionConfig

struct ModelLoadError(Error):
    pass

struct Model:
    var _graph: OpGraph
    var _path: String

    def __init__(out self, graph: OpGraph, path: String):
        self._graph = graph
        self._path = path

    @staticmethod
    fn from_file(path: String) raises -> Model:
        let config = LoadConfig.default()
        return Model.from_file_with_config(path, config)

    @staticmethod
    fn from_file_with_config(path: String, config: LoadConfig) raises -> Model:
        if GGUFLoader.can_load(path):
            let loader = GGUFLoader()
            let graph = loader.load(path, config)
            return Model(graph, path)
        elif SafeTensorsLoader.can_load(path):
            let loader = SafeTensorsLoader()
            let graph = loader.load(path, config)
            return Model(graph, path)
        elif APGLoader.can_load(path):
            let loader = APGLoader()
            let graph = loader.load(path, config)
            return Model(graph, path)

        raise ModelLoadError("Unsupported model format: " + path)

    fn create_session(ref self, config: SessionConfig) raises:
        pass

    fn graph(ref self) -> OpGraph:
        return self._graph

    fn path(ref self) -> String:
        return self._path