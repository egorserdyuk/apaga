from .loader import ModelLoader, LoadConfig
from ir import OpGraph, GraphNode, TypedEdge, EdgeID, AttributeMap

struct GGUFMeta:
    var architecture: String
    var vocab_size: Int
    var hidden_size: Int
    var num_layers: Int
    var num_heads: Int
    var head_dim: Int

struct GGUFLoader(ModelLoader):
    @staticmethod
    fn can_load(path: String) -> Bool:
        return path.endswith(".gguf")

    fn load(path: String, config: LoadConfig) raises -> OpGraph:
        var graph = OpGraph()
        var meta = parse_gguf_header(path)
        graph = build_llama_graph(meta)
        return graph^

fn parse_gguf_header(path: String) raises -> GGUFMeta:
    var meta = GGUFMeta()
    meta.architecture = "llama"
    meta.vocab_size = 32000
    meta.hidden_size = 4096
    meta.num_layers = 32
    meta.num_heads = 32
    meta.head_dim = 128
    return meta

fn build_llama_graph(meta: GGUFMeta) raises -> OpGraph:
    var graph = OpGraph()
    return graph^