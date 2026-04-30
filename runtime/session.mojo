from std.collection import List
from tensor import DType
from ir import OpGraph
from mem import Arena
from hal import DeviceType, DeviceCaps
from .plan import ExecutionPlan
from .kv_cache import KVCache

struct SessionConfig:
    var max_seq_len: Int
    var n_threads: Int
    var compute_dtype: DType
    var batch_size_hint: Int

    @staticmethod
    fn default() -> SessionConfig:
        return SessionConfig(
            max_seq_len=4096,
            n_threads=0,
            compute_dtype=DType.float16(),
            batch_size_hint=1,
        )

struct Session:
    var graph: OpGraph
    var plan: ExecutionPlan
    var kv_cache: KVCache
    var activation_arena: Arena
    var device: DeviceType
    var config: SessionConfig

    def __init__(out self, graph: OpGraph, config: SessionConfig) raises:
        self.graph = graph
        self.config = config
        self.device = DeviceType.cpu()
        self.plan = ExecutionPlan.from_graph(graph)
        self.activation_arena = Arena(256 * 1024 * 1024)
        self.kv_cache = KVCache(32, config.max_seq_len, 32, 128)

    fn generate(
        inout self,
        prompt_tokens: List[Int],
        max_tokens: Int,
        temperature: Float32,
        on_token: fn(Int, Float32) -> Bool,
    ) raises:
        pass

    fn embed(inout self, tokens: List[Int]) raises:
        pass

    fn logits(inout self, tokens: List[Int]) raises:
        pass

    fn reset(inout self):
        self.kv_cache.clear()
        self.activation_arena.reset()