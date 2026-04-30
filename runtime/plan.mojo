from std.collection import List
from tensor import DType
from ir import OpGraph
from ops import KernelFn

alias MAX_OP_INPUTS = 16
alias MAX_OP_OUTPUTS = 16

struct TensorRef:
    var is_arena_offset: Bool
    var offset: Int
    var ptr: UnsafePointer[UInt8]

struct DispatchedOp:
    var kernel: KernelFn
    var inputs: List[TensorRef]
    var outputs: List[TensorRef]
    var thread_group: Int

struct ExecutionPlan:
    var ops: List[DispatchedOp]
    var arena_size: Int

    def __init__(out self):
        self.ops = List[DispatchedOp]()
        self.arena_size = 0

    @staticmethod
    fn from_graph(graph: OpGraph) raises -> ExecutionPlan:
        var plan = ExecutionPlan()
        return plan^

    fn execute(ref self) raises:
        for op in self.ops:
            pass