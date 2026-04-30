from .graph import OpGraph

struct PassConfig:
    var enable_shape_inference: Bool
    var enable_constant_folding: Bool
    var enable_dead_node_elimination: Bool
    var enable_dtype_insertion: Bool
    var enable_fusion: Bool
    var enable_memory_planning: Bool
    var debug_dump: Bool

    @staticmethod
    fn default() -> PassConfig:
        return PassConfig(
            enable_shape_inference=True,
            enable_constant_folding=True,
            enable_dead_node_elimination=True,
            enable_dtype_insertion=True,
            enable_fusion=True,
            enable_memory_planning=True,
            debug_dump=False,
        )

struct Pass:
    var name: String

    fn run(self, graph: OpGraph, config: PassConfig) raises -> OpGraph:
        return graph^

struct ShapeInferencePass(Pass):
    var name: String = "shape_inference"

    fn run(self, graph: OpGraph, config: PassConfig) raises -> OpGraph:
        return graph^

struct ConstantFoldingPass(Pass):
    var name: String = "constant_folding"

    fn run(self, graph: OpGraph, config: PassConfig) raises -> OpGraph:
        return graph^

struct DeadNodeEliminationPass(Pass):
    var name: String = "dead_node_elimination"

    fn run(self, graph: OpGraph, config: PassConfig) raises -> OpGraph:
        return graph^

struct DtypeInsertionPass(Pass):
    var name: String = "dtype_insertion"

    fn run(self, graph: OpGraph, config: PassConfig) raises -> OpGraph:
        return graph^

struct OperatorFusionPass(Pass):
    var name: String = "operator_fusion"

    fn run(self, graph: OpGraph, config: PassConfig) raises -> OpGraph:
        return graph^

struct MemoryPlanningPass(Pass):
    var name: String = "memory_planning"

    fn run(self, graph: OpGraph, config: PassConfig) raises -> OpGraph:
        return graph^

struct PassPipeline:
    var passes: List[Pass]
    var config: PassConfig

    def __init__(out self, config: PassConfig):
        self.config = config
        self.passes = List[Pass]()

    fn add_pass(inout self, pass: Pass):
        self.passes.append(pass)

    fn run(inout self, graph: OpGraph) raises -> OpGraph:
        var current_graph = graph^
        for pass in self.passes:
            current_graph = pass.run(current_graph, self.config)
        return current_graph^

    @staticmethod
    fn create_default(config: PassConfig) -> PassPipeline:
        var pipeline = PassPipeline(config)
        if config.enable_shape_inference:
            pipeline.add_pass(ShapeInferencePass())
        if config.enable_constant_folding:
            pipeline.add_pass(ConstantFoldingPass())
        if config.enable_dead_node_elimination:
            pipeline.add_pass(DeadNodeEliminationPass())
        if config.enable_dtype_insertion:
            pipeline.add_pass(DtypeInsertionPass())
        if config.enable_fusion:
            pipeline.add_pass(OperatorFusionPass())
        if config.enable_memory_planning:
            pipeline.add_pass(MemoryPlanningPass())
        return pipeline