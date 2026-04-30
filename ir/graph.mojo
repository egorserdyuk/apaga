from std.collection import List, Dict
from tensor import DType, Shape

alias MAX_ATTRS = 32

struct EdgeID:
    var value: Int

    @staticmethod
    fn from_index(i: Int) -> EdgeID:
        return EdgeID(i)

    fn __eq__(ref self, other: EdgeID) -> Bool:
        return self.value == other.value

struct AttributeMap:
    var keys: List[String]
    var int_values: List[Int]
    var float_values: List[float]
    var string_values: List[String]

    def __init__(out self):
        self.keys = List[String]()
        self.int_values = List[Int]()
        self.float_values = List[float]()
        self.string_values = List[String]()

    fn set_int(inout self, key: String, value: Int):
        self.keys.append(key)
        self.int_values.append(value)

    fn get_int(ref self, key: String) -> Int:
        for i in range(self.keys.len):
            if self.keys[i] == key:
                return self.int_values[i]
        return 0

struct GraphNode:
    var id: Int
    var op_name: String
    var attrs: AttributeMap
    var input_edges: List[EdgeID]
    var output_edges: List[EdgeID]

    def __init__(out self, id: Int, op_name: String):
        self.id = id
        self.op_name = op_name
        self.attrs = AttributeMap()
        self.input_edges = List[EdgeID]()
        self.output_edges = List[EdgeID]()

struct TypedEdge:
    var id: EdgeID
    var dtype: DType
    var shape: Shape
    var is_constant: Bool

    def __init__(out self, id: EdgeID, dtype: DType):
        self.id = id
        self.dtype = dtype
        self.shape = Shape()
        self.is_constant = False

struct OpGraph:
    var nodes: List[GraphNode]
    var edges: List[TypedEdge]
    var inputs: List[EdgeID]
    var outputs: List[EdgeID]

    def __init__(out self):
        self.nodes = List[GraphNode]()
        self.edges = List[TypedEdge]()
        self.inputs = List[EdgeID]()
        self.outputs = List[EdgeID]()

    fn add_node(inout self, node: GraphNode):
        self.nodes.append(node)

    fn add_edge(inout self, edge: TypedEdge):
        self.edges.append(edge)

    fn topological_order(ref self) raises -> List[Int]:
        var in_degree = List[Int](repeating=0)
        for i in range(self.nodes.len):
            in_degree.append(0)

        for node in self.nodes:
            for input_edge in node.input_edges:
                in_degree[node.id] += 1

        var queue = List[Int]()
        for i in range(self.nodes.len):
            if in_degree[i] == 0:
                queue.append(i)

        var result = List[Int]()
        while queue.len > 0:
            let node_id = queue.pop()
            result.append(node_id)

        if result.len != self.nodes.len:
            raise Error("Graph has cycles")

        return result

    fn validate(ref self) raises:
        for node in self.nodes:
            if node.id < 0 or node.id >= self.nodes.len:
                raise Error("Invalid node ID: " + String(node.id))

        for edge in self.edges:
            if edge.id.value < 0 or edge.id.value >= self.edges.len:
                raise Error("Invalid edge ID")