"""Graph traversal and manipulation utilities for ONNX models.

Builds a directed graph representation from an ONNX ModelProto,
enabling node insertion, removal, reconnection, and subgraph extraction.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from onnx import ModelProto, helper


@dataclass
class GraphNode:
    """Internal representation of a single ONNX graph node."""
    index: int
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    domain: str = ""


@dataclass
class SurgeryGraph:
    """A mutable graph built from an ONNX model.

    Maintains adjacency lists and provides methods for common
    graph surgery operations.
    """
    nodes: list[GraphNode] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    initializers: set[str] = field(default_factory=set)
    # name -> list of consumer node indices
    consumers: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    # name -> producer node index (or -1 if input/init)
    producers: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_model(cls, model: ModelProto) -> "SurgeryGraph":
        """Build a SurgeryGraph from an ONNX ModelProto."""
        graph = model.graph
        sg = cls()

        # Track initializer names
        sg.initializers = {init.name for init in graph.initializer}
        sg.inputs = [v.name for v in graph.input]
        sg.outputs = [v.name for v in graph.output]

        # Build nodes
        for i, n in enumerate(graph.node):
            node = GraphNode(
                index=i,
                name=n.name or f"node_{i}",
                op_type=n.op_type,
                inputs=list(n.input),
                outputs=list(n.output),
                domain=n.domain or "",
            )
            sg.nodes.append(node)

            # Map outputs -> producer node
            for out in n.output:
                if out:
                    sg.producers[out] = i

        # Build consumer adjacency (input tensor -> which nodes consume it)
        for i, n in enumerate(graph.node):
            for inp in n.input:
                if inp:
                    sg.consumers[inp].append(i)

        return sg

    def find_node(self, name: str) -> int | None:
        """Find a node index by name."""
        for i, n in enumerate(self.nodes):
            if n.name == name:
                return i
        return None

    def find_nodes_by_op(self, *op_types: str) -> list[int]:
        """Find all node indices matching given op types."""
        return [i for i, n in enumerate(self.nodes) if n.op_type in op_types]

    def remove_node(self, index: int) -> None:
        """Remove a node from the graph, leaving its inputs connected to its consumers."""
        if index < 0 or index >= len(self.nodes):
            raise IndexError(f"Node index {index} out of range")
        self.nodes.pop(index)

    def node_count(self) -> int:
        return len(self.nodes)

    def topological_order(self) -> list[int]:
        """Return node indices in topological order (BFS from inputs)."""
        in_degree = [0] * len(self.nodes)
        adj: dict[int, list[int]] = defaultdict(list)

        # Build adjacency: which nodes consume outputs of which nodes
        for name, consumers in self.consumers.items():
            producer = self.producers.get(name)
            if producer is not None:
                for c in consumers:
                    if c < len(self.nodes):
                        adj[producer].append(c)
                        in_degree[c] += 1

        queue = deque([i for i in range(len(self.nodes)) if in_degree[i] == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def subgraph(self, start_nodes: list[int], end_nodes: list[int]) -> "SurgeryGraph":
        """Extract a subgraph between start and end node indices."""
        keep = set(end_nodes)
        queue = deque(end_nodes)
        visited = set()

        # Walk backwards from end nodes
        while queue:
            n = queue.popleft()
            if n in visited or n >= len(self.nodes):
                continue
            visited.add(n)
            for inp in self.nodes[n].inputs:
                producer = self.producers.get(inp)
                if producer is not None and producer not in visited:
                    queue.append(producer)
                    keep.add(producer)

        sub = SurgeryGraph()
        old_to_new = {}
        for i, n in enumerate(self.nodes):
            if i in keep:
                old_to_new[i] = len(sub.nodes)
                sub.nodes.append(n)

        sub.inputs = [n for n in self.inputs if n in self.consumers]
        sub.outputs = self.outputs
        sub.initializers = self.initializers.copy()
        sub.producers = {k: old_to_new[v] for k, v in self.producers.items() if v in old_to_new}
        for name, consumers in self.consumers.items():
            filtered = [c for c in consumers if c in old_to_new]
            if filtered:
                sub.consumers[name] = filtered

        return sub

    def to_model(self, original: ModelProto) -> ModelProto:
        """Convert this graph back into an ONNX ModelProto.

        Preserves original opset imports, metadata, and initializers.
        Only the graph nodes, inputs, and outputs are replaced.
        """
        new_model = ModelProto()
        new_model.CopyFrom(original)
        new_model.graph.ClearField("node")
        new_model.graph.ClearField("output")

        for n in self.nodes:
            new_node = helper.make_node(
                op_type=n.op_type,
                inputs=n.inputs,
                outputs=n.outputs,
                name=n.name,
                domain=n.domain or "",
            )
            new_model.graph.node.append(new_node)

        return new_model
