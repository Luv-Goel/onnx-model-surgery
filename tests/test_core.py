"""Tests for ONNX Model Surgery core."""

import pytest
import onnx
from onnx import helper, TensorProto
import numpy as np
from pathlib import Path
import tempfile


@pytest.fixture
def simple_model():
    """Create a minimal ONNX model: input -> Relu -> output."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10])
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [10, 5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 5])

    matmul = helper.make_node("MatMul", ["X", "W"], ["matmul_out"], name="matmul_1")
    relu = helper.make_node("Relu", ["matmul_out"], ["relu_out"], name="relu_1")
    identity = helper.make_node("Identity", ["relu_out"], ["Y"], name="identity_1")

    graph = helper.make_graph(
        nodes=[matmul, relu, identity],
        name="test_graph",
        inputs=[X, W],
        outputs=[Y],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    return model


def test_load_model(simple_model, tmp_path):
    path = tmp_path / "test.onnx"
    onnx.save(simple_model, str(path))
    from onnx_surgery.core import load_model
    loaded = load_model(str(path))
    assert len(loaded.graph.node) == 3


def test_model_summary(simple_model):
    from onnx_surgery.core import model_summary
    s = model_summary(simple_model)
    assert s["node_count"] == 3
    assert s["input_count"] == 2
    assert s["output_count"] == 1
    assert "Relu" in s["op_types"]


def test_list_nodes(simple_model):
    from onnx_surgery.core import list_nodes
    nodes = list_nodes(simple_model)
    assert len(nodes) == 3
    assert nodes[0]["op_type"] == "MatMul"


def test_surgery_graph(simple_model):
    from onnx_surgery.core import SurgeryGraph
    g = SurgeryGraph.from_model(simple_model)
    assert g.node_count() == 3
    assert len(g.find_nodes_by_op("Relu")) == 1
    assert g.find_node("matmul_1") == 0


def test_prune_op_types(simple_model):
    from onnx_surgery.tools.prune import prune_nodes
    result = prune_nodes(simple_model, op_types=["Identity"])
    assert len(result.graph.node) == 2


def test_export_validate(simple_model):
    from onnx_surgery.tools.export import validate
    issues = validate(simple_model)
    assert len(issues) == 0


def test_strip_initializers(simple_model):
    from onnx_surgery.tools.prune import strip_initializers
    result = strip_initializers(simple_model)
    assert result is not None


def test_optimize_identity(simple_model):
    from onnx_surgery.tools.export import optimize
    result = optimize(simple_model, level="basic")
    # Identity should be removed
    ops = [n.op_type for n in result.graph.node]
    assert "Identity" not in ops


def test_inspect(simple_model):
    from onnx_surgery.tools.inspect import inspect, to_json
    report = inspect(simple_model, detailed=False)
    assert "test_graph" in report
    json_out = to_json(simple_model)
    assert "MatMul" in json_out


def test_graphviz_dot(simple_model):
    from onnx_surgery.core.visualization import generate_graphviz
    dot = generate_graphviz(simple_model)
    assert "digraph" in dot
    assert "MatMul" in dot


def test_ascii_graph(simple_model):
    from onnx_surgery.core import ascii_graph
    out = ascii_graph(simple_model)
    assert "Relu" in out


def test_prune_by_threshold(simple_model):
    from onnx_surgery.tools.prune import prune_by_threshold
    result = prune_by_threshold(simple_model, min_node_count=4)
    assert len(result.graph.node) <= 3
