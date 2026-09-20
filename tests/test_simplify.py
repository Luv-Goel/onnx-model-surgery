import pytest
import onnx
from onnx import helper, TensorProto
from onnx_surgery.tools.simplify import simplify

@pytest.fixture
def simple_model():
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

def test_simplify_basic(simple_model):
    simplified = simplify(simple_model)
    # Check that it runs without breaking the model
    ops = [n.op_type for n in simplified.graph.node]
    assert "Identity" not in ops
    assert "MatMul" in ops
