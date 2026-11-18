import ast
from pathlib import Path


def test_training_has_no_physical_gpu_id():
    source = Path(__file__).parents[1] / "training/train_v2.py"
    tree = ast.parse(source.read_text())
    literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert not any("physical GPU" in value for value in literals)
