import ast
import hashlib
from pathlib import Path

class ShapeNormalizer(ast.NodeTransformer):
    """Erase names/docstrings/literal values while retaining program shape."""

    def visit_Constant(self, node: ast.Constant):
        node.value = f"<constant:{type(node.value).__name__}>"
        return node

    def visit_Name(self, node: ast.Name):
        node.id = "_NAME_"
        return node

    def visit_arg(self, node: ast.arg):
        node.arg = "_ARG_"
        node.annotation = self.visit(node.annotation) if node.annotation is not None else None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.name = "_FUNCTION_"
        return self._visit_body(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        node.name = "_FUNCTION_"
        return self._visit_body(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        node.name = "_CLASS_"
        return self._visit_body(node)

    def _visit_body(self, node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return self.generic_visit(node)


def normalized_ast(path: Path) -> tuple[str, dict[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    normalized = ShapeNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    structure = ast.dump(normalized, include_attributes=False)
    counts = {
        "functions": sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)),
        "for_loops": sum(isinstance(n, (ast.For, ast.AsyncFor)) for n in ast.walk(tree)),
        "while_loops": sum(isinstance(n, ast.While) for n in ast.walk(tree)),
        "branches": sum(isinstance(n, ast.If) for n in ast.walk(tree)),
        "loads": sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "load" for n in ast.walk(tree)),
        "stores": sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "store" for n in ast.walk(tree)),
        "launches": sum(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id.startswith("_") for n in ast.walk(tree)),
    }
    return hashlib.sha256(structure.encode("utf-8")).hexdigest(), counts


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
