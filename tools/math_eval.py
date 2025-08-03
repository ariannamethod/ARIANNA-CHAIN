from __future__ import annotations

import ast
import operator as op
from typing import Dict, Any


def math_eval(expr: str) -> str:
    allowed = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.Mod: op.mod,
        ast.FloorDiv: op.floordiv,
    }

    def eval_(node):
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](eval_(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](eval_(node.left), eval_(node.right))
        raise ValueError("unsupported expression")

    if "**" in expr and "e" in expr.lower():
        raise ValueError("disallowed form")

    node = ast.parse(expr, mode="eval").body
    result = eval_(node)
    if isinstance(result, (int, float)) and abs(result) > 1e12:
        raise ValueError("result too large")
    return str(result)


TOOL_SPEC: Dict[str, Any] = {
    "name": "math.eval",
    "desc": "evaluate a safe arithmetic expression.",
    "args": {"expr": "string"},
    "func": math_eval,
}
