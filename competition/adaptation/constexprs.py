"""Catch the packed-constexpr trap before the platform does.

Real-hardware failure this exists for
-------------------------------------
Submission ``task111-2026-10-02T08:07:24+08:00-08090c505f03`` routed Iluvatar to
a member that passed the nineteen launch-time scalars as four ``tl.constexpr``
tuples and unpacked them inside the kernel with plain assignments::

    CACHE = SHAPE[0]
    ROW_BLOCK = SHAPE[4]
    ...
    row = chunk * ROW_BLOCK + tl.arange(0, ROW_BLOCK)

Every case failed to compile::

    ValueError: arange's arguments must be of type tl.constexpr
    CompilationError: arange's arguments must be of type tl.constexpr

The mechanism is that ``constexpr.__getitem__`` is defined as
``return self.value.__getitem__(*args)`` -- it hands back the *raw* element, not
another ``constexpr``.  A literal written in the kernel body is wrapped by the
frontend, but a name bound from a constexpr subscript is a plain Python int by
the time ``tl.arange`` sees it, and this build requires the mark.

What is checked
---------------
For every ``@triton.jit`` function: a value that (a) is a subscript of a
``tl.constexpr`` parameter, or (b) is a name assigned from such a value, must not
be passed to ``tl.arange`` unless it is re-wrapped by ``tl.constexpr(...)``.
The rule is deliberately conservative -- it does not try to model control flow --
because the remedy (wrap it) is always correct and costs nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

WRAPPERS = {"tl.constexpr", "constexpr", "language.constexpr"}
ARANGE = "arange"


def _annotation_constexpr(node: ast.AST | None) -> bool:
    if node is None:
        return False
    try:
        text = ast.unparse(node)
    except Exception:  # pragma: no cover - unparse is best effort
        return False
    return "constexpr" in text


def _is_constexpr_call(node: ast.AST) -> bool:
    """True when ``node`` re-wraps its argument with ``tl.constexpr(...)``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "constexpr"
    )


def _contains_subscript_of(node: ast.AST, names: set[str]) -> bool:
    """True when a bare ``CONSTPARAM[...]`` appears outside a ``tl.constexpr``.

    ``tl.constexpr(SHAPE[4])`` re-wraps the element, so the subscript inside the
    wrapper is fine and its subtree is skipped.
    """
    if _is_constexpr_call(node):
        return False
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        if node.value.id in names:
            return True
    return any(_contains_subscript_of(child, names) for child in ast.iter_child_nodes(node))


def _arange_arguments(node: ast.AST):
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == ARANGE
        ):
            for arg in child.args:
                yield child, arg


def check_source(path: str | Path) -> list[str]:
    """Return one message per unpacked constexpr used as an ``arange`` bound."""
    path = Path(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for function in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        constparams = {
            arg.arg
            for arg in list(function.args.args) + list(function.args.kwonlyargs)
            if _annotation_constexpr(arg.annotation)
        }
        if not constparams:
            continue
        unwrapped: set[str] = set()
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and _contains_subscript_of(
                node.value, constparams
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        unwrapped.add(target.id)
        if not unwrapped:
            continue
        for call, argument in _arange_arguments(function):
            if _is_constexpr_call(argument):
                continue
            names = {n.id for n in ast.walk(argument) if isinstance(n, ast.Name)}
            offenders = names & unwrapped
            if offenders:
                errors.append(
                    f"{path.name}:{call.lineno}: tl.arange bound "
                    f"{sorted(offenders)} came from a tl.constexpr parameter "
                    f"subscript, which returns a plain value; wrap it with "
                    f"tl.constexpr(...) or the target compiler rejects the kernel "
                    f'with "arange\'s arguments must be of type tl.constexpr"'
                )
    return errors
