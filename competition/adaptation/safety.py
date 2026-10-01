"""Static rules the submission platform enforces, learned from real rejections.

Every rule here exists because a submission was **accepted by local checks and
rejected by the platform**, and a rejection costs a submission slot and a day of
feedback latency.  A rule that is only in someone's head is a rule that gets
broken again, so each one names the incident it came from.

Rule 1 -- no ``try``/``except`` anywhere in the module.
    Anti-cheat: an implicit fallback would hide a failing kernel behind a
    different code path.

Rule 2 -- no module-level mutable **dict or set**.
    Incident: Task 111 submission 10-02 07:04 was rejected on every chip with

        Code safety validation failed: Module-level mutable container detected:
        '_PLANS'.  Global dict/set variables can cache results across benchmark
        iterations.  Use local variables instead.

    ``_PLANS`` was the plan cache that every optimisation since variant E leaned
    on, so the whole candidate scored 0/7 on a rule we could have checked
    locally in a microsecond.  ``__all__`` is exempt: Task 112's package carries
    a module-level ``__all__`` list and passes, so the platform's rule targets
    dicts/sets, not lists.

Rule 3 -- ``torch.<attr>(...)`` is limited to allocation and metadata.
    ``torch.tensor`` is allowed only with a literal list first argument (the
    stride/shape metadata buffer); anything computed from a tensor is operator
    data being processed in torch, which the task forbids.

Known scope limit, asserted in the tests rather than assumed away: the scan sees
the ``torch`` namespace only, so compute reached through a tensor *method*
(``dst.clone()``, ``src.to(...)``) is not matched here.  A reference-shaped
implementation is caught by the structural-delta gate instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Allocation and dtype metadata only; no operator data is processed.
ALLOWED_TORCH_ATTRS = {"empty", "empty_like", "new_empty", "promote_types"}

#: Module-level names that may hold a container literal.  ``__all__`` is the
#: package's export list and is present in accepted Task 112 submissions.
CONTAINER_EXEMPT_NAMES = {"__all__"}


def module_level_containers(tree: ast.Module) -> list[tuple[str, int, str]]:
    """``(name, lineno, kind)`` for module-level dict/set assignments.

    Both literals (``{}``, ``{1, 2}``) and the ``dict()``/``set()`` constructors
    count: ``set()`` is how people actually write a module-level set, and a
    ruleset that only matched ``ast.Set`` would have waved it through.
    """
    found = []
    for node in tree.body:
        targets, value = [], None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if value is None:
            continue
        kind = None
        if isinstance(value, ast.Dict):
            kind = "dict"
        elif isinstance(value, ast.Set):
            kind = "set"
        elif (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in ("dict", "set")
        ):
            kind = value.func.id
        if kind is None:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Name)
                and target.id not in CONTAINER_EXEMPT_NAMES
            ):
                found.append((target.id, node.lineno, kind))
    return found


def check_source(path: str | Path) -> list[str]:
    """All known platform-safety errors for one module, in submission order."""
    path = Path(path)
    name = path.name
    try:
        tree = ast.parse(path.read_text(), filename=name)
    except (
        SyntaxError
    ) as exc:  # a module that will not parse cannot be fixed later
        return [f"cannot parse: {name}: {exc}"]

    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            errors.append(
                f"implicit fallback through try/except is forbidden: {name}:{node.lineno}"
            )

    for held, lineno, kind in module_level_containers(tree):
        errors.append(
            f"module-level mutable container detected: {held!r} ({kind}) at "
            f"{name}:{lineno}; the platform rejects global dict/set variables "
            "because they can cache results across benchmark iterations"
        )

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr not in ALLOWED_TORCH_ATTRS
        ):
            if node.func.attr == "tensor":
                if node.args and isinstance(node.args[0], ast.List):
                    continue
                errors.append(
                    f"torch.tensor may only build a literal metadata buffer: "
                    f"{name}:{node.lineno}"
                )
                continue
            errors.append(
                f"native torch compute call requires removal: {name}:{node.lineno}"
            )
    return errors
