"""Who is in a submission package.

A competition package is ``<op>.py`` plus optional ``<op>_<chip>.py`` files.
The platform routes a target to its suffixed file when one exists and to the
generic file otherwise.  That routing is observed behaviour, not documented
behaviour: on Task 103 the generic file gave Kunlunxin 0.09x while the
suffixed file gave 1.85x, with the generic file byte-identical in both.

The member list is easy to get wrong by hand, and the failure is silent.  A
Task 103 package that shipped three files and scored 7/7 existed while
``profile.json`` declared a single member; ``make_package`` writes exactly the
declared members, so any undeclared chip file is dropped without an error and
the resulting ZIP still validates against the shrunken contract.

This module makes the layout explicit and fails closed.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path


def dedicated_name(operator, chip):
    """The suffixed module name a target's dedicated file must use."""
    return f"{operator}_{chip}.py"


def discover(operator, directory, targets, declared):
    """Return the layout of a source directory.

    ``dedicated`` maps a chip to its file only when the suffix is one of this
    task's targets; it is still recorded under its raw suffix when it is not,
    so :func:`audit` can reject it instead of ignoring it.
    """
    directory = Path(directory)
    declared = list(declared or [])
    names = (
        sorted(path.name for path in directory.glob("*.py"))
        if directory.is_dir()
        else []
    )
    generic = f"{operator}.py"
    layout = {
        "generic": generic if generic in names else None,
        "dedicated": {},
        "unknown_chip": {},
        "foreign": [],
    }
    for name in names:
        if name == generic:
            continue
        if name.startswith(f"{operator}_") and name.endswith(".py"):
            suffix = name[len(operator) + 1 : -3]
            key = "dedicated" if suffix in targets else "unknown_chip"
            layout[key][suffix] = name
        else:
            layout["foreign"].append(name)
    layout["declared"] = declared
    return layout


def target_sources(operator, targets, layout):
    """Which file serves each target, and which targets have no file at all."""
    resolved, unresolved = {}, []
    for chip in targets:
        name = layout["dedicated"].get(chip) or layout["generic"]
        if name is None:
            unresolved.append(chip)
        else:
            resolved[chip] = name
    return resolved, unresolved


def audit(operator, directory, targets, declared, package_sha256=None):
    """Fail-closed audit of one package layout.

    Errors are conditions that change what gets shipped or routed.  Warnings
    are informational: a target without a dedicated file is normal (it runs the
    generic implementation), but the count belongs in the report because that
    is where the remaining optimisation headroom is.
    """
    directory = Path(directory)
    declared = list(declared or [])
    layout = discover(operator, directory, targets, declared)
    errors, warnings = [], []
    if not declared:
        errors.append(
            "the task contract declares no package members: the package "
            "layout would be undefined"
        )
    expected = set(layout["dedicated"].values()) | set(
        layout["unknown_chip"].values()
    )
    if layout["generic"]:
        expected.add(layout["generic"])
    for suffix, name in sorted(layout["unknown_chip"].items()):
        errors.append(
            f"{name}: chip suffix {suffix!r} is not one of this task's "
            f"targets {sorted(targets)}; it would never be routed"
        )
    for name in sorted(expected - set(declared)):
        errors.append(
            f"{name}: present in the source directory but not declared in "
            "package_members, so packaging would silently drop it"
        )
    for name in declared:
        if not (directory / name).is_file():
            errors.append(f"{name}: declared in package_members but missing")
    for name in declared:
        if name == layout["generic"]:
            continue
        if name not in expected:
            errors.append(
                f"{name}: declared member is neither the generic module "
                f"{operator}.py nor a <op>_<chip>.py module of this task"
            )
    if layout["generic"] is None:
        errors.append(f"{operator}.py: the generic module is required")
    resolved, unresolved = target_sources(operator, targets, layout)
    for chip in unresolved:
        errors.append(f"{chip}: no module can serve this target")
    without = [chip for chip in targets if chip not in layout["dedicated"]]
    if without:
        warnings.append(
            f"{len(without)} of {len(targets)} target(s) share the generic "
            f"module: {without}"
        )
    if layout["foreign"]:
        warnings.append(
            f"ignored non-module file(s) in the source directory: "
            f"{layout['foreign']}"
        )
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "operator": operator,
        "generic": layout["generic"],
        "dedicated": layout["dedicated"],
        "unknown_chip": layout["unknown_chip"],
        "declared": declared,
        "target_sources": resolved,
        "targets_without_dedicated": without,
        "package_sha256": package_sha256,
    }


def recent_packages(task, operator, local):
    """Member sets of the packages this machine already submitted for a task.

    Used to detect contract drift: if a recently submitted package carried
    files the current contract no longer declares, the next package would
    silently be smaller than the one that scored.

    ``local`` is the ignored ``competition/.local`` directory; it is passed in
    rather than imported so this module stays free of a circular import.
    """
    observed = []
    for directory in sorted((Path(local) / "adaptations").glob("adapt-*")):
        manifest, archive = (
            directory / "adaptation.json",
            directory / "package.zip",
        )
        if not (manifest.is_file() and archive.is_file()):
            continue
        try:
            meta = json.loads(manifest.read_text())
        except ValueError:
            continue
        if meta.get("task_id") != task:
            continue
        try:
            with zipfile.ZipFile(archive) as stream:
                members = sorted(stream.namelist())
        except zipfile.BadZipFile:
            continue
        if f"{operator}.py" not in members:
            continue
        observed.append(
            {
                "adaptation_id": directory.name,
                "created_at": meta.get("created_at"),
                "members": members,
            }
        )
    observed.sort(key=lambda item: item.get("created_at") or "")
    return observed


def drift(task, contract, local):
    """Compare the current contract against recently submitted packages."""
    operator = contract["operator"]
    declared = sorted(contract.get("package_members") or [])
    packages = recent_packages(task, operator, local)
    missing, latest = {}, packages[-1] if packages else None
    for package in packages:
        for name in package["members"]:
            if name not in declared:
                missing.setdefault(name, []).append(package["adaptation_id"])
    return {
        "task_id": task,
        "operator": operator,
        "declared": declared,
        "packages_observed": len(packages),
        "latest_package": latest,
        "shipped_but_not_declared": {
            name: ids for name, ids in sorted(missing.items())
        },
        "drifted": bool(missing),
    }
