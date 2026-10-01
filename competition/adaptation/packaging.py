"""Task-declared ZIP contents and historical artifact lookup."""

from __future__ import annotations
import ast
from pathlib import Path
import zipfile
from competition import members
from competition.adaptation import safety
from competition.experiments.store import ROOT, LOCAL, digest, load_json


def resolve_artifact(task_root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path must be relative")
    for path in (
        Path(task_root) / relative,
        ROOT
        / "competition/archive/legacy/competition"
        / Path(task_root).name
        / relative,
        LOCAL / "archive/competition" / Path(task_root).name / relative,
    ):
        if path.is_file():
            return path
    return Path(task_root) / relative


def validate_package(source, archive, contract):
    expected = set(contract["package_members"])
    errors = []
    layout = members.audit(
        contract["operator"],
        source,
        contract["targets"],
        contract["package_members"],
    )
    errors.extend(layout["errors"])
    try:
        with zipfile.ZipFile(archive) as stream:
            names = stream.namelist()
            if len(names) != len(set(names)) or set(names) != expected:
                errors.append(
                    "ZIP must contain exactly the task-declared root members, without duplicates"
                )
            for name in expected:
                path = Path(source) / name
                if name not in names or not path.is_file():
                    errors.append(f"missing {name}")
                    continue
                data = stream.read(name)
                if data != path.read_bytes():
                    errors.append(f"source/package mismatch: {name}")
                tree = ast.parse(data, filename=name)
                functions = [
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == contract["operator"]
                ]
                signature = contract["entrypoint"].split("(", 1)[1].rstrip(")")
                args = [p.strip() for p in signature.split(",") if p.strip()]
                if (
                    len(functions) != 1
                    or [arg.arg for arg in functions[0].args.args] != args
                ):
                    errors.append(f"public entry/signature mismatch: {name}")
                # One implementation of the platform-safety rules, shared with
                # the task tests.  No copy here: two copies of one rule already
                # drifted once and silently blocked every task111 candidate.
                errors.extend(safety.check_source(Path(source) / name))
    except (OSError, zipfile.BadZipFile, SyntaxError) as exc:
        errors.append(str(exc))
    return {
        "passed": not errors,
        "errors": errors,
        "archive_sha256": digest(archive) if Path(archive).is_file() else None,
    }


def make_package(source, archive, contract):
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED
    ) as stream:
        for name in contract["package_members"]:
            data = (Path(source) / name).read_bytes()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            stream.writestr(info, data)
    return validate_package(source, archive, contract)
