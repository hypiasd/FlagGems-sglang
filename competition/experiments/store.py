"""Immutable experiment inputs and append-only execution attempts."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from competition import members

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "competition/.local"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )
    temporary.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def profile(task):
    if not re.fullmatch(r"task\d+", task):
        raise ValueError("task must be taskNN")
    return load_json(ROOT / "competition" / task / "profile.json")


def run_path(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("invalid run ID")
    return LOCAL / "runs" / run_id


def create(task, source, hypothesis, parent=None, seed=0, requirements=None):
    if not hypothesis.strip():
        raise ValueError("a concrete hypothesis is required")
    contract = profile(task)
    if requirements is not None and (
        not isinstance(requirements, dict)
        or any(
            key != "bf16_tensor_core" or not isinstance(value, bool)
            for key, value in requirements.items()
        )
    ):
        raise ValueError("supported requirement: bf16_tensor_core boolean")
    if parent and load_json(run_path(parent) / "run.json")["task_id"] != task:
        raise ValueError("parent belongs to another task")
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    root = run_path(run_id)
    source = Path(source).resolve()
    if source.is_file():
        files = [source]
        if source.name not in contract["package_members"]:
            raise ValueError(f"{source.name} is not a declared package member")
    else:
        layout = members.audit(
            contract["operator"],
            source,
            contract["targets"],
            contract["package_members"],
        )
        if not layout["passed"]:
            raise ValueError(
                "package layout is inconsistent: "
                + "; ".join(layout["errors"])
            )
        files = [
            source / name
            for name in contract["package_members"]
            if (source / name).is_file()
        ]
    if not files or contract["operator"] + ".py" not in [
        p.name for p in files
    ]:
        raise ValueError("source must include the public operator module")
    root.mkdir(parents=True)
    for path in files:
        if path.is_symlink() or path.name not in contract["package_members"]:
            raise ValueError(
                "only declared regular source files may be frozen"
            )
        (root / "source").mkdir(exist_ok=True)
        shutil.copy2(path, root / "source" / path.name)
    # Freeze the harness as well: later changes cannot silently reinterpret a run.
    harness = root / "harness"
    patterns = (
        "competition/*.py",
        "competition/experiments/*.py",
        f"competition/{task}/*.py",
    )
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            if path.name.startswith("test_"):
                continue
            if path.parent.name == task and path.name not in {
                "__init__.py",
                "adapter.py",
                "validate_cpu.py",
            }:
                continue
            dest = harness / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    write_json(root / "contract.json", contract)
    hashes = {
        p.relative_to(root).as_posix(): digest(p)
        for folder in (root / "source", harness)
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }
    hashes["contract.json"] = digest(root / "contract.json")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "task_id": task,
        "created_at": now(),
        "hypothesis": hypothesis,
        "parent": parent,
        "seed": seed,
        "source_method": "agent-authored",
        "requirements": (
            requirements
            if requirements is not None
            else contract["hardware_requirements"]
        ),
        "hashes": hashes,
        "snapshot_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
    }
    write_json(root / "run.json", manifest)
    (root / "snapshot.sha256").write_text(digest(root / "run.json") + "\n")
    event(root, "created", {"snapshot_sha256": manifest["snapshot_sha256"]})
    return manifest


def verify(root):
    root = Path(root)
    manifest = load_json(root / "run.json")
    if (root / "snapshot.sha256").read_text().strip() != digest(
        root / "run.json"
    ):
        raise ValueError(
            "experiment metadata changed; create a new experiment"
        )
    expected = manifest["hashes"]
    actual_paths = {
        p.relative_to(root).as_posix()
        for folder in (root / "source", root / "harness")
        for p in folder.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    if actual_paths | {"contract.json"} != set(expected):
        raise ValueError("frozen snapshot file set changed")
    if any(
        not (root / path).is_file() or digest(root / path) != value
        for path, value in expected.items()
    ):
        raise ValueError(
            "frozen snapshot hash mismatch; create a new experiment"
        )
    if (
        hashlib.sha256(
            json.dumps(expected, sort_keys=True).encode()
        ).hexdigest()
        != manifest["snapshot_sha256"]
    ):
        raise ValueError("snapshot manifest hash mismatch")
    return manifest


def event(root, kind, payload):
    with (Path(root) / "events.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {"at": now(), "event": kind, **payload}, ensure_ascii=False
            )
            + "\n"
        )


def attempts(root):
    return [
        load_json(path)
        for path in sorted((Path(root) / "attempts").glob("*/report.json"))
    ]
