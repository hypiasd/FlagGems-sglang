"""Build an upstream-ready competition PR bundle from an awarded submission.

A bundle is the answer to "the platform accepted this package, now what does
the repository need?".  It holds the same sources re-homed into the dispatcher
tree, formatted the way upstream CI demands, plus the provenance needed to
write an honest PR description.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

from competition.experiments.store import (
    LOCAL,
    ROOT,
    digest,
    load_json,
    now,
    write_json,
)

from . import spec

PR_ROOT = LOCAL / "pr"


def _suffix_of(op: str, stem: str) -> str | None:
    if stem == op:
        return None
    prefix = op + "_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    raise ValueError(f"{stem}.py does not belong to operator {op!r}")


def _ensure_prologue(text: str, op: str) -> str:
    if not text.startswith(spec.HEADER):
        text = spec.HEADER + "\n" + text.lstrip("\n")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines).rstrip("\n") + "\n"
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ValueError(f"awarded source does not parse: {exc}") from exc
    assigned = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets
        )
    ]
    if assigned:
        try:
            value = ast.literal_eval(assigned[0].value)
        except ValueError:
            value = None
        if value != [op]:
            raise ValueError(
                f"existing __all__ is {value!r}, expected [{op!r}]"
            )
        return text
    return text + f'\n\n__all__ = ["{op}"]\n'


def _reflow_long_text(
    text: str, limit: int = spec.FLAKE8_MAX_LINE, width: int = 100
) -> tuple[str, list[int]]:
    """Wrap over-long comment and module-docstring lines.

    Black does not touch comments or strings, but CI's flake8 (E501 at 120)
    still rejects them, so a competition source can be formatter-clean and
    still fail code-style.  Only comments and the *module* docstring are
    touched: the AST comparison ignores both, and every rewrapped line is
    recorded so the PR description can disclose it.
    """
    out, changed = [], []
    in_doc = done = False
    for number, line in enumerate(text.split("\n"), 1):
        doc_line = in_doc
        if not done:
            quotes = line.count('"""')
            if quotes % 2 == 1:
                if in_doc:
                    in_doc, done = False, True
                else:
                    in_doc, doc_line = True, True
            elif quotes:
                doc_line, done = True, True
            elif in_doc:
                doc_line = True
        is_comment = line.lstrip().startswith("#")
        if len(line) <= limit or not (is_comment or doc_line):
            out.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        marker = "# " if is_comment else ""
        body = line.strip()
        if is_comment:
            body = body.lstrip("#").strip()
        prefix = indent + marker if is_comment else indent
        current = prefix
        chunks = []
        for word in body.split(" "):
            candidate = (
                (current + " " + word) if current.strip() else prefix + word
            )
            if len(candidate) > width and current.strip():
                chunks.append(current.rstrip())
                current = prefix + word
            else:
                current = candidate
        chunks.append(current.rstrip())
        out.extend(chunks)
        changed.append(number)
    return "\n".join(out), changed


def _rename_tokens(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    """Token-level identifier rename; never touches strings or comments."""
    import io
    import tokenize

    lines = text.split("\n")
    edits = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.NAME and tok.string in mapping:
            edits.append((tok.start, tok.end, mapping[tok.string]))
    for (srow, scol), (erow, ecol), new in sorted(
        edits, key=lambda e: (e[0][0], e[0][1]), reverse=True
    ):
        if srow != erow:
            continue
        line = lines[srow - 1]
        lines[srow - 1] = line[:scol] + new + line[ecol:]
    return "\n".join(lines), len(edits)


def _format(files: list[Path], tools: Path | None) -> dict:
    def tool(name):
        if tools is not None and (Path(tools) / name).is_file():
            return str(Path(tools) / name)
        return shutil.which(name)

    applied, missing = {}, []
    for name, argv in (
        ("isort", [*spec.ISORT_ARGS, *map(str, files)]),
        (
            "black",
            ["--line-length", str(spec.BLACK_LINE_LENGTH), *map(str, files)],
        ),
    ):
        binary = tool(name)
        if binary is None:
            missing.append(name)
            continue
        proc = subprocess.run([binary, *argv], capture_output=True, text=True)
        applied[name] = {
            "returncode": proc.returncode,
            "output": (proc.stdout + proc.stderr).strip()[:500],
        }
    return {"applied": applied, "missing": missing}


def build(
    task: str,
    adaptation: str,
    op: str | None = None,
    tools: Path | None = None,
    submission_id: str | None = None,
    note: str | None = None,
    renames: dict[str, str] | None = None,
) -> dict:
    profile = load_json(ROOT / "competition" / task / "profile.json")
    op = op or profile["operator"]
    src_dir = LOCAL / "adaptations" / adaptation
    if not src_dir.is_dir():
        raise ValueError(f"no adaptation at {src_dir}")
    manifest = load_json(src_dir / "adaptation.json")
    sources = sorted((src_dir / "source").glob("*.py"))
    if not sources:
        raise ValueError(f"{src_dir}/source has no .py files")

    dest = PR_ROOT / task / adaptation
    if dest.exists():
        shutil.rmtree(dest)
    entries = []
    for path in sources:
        suffix = _suffix_of(op, path.stem)
        tier, rel = spec.tier_path(op, suffix)
        if suffix is not None and spec.vendor_for(suffix) is None:
            raise ValueError(
                f"{path.name}: chip suffix {suffix!r} has no upstream vendor tier"
            )
        original = path.read_text()
        bundled = _ensure_prologue(original, op)
        renamed, rename_count = _rename_tokens(bundled, renames or {})
        bundled = renamed
        write = dest / "files" / rel
        write.parent.mkdir(parents=True, exist_ok=True)
        write.write_text(bundled)
        awarded = dest / "awarded" / path.name
        awarded.parent.mkdir(parents=True, exist_ok=True)
        awarded.write_text(original)
        entries.append(
            {
                "name": path.name,
                "suffix": suffix,
                "tier": tier,
                "path": rel,
                "awarded_sha256": digest(awarded),
                "bundled_sha256": digest(write),
                "renamed_identifiers": rename_count,
            }
        )

    formatting = _format([dest / "files" / e["path"] for e in entries], tools)
    reflowed = {}
    for entry in entries:
        path = dest / "files" / entry["path"]
        text, changed = _reflow_long_text(path.read_text())
        if changed:
            path.write_text(text)
            reflowed[entry["path"]] = changed
    formatting["reflowed_text_lines"] = reflowed
    for entry in entries:
        entry["bundled_sha256"] = digest(dest / "files" / entry["path"])
        entry["awarded_sha256"] = digest(dest / "awarded" / entry["name"])

    official = _official(manifest.get("package_sha256"))
    bundle = {
        "schema_version": 1,
        "task_id": task,
        "op": op,
        "adaptation_id": adaptation,
        "created_at": now(),
        "files": entries,
        "provenance": {
            "platform_package_sha256": manifest.get("package_sha256"),
            "submitted_at": manifest.get("submitted_at"),
            "platform_status": manifest.get("platform_status"),
            "submission_evidence": manifest.get("submission_evidence"),
            "submission_id": submission_id,
            "preflight": manifest.get("preflight"),
            "note": note,
        },
        "official": official,
        "formatting": formatting,
        "renames": dict(renames or {}),
    }
    write_json(dest / "bundle.json", bundle)
    return bundle


def _official(package_sha256: str | None) -> dict:
    if not package_sha256:
        return {
            "available": False,
            "reason": "adaptation has no platform package hash",
        }
    try:
        from competition.adaptation import ledger
    except Exception as exc:  # pragma: no cover - import guard
        return {"available": False, "reason": f"ledger unavailable: {exc}"}
    rows = [
        row
        for row in ledger.rows("task103")
        if row.get("local_archive_sha256") == package_sha256
    ]
    if not rows:
        return {
            "available": False,
            "reason": "no official observation is bound to this package hash",
            "package_sha256": package_sha256,
        }
    latest = rows[-1]
    return {"available": True, "record": latest, "revisions": len(rows) - 1}


def load(bundle_dir: Path) -> tuple[dict, dict[str, str], dict[str, str]]:
    bundle_dir = Path(bundle_dir)
    bundle = load_json(bundle_dir / "bundle.json")
    texts = {
        entry["path"]: (bundle_dir / "files" / entry["path"]).read_text()
        for entry in bundle["files"]
    }
    awarded = {
        entry["name"]: (bundle_dir / "awarded" / entry["name"]).read_text()
        for entry in bundle["files"]
    }
    return bundle, texts, awarded


def materialize(
    bundle_dir: Path, ref: str = "upstream/master", tree: Path | None = None
) -> Path:
    """Unpack the upstream ref and copy the bundle files into it."""
    bundle_dir = Path(bundle_dir)
    bundle = load_json(bundle_dir / "bundle.json")
    tree = (
        Path(tree)
        if tree
        else PR_ROOT / bundle["task_id"] / bundle["adaptation_id"] / "repo"
    )
    if tree.exists():
        shutil.rmtree(tree)
    tree.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "archive", ref], cwd=ROOT, capture_output=True
    )
    if archive.returncode != 0:
        raise ValueError(
            f"git archive {ref} failed: {archive.stderr.decode(errors='replace').strip()}"
        )
    unpack = subprocess.run(
        ["tar", "-x", "-C", str(tree)],
        input=archive.stdout,
        capture_output=True,
    )
    if unpack.returncode != 0:
        raise ValueError(
            f"tar failed: {unpack.stderr.decode(errors='replace').strip()}"
        )
    for entry in bundle["files"]:
        target = tree / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle_dir / "files" / entry["path"], target)
    return tree
