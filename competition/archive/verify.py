"""Check every pre-refactor asset against its archived SHA-256."""
import argparse
import json
from pathlib import Path
from competition.experiments.store import ROOT, LOCAL, digest, load_json


def verify(include_local=False):
    manifest = load_json(ROOT / "competition/archive/migration.json")
    items = manifest["tracked"]
    if include_local:
        items += load_json(LOCAL / "archive-manifest.json")["files"]
    errors = []
    for item in items:
        path = ROOT / item["archive_path"]
        if not path.is_file() or digest(path) != item["sha256"]:
            errors.append(item["old_path"])
    return {"passed": not errors, "checked": len(items), "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--include-local", action="store_true")
    report = verify(parser.parse_args().include_local)
    print(json.dumps(report, indent=2)); raise SystemExit(0 if report["passed"] else 1)
