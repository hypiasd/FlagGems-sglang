#!/usr/bin/env python3
"""Non-target GPU smoke orchestrator for Task 78 candidate sources.

This is the local half of the GPU smoke layer.  It copies a candidate source
directory to a remote machine that has a real Triton-capable accelerator, runs
``gpu_smoke_remote.py`` there, and writes a hash-bound evidence file locally.

The smoke layer exists for exactly one job: catch Triton/compiler/launch-level
defects that the CPU execution model in ``validate_cpu.py`` structurally cannot
see, before a scarce FlagOS submission is spent.  It is deliberately NOT wired
into ``target_validated`` or ``measured``: the remote device is not one of the
eight declared Task 78 targets, so its result is labelled
``evidence_class: nvidia-smoke-nontarget`` and is never target evidence.

Credentials live only in the ignored local config
(``competition/.autopilot/gpu-smoke.json``).  They are passed to ``sshpass``
through the environment, never on a command line, never written to the evidence
file, and never printed.

Exit codes: 0 pass, 1 smoke failure, 2 infrastructure/transport failure,
3 disabled or not configured.

    python3 gpu_smoke.py --source-dir <dir> --json <report.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

EVIDENCE_CLASS = "nvidia-smoke-nontarget"
DEFAULT_CONFIG = (Path(__file__).resolve().parents[2] / ".autopilot" / "gpu-smoke.json")
DRIVER = Path(__file__).resolve().with_name("gpu_smoke_remote.py")
VALIDATOR = Path(__file__).resolve().parents[1] / "validate_cpu.py"
DEFAULT_SSH_OPTIONS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=20",
    "-o", "ServerAliveInterval=15",
    "-o", "LogLevel=ERROR",
]
NOTICE = (
    "Non-target GPU smoke only. The remote device is not one of the eight declared "
    "Task 78 targets. This evidence is diagnostic for compilation and correctness; "
    "it never satisfies target_validated or measured and never predicts a target "
    "chip's score."
)


def log(message: str) -> None:
    print(f"[gpu-smoke] {message}", file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_sources(source_dir: Path) -> list[Path]:
    return sorted(path for path in source_dir.glob("*.py") if not path.name.startswith("_"))


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"no GPU smoke config at {path}; create it with host/port/user/auth "
            "(the .autopilot directory is git-ignored)"
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("GPU smoke config must be a JSON object")
    for field in ("host", "user"):
        if not config.get(field):
            raise ValueError(f"GPU smoke config is missing '{field}'")
    return config


def auth_prefix(config: dict) -> tuple[list[str], dict]:
    """Return the command prefix and extra environment for authentication.

    The password is supplied through ``SSHPASS`` so it never appears in argv,
    in a process listing, or in any log this tool writes.
    """
    auth = config.get("auth") or {}
    method = auth.get("method", "password")
    env = dict(os.environ)
    if method == "password":
        password = auth.get("password")
        if not password:
            raise ValueError("auth.method is 'password' but no password is configured")
        env["SSHPASS"] = password
        return ["sshpass", "-e"], env
    if method == "key":
        identity = auth.get("identity_file")
        if not identity:
            raise ValueError("auth.method is 'key' but no identity_file is configured")
        return [], env
    raise ValueError(f"unsupported auth.method: {method!r}")


def build_commands(config: dict) -> tuple[list[str], list[str]]:
    options = list(config.get("ssh_options") or DEFAULT_SSH_OPTIONS)
    auth = config.get("auth") or {}
    if auth.get("method", "password") == "key":
        options = ["-i", str(auth["identity_file"]), *options]
    target = f"{config['user']}@{config['host']}"
    port = str(config.get("port") or 22)
    prefix, env = auth_prefix(config)
    ssh = [*prefix, "ssh", *options, "-p", port, target]
    scp = [*prefix, "scp", *options, "-P", port]
    return ssh, scp, env


def run(command: list[str], env: dict, timeout: int, what: str):
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"{what} failed (exit {result.returncode}): "
            f"{(result.stderr or result.stdout).strip()[:600]}"
        )
    return result


def redact(config: dict, value: str | None) -> str | None:
    if value is None or not config.get("redact_host"):
        return value
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--json", dest="json_path", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--filter", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--timeout", type=int, default=None,
                        help="overall remote command timeout in seconds")
    parser.add_argument("--keep-remote", action="store_true",
                        help="do not delete the remote scratch directory")
    args = parser.parse_args(argv)

    source_dir = args.source_dir.resolve()
    sources = candidate_sources(source_dir)
    report: dict = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dir": str(source_dir),
        "source_hashes": {path.name: sha256_file(path) for path in sources},
        "remote": {},
        "suite": {},
        "results": {},
        "negative_control": None,
        "passed": False,
        "status": "infrastructure_error",
        "notice": NOTICE,
    }

    def finish(status: str, code: int, error: str | None = None) -> int:
        report["status"] = status
        report["passed"] = status == "pass"
        if error:
            report["error"] = error
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
        return code

    if not sources:
        return finish("infrastructure_error", 2, f"no candidate sources under {source_dir}")
    if not DRIVER.is_file() or not VALIDATOR.is_file():
        return finish("infrastructure_error", 2,
                      f"missing driver or validator: {DRIVER} / {VALIDATOR}")

    try:
        config = load_config(args.config.resolve())
    except (OSError, ValueError) as exc:
        log(f"DISABLED: {exc}")
        return finish("disabled", 3, str(exc))
    if not config.get("enabled", True):
        log("DISABLED: config sets enabled=false")
        return finish("disabled", 3, "config sets enabled=false")

    timeout = args.timeout or int(config.get("timeout_seconds") or 1800)
    remote_root = str(config.get("remote_root") or "/tmp/flagos-gpu-smoke")
    remote_dir = f"{remote_root}/{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    remote_src = f"{remote_dir}/source"

    try:
        ssh, scp, env = build_commands(config)
    except ValueError as exc:
        log(f"DISABLED: {exc}")
        return finish("disabled", 3, str(exc))

    report["remote"] = {
        "host": redact(config, config["host"]),
        "port": config.get("port"),
        "user": config["user"],
        "transport": config.get("transport", "ssh"),
        "auth_method": (config.get("auth") or {}).get("method", "password"),
    }

    try:
        log(f"ENABLED: non-target GPU smoke via {config['host']}:{config.get('port', 22)}")
        log(f"pushing {len(sources)} candidate source(s) + driver + validator")
        run([*ssh, f"mkdir -p {shlex.quote(remote_src)}"], env, timeout, "remote mkdir")
        # Keep the driver and validator out of the candidate directory: anything
        # sitting next to the candidate sources is treated as a candidate file.
        run([*scp, *[str(path) for path in sources],
             f"{config['user']}@{config['host']}:{shlex.quote(remote_src)}/"], env, timeout, "scp sources")
        run([*scp, str(DRIVER), str(VALIDATOR),
             f"{config['user']}@{config['host']}:{shlex.quote(remote_dir)}/"], env, timeout, "scp tooling")

        exports = "".join(
            f"export {name}={shlex.quote(str(value))}; "
            for name, value in (config.get("env") or {}).items()
        )
        driver_args = [
            config.get("remote_python", "python3"),
            f"{remote_dir}/gpu_smoke_remote.py",
            "--source-dir", remote_src,
            "--validator", f"{remote_dir}/validate_cpu.py",
            "--out", f"{remote_dir}/remote-report.json",
            "--seed", str(args.seed),
        ]
        if args.filter:
            driver_args += ["--filter", args.filter]
        if args.max_cases is not None:
            driver_args += ["--max-cases", str(args.max_cases)]
        remote_command = (
            f"{exports}cd {shlex.quote(remote_dir)} && "
            + " ".join(shlex.quote(item) for item in driver_args)
        )
        log("running remote smoke (real Triton compilation + execution)")
        result = subprocess.run([*ssh, remote_command], env=env, text=True,
                                capture_output=True, timeout=timeout)
        for line in (result.stderr or "").splitlines():
            if line.startswith("[gpu-smoke]"):
                log(line[len("[gpu-smoke] "):])

        cat = subprocess.run([*ssh, f"cat {shlex.quote(remote_dir)}/remote-report.json"],
                             env=env, text=True, capture_output=True, timeout=timeout)
        if cat.returncode != 0:
            raise RuntimeError(
                "remote smoke produced no report "
                f"(driver exit {result.returncode}): {(result.stderr or '')[-800:]}"
            )
        remote_report = json.loads(cat.stdout)
    except subprocess.TimeoutExpired:
        return finish("infrastructure_error", 2, f"remote smoke exceeded {timeout}s")
    except (RuntimeError, json.JSONDecodeError, OSError) as exc:
        return finish("infrastructure_error", 2, str(exc))
    finally:
        if not args.keep_remote:
            try:
                subprocess.run([*ssh, f"rm -rf {shlex.quote(remote_dir)}"], env=env,
                               text=True, capture_output=True, timeout=120)
            except Exception:  # noqa: BLE001 - cleanup is best effort
                pass

    # Integrity: the evidence must describe the exact local candidate bytes.
    remote_hashes = remote_report.get("source_hashes") or {}
    if remote_hashes != report["source_hashes"]:
        mismatched = sorted(
            name for name in set(remote_hashes) | set(report["source_hashes"])
            if remote_hashes.get(name) != report["source_hashes"].get(name)
        )
        report["results"] = remote_report.get("results", {})
        return finish("infrastructure_error", 2,
                      f"remote source hashes do not match the local candidate: {mismatched}")

    report["remote"].update(remote_report.get("device") or {})
    report["suite"] = remote_report.get("suite") or {}
    report["results"] = remote_report.get("results") or {}
    report["negative_control"] = remote_report.get("negative_control")
    if remote_report.get("infrastructure_error"):
        return finish("infrastructure_error", 2, str(remote_report["infrastructure_error"]))

    status = "pass" if remote_report.get("passed") is True else "fail"
    log(f"{status.upper()}: {sum(1 for item in report['results'].values() if item.get('passed'))}"
        f"/{len(report['results'])} source file(s) passed. {NOTICE}")
    return finish(status, 0 if status == "pass" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
