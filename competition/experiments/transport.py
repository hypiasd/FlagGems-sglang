"""One batch transfer and isolated remote process, with secret-free reports."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

from .store import LOCAL, ROOT, load_json, write_json


def config_path(device):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", device):
        raise ValueError("invalid device alias")
    return LOCAL / "devices" / f"{device}.json"


def connection(device):
    path = config_path(device)
    if not path.is_file():
        raise ValueError(f"missing local device configuration for {device}")
    if os.name != "nt" and path.stat().st_mode & 0o077:
        raise ValueError("device configuration must have mode 0600")
    config = load_json(path)
    env = dict(os.environ)
    auth = config.get("auth", {})
    prefix = []
    options = [
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "LogLevel=ERROR",
    ]
    if os.name != "nt":
        # macOS Unix socket paths are short; keep the multiplex socket out of a deep checkout.
        control = Path("/tmp") / (
            "flagos-ssh-"
            + str(os.getuid())
            + "-"
            + hashlib.sha256(str(ROOT).encode()).hexdigest()[:8]
        )
        control.mkdir(parents=True, exist_ok=True, mode=0o700)
        if (
            control.stat().st_uid != os.getuid()
            or control.stat().st_mode & 0o077
        ):
            raise ValueError(
                "SSH control directory must be private and owned by this user"
            )
        options += [
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=600",
            "-o",
            "ControlPath=" + str(control / "connection-%C"),
        ]
    # The device config may pin its own SSH policy (for example a checkout-local
    # known_hosts file when the remote instance is rebuilt and its host keys
    # change).  Appending them last lets them override the defaults above -- which
    # they must, because a config that declares `ssh_options` and is then ignored
    # is how a rebuilt host looks like an unreachable one.
    options += [str(item) for item in config.get("ssh_options") or []]
    if auth.get("method") == "password":
        password = os.environ.get(auth.get("password_env", "")) or auth.get(
            "password"
        )
        if not password:
            raise ValueError("missing password in local config/environment")
        env["SSHPASS"] = password
        prefix = ["sshpass", "-e"]
    elif auth.get("method") == "key":
        options += ["-i", str(auth["identity_file"])]
    else:
        raise ValueError("auth.method must be key or password")
    target = f"{config['user']}@{config['host']}"
    port = str(config.get("port", 22))
    return (
        config,
        [*prefix, "ssh", *options, "-p", port, target],
        [*prefix, "scp", *options, "-P", port],
        env,
        target,
    )


def administrative(device, script, timeout=60):
    """For explicitly authorized device setup; caller must keep outputs secret-free."""
    _, ssh, _, env, _ = connection(device)
    return subprocess.run(
        [*ssh, "sh", "-s"],
        input=script,
        text=True,
        env=env,
        capture_output=True,
        timeout=timeout,
    )


def execute(device, job, output, timeout, root=None):
    config, ssh, scp, env, target = connection(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    remote_base = config.get("remote_root", "/tmp/flagos-experiments").rstrip(
        "/"
    )
    if not remote_base.startswith("/") or remote_base in {
        "/",
        "/tmp",
        "/home",
        "/root",
    }:
        raise ValueError(
            "remote_root must name a dedicated absolute scratch directory"
        )
    token = uuid.uuid4().hex
    remote = remote_base + "/experiment-" + token
    python = config.get("remote_python", "python3")
    exports = []
    for key, value in config.get("env", {}).items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError("invalid environment variable name")
        exports.append(f"export {key}={shlex.quote(str(value))}")
    gpu = int(config.get("gpu", 0))
    job = {**job, "root": remote + "/experiment", "gpu": gpu}
    phase = time.perf_counter()
    status = None
    with tempfile.TemporaryDirectory(prefix="flagos-bundle-") as temporary:
        temporary = Path(temporary)
        bundle_dir = temporary / "bundle"
        bundle_dir.mkdir()
        if root:
            for name in ("source", "harness"):
                shutil.copytree(
                    Path(root) / name, bundle_dir / "experiment" / name
                )
            for name in ("run.json", "contract.json", "snapshot.sha256"):
                shutil.copy2(
                    Path(root) / name, bundle_dir / "experiment" / name
                )
        else:
            for path in (ROOT / "competition/experiments").glob("*.py"):
                dest = (
                    bundle_dir
                    / "experiment/harness/competition/experiments"
                    / path.name
                )
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
            for path in (ROOT / "competition").glob("*.py"):
                if path.name.startswith("test_"):
                    continue
                dest = (
                    bundle_dir / "experiment/harness/competition" / path.name
                )
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
        write_json(bundle_dir / "job.json", job)
        archive = temporary / "bundle.tgz"
        with tarfile.open(archive, "w:gz") as stream:
            for path in bundle_dir.rglob("*"):
                if path.is_file():
                    stream.add(
                        path, arcname=path.relative_to(bundle_dir).as_posix()
                    )

        def call(argv, seconds):
            return subprocess.run(
                argv, capture_output=True, text=True, env=env, timeout=seconds
            )

        created = call([*ssh, "mkdir -p " + shlex.quote(remote)], 30)
        if created.returncode:
            raise RuntimeError(
                "SSH scratch creation failed; connection details are withheld"
            )
        try:
            copied = call(
                [
                    *scp,
                    str(archive),
                    target + ":" + shlex.quote(remote + "/bundle.tgz"),
                ],
                60,
            )
            if copied.returncode:
                raise RuntimeError(
                    "batch upload failed; connection details are withheld"
                )
            upload_seconds = time.perf_counter() - phase
            flags = (
                "--doctor --gpu " + str(gpu)
                if job["command"] == "doctor"
                else "--job " + shlex.quote(remote + "/job.json")
            )
            cache = remote_base + "/triton-cache"
            lock = (
                ""
                if job["command"] == "doctor"
                else "flock -n -E 75 "
                + shlex.quote(remote_base + "/gpu-" + str(gpu) + ".lock")
                + " "
            )
            script = "\n".join(
                [
                    "set -u",
                    *exports,
                    "mkdir -p "
                    + shlex.quote(remote + "/out")
                    + " "
                    + shlex.quote(cache),
                    "tar xzf "
                    + shlex.quote(remote + "/bundle.tgz")
                    + " -C "
                    + shlex.quote(remote),
                    "export PYTHONPATH="
                    + shlex.quote(remote + "/experiment/harness"),
                    "export PYTHONDONTWRITEBYTECODE=1",
                    "export TRITON_CACHE_DIR=" + shlex.quote(cache),
                    "timeout --signal=TERM --kill-after=5s "
                    + str(timeout)
                    + "s "
                    + lock
                    + shlex.quote(python)
                    + " -m competition.experiments.worker "
                    + flags
                    + " --out "
                    + shlex.quote(remote + "/out/report.json")
                    + " > "
                    + shlex.quote(remote + "/out/worker.log")
                    + " 2>&1",
                    "code=$?",
                    "tar czf "
                    + shlex.quote(remote + "/results.tgz")
                    + " -C "
                    + shlex.quote(remote + "/out")
                    + " .",
                    'exit "$code"',
                ]
            )
            running = time.perf_counter()
            try:
                process = call([*ssh, script], timeout + 15)
                status = process.returncode
            except subprocess.TimeoutExpired:
                status = 124
            remote_seconds = time.perf_counter() - running
            result_archive = temporary / "results.tgz"
            pulled = call(
                [
                    *scp,
                    target + ":" + shlex.quote(remote + "/results.tgz"),
                    str(result_archive),
                ],
                90,
            )
            if pulled.returncode == 0:
                with tarfile.open(result_archive) as stream:
                    # Never trust paths or symlinks in a returned archive.
                    for member in stream.getmembers():
                        path = Path(member.name)
                        if (
                            member.isfile()
                            and not path.is_absolute()
                            and ".." not in path.parts
                        ):
                            dest = output / path
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            with stream.extractfile(member) as source:
                                dest.write_bytes(source.read())
            elif status == 124:
                # A disconnected SSH client may precede the server-side timeout.
                call(
                    [
                        *scp,
                        target
                        + ":"
                        + shlex.quote(remote + "/out/report.json"),
                        str(output / "report.json"),
                    ],
                    30,
                )
            report = (
                load_json(output / "report.json")
                if (output / "report.json").exists()
                else {
                    "status": "failed",
                    "reason": "remote report unavailable",
                }
            )
            if status == 75:
                report.update(
                    status="busy",
                    reason="another experiment owns this device lock; retry after it finishes",
                    full_matrix_passed=False,
                )
            if status in {124, 137}:
                report["partial_status"] = report.get("status")
                report["status"] = "timeout"
                report["full_matrix_passed"] = False
            report.update(
                command=job["command"],
                device_alias=device,
                transport={
                    "upload_seconds": upload_seconds,
                    "remote_seconds": remote_seconds,
                    "exit_code": status,
                },
                run_id=job.get("run_id", report.get("run_id")),
            )
            write_json(output / "report.json", report)
            return report
        finally:
            # Only this UUID scratch directory; persistent compilation cache survives.
            try:
                call([*ssh, "rm -rf -- " + shlex.quote(remote)], 30)
            except subprocess.TimeoutExpired:
                pass  # Server-side timeout still bounds the child process lifetime.
