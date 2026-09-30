#!/usr/bin/env python3
"""Tests for the non-target GPU smoke layer.

The layer exists to catch Triton/compiler/launch defects before a scarce FlagOS
submission, without ever letting a non-target device act as target evidence.
These tests are offline: they exercise the evidence contract, the hash binding,
the credential handling, and the remote file layout. The actual GPU execution is
covered by running ``gpu_smoke.py`` against a real accelerator.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpu_smoke
import run_candidate_gate as gate


def write_config(directory: Path, **overrides) -> Path:
    config = {
        "schema_version": 1,
        "enabled": True,
        "host": "example.invalid",
        "port": 2200,
        "user": "root",
        "auth": {"method": "password", "password": "hunter2-secret"},
        "env": {"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64"},
    }
    config.update(overrides)
    path = directory / "gpu-smoke.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def make_report(source_hashes: dict, **overrides) -> dict:
    report = {
        "schema_version": 1,
        "evidence_class": gpu_smoke.EVIDENCE_CLASS,
        "source_hashes": dict(source_hashes),
        "status": "pass",
        "passed": True,
        "negative_control": {"passed": True},
        "results": {name: {"passed": True} for name in source_hashes},
    }
    report.update(overrides)
    return report


class EvidenceClassTests(unittest.TestCase):
    def test_both_halves_agree_on_the_evidence_class(self):
        self.assertEqual(gpu_smoke.EVIDENCE_CLASS, gate.GPU_SMOKE_EVIDENCE_CLASS)

    def test_evidence_class_marks_the_device_as_a_non_target(self):
        self.assertIn("nontarget", gpu_smoke.EVIDENCE_CLASS)


class GateBindingTests(unittest.TestCase):
    def setUp(self):
        self.hashes = {"default": "a" * 64, "ascend": "b" * 64}
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def report_path(self, report) -> Path:
        path = self.dir / "gpu-smoke.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def expected(self):
        return {gate.source_name(backend): value for backend, value in self.hashes.items()}

    def test_matching_report_passes_the_gate_check(self):
        result = gate.gpu_smoke_check(
            self.report_path(make_report(self.expected())), self.hashes, require=True)
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["blocking"])

    def test_non_target_label_is_required(self):
        result = gate.gpu_smoke_check(
            self.report_path(make_report(self.expected(), evidence_class="target")),
            self.hashes, require=True)
        self.assertFalse(result["passed"])
        self.assertIn("evidence_class", " ".join(result["errors"]))

    def test_failed_smoke_blocks_only_when_required(self):
        report = make_report(self.expected(), status="fail", passed=False)
        advisory = gate.gpu_smoke_check(self.report_path(report), self.hashes, require=False)
        required = gate.gpu_smoke_check(self.report_path(report), self.hashes, require=True)
        # The report itself failed in both modes; only the required mode turns
        # that into a blocking gate verdict.
        self.assertFalse(advisory["passed"])
        self.assertFalse(advisory["blocking"])
        self.assertFalse(required["passed"])
        self.assertTrue(required["blocking"])

    def test_advisory_mode_still_records_the_report_status(self):
        report = make_report(self.expected())
        result = gate.gpu_smoke_check(self.report_path(report), self.hashes, require=False)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["evidence_class"], gate.GPU_SMOKE_EVIDENCE_CLASS)
        self.assertFalse(result["blocking"])

    def test_missing_negative_control_blocks(self):
        report = make_report(self.expected(), negative_control={"passed": False})
        result = gate.gpu_smoke_check(self.report_path(report), self.hashes, require=True)
        self.assertFalse(result["passed"])
        self.assertIn("negative control", " ".join(result["errors"]))

    def test_hash_binding_rejects_a_stale_report(self):
        stale = {"concat_and_cast_mha_k.py": "c" * 64,
                 "concat_and_cast_mha_k_ascend.py": "d" * 64}
        result = gate.gpu_smoke_check(self.report_path(make_report(stale)),
                                      self.hashes, require=True)
        self.assertFalse(result["passed"])
        self.assertIn("hashes", " ".join(result["errors"]))

    def test_required_without_a_report_blocks(self):
        result = gate.gpu_smoke_check(None, self.hashes, require=True)
        self.assertFalse(result["passed"])
        self.assertIn("--gpu-smoke-json", result["error"])

    def test_optional_without_a_report_is_not_blocking(self):
        result = gate.gpu_smoke_check(None, self.hashes, require=False)
        self.assertTrue(result["passed"])
        self.assertFalse(result["blocking"])


class CredentialHandlingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_password_goes_through_the_environment_only(self):
        config = json.loads(write_config(self.dir).read_text(encoding="utf-8"))
        ssh, scp, env = gpu_smoke.build_commands(config)
        self.assertEqual(env["SSHPASS"], config["auth"]["password"])
        for command in (ssh, scp):
            for item in command:
                self.assertNotIn(config["auth"]["password"], item)

    def test_password_auth_without_a_password_is_rejected(self):
        config = json.loads(write_config(self.dir).read_text(encoding="utf-8"))
        del config["auth"]["password"]
        with self.assertRaises(ValueError):
            gpu_smoke.build_commands(config)

    def test_key_auth_does_not_require_sshpass(self):
        config = json.loads(write_config(
            self.dir, auth={"method": "key", "identity_file": "/tmp/id_ed25519"},
        ).read_text(encoding="utf-8"))
        ssh, _, env = gpu_smoke.build_commands(config)
        self.assertNotIn("sshpass", ssh)
        self.assertIn("/tmp/id_ed25519", ssh)
        self.assertNotIn("SSHPASS", env)

    def test_config_requires_host_and_user(self):
        path = write_config(self.dir)
        config = json.loads(path.read_text(encoding="utf-8"))
        del config["host"]
        path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaises(ValueError):
            gpu_smoke.load_config(path)


class SourceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_private_and_non_python_files_are_excluded(self):
        for name in ("b.py", "a.py", "_helper.py", "notes.md", "__pycache__"):
            (self.dir / name).touch()
        self.assertEqual([path.name for path in gpu_smoke.candidate_sources(self.dir)],
                         ["a.py", "b.py"])


class RemoteLayoutTests(unittest.TestCase):
    """Regression: tooling must not land inside the candidate source directory.

    Anything sitting next to the candidate files is treated as a candidate, so a
    driver copied into that directory would be reported as a broken operator.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.source = self.dir / "a.py"
        self.source.write_text("def concat_and_cast_mha_k(k, k_nope, k_rope):\n    return k\n",
                               encoding="utf-8")

    def test_tooling_is_copied_beside_not_into_the_candidate_dir(self):
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        report = make_report({"a.py": digest})
        captured = []

        class Result:
            returncode = 0
            stderr = ""
            stdout = ""

        def fake_run(command, env, timeout, what):
            captured.append((what, list(command)))
            return Result()

        def fake_subprocess(command, **kwargs):
            captured.append(("shell", list(command)))
            result = Result()
            result.stdout = json.dumps(report)
            return result

        config = write_config(self.dir)
        out = self.dir / "out.json"
        with mock.patch.object(gpu_smoke, "run", fake_run), \
                mock.patch.object(gpu_smoke.subprocess, "run", fake_subprocess):
            code = gpu_smoke.main(["--source-dir", str(self.dir), "--json", str(out),
                                   "--config", str(config)])
        self.assertEqual(code, 0, out.read_text(encoding="utf-8"))

        def last_arg(what):
            return next(args[-1] for name, args in captured if name == what)

        self.assertTrue(last_arg("scp sources").endswith("/source/"))
        tooling_target = last_arg("scp tooling")
        self.assertFalse(tooling_target.endswith("/source/"))
        self.assertTrue(tooling_target.endswith("/"))

        shell = next(args for name, args in captured if name == "shell")[-1]
        self.assertIn("--source-dir", shell)
        self.assertIn("/source", shell)

    def test_hash_mismatch_is_an_infrastructure_error(self):
        report = make_report({"a.py": "f" * 64})
        captured = []

        class Result:
            returncode = 0
            stderr = ""
            stdout = ""

        def fake_run(command, env, timeout, what):
            captured.append((what, list(command)))
            return Result()

        def fake_subprocess(command, **kwargs):
            result = Result()
            result.stdout = json.dumps(report)
            return result

        config = write_config(self.dir)
        out = self.dir / "out.json"
        with mock.patch.object(gpu_smoke, "run", fake_run), \
                mock.patch.object(gpu_smoke.subprocess, "run", fake_subprocess):
            code = gpu_smoke.main(["--source-dir", str(self.dir), "--json", str(out),
                                   "--config", str(config)])
        self.assertEqual(code, 2)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "infrastructure_error")
        self.assertIn("hashes", payload["error"])


class DisabledConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.source = self.dir / "a.py"
        self.source.write_text("x = 1\n", encoding="utf-8")

    def test_disabled_config_reports_disabled_without_contacting_the_remote(self):
        config = write_config(self.dir, enabled=False)
        out = self.dir / "out.json"
        with mock.patch.object(gpu_smoke.subprocess, "run",
                               side_effect=AssertionError("must not contact the remote")):
            code = gpu_smoke.main(["--source-dir", str(self.dir), "--json", str(out),
                                   "--config", str(config)])
        self.assertEqual(code, 3)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "disabled")
        self.assertFalse(payload["passed"])

    def test_missing_config_reports_disabled(self):
        out = self.dir / "out.json"
        code = gpu_smoke.main(["--source-dir", str(self.dir), "--json", str(out),
                               "--config", str(self.dir / "absent.json")])
        self.assertEqual(code, 3)
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["status"], "disabled")


if __name__ == "__main__":
    unittest.main()
