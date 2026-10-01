"""Offline tests for the submission-package member rule.

Stdlib only.  The behaviours under test are the ones that silently changed
what got shipped: an undeclared chip file must fail the layout instead of
being dropped, and a contract that no longer matches recent packages must be
reported as drift.

Run with ``python -m unittest competition.test_members``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from competition import members

OP = "recompute_w_u"
TARGETS = [
    "iluvatar",
    "metax",
    "hygon",
    "kunlunxin",
    "ascend",
    "intl_a",
    "intl_b",
]


def write(directory, names):
    for name in names:
        (Path(directory) / name).write_text("def recompute_w_u():\n    pass\n")


class DiscoverTest(unittest.TestCase):
    def test_generic_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write(directory, [f"{OP}.py"])
            layout = members.discover(OP, directory, TARGETS, [f"{OP}.py"])
            self.assertEqual(layout["generic"], f"{OP}.py")
            self.assertEqual(layout["dedicated"], {})

    def test_dedicated_and_unknown_suffixes_are_separated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write(
                directory,
                [
                    f"{OP}.py",
                    f"{OP}_kunlunxin.py",
                    f"{OP}_tianshu.py",
                    "helper.py",
                ],
            )
            layout = members.discover(OP, directory, TARGETS, [f"{OP}.py"])
            self.assertEqual(
                layout["dedicated"], {"kunlunxin": f"{OP}_kunlunxin.py"}
            )
            self.assertEqual(
                layout["unknown_chip"], {"tianshu": f"{OP}_tianshu.py"}
            )
            self.assertEqual(layout["foreign"], ["helper.py"])

    def test_dedicated_name(self) -> None:
        self.assertEqual(
            members.dedicated_name(OP, "intl_a"), f"{OP}_intl_a.py"
        )


class TargetSourceTest(unittest.TestCase):
    def test_dedicated_file_wins_over_generic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write(directory, [f"{OP}.py", f"{OP}_kunlunxin.py"])
            layout = members.discover(OP, directory, TARGETS, [f"{OP}.py"])
            resolved, unresolved = members.target_sources(OP, TARGETS, layout)
            self.assertEqual(resolved["kunlunxin"], f"{OP}_kunlunxin.py")
            self.assertEqual(resolved["metax"], f"{OP}.py")
            self.assertEqual(unresolved, [])


class AuditTest(unittest.TestCase):
    def audit(self, directory, declared):
        return members.audit(OP, directory, TARGETS, declared)

    def test_consistent_three_file_package(self) -> None:
        names = [f"{OP}.py", f"{OP}_iluvatar.py", f"{OP}_kunlunxin.py"]
        with tempfile.TemporaryDirectory() as directory:
            write(directory, names)
            result = self.audit(directory, names)
            self.assertTrue(result["passed"])
            self.assertEqual(
                result["dedicated"],
                {
                    "iluvatar": f"{OP}_iluvatar.py",
                    "kunlunxin": f"{OP}_kunlunxin.py",
                },
            )

    def test_shared_generic_is_reported_as_headroom(self) -> None:
        names = [f"{OP}.py", f"{OP}_kunlunxin.py"]
        with tempfile.TemporaryDirectory() as directory:
            write(directory, names)
            result = self.audit(directory, names)
            self.assertTrue(result["passed"])
            self.assertEqual(
                result["targets_without_dedicated"],
                ["iluvatar", "metax", "hygon", "ascend", "intl_a", "intl_b"],
            )
            self.assertIn(
                "share the generic module", " ".join(result["warnings"])
            )

    def test_undeclared_chip_file_fails_closed(self) -> None:
        names = [f"{OP}.py", f"{OP}_iluvatar.py", f"{OP}_kunlunxin.py"]
        with tempfile.TemporaryDirectory() as directory:
            write(directory, names)
            result = self.audit(directory, [f"{OP}.py"])
            self.assertFalse(result["passed"])
            joined = " ".join(result["errors"])
            self.assertIn("silently drop", joined)
            self.assertIn(f"{OP}_kunlunxin.py", joined)

    def test_declared_but_missing_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write(directory, [f"{OP}.py"])
            result = self.audit(directory, [f"{OP}.py", f"{OP}_kunlunxin.py"])
            self.assertFalse(result["passed"])
            self.assertIn("missing", " ".join(result["errors"]))

    def test_missing_generic_module_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write(directory, [f"{OP}_kunlunxin.py"])
            result = self.audit(directory, [f"{OP}_kunlunxin.py"])
            self.assertFalse(result["passed"])
            self.assertIn(
                "generic module is required", " ".join(result["errors"])
            )

    def test_unknown_chip_suffix_fails(self) -> None:
        names = [f"{OP}.py", f"{OP}_tianshu.py"]
        with tempfile.TemporaryDirectory() as directory:
            write(directory, names)
            result = self.audit(directory, names)
            self.assertFalse(result["passed"])
            self.assertIn("never be routed", " ".join(result["errors"]))

    def test_declared_member_that_is_not_a_module_fails(self) -> None:
        names = [f"{OP}.py", "notes.py"]
        with tempfile.TemporaryDirectory() as directory:
            write(directory, names)
            result = self.audit(directory, names)
            self.assertFalse(result["passed"])
            self.assertIn(
                "neither the generic module", " ".join(result["errors"])
            )

    def test_empty_declaration_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write(directory, [f"{OP}.py"])
            result = self.audit(directory, [])
            self.assertFalse(result["passed"])
            self.assertIn("no package members", " ".join(result["errors"]))


class DriftTest(unittest.TestCase):
    def build_local(self, root, task, entries):
        for identifier, names in entries:
            directory = Path(root) / "adaptations" / identifier
            directory.mkdir(parents=True)
            (directory / "adaptation.json").write_text(
                json.dumps(
                    {
                        "task_id": task,
                        "created_at": f"2026-10-01T0{len(names)}:00:00",
                    }
                )
            )
            with zipfile.ZipFile(directory / "package.zip", "w") as stream:
                for name in names:
                    stream.writestr(name, "def recompute_w_u():\n    pass\n")

    def test_drift_detects_a_shrunken_contract(self) -> None:
        contract = {"operator": OP, "package_members": [f"{OP}.py"]}
        with tempfile.TemporaryDirectory() as local:
            self.build_local(
                local,
                "task103",
                [
                    ("adapt-one", [f"{OP}.py"]),
                    (
                        "adapt-two",
                        [f"{OP}.py", f"{OP}_kunlunxin.py"],
                    ),
                ],
            )
            result = members.drift("task103", contract, local)
            self.assertTrue(result["drifted"])
            self.assertEqual(
                list(result["shipped_but_not_declared"]),
                [f"{OP}_kunlunxin.py"],
            )
            self.assertEqual(
                result["latest_package"]["adaptation_id"], "adapt-two"
            )

    def test_no_drift_when_the_contract_matches(self) -> None:
        contract = {
            "operator": OP,
            "package_members": [f"{OP}.py", f"{OP}_kunlunxin.py"],
        }
        with tempfile.TemporaryDirectory() as local:
            self.build_local(
                local,
                "task103",
                [("adapt-one", [f"{OP}.py", f"{OP}_kunlunxin.py"])],
            )
            result = members.drift("task103", contract, local)
            self.assertFalse(result["drifted"])
            self.assertEqual(result["shipped_but_not_declared"], {})

    def test_other_tasks_are_ignored(self) -> None:
        contract = {"operator": OP, "package_members": [f"{OP}.py"]}
        with tempfile.TemporaryDirectory() as local:
            self.build_local(
                local,
                "task78",
                [("adapt-other", [f"{OP}.py", f"{OP}_kunlunxin.py"])],
            )
            result = members.drift("task103", contract, local)
            self.assertEqual(result["packages_observed"], 0)
            self.assertFalse(result["drifted"])


if __name__ == "__main__":
    unittest.main()
