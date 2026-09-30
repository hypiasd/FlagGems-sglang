"""Behavioral regression checks that do not require a GPU or PyTorch."""
import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from competition.experiments import store
from competition.experiments.timing import Budget, repeats, sample_pair
from competition.experiments.__main__ import comparison
from competition.experiments.worker import select
from competition.adaptation import ledger
from competition.adaptation.packaging import validate_package, make_package
from competition.adaptation.__main__ import publication_check
from competition.adaptation.structure import normalized_ast


class Clock:
    def __init__(self): self.value = 0.0
    def __call__(self): return self.value
    def timer(self, fn, count):
        cost = fn()
        self.value += cost * count / 1000
        return cost


class TimingTests(unittest.TestCase):
    def test_slow_reference_does_not_repeat_hundreds_of_times(self):
        clock = Clock()
        report = sample_pair(lambda: 1, lambda: 2000, clock.timer, Budget(30, clock))
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["reference"]["repeats"], [1, 1, 1])
        self.assertLess(clock.value, 30)

    def test_single_call_overrun_is_explicit_and_not_a_speedup(self):
        clock = Clock()
        report = sample_pair(lambda: 1, lambda: 40000, clock.timer, Budget(30, clock))
        self.assertEqual(report["status"], "insufficient")
        self.assertIsNone(report["speedup"])
        self.assertGreater(report["budget_overrun_seconds"], 10)

    def test_exhausted_budget_enqueues_nothing(self):
        clock = Clock(); budget = Budget(1, clock); clock.value = 2
        report = sample_pair(lambda: self.fail(), lambda: self.fail(), clock.timer, budget)
        self.assertEqual(report["candidate"]["samples_ms"], [])

    def test_fast_calls_are_capped(self):
        self.assertEqual(repeats(0.00001, 10, 256, 30), 256)

    def test_invalid_budgets(self):
        for seconds in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError): Budget(seconds)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.contract = {"task_id": "task1", "operator": "op", "entrypoint": "op(x)", "package_members": ["op.py"], "hardware_requirements": {}, "targets": ["a", "b"], "review_policy": {"independent_reviews": 2}, "publication_policy": None}
        p = self.root / "competition/task1"; p.mkdir(parents=True)
        store.write_json(p / "profile.json", self.contract)
        self.source = self.root / "candidate"; self.source.mkdir(); (self.source / "op.py").write_text("def op(x):\n    return x\n")
        self.patches = [patch.object(store, "ROOT", self.root), patch.object(store, "LOCAL", self.root / "local")]
        for p in self.patches: p.start()
    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()
    def make_run(self):
        m = store.create("task1", self.source, "test isolated source snapshot")
        return m, store.run_path(m["run_id"])
    def test_source_is_copied_not_linked(self):
        m, root = self.make_run(); (self.source / "op.py").write_text("changed")
        self.assertEqual(store.verify(root)["run_id"], m["run_id"])
    def test_source_mutation_rejected(self):
        _, root = self.make_run(); (root / "source/op.py").write_text("changed")
        with self.assertRaises(ValueError): store.verify(root)
    def test_added_file_rejected(self):
        _, root = self.make_run(); (root / "source/extra.py").write_text("pass")
        with self.assertRaises(ValueError): store.verify(root)
    def test_metadata_mutation_rejected(self):
        m, root = self.make_run(); m["seed"] = 2; store.write_json(root / "run.json", m)
        with self.assertRaises(ValueError): store.verify(root)
    def test_new_child_preserves_parent(self):
        m, root = self.make_run(); child = store.create("task1", self.source, "child", m["run_id"])
        self.assertEqual(child["parent"], m["run_id"]); store.verify(root)
    def test_unknown_task_or_empty_hypothesis_rejected(self):
        for task, hypothesis in [("../../task1", "x"), ("task1", "")]:
            with self.assertRaises(ValueError): store.create(task, self.source, hypothesis)
    def test_package_exact_bytes_and_signature(self):
        archive = self.root / "p.zip"
        self.assertTrue(make_package(self.source, archive, self.contract)["passed"])
        (self.source / "op.py").write_text("def op(y):\n    return y\n")
        self.assertFalse(validate_package(self.source, archive, self.contract)["passed"])
    def test_bad_zip_member_rejected(self):
        archive = self.root / "p.zip"
        with zipfile.ZipFile(archive, "w") as z: z.writestr("../op.py", "pass")
        self.assertFalse(validate_package(self.source, archive, self.contract)["passed"])
    def test_dtype_metadata_allowed_but_torch_tensor_compute_rejected(self):
        archive = self.root / "p.zip"
        (self.source / "op.py").write_text("import torch\ndef op(x):\n    kind = torch.promote_types(x.dtype, x.dtype)\n    return torch.empty(x.shape, dtype=kind)\n")
        self.assertTrue(make_package(self.source, archive, self.contract)["passed"])
        (self.source / "op.py").write_text("import torch\ndef op(x):\n    return torch.add(x, x)\n")
        self.assertFalse(make_package(self.source, archive, self.contract)["passed"])
    def test_missing_publication_evidence_rejected(self):
        store.write_json(self.root / "contract.json", self.contract)
        self.assertFalse(publication_check(self.root)["passed"])
    def test_official_records_are_idempotent_and_immutable(self):
        row = {"record_id": "r", "submitted_at": "2026-10-01", "evidence_class": "official-platform", "evidence": "fixture", "local_archive_sha256": "a" * 64,
               "status": "completed", "targets": {"a": {"status": "pass", "speedup": 1}, "b": {"status": "pass", "speedup": 2}}, "pass_count": 2, "aggregate_speedup": 1.5}
        with patch.object(ledger, "ROOT", self.root):
            self.assertEqual(ledger.record("task1", row)["status"], "recorded")
            self.assertEqual(ledger.record("task1", row)["status"], "unchanged")
            row["aggregate_speedup"] = 2
            with self.assertRaises(ValueError): ledger.record("task1", row)
    def test_partial_matrix_cannot_have_aggregate(self):
        row = {"record_id": "r", "submitted_at": "t", "evidence_class": "official-platform", "evidence": "fixture", "local_archive_sha256": "a" * 64,
               "targets": {"a": {"status": "pass", "speedup": 1}, "b": {"status": "fail", "speedup": None}}, "pass_count": 1, "aggregate_speedup": 1}
        self.assertTrue(ledger.validate(row, self.contract))
    def test_task_target_count_is_not_global_eight(self):
        row = {"record_id": "r", "submitted_at": "t", "evidence_class": "official-platform", "evidence": "fixture", "local_archive_sha256": "a" * 64,
               "targets": {"a": {"status": "pass", "speedup": 1}, "b": {"status": "pass", "speedup": 2}}, "pass_count": 2, "aggregate_speedup": 1.5}
        self.assertEqual(ledger.validate(row, self.contract), [])
    def test_constant_only_change_is_not_structural(self):
        baseline = self.root / "base.py"; candidate = self.root / "next.py"
        baseline.write_text("def op(x):\n    return x + 1\n")
        candidate.write_text("def op(x):\n    return x + 2\n")
        self.assertEqual(normalized_ast(baseline)[0], normalized_ast(candidate)[0])
        candidate.write_text("def op(x):\n    if x:\n        return x + 2\n    return x\n")
        self.assertNotEqual(normalized_ast(baseline)[0], normalized_ast(candidate)[0])
    def test_evaluating_revision_can_be_retried(self):
        row = {"record_id": "r", "submitted_at": "t", "evidence_class": "official-platform", "evidence": "fixture", "local_archive_sha256": "a" * 64, "status": "evaluating",
               "targets": {"a": {"status": "pending", "speedup": None}, "b": {"status": "pending", "speedup": None}}, "pass_count": 0, "aggregate_speedup": None}
        with patch.object(ledger, "ROOT", self.root):
            ledger.record("task1", row)
            final = copy.deepcopy(row); final.update(status="completed", pass_count=2, aggregate_speedup=1.0)
            final["targets"] = {key: {"status": "pass", "speedup": 1} for key in ("a", "b")}
            ledger.record("task1", final)
            self.assertEqual(ledger.record("task1", final)["status"], "unchanged")


class ComparisonTests(unittest.TestCase):
    def test_different_device_or_partial_results_are_rejected(self):
        base = {"command": "bench", "status": "passed", "device": {"name": "T4", "device_id": "1"}, "rows": [], "run_id": "a"}
        other = copy.deepcopy(base); other["device"]["device_id"] = "2"
        with self.assertRaises(ValueError): comparison([base, other])
        other = copy.deepcopy(base); other["status"] = "incomplete"
        with self.assertRaises(ValueError): comparison([base, other])
    def test_empty_or_unknown_case_selection_rejected(self):
        class Adapter:
            @staticmethod
            def cases(): return [{"id": "small"}]
            @staticmethod
            def quick_ids(): return []
        with self.assertRaises(ValueError): select(Adapter, None, True)
        with self.assertRaises(ValueError): select(Adapter, ["unknown"], True)


if __name__ == "__main__": unittest.main()
