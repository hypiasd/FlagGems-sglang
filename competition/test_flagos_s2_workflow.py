#!/usr/bin/env python3
"""Tests for the shared FlagOS S2 run-state gates."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flagos_s2_workflow as workflow


class PreparedDiagnosticRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name) / "task78" / "run-001"
        self.run_dir.mkdir(parents=True)
        profile_bytes = (HERE / "workflow" / "tasks" / "task78.json").read_bytes()
        (self.run_dir / "task-profile.json").write_bytes(profile_bytes)
        (self.run_dir / "run.json").write_text(json.dumps({
            "profile_snapshot": "task-profile.json",
            "profile_sha256": workflow.sha256_bytes(profile_bytes),
        }), encoding="utf-8")
        profile = json.loads(profile_bytes)
        self.evidence = {
            "service_state": "callable",
            "tool_name": "optimize_kernel",
            "baseline_kind": "candidate_source",
            "baseline_hashes": {target: "a" * 64 for target in profile["targets"]},
            "forecast_status": "pending",
            "pre_generation_hypothesis": "Fuse segment loads into one target-specific output tile.",
            "generation_plan_sha256": "b" * 64,
            "frozen_task_goal": "Pass all targets and clear the adapter release hurdle.",
            "goal_source": "competition/workflow/tasks/task78.json",
        }

    def test_pending_forecast_allows_diagnostic_generation_with_frozen_plan(self):
        workflow.validate_transition(
            "task78", self.run_dir, "inconclusive", "prepared", self.evidence
        )

    def test_pending_forecast_requires_hypothesis_and_plan_hash(self):
        self.evidence.pop("generation_plan_sha256")
        with self.assertRaisesRegex(workflow.WorkflowError, "frozen structural hypothesis"):
            workflow.validate_transition(
                "task78", self.run_dir, "inconclusive", "prepared", self.evidence
            )

    def test_failed_forecast_cannot_prepare_a_run(self):
        self.evidence["forecast_status"] = "fail"
        with self.assertRaisesRegex(workflow.WorkflowError, "must be pass or pending"):
            workflow.validate_transition(
                "task78", self.run_dir, "inconclusive", "prepared", self.evidence
            )

    def test_prepared_self_checkpoint_records_generation_start(self):
        self.evidence.update({
            "activity": "generation_started",
            "source_generation_started": True,
        })
        workflow.validate_transition(
            "task78", self.run_dir, "prepared", "prepared", self.evidence
        )

    def test_prepared_self_checkpoint_rejects_missing_generation_marker(self):
        self.evidence["activity"] = "generation_started"
        with self.assertRaisesRegex(workflow.WorkflowError, "durable generation-start marker"):
            workflow.validate_transition(
                "task78", self.run_dir, "prepared", "prepared", self.evidence
            )


class GeneratedReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name) / "task78" / "run-001"
        self.run_dir.mkdir(parents=True)
        profile_bytes = (HERE / "workflow" / "tasks" / "task78.json").read_bytes()
        (self.run_dir / "task-profile.json").write_bytes(profile_bytes)
        (self.run_dir / "run.json").write_text(json.dumps({
            "profile_snapshot": "task-profile.json",
            "profile_sha256": workflow.sha256_bytes(profile_bytes),
        }), encoding="utf-8")
        profile = json.loads(profile_bytes)
        self.members = profile["package"]["root_members"]
        self.evidence = {
            "request_sha256": "a" * 64,
            "source_hashes": {member: "b" * 64 for member in self.members},
            "service_call_receipt_sha256": "c" * 64,
            "tool_name": "optimize_kernel",
        }

    def test_sync_kernelgen_call_can_use_saved_receipt_hash(self):
        workflow.validate_transition(
            "task78", self.run_dir, "prepared", "generated", self.evidence
        )

    def test_sync_kernelgen_call_requires_receipt_when_job_id_is_absent(self):
        self.evidence.pop("service_call_receipt_sha256")
        with self.assertRaisesRegex(workflow.WorkflowError, "service job id"):
            workflow.validate_transition(
                "task78", self.run_dir, "prepared", "generated", self.evidence
            )

    def test_multi_source_sync_calls_bind_receipts_to_every_member(self):
        self.evidence.pop("service_call_receipt_sha256")
        self.evidence["service_call_receipts_sha256"] = {
            member: "c" * 64 for member in self.members
        }
        workflow.validate_transition(
            "task78", self.run_dir, "prepared", "generated", self.evidence
        )

    def test_multi_source_sync_calls_reject_a_partial_receipt_bundle(self):
        self.evidence.pop("service_call_receipt_sha256")
        self.evidence["service_call_receipts_sha256"] = {
            self.members[0]: "c" * 64
        }
        with self.assertRaisesRegex(workflow.WorkflowError, "service job id"):
            workflow.validate_transition(
                "task78", self.run_dir, "prepared", "generated", self.evidence
            )


if __name__ == "__main__":
    unittest.main()
