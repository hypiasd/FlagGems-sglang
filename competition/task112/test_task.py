"""Torch-free checks for the Task 112 task package.

This machine has no torch and no reachable device, so nothing here can run the
reference or the kernel numerically.  What it can do is catch the mistakes
that are decidable from the text: a profile that disagrees with the official
contract, a development case table that misses a semantic branch the task
names, an adapter that does not expose the harness interface, or a candidate
whose public entry cannot match the declared contract.

Run with ``python -m unittest competition.task112.test_task``.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from competition import members

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "competition/task112"
PROFILE = json.loads((TASK / "profile.json").read_text())
ADAPTER = (TASK / "adapter.py").read_text()
CANDIDATE = (TASK / "dcp_lse_combine.py").read_text()

OFFICIAL_TARGETS = [
    "iluvatar",
    "metax",
    "hygon",
    "kunlunxin",
    "ascend",
    "intl_a",
    "intl_b",
]
OFFICIAL_ARGS = ["recv_output", "recv_lse", "is_lse_base_on_e", "return_lse"]
HARNESS_API = (
    "cases",
    "quick_ids",
    "inputs",
    "reference",
    "check",
    "metadata",
)


def top_level_functions(source):
    tree = ast.parse(source)
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


class ProfileTest(unittest.TestCase):
    def test_contract_matches_the_official_endpoint(self) -> None:
        self.assertEqual(PROFILE["task_id"], "task112")
        self.assertEqual(PROFILE["operator"], "dcp_lse_combine")
        self.assertEqual(PROFILE["targets"], OFFICIAL_TARGETS)
        self.assertIn(
            "operator-tasks/dcp_lse_combine", PROFILE["contract_source"]["url"]
        )

    def test_entrypoint_names_the_operator_with_the_official_arguments(
        self,
    ) -> None:
        entry = PROFILE["entrypoint"]
        name, rest = entry.split("(", 1)
        self.assertEqual(name, PROFILE["operator"])
        self.assertEqual(
            [part.strip() for part in rest.rstrip(")").split(",")],
            OFFICIAL_ARGS,
        )

    def test_case_coverage_is_not_claimed_as_official(self) -> None:
        self.assertEqual(PROFILE["case_coverage"], "development-assumptions")
        self.assertFalse(PROFILE["official_cases_complete"])

    def test_no_publication_policy_yet(self) -> None:
        self.assertIsNone(PROFILE["publication_policy"])

    def test_package_members_cover_the_operator_module(self) -> None:
        self.assertEqual(
            PROFILE["package_members"], [f"{PROFILE['operator']}.py"]
        )
        layout = members.audit(
            PROFILE["operator"],
            TASK,
            PROFILE["targets"],
            PROFILE["package_members"],
        )
        self.assertTrue(layout["passed"], layout["errors"])


class AdapterTest(unittest.TestCase):
    def test_harness_interface_is_present(self) -> None:
        functions = top_level_functions(ADAPTER)
        for name in HARNESS_API:
            self.assertIn(name, functions, name)

    def test_reference_has_the_official_signature(self) -> None:
        node = top_level_functions(ADAPTER)["reference"]
        self.assertEqual([arg.arg for arg in node.args.args], OFFICIAL_ARGS)

    def test_reference_handles_every_semantic_branch(self) -> None:
        for needle in (
            "isnan",
            'float("inf")',
            "-inf",
            "amax",
            "exp2",
            "log2",
            "return_lse",
            "float32",
        ):
            self.assertIn(needle, ADAPTER, needle)

    def test_check_accepts_a_none_lse_only_when_expected(self) -> None:
        self.assertIn("want[1] is None", ADAPTER)
        self.assertIn("return_lse is False but a combined LSE", ADAPTER)

    def test_no_try_except_in_the_adapter(self) -> None:
        self.assertFalse(
            [n for n in ast.walk(ast.parse(ADAPTER)) if isinstance(n, ast.Try)]
        )


class CaseTableTest(unittest.TestCase):
    def setUp(self) -> None:
        namespace = {"__name__": "task112.adapter_probe"}
        # The adapter imports torch only inside its functions, so the module
        # can be executed here without a tensor library.
        exec(compile(ADAPTER, "adapter.py", "exec"), namespace)
        self.cases = namespace["cases"]()
        self.quick = namespace["quick_ids"]()
        self.namespace = namespace

    def test_ids_are_unique_and_quick_is_a_subset(self) -> None:
        ids = [case["id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(set(self.quick) <= set(ids))
        self.assertLessEqual(len(self.quick), 3)

    def test_every_named_semantic_branch_is_covered(self) -> None:
        flags = {(case["base_e"], case["return_lse"]) for case in self.cases}
        self.assertIn((True, False), flags)
        self.assertIn((True, True), flags)
        self.assertIn((False, True), flags)
        self.assertTrue(any(case["dead_shards"] for case in self.cases))
        self.assertTrue(any(case["n"] == 1 for case in self.cases))

    def test_shapes_are_legal_and_match_the_reference_layout(self) -> None:
        for case in self.cases:
            self.assertGreaterEqual(case["n"], 1)
            self.assertGreaterEqual(case["d"], 1)
            self.assertEqual(case["d"] & (case["d"] - 1), 0, case["d"])

    def test_quick_cases_are_the_small_representatives(self) -> None:
        by_id = {case["id"]: case for case in self.cases}
        sizes = [
            by_id[name]["n"] * by_id[name]["b"] * by_id[name]["d"]
            for name in self.quick
        ]
        self.assertLess(max(sizes), 200000)

    def test_metadata_marks_the_coverage(self) -> None:
        for case in self.cases:
            self.assertEqual(
                self.namespace["metadata"](case)["coverage"],
                "development-assumptions",
            )


class CandidateTest(unittest.TestCase):
    def test_module_exports_exactly_the_operator(self) -> None:
        tree = ast.parse(CANDIDATE)
        assigned = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
        ]
        self.assertEqual(len(assigned), 1)
        self.assertEqual(
            ast.literal_eval(assigned[0].value), [PROFILE["operator"]]
        )

    def test_public_entry_matches_the_declared_contract(self) -> None:
        functions = top_level_functions(CANDIDATE)
        self.assertIn(PROFILE["operator"], functions)
        self.assertEqual(
            [arg.arg for arg in functions[PROFILE["operator"]].args.args],
            OFFICIAL_ARGS,
        )

    def test_no_fallback_shapes_are_present(self) -> None:
        tree = ast.parse(CANDIDATE)
        self.assertFalse(
            [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
            ):
                self.assertIn(
                    node.func.attr,
                    {"empty", "empty_like", "new_empty", "promote_types"},
                    node.func.attr,
                )

    def test_kernel_is_triton_jitted(self) -> None:
        self.assertIn("@triton.jit", CANDIDATE)
        self.assertIn("tl.static_range", CANDIDATE)


if __name__ == "__main__":
    unittest.main()
