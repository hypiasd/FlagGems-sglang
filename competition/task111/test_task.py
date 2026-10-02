"""Torch-free checks for the Task 111 task package.

Nothing here runs the reference or the kernel numerically -- that is what
``validate_cpu.py`` does.  What these checks can decide from the text alone is
whether the profile still matches the official contract, whether the adapter
exposes the harness interface, whether the development table covers the
branches the task names (overlapping strided source, all-invalid, non-power-of
-two extents), and whether the candidate smuggles in a fallback.

Run with ``python -m unittest competition.task111.test_task``.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "competition/task111"
PROFILE = json.loads((TASK / "profile.json").read_text())
ADAPTER_TEXT = (TASK / "adapter.py").read_text()
CANDIDATE = (TASK / "conv_window_scatter_with_mask.py").read_text()

OFFICIAL_TARGETS = [
    "iluvatar",
    "metax",
    "hygon",
    "kunlunxin",
    "ascend",
    "intl_a",
    "intl_b",
]
OFFICIAL_ARGS = ["dst", "src", "dst_indices_raw", "step_indices_raw"]
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


def load_adapter():
    namespace = {}
    exec(compile(ADAPTER_TEXT, "adapter.py", "exec"), namespace)
    return namespace


class ProfileTest(unittest.TestCase):
    def test_contract_matches_the_official_endpoint(self) -> None:
        self.assertEqual(PROFILE["task_id"], "task111")
        self.assertEqual(PROFILE["operator"], "conv_window_scatter_with_mask")
        self.assertEqual(PROFILE["targets"], OFFICIAL_TARGETS)
        self.assertIn(
            "operator-tasks/conv_window_scatter_with_mask",
            PROFILE["contract_source"]["url"],
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

    def test_core_computation_is_triton_only(self) -> None:
        self.assertEqual(PROFILE["core_computation"], "triton-only")


class AdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_adapter()
        cls.cases = cls.namespace["cases"]()
        cls.quick = cls.namespace["quick_ids"]()

    def test_harness_api_is_complete(self) -> None:
        for name in HARNESS_API:
            self.assertTrue(callable(self.namespace.get(name)), name)

    def test_case_ids_are_unique_and_quick_is_a_subset(self) -> None:
        ids = [case["id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(set(self.quick) <= set(ids))
        self.assertLessEqual(len(self.quick), 3)

    def test_reference_transcribes_the_official_definition(self) -> None:
        source = ADAPTER_TEXT
        # clone + validity mask + nonzero + advanced-index scatter, exactly as
        # the official statement writes it.
        for fragment in (
            "out = dst.clone()",
            "step_indices_raw >= 0",
            "torch.nonzero",
            "out[:, dst_idx] = src[:, req, step_idx]",
        ):
            self.assertIn(fragment, source)

    def test_overlapping_strided_source_is_built_not_faked(self) -> None:
        """The adapter must construct the official layout, not a dense copy."""
        self.assertIn("as_strided", ADAPTER_TEXT)
        self.assertIn("draft + window - 1", ADAPTER_TEXT)
        case = next(c for c in self.cases if c["id"] == "l2-c16-r5-d3-dim8-w3")
        dst, src, dst_idx, step_idx = self.namespace["inputs"](case, "cpu", 0)
        # Step axis and window axis share one contiguous axis, and consecutive
        # dim rows are draft + window - 1 apart: the defining property.
        draft, window, dim = (
            case["draft"],
            case["window"],
            case["dim"],
        )
        self.assertEqual(src.stride(2), 1)
        self.assertEqual(src.stride(4), 1)
        self.assertEqual(src.stride(3), draft + window - 1)
        self.assertEqual(
            src.shape, (case["layers"], case["requests"], draft, dim, window)
        )
        self.assertEqual(
            dst.shape, (case["layers"], case["cache"], dim, window)
        )
        self.assertEqual(int(dst_idx.max()), case["cache"] - 1)

    def test_slots_are_unique_because_duplicates_are_unscoreable(self) -> None:
        """Duplicate writers make the official reference arbitrary (adapter.py)."""
        self.assertIn("randperm", ADAPTER_TEXT)
        for case in self.cases:
            _, _, dst_idx, step_idx = self.namespace["inputs"](case, "cpu", 0)
            valid = [int(v) for v in dst_idx[step_idx >= 0].tolist()]
            self.assertEqual(len(valid), len(set(valid)), case["id"])

    def test_every_named_branch_is_covered(self) -> None:
        self.assertTrue(any(case["all_invalid"] for case in self.cases))
        self.assertTrue(any(case["requests"] == 1 for case in self.cases))
        self.assertTrue(any(case["layers"] == 1 for case in self.cases))
        self.assertTrue(any(case["dtype"] == "float16" for case in self.cases))
        # Non-power-of-two extents exercise the row/chunk masks.
        for field in ("cache", "dim", "requests"):
            values = [case[field] for case in self.cases]
            self.assertTrue(
                any(value & (value - 1) for value in values), field
            )

    def test_quick_cases_are_the_small_representatives(self) -> None:
        by_id = {case["id"]: case for case in self.cases}
        sizes = [
            by_id[name]["layers"]
            * by_id[name]["cache"]
            * by_id[name]["dim"]
            * by_id[name]["window"]
            for name in self.quick
        ]
        self.assertLess(max(sizes), 300000)

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
                if node.func.attr == "tensor":
                    # Metadata only: a literal list of stride/shape integers.
                    # Anything else would be computing on operator data in
                    # torch, which the task forbids.
                    self.assertTrue(
                        node.args and isinstance(node.args[0], ast.List),
                        "torch.tensor may only build a literal metadata buffer",
                    )
                    continue
                self.assertIn(
                    node.func.attr,
                    {"empty", "empty_like", "new_empty", "promote_types"},
                    node.func.attr,
                )

    def test_kernel_is_triton_jitted_and_addresses_the_source_by_stride(
        self,
    ) -> None:
        self.assertIn("@triton.jit", CANDIDATE)
        # The source's five real strides must come from the tensor, never from a
        # shape assumption.  Either form proves it: the indexed form
        # ``src.stride(4)`` or the whole-tuple form ``src.stride()``, which the
        # current member uses because one tuple call replaced thirteen indexed
        # calls on a host-bound path (measured +1.9x on T4, 2026-10-02).
        self.assertTrue(
            "src.stride(4)" in CANDIDATE or "src.stride()" in CANDIDATE
        )
        self.assertIn("tl.program_id", CANDIDATE)

    def test_kernel_reads_the_request_table_in_kernel(self) -> None:
        """The reverse mapping must happen inside the kernel, not in torch."""
        self.assertIn("for i in range(REQUESTS)", CANDIDATE)
        self.assertIn("tl.load(dst_idx_ptr + i)", CANDIDATE)
