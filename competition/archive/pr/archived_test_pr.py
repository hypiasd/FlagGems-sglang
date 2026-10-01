"""Offline tests for the upstream competition-PR channel.

Stdlib only, no GPU and no torch: everything here is text, AST and gate
plumbing.  Run with ``python -m unittest competition.archive.pr.test_pr``.
"""

from __future__ import annotations

import ast
import unittest

from . import bundle, description, gates, spec

AWARDED = '''"""Docstring."""

import torch
import triton
import triton.language as tl


def recompute_w_u(k, v, beta, g_cumsum, A, cu_seqlens):
    O = k + v
    return O, beta
'''

ENTRY = "recompute_w_u(k, v, beta, g_cumsum, A, cu_seqlens)"


class MappingTest(unittest.TestCase):
    def test_generic_and_vendor_tiers(self) -> None:
        self.assertEqual(
            spec.tier_path("op", None),
            ("generic", "src/flaggems_sglang/ops/op.py"),
        )
        self.assertEqual(
            spec.tier_path("op", "kunlunxin"),
            (
                "vendor",
                "src/flaggems_sglang/runtime/backend/_kunlunxin/ops/op.py",
            ),
        )
        self.assertEqual(
            spec.tier_path("op", "tianshu"),
            (
                "vendor",
                "src/flaggems_sglang/runtime/backend/_iluvatar/ops/op.py",
            ),
        )

    def test_unknown_chip_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            spec.tier_path("op", "card_a")


class PrologueTest(unittest.TestCase):
    def test_header_all_and_whitespace(self) -> None:
        text = bundle._ensure_prologue(AWARDED, "recompute_w_u")
        self.assertTrue(text.startswith(spec.HEADER))
        tree = ast.parse(text)
        assigned = [n for n in tree.body if isinstance(n, ast.Assign)]
        self.assertEqual(
            ast.literal_eval(assigned[-1].value), ["recompute_w_u"]
        )
        self.assertTrue(text.endswith('\n__all__ = ["recompute_w_u"]\n'))

    def test_idempotent_and_conflicting_all_rejected(self) -> None:
        once = bundle._ensure_prologue(AWARDED, "recompute_w_u")
        self.assertEqual(bundle._ensure_prologue(once, "recompute_w_u"), once)
        with self.assertRaises(ValueError):
            bundle._ensure_prologue(
                AWARDED + '\n__all__ = ["other"]\n', "recompute_w_u"
            )


class PreservationTest(unittest.TestCase):
    def test_equal_modulo_prologue(self) -> None:
        bundled = bundle._ensure_prologue(AWARDED, "recompute_w_u")
        equal, _ = gates.ast_equal_modulo_prologue(AWARDED, bundled)
        self.assertTrue(equal)

    def test_real_change_detected(self) -> None:
        bundled = bundle._ensure_prologue(AWARDED, "recompute_w_u").replace(
            "k + v", "k - v"
        )
        equal, detail = gates.ast_equal_modulo_prologue(AWARDED, bundled)
        self.assertFalse(equal)
        self.assertIn("changed", detail)

    def test_declared_rename_is_normalised(self) -> None:
        bundled = bundle._ensure_prologue(AWARDED, "recompute_w_u")
        renamed, count = bundle._rename_tokens(bundled, {"O": "out_ptr"})
        self.assertEqual(count, 2)
        equal, detail = gates.ast_equal_modulo_prologue(
            AWARDED, renamed, {"O": "out_ptr"}
        )
        self.assertTrue(equal)
        self.assertIn("renames", detail)

    def test_rename_never_touches_strings_or_comments(self) -> None:
        text = '# O stays\nx = "O stays too"\nO = 1\n'
        renamed, count = bundle._rename_tokens(text, {"O": "out_ptr"})
        self.assertEqual(count, 1)
        self.assertIn("# O stays", renamed)
        self.assertIn('"O stays too"', renamed)
        self.assertIn("out_ptr = 1", renamed)


class ReflowTest(unittest.TestCase):
    def test_wraps_docstring_and_comments_only(self) -> None:
        long_doc = '"""' + "word " * 40 + '"""'
        long_comment = "# " + "comment " * 30
        short_code = "value = 1"
        text, changed = bundle._reflow_long_text(
            f"{long_doc}\n{long_comment}\n{short_code}\n"
        )
        self.assertEqual(changed, [1, 2])
        lines = text.split("\n")
        self.assertTrue(
            all(len(line) <= spec.FLAKE8_MAX_LINE for line in lines[:-2])
        )
        self.assertIn(short_code, lines)

    def test_leaves_long_code_lines_alone(self) -> None:
        long_code = "value = " + " + ".join(["a"] * 40)
        text, changed = bundle._reflow_long_text(long_code + "\n")
        self.assertEqual(changed, [])
        self.assertIn(long_code, text)


class GateTest(unittest.TestCase):
    def _bundle(self, path):
        return {
            "op": "recompute_w_u",
            "files": [
                {
                    "path": path,
                    "suffix": "kunlunxin",
                    "name": "recompute_w_u_kunlunxin.py",
                }
            ],
        }

    def test_structure_accepts_and_rejects(self) -> None:
        good = bundle._ensure_prologue(AWARDED, "recompute_w_u")
        rel = "src/flaggems_sglang/runtime/backend/_kunlunxin/ops/recompute_w_u.py"
        self.assertEqual(
            gates.structure_gate(self._bundle(rel), {rel: good}, ENTRY)[
                "status"
            ],
            gates.PASS,
        )
        bad = {
            rel: good.replace(
                '__all__ = ["recompute_w_u"]', '__all__ = ["wrong"]'
            )
        }
        self.assertEqual(
            gates.structure_gate(self._bundle(rel), bad, ENTRY)["status"],
            gates.FAIL,
        )
        misplaced = {"src/flaggems_sglang/ops/recompute_w_u.py": good}
        self.assertEqual(
            gates.structure_gate(self._bundle(rel), misplaced, ENTRY)[
                "status"
            ],
            gates.FAIL,
        )

    def test_hygiene_flags_ci_violations(self) -> None:
        good = bundle._ensure_prologue(AWARDED, "recompute_w_u")
        self.assertEqual(
            gates.hygiene_gate({"a.py": good})["status"], gates.PASS
        )
        damaged = good.replace("# Copyright 2026", "# copyright", 1)
        self.assertEqual(
            gates.hygiene_gate({"a.py": damaged})["status"], gates.FAIL
        )
        no_final_newline = good.rstrip("\n")
        self.assertEqual(
            gates.hygiene_gate({"a.py": no_final_newline})["status"],
            gates.FAIL,
        )
        trailing = good.replace("import torch", "import torch   ")
        self.assertEqual(
            gates.hygiene_gate({"a.py": trailing})["status"], gates.FAIL
        )
        long_line = good + "y = '" + "z" * 200 + "'\n"
        self.assertEqual(
            gates.hygiene_gate({"a.py": long_line})["status"], gates.FAIL
        )


class DescriptionTest(unittest.TestCase):
    def test_body_carries_competition_evidence(self) -> None:
        built = {
            "op": "recompute_w_u",
            "task_id": "task103",
            "adaptation_id": "adapt-x",
            "files": [
                {
                    "path": "src/flaggems_sglang/ops/recompute_w_u.py",
                    "tier": "generic",
                    "suffix": None,
                }
            ],
            "provenance": {
                "platform_package_sha256": "a" * 64,
                "submitted_at": "2026-10-01T19:44:56+08:00",
                "submission_evidence": "queued",
                "submission_id": None,
            },
            "official": {
                "available": True,
                "revisions": 1,
                "record": {
                    "submitted_at": "2026-10-01T19:44:56+08:00",
                    "status": "completed",
                    "pass_count": 7,
                    "aggregate_speedup": 4.8,
                    "targets": {
                        "kunlunxin": {"status": "pass", "speedup": 1.85},
                        "iluvatar": {"status": "pass", "speedup": 0.17},
                        "metax": {"status": "pass", "speedup": 4.98},
                        "hygon": {"status": "pass", "speedup": 18.46},
                        "ascend": {"status": "pass", "speedup": 3.44},
                        "intl_a": {"status": "pass", "speedup": 0.89},
                        "intl_b": {"status": "pass", "speedup": 3.81},
                    },
                },
            },
            "renames": {"O": "out_ptr"},
            "formatting": {
                "reflowed_text_lines": {
                    "src/flaggems_sglang/ops/recompute_w_u.py": [18]
                }
            },
        }
        body = description.render(built)
        self.assertIn("FlagOS x SGLang competition", body)
        self.assertIn("`O` -> `out_ptr`", body)
        self.assertIn("1.85", body)
        self.assertIn("**7/7**", body)
        self.assertIn("4.8", body)
        self.assertIn("No `tests/test_recompute_w_u.py`", body)
        self.assertIn("section 9", body)


if __name__ == "__main__":
    unittest.main()
