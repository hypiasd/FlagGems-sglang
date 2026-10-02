"""Numerical CPU-model coverage for the Task 111 candidate.

``test_task.py`` decides things from the text alone.  This file runs the member
through the shared CPU semantic model (the same machinery ``validate_cpu.py``
drives) for the shapes the development table does *not* contain, so that branches
the table cannot reach are still executed before they reach a device.

Right now that means one branch: the member derives the dst/out strides from the
shape only when ``dst`` is dense, and carries them explicitly otherwise.  Every
case in ``adapter.cases()`` has a dense ``dst``, so without this test the
explicit-stride path would never run locally -- and it is the path that decides
whether a caller handing us a strided destination still gets the right answer.

The candidate module is loaded **once per process**: re-executing it re-registers
torch's ``c10d_functional`` kernels and fails with "Only a single TORCH_LIBRARY
can be used to register the namespace", which has nothing to do with the member.

Run with ``python -m unittest competition.task111.test_cpu_semantics``.
"""

from __future__ import annotations

import unittest

from competition.task111 import validate_cpu


def _case(**overrides):
    case = {
        "id": "l2-c16-r5-d3-dim8-w3-strided-dst",
        "layers": 2,
        "cache": 16,
        "requests": 5,
        "draft": 3,
        "dim": 8,
        "window": 3,
        "dtype": "float32",
        "all_invalid": False,
        "pad": 0,
        "strided_dst": True,
        "int64_indices": False,
        "coverage": "development-assumptions",
    }
    case.update(overrides)
    return case


class NonContiguousDestination(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if validate_cpu.torch is None:
            raise unittest.SkipTest("PyTorch is unavailable")
        cls.cpu_model = validate_cpu.load_module(
            "_flagos_shared_cpu_model", validate_cpu.MODEL_PATH
        )
        cls.adapter = validate_cpu.load_module(
            "_task111_adapter_strided", validate_cpu.TASK / "adapter.py"
        )
        cls.model = cls.cpu_model.CPUModel()
        with cls.model.installed():
            # staticmethod: a plain function stored on the class would be bound
            # to ``self`` on attribute access and receive an extra argument.
            cls.wrapper = staticmethod(
                validate_cpu.load_candidate(
                    validate_cpu.TASK / "conv_window_scatter_with_mask.py", cls.model
                ).conv_window_scatter_with_mask
            )

    def _validate(self, case, wrapper=None):
        return validate_cpu.run_case(
            self.adapter,
            self.cpu_model,
            self.model,
            wrapper or self.wrapper,
            case,
            0,
            False,
        )

    def test_the_adapter_really_builds_a_non_dense_destination(self) -> None:
        # Guard the guard: if this ever becomes contiguous, the tests below stop
        # covering anything and would silently keep passing.
        dst, _, _, _ = self.adapter.inputs(_case(), "cpu", 0)
        self.assertFalse(dst.is_contiguous())
        self.assertGreater(dst.stride(-1), 1)

    def test_a_strided_destination_still_matches_the_reference(self) -> None:
        # run_case raises on any mismatch, wrong write count, mutated input or
        # out-of-view store, so reaching the end is the assertion.
        self.assertGreater(self._validate(_case()), 0)

    def test_a_strided_destination_with_int64_index_inputs(self) -> None:
        # The pointer-width selection is its own constexpr; exercise it together
        # with the stride fallback so neither can regress unnoticed.
        self.assertGreater(self._validate(_case(int64_indices=True)), 0)

    def test_an_all_invalid_strided_destination_is_a_plain_copy(self) -> None:
        # All-invalid means "return an unmodified copy", which for a strided
        # destination is the case most likely to be got wrong by a stride mix-up.
        self.assertGreater(
            self._validate(_case(all_invalid=True)), 0
        )

    def test_the_dense_destination_still_takes_the_shape_derived_path(self) -> None:
        # The counterpart of the tests above: with a dense ``dst`` the member
        # takes the other branch.  Both must produce the reference result.
        self.assertGreater(self._validate(_case(strided_dst=False)), 0)


if __name__ == "__main__":
    unittest.main()
