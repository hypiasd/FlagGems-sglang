"""Official FlagOS x SGLang competition submission contract.

Distilled from the upstream repository's ``docs/CONTRIBUTING.md`` section 9
(competition submissions) plus the conventions observable in merged
competition PRs.  The platform ZIP channel and this channel are different
contracts: the platform wants ``<op>.py`` / ``<op>_<chip>.py`` inside one
archive, the repository wants the very same sources placed in the dispatcher
tree with an Apache header and ``__all__``.
"""

from __future__ import annotations

TIERS = {
    "generic": "src/flaggems_sglang/ops/{op}.py",
    "vendor": "src/flaggems_sglang/runtime/backend/{vendor}/ops/{op}.py",
    "arch": "src/flaggems_sglang/runtime/backend/{vendor}/{arch}/ops/{op}.py",
}

# Vendor directories that exist upstream.  A platform chip suffix maps onto one
# of them; the international cards have no vendor folder and use the generic
# tier, which is exactly what the platform's unsuffixed file is for.
VENDOR_DIRS = (
    "_amd",
    "_ascend",
    "_enflame",
    "_hygon",
    "_iluvatar",
    "_kunlunxin",
    "_metax",
    "_mthreads",
    "_nvidia",
    "_thead",
)

# Platform chip suffix (as used by the submission ZIP and the task contract)
# to upstream vendor directory.  Identity where the names already agree.
CHIP_TO_VENDOR = {v.lstrip("_"): v for v in VENDOR_DIRS}
CHIP_TO_VENDOR.update(
    {
        "tianshu": "_iluvatar",
        "muxi": "_metax",
        "haiguang": "_hygon",
        "huawei": "_ascend",
        "kunlunxin": "_kunlunxin",
    }
)
# The two international card entries are deliberately absent: they are served
# by the generic tier.

HEADER = """\
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

# Hard gates from section 9.  Everything here is mandatory for a competition
# PR; tests, benchmarks, docs and operators.yaml entries are explicitly *not*
# required (the maintainers run the held-out harness).
REQUIRED = (
    "file sits in the correct dispatcher tier",
    "Apache 2.0 header on every new file",
    '__all__ = ["<op>"] in every file',
    "import flaggems_sglang succeeds and all_registered_ops() lists the op",
    "pre-commit run --all-files passes (black 79, isort --profile black, flake8)",
    "CLA signed by every committer",
)
NOT_REQUIRED = (
    "tests/test_<op>.py",
    "benchmark/test_<op>.py or shapes in benchmark/attri_util.py",
    "PR-description performance numbers (the harness collects them)",
    "docs/tasks/<op>.md",
    "conf/operators.yaml entry",
)

# Used by gates.run_style_gate; matches .pre-commit-config.yaml.
ISORT_ARGS = ("--profile", "black", "--line-length", "80")
FLAKE8_ARGS = ("--ignore=F405,E731,W503,E203,E704", "--max-line-length=120")
BLACK_LINE_LENGTH = 79
# flake8's effective limit in .pre-commit-config.yaml (E501 is not ignored).
FLAKE8_MAX_LINE = 120


def vendor_for(suffix: str) -> str | None:
    """Map a platform chip suffix (``kunlunxin``) to a vendor dir (``_kunlunxin``)."""
    return CHIP_TO_VENDOR.get(suffix.lstrip("_").lower())


def tier_path(op: str, suffix: str | None) -> tuple[str, str]:
    """Return ``(tier, repo-relative path)`` for an operator source file."""
    if suffix is None:
        return "generic", TIERS["generic"].format(op=op)
    vendor = vendor_for(suffix)
    if vendor is None:
        raise ValueError(
            f"unknown chip suffix {suffix!r}: no upstream vendor tier"
        )
    return "vendor", TIERS["vendor"].format(op=op, vendor=vendor)


def rules() -> dict:
    """Machine-readable copy of the competition submission rules."""
    return {
        "source": "https://github.com/flagos-ai/FlagGems-sglang/blob/master/docs/CONTRIBUTING.md#9",
        "tiers": TIERS,
        "vendor_dirs": list(VENDOR_DIRS),
        "chip_to_vendor": dict(sorted(CHIP_TO_VENDOR.items())),
        "required": list(REQUIRED),
        "not_required": list(NOT_REQUIRED),
        "code_style": {
            "black_line_length": BLACK_LINE_LENGTH,
            "isort": list(ISORT_ARGS),
            "flake8": list(FLAKE8_ARGS),
        },
    }
