"""Task 111 ``conv_window_scatter_with_mask`` reference and development shapes.

``reference`` is a direct transcription of the official task text
(``.../operator-tasks/conv_window_scatter_with_mask``): clone ``dst``, then for
every request whose ``step_indices_raw`` is not negative, scatter
``src[:, i, step]`` onto slot ``dst_indices_raw[i]``.  Two properties of the
official statement drive the design:

* ``src`` is an **overlapping** ``as_strided`` view: element ``(l, i, s, d, k)``
  lives at ``buffer[l, i, d, s + k]``.  Adjacent steps therefore overlap, the
  window axis and the step axis share one contiguous axis, and consecutive
  ``dim`` rows are ``draft + K - 1`` apart rather than ``K - 1``.  A kernel that
  assumes a flat contiguous row is wrong even though the values would fit.
* when no request is valid the result is exactly ``dst.clone()``.

Duplicate ``dst_indices_raw`` entries are deliberately **not** in the case
table, and that is now a measured requirement rather than a guess.  The
official reference resolves duplicates through ``index_put_``; on torch 2.14.1
CPU the winner is *layer dependent* -- with slot 31 written by requests 1 and 5,
layer 0 keeps the later writer while layers 1 and 2 keep the earlier one, and
the result is reproducible but arbitrary (it follows the parallel partition of
the write).  A hidden case with duplicate slots would be unscoreable for every
implementation, so this table only generates **unique** slots via ``randperm``.
The candidate implements "last valid request wins", deterministically, and
nothing here asserts the arbitrary reference behaviour.

The case table is a development assumption, not the official one: the platform
publishes no ``cases.py``/``baseline.py`` for this task.  Shapes follow the
upstream SGLang unit test for the conv-window variant; they are a magnitude
reference only.
"""

from competition.experiments.checking import check_output

# (id, layers, cache, requests, draft, dim, window, dtypes, all_invalid)
_CASES = (
    ("l2-c16-r5-d3-dim8-w3", 2, 16, 5, 3, 8, 3, "float32", False),
    ("l4-c64-r16-d4-dim64-w8", 4, 64, 16, 4, 64, 8, "float32", False),
    ("l2-c13-r7-d5-dim96-w5", 2, 13, 7, 5, 96, 5, "float32", False),
    ("l1-c32-r1-d2-dim32-w2", 1, 32, 1, 2, 32, 2, "float32", False),
    ("l3-c64-r8-d4-dim128-w16", 3, 64, 8, 4, 128, 16, "float32", False),
    ("l2-c16-r5-d3-dim8-w3-all-invalid", 2, 16, 5, 3, 8, 3, "float32", True),
    ("l2-c512-r32-d16-dim128-w16", 2, 512, 32, 16, 128, 16, "float32", False),
    ("l2-c16-r5-d3-dim8-w3-fp16", 2, 16, 5, 3, 8, 3, "float16", False),
    ("l2-c10-r3-d2-dim24-w4-live", 2, 10, 3, 2, 24, 4, "float32", False),
)

_QUICK = (
    "l2-c16-r5-d3-dim8-w3",
    "l4-c64-r16-d4-dim64-w8",
    "l2-c13-r7-d5-dim96-w5",
)


def cases():
    return [
        {
            "id": case_id,
            "layers": layers,
            "cache": cache,
            "requests": requests,
            "draft": draft,
            "dim": dim,
            "window": window,
            "dtype": dtype,
            "all_invalid": all_invalid,
        }
        for case_id, layers, cache, requests, draft, dim, window, dtype, all_invalid in _CASES
    ]


def quick_ids():
    return list(_QUICK)


def inputs(case, device, seed):
    """Build the official overlapping view, never a contiguous lookalike."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    layers = case["layers"]
    cache = case["cache"]
    requests = case["requests"]
    draft = case["draft"]
    dim = case["dim"]
    window = case["window"]
    dtype = getattr(torch, case["dtype"])

    # Shared sliding-window buffer: [layers, requests, dim, draft + window - 1].
    width = draft + window - 1
    buffer = torch.randn(
        layers, requests, dim, width, generator=gen, dtype=torch.float32
    ).to(dtype)
    src = buffer.as_strided(
        size=(layers, requests, draft, dim, window),
        stride=(buffer.stride(0), buffer.stride(1), 1, buffer.stride(2), 1),
    )
    assert src.stride() == (
        buffer.stride(0),
        buffer.stride(1),
        1,
        buffer.stride(2),
        1,
    )

    dst = torch.randn(layers, cache, dim, window, generator=gen).to(dtype)
    # Unique slots: duplicate writers make the official reference arbitrary
    # (see the module docstring), so a scored case cannot contain them.
    assert requests <= cache, "case needs at least as many slots as requests"
    dst_indices_raw = torch.randperm(cache, generator=gen)[:requests].to(
        torch.int32
    )
    if case["all_invalid"]:
        step_indices_raw = torch.full(
            (requests,), -1, dtype=torch.int32, device=torch.device("cpu")
        )
    else:
        step_indices_raw = torch.randint(
            0, draft, (requests,), generator=gen, dtype=torch.int32
        )
        # At least one request must be valid so the scatter is exercised.
        step_indices_raw[0] = 0
        if requests > 1:
            step_indices_raw[1] = draft - 1
    return (
        dst.to(device),
        src.to(device),
        dst_indices_raw.to(device),
        step_indices_raw.to(device),
    )


def reference(dst, src, dst_indices_raw, step_indices_raw):
    import torch

    out = dst.clone()
    valid = step_indices_raw >= 0
    if not bool(valid.any()):
        return out
    req = torch.nonzero(valid, as_tuple=True)[0]
    dst_idx = dst_indices_raw[req].to(torch.int64)
    step_idx = step_indices_raw[req].to(torch.int64)
    out[:, dst_idx] = src[:, req, step_idx]
    return out


def check(got, want):
    """Pure copy: the official statement allows per-dtype tolerance, exact passes."""
    check_output(got, want, atol=0.0, rtol=0.0)


def metadata(case):
    return {**case, "coverage": "development-assumptions"}
