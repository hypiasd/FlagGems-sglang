"""Task 112 ``dcp_lse_combine`` reference and development shapes.

``reference`` is a direct transcription of the official task text
(``.../operator-tasks/dcp_lse_combine``), including the three semantics that
are easy to get wrong:

* a NaN or ``+inf`` shard LSE is replaced by ``-inf``, so that shard
  contributes nothing;
* if every shard is ``-inf`` then ``m`` is forced to 0 -- the task states its
  cases never contain such a ``(batch, head)``, so the reference itself is
  allowed to divide 0/0 there and produce NaN;
* ``is_lse_base_on_e=False`` switches ``exp`` to ``exp2`` and ``log`` to
  ``log2`` (the FlashInfer convention).

The case table is a development assumption, not the official one: the platform
publishes no ``cases.py``/``baseline.py`` for this task.  Shapes follow the
upstream SGLang unit test ``test/registered/kernels/ops/attention/
test_dcp_lse_combine.py`` at commit ``41cbe65d``; they are a magnitude
reference only.
"""

from competition.experiments.checking import check_output

# (id, N, B, H, D, base_e, return_lse, dead_shards)
_CASES = (
    ("n2-base-e", 2, 4, 8, 64, True, False, False),
    ("n4-base-e-lse", 4, 8, 16, 128, True, True, False),
    ("n8-base2-lse", 8, 4, 8, 128, False, True, False),
    ("n8-d512", 8, 4, 8, 512, True, False, False),
    ("n2-b64", 2, 64, 16, 128, True, False, False),
    ("n4-d512-base2", 4, 8, 8, 512, False, True, False),
    ("n1-single", 1, 4, 8, 64, True, True, False),
    ("n4-dead-shard", 4, 4, 8, 128, True, False, True),
    # The two cases below exist because of a gap the CPU loop exposed: every
    # published shape has batch * head divisible by any plausible per-program
    # grouping, so a variant that folds several positions into one block never
    # exercises its row mask.  A negative control that deleted that mask
    # therefore still passed.  `n3-b1h3-d96` has 3 positions (one masked row
    # under a 4-row block) and a non-power-of-two head dim (masked columns);
    # `n7-b7h1-base2` has 7 positions (one masked row under an 8-row block).
    ("n3-b1h3-d96", 3, 1, 3, 96, True, True, False),
    ("n7-b7h1-base2", 7, 7, 1, 64, False, True, False),
)

_QUICK = ("n2-base-e", "n4-base-e-lse", "n8-base2-lse")


def cases():
    return [
        {
            "id": case_id,
            "n": n,
            "b": b,
            "h": h,
            "d": d,
            "base_e": base_e,
            "return_lse": return_lse,
            "dead_shards": dead,
        }
        for case_id, n, b, h, d, base_e, return_lse, dead in _CASES
    ]


def quick_ids():
    return list(_QUICK)


def inputs(case, device, seed):
    import torch

    gen = torch.Generator().manual_seed(seed)
    n, b, h, d = case["n"], case["b"], case["h"], case["d"]
    recv_output = torch.randn(n, b, h, d, generator=gen).to(device).bfloat16()
    recv_lse = torch.randn(n, b, h, generator=gen).to(device).float()
    if case.get("dead_shards"):
        # NaN and +inf shards must vanish; the remaining shards decide the
        # result.  Both are sanitized to -inf by the reference.
        recv_lse[0].fill_(float("nan"))
        recv_lse[1].fill_(float("inf"))
    return (
        recv_output,
        recv_lse,
        bool(case["base_e"]),
        bool(case["return_lse"]),
    )


def reference(recv_output, recv_lse, is_lse_base_on_e, return_lse):
    import torch

    lse = recv_lse.to(torch.float32)
    lse = torch.where(
        torch.isnan(lse) | (lse == float("inf")), float("-inf"), lse
    )
    lse_max = lse.amax(dim=0)
    lse_max = torch.where(
        lse_max == float("-inf"), torch.zeros_like(lse_max), lse_max
    )
    centered = lse - lse_max[None]
    w = torch.exp(centered) if is_lse_base_on_e else torch.exp2(centered)
    weight_sum = w.sum(dim=0)
    acc = (recv_output.to(torch.float32) * w[..., None]).sum(dim=0) / (
        weight_sum[..., None]
    )
    out = acc.to(recv_output.dtype)
    if not return_lse:
        return out, None
    log = torch.log if is_lse_base_on_e else torch.log2
    return out, (log(weight_sum) + lse_max).to(recv_lse.dtype)


def check(got, want):
    """Compare against the reference using the task's own tolerance split."""
    import torch

    if not isinstance(got, (tuple, list)) or len(got) != 2:
        raise AssertionError("dcp_lse_combine must return exactly two values")
    if want[1] is None:
        if got[1] is not None:
            raise AssertionError(
                "return_lse is False but a combined LSE was returned"
            )
    else:
        if got[1] is None:
            raise AssertionError("return_lse is True but no LSE was returned")
        # The task compares the combined LSE in fp32 regardless of its
        # storage dtype.
        check_output(
            got[1].to(torch.float32),
            want[1].to(torch.float32),
            atol=0.0001,
            rtol=0.0001,
        )
    check_output(got[0], want[0], atol=0.015, rtol=0.015)


def metadata(case):
    return {**case, "coverage": "development-assumptions"}
