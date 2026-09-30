# v18 local validation record

Run on 2026-09-18 with Python 3.12 / PyTorch 2.13.0 CPU.
Command: `python3 competition/task78/validate_cpu.py --all`.

The validator's negative controls passed, then all seven source files passed
189 cases. This is CPU source emulation, not target compilation or benchmarking.

| File suffix | SHA-256 prefix of tested source | Cases passed | Nonempty launches | Max programs/launch |
| --- | --- | ---: | ---: | ---: |
| default | 74aed7237a34 | 189/189 | 181 | 975 |
| ascend | 7ed3761a57b3 | 189/189 | 181 | 32 |
| enflame | 0d20e41e30c9 | 189/189 | 181 | 656 |
| hygon | 88c44d466c3c | 189/189 | 181 | 585 |
| iluvatar | 29bb8601b4cb | 189/189 | 181 | 975 |
| kunlunxin | 47a5a562d381 | 189/189 | 181 | 1312 |
| metax | dcc1f4c2f6df | 189/189 | 181 | 840 |

All 1,323 cases passed. Every active input access was in the passed view,
input backing storage remained unchanged, and every output element was written
exactly once. Nonempty calls each issued one kernel launch; empty outputs none.
The Ascend launch-product assertion enforced 32 total programs, not merely a
32-entry first dimension.

Not validated here: backend compiler support, Triton integer-lowering details,
device memory layout, register/UB requirements, hardware concurrency, bitwise
NaN payload/signed-zero preservation or performance.
