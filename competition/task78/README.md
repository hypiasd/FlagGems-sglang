# Task 78: `concat_and_cast_mha_k`

Current candidate: [v18 design and per-chip matrix](v18/README.md).
Submission package: `flagos-task78-v18.zip`, containing the seven operator
files in this directory. All files expose `concat_and_cast_mha_k(k, k_nope, k_rope)`.

v17 finished with 7/8 (Kunlunxin failed); the last complete run was v16 at
1.18x. The best complete recorded run remains v12 at 1.27x.

v18 uses one launch per nonempty call: dense segment jobs, hybrid dense NoPE
and broadcast RoPE jobs, or Hygon's serial two-token tiles. Ascend caps the
entire one-dimensional persistent grid at 32 programs. All variants preserve
cat-then-cast dtype promotion and support strided input tensors.

Run `python3 competition/task78/validate_cpu.py --all` from the repository
root for direct-source CPU semantic checks. This does not compile Triton or
prove device performance. The local [validation record](v18/validation.md)
contains the tested source hashes and results.

Historical version directories and ZIPs remain reproducible references.
The user uploads candidates to FlagOS for real eight-chip evaluation.
