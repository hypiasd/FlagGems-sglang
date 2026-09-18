# Task 78: `concat_and_cast_mha_k`

Current candidate: [v19 design and per-chip matrix](v19/README.md).
Submission package: `flagos-task78-v19.zip`, containing the seven operator
files in this directory. All files expose `concat_and_cast_mha_k(k, k_nope, k_rope)`.

v18 finished with 6/8 (Enflame/Kunlunxin failed, Ascend 0.02x); the last complete run was v16 at
1.18x. The best complete recorded run remains v12 at 1.27x.

v19 uses fused output stores on GPU paths, one-dimensional serial-head
RoPE reuse on Enflame/Kunlunxin, and persistent token blocks with hoisted
RoPE loads on Ascend. Ascend caps the entire grid at 32 programs.
All variants preserve cat-then-cast dtype promotion and source strides.
Every nonempty shape enters the new structure. Before submission, performance
was unmeasured.

v19 completed with 8/8 and an average of 1.25x: Iluvatar 2.36x, MetaX 1.45x,
Enflame 0.25x, Hygon 2.45x, Kunlunxin 0.27x, Ascend 0.14x, and International
A/B 1.60/1.51x. It restores both v18 failures and improves Iluvatar, MetaX and
Hygon, but remains 0.02x below the complete v12 result of 1.27x. The v19
device result supersedes the unmeasured wording above.

Run `python3 competition/task78/validate_cpu.py --all` from the repository
root for direct-source CPU semantic checks. This does not compile Triton or
prove device performance. The local [validation record](v19/validation.md)
contains the tested source hashes and results.

Historical version directories and ZIPs remain reproducible references.
The user uploads candidates to FlagOS for real eight-chip evaluation.
