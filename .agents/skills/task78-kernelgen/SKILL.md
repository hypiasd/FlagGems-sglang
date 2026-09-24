---
name: task78-kernelgen
description: Optimize FlagOS S2 Task 78 using the shared task-neutral workflow and the Task 78 operator adapter.
---

# Task 78 compatibility entry

For all current work, use the shared [`flagos-s2-autopilot`](../flagos-s2-autopilot/SKILL.md)
workflow and Task 78 adapter [`competition/task78/kernelgen/README.md`](../../../competition/task78/kernelgen/README.md).
This entry remains for callers that explicitly name the older Task 78 workflow.

Task 78-specific constraints remain:

- Preserve exact `concat_and_cast_mha_k(k, k_nope, k_rope)` semantics,
  including cat-then-cast promotion, supported strides, empty/tail behavior,
  and input immutability.
- Freeze per-chip baseline source and record exact hashes. International A and
  B are separate targets even though both use the default source.
- Require a structural optimization for each of the seven implementation
  files. The Task 78 forecast hurdle and deterministic/compiler-risk gates stay
  defined by the adapter and `competition/task78/kernelgen/` tools.
- Run `competition/task78/validate_cpu.py` as local semantic evidence only;
  it does not validate target compilation, device correctness, or performance.
- Submit through Chrome under the active FlagOS S2 session authorization and
  record the official FlagOS result before promotion.
