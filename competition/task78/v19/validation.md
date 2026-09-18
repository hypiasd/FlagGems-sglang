# v19 validation record

The final seven source files passed Python syntax, unique public-entry and
static operation checks, plus the existing restricted CPU execution model:
189 self-created cases per source, 1323 total. This is NOT the repository's
official Task78 test suite, a Triton interpreter, device compilation, or a
benchmark. The checkout contains no Task78 official test.

Tests exercise mixed fp16/bf16/fp32 promotion, odd token/head tails, column
splits at 512/1024, noncontiguous and zero-stride views, storage offsets,
empty outputs and empty segments. Active source/destination bounds,
single output writes, input immutability and exact numeric results are checked.
NaN payload bits and signed-zero representation are not certified.

All 181 nonempty cases per source invoked its new v19 kernel exactly once.
Ascend's maximum observed grid product was 32. The simulator uses int64
indices, so the wrapper's WIDE policy still needs real compiler validation.
Two elementary simulator operations (`where`, `maximum`) were added to
execute the new GPU data flow; acceptance criteria were not relaxed.

| File | SHA-256 |
| --- | --- |
| concat_and_cast_mha_k.py | c268a0ae4dc6970e38586f3272e26f9e46058f275418831e232c951b52f8980d |
| concat_and_cast_mha_k_ascend.py | 0846663e360cd6c265a2f9e5f37b763f7afd8cbcf5be4588ba897717b407beb9 |
| concat_and_cast_mha_k_enflame.py | 58673fba5dd8aabffd5501fc1a8131bb6933e7648fe4f42fa130658a36cf269e |
| concat_and_cast_mha_k_hygon.py | 2680ec5c62feeebfdda5a9045eb80744b8e5cf03bd638ce82076d7595c7042a6 |
| concat_and_cast_mha_k_iluvatar.py | 7e95aaa5a2ac2fcda85c54f6eae9dca04f0feb1ec3d1e63cb1a8a66bcb851755 |
| concat_and_cast_mha_k_kunlunxin.py | 7d8f1aadae0a60ec36a571e59924fdca56ac3891504d76e58e77c69aa4e6322e |
| concat_and_cast_mha_k_metax.py | 1c362d1b08f250b7bd28c16ec673b8cf0884cf2183ef1f8dca04007771e9ae25 |

Reproduce from the project root:

```sh
python3 competition/task78/validate_cpu.py --all
unzip -t competition/task78/flagos-task78-v19.zip
git diff --check
```

The archive must contain exactly these seven source files, byte-identical
to this hash list. Device correctness, compatibility and score remain unknown.
