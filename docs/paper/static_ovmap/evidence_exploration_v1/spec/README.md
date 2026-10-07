# Evidence exploration: execution package

Start with **CODEX_FINAL_EXECUTION_EN.md**. It is the authoritative, self-contained
execution instruction. `PROTOCOL_SPEC.json` fixes IDs, constants, bounds and cohort
membership; the two must agree. `SOURCE_EVIDENCE.md` separates verified existing
behavior from proposed experiments. `CODE_REVIEW_AND_PLAN_ZH.md` is a Chinese
review summary, not an alternative set of rules.

This package instructs Codex to implement and run the study on the user's server,
then publish code/results/handoff to `Orangekostar/oviovo`. Creating this package
has NOT run any scene, FC, AnyUp, CUDA or official evaluator experiment. The NumPy
reference kernels only test proposed small mathematical contracts.

No new maps or trained model are required. AnyUp is an additional **pretrained**
model and incurs real inference. Reusing cached dense FC features is permitted
for scientific acquisition but never presented as zero standalone method cost.

Run the local package checks with:
```bash
python -m unittest -v test_reference_kernels.py
```

The final task keeps the two existing benchmarks. Nine fixed outputs include two
inherited baselines and seven new conditions: 234 scene/method rows, 18 full pools.
This is exploratory selection on already exposed data, not a new blind test.
