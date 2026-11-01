# Fixed A and a supplied calculation opening

This follow-up uses the original 50 question/correction pairs and the pinned Qwen3-0.6B checkpoints at steps 0, 77, and 251. It reuses the first ten saved baseline A spans for each question (500 distinct reference spans), scoring exactly the same A IDs at every checkpoint. Input scores therefore describe teacher-forced reference text rather than fresh A samples.

B begins with `The calculation is`. The opening is assistant content appended after the normal non-thinking generation suffix, identically in the full, empty-injection, and empty-A contexts. The runner scores and samples 20 NEW tokens after the opening. Supplied opening tokens are not scored, counted in that window, or included in `b_token_ids`. Their text/IDs are stored separately. No numbers, operators, or answers are supplied. The opening encourages calculation but does not guarantee arithmetic or correction adoption.

Question, reference A, correction, and the opening stay identical across checkpoints. B is still sampled separately at each checkpoint. Subsequent B prefixes are shared between ablations within a rollout but can differ across checkpoints; this is not full B teacher forcing. This design changes both A reuse and the B opening relative to the published natural-generation study, so a difference from that study cannot be attributed solely to the opening. An optional matched control omits `--b-opening` while retaining the same references.

## Planned run (not launched)

From this checkout, using a Python environment with requirements installed:

```sh
python sampled_rlvr_experiment.py \
  --experiment-config configs/qwen06b_gsm8k200.json \
  --limit 50 --rollouts 10 \
  --reference-rollouts results/qwen06b_gsm8k200_20rollouts/rollouts.parquet \
  --reference-step 0 \
  --b-opening 'The calculation is' \
  --output-dir runs/qwen06b_original50_fixed_a_calculation
```

This records 1,500 checkpoint evaluations. No model training occurs. The reference file already supplies A; incomplete, missing, duplicate, or mismatched references cause an error instead of resampling. Reference checkpoint revisions and chat mode must match the current manifest. Checkpoint vocabularies, rendered question/correction contexts, and opening token IDs are checked. EOS remains enabled for new B tokens; early B completions have null primary scores. Reference provenance and opening settings are included in resume configuration.

The existing commands retain sampled A and no supplied opening by default. Historical published artifacts are unchanged. Run `python -m unittest -v` to check cache equivalence, context construction, score exclusions, reference alignment, stopping, and existing analysis behavior without running a Qwen experiment.
