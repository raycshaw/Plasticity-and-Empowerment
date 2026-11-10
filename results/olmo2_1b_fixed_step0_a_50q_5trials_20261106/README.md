# OLMo 2 1B — fixed step-0 A, five trials (stopped early)

Planned: 50 questions × five trials × four checkpoints = 1,000 responses.
Saved: **993**, with 250/250/250/243 at steps 0/200/1400/2600. The run was stopped
at the user's request after inspecting preliminary results; it is not complete.

For each question/trial, every checkpoint receives identical saved step-0 A
content token IDs, up to 100 tokens. B is independently generated per checkpoint
to EOS or a 2,048-token cap. Scores use the first 100 available non-EOS B tokens;
the earlier-output baseline removes all available A content, retaining framing.

`rollouts.parquet` consolidates the original saved JSONL records without changing
their content. `run_status.json` records completion counts. `configs/` and
`source_snapshot/` preserve checkpoint revisions, settings, reference A token IDs,
and original scoring code. Historical `a_complete`/`b_complete` fields describe
length thresholds, not whether the rollout is included in available-token scoring.

The balanced analysis selects 48 questions with all five trials at all four
checkpoints: **960 saved responses**, not 960 new generations. Its entropy and
JSD results are in [the matched rescore](../olmo2_fixed_a_jsd_20261109/README.md).

Because the stopping decision followed preliminary results, findings are
exploratory. Fixed A controls source text, not the generated B paths or model's
interpretation of A. [Full update](../../docs/updates/2026-11-10.md).
