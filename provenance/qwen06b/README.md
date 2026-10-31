# Qwen3-0.6B publication provenance and audit

Published October 31, 2026. The completed run was copied from the local execution
folder into [results/qwen06b_gsm8k200_20rollouts](../../results/qwen06b_gsm8k200_20rollouts).
The raw rollout and question-summary Parquet files and configuration are byte-for-
byte copies. All three model revisions and the 200-question input hash are
recorded in the configuration and embedded in the raw tables.

## Preserved source and execution history

`source_snapshot/` contains the files archived at launch and its original
`sha256.json` inventory. The runner and configuration-module hashes agree with
those recorded in the completed run. It also includes the pinned checkpoint
manifest, dataset, selection and correction records, analysis dependencies, and
analysis scripts. The archived plot script uses the original panel titles; the
published plot script uses the user's requested source-to-target titles. This
label change did not alter any numerical data.

`execution.json` records the interruption after 4,000 saved rollouts, offline
resume with unchanged code and settings, and completion. It omits machine-local
paths and process IDs. No network connection was required for the remaining
inference after resume because the pinned files were already cached.

The archived Python files are provenance snapshots, not separate supported
entry points. Use the scripts in the repository root for published-data analysis.

## Independent audit

- `audit_summary.json`: checks performed and limitations.
- `independent_aggregation.json`: separate calculation of Output A to Output B
  means and endpoint tests directly from the raw entropy arrays in both runs.
- `independent_replay.json`: 15 teacher-forced trajectory replays across all six
  old/new checkpoints, with saved scores, independently rescored values, and errors.
- `replay_saved_tokens.py.txt`: exact script used for those replays. Its original
  paths refer to the local run layout; it is preserved as evidence, not an
  additional default runner. The original questions at source rows 1 and 25 were
  checked in every old checkpoint, and rows 1, 25, and 149 in every new checkpoint.

The replay scorer used explicit continuation framing, full-sequence forward
passes without KV caches, float16 model weights on MPS, and float64 CPU entropy
calculations. The largest absolute Output A to Output B score discrepancy was
0.00210 bits/token. This supports the calculations in the checked examples but
is neither bitwise reproduction nor a complete inference audit.

The raw-data audit validated all 15,000 old/new rollout records and recomputed
question means, bootstrap intervals, paired tests, and adjusted p-values. These
matched the saved results and plot-input tables. All 14 existing tests passed.
The audit does not establish that every source question or generated response
is correct, or that this entropy statistic measures behavioral plasticity.

The previous 1.7B run's historical generating-script hash mismatch remains
unresolved. See [the original provenance record](../README.md). Spot checks do
not retrospectively certify that older code identity.

## File integrity

`artifact_sha256.json` covers this publication's result files, source snapshots,
input files, and audit evidence. It does not modify the original experiment's
separate manifest. Check it from the repository root:

```sh
python3 - <<'PY'
from pathlib import Path
import hashlib, json
manifest = Path('provenance/qwen06b/artifact_sha256.json')
for name, expected in json.loads(manifest.read_text()).items():
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    assert actual == expected, name
print('All Qwen publication artifact hashes match.')
PY
```

Raw-data and statistical validation commands are in the
[results README](../../results/qwen06b_gsm8k200_20rollouts/README.md#files-and-reproduction).
Re-rendering a PDF can change its file hash because of creation metadata even
when the numerical data and visible figure are identical.
