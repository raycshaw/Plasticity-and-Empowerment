# Qwen3-0.6B across additional GSM8K training

Completed October 31, 2026: [results, figures, and interpretation](results/qwen06b_gsm8k200_20rollouts/README.md).

This experiment uses the published Qwen3-0.6B model and two descendants from
Cristian Patino's sequential GSM8K GRPO campaign. The selected public weights
are pinned in `configs/qwen06b_gsm8k200.json`.

| Model | Cumulative additional optimizer steps | Nominal prompt presentations | Publisher's GSM8K accuracy |
|---|---:|---:|---:|
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 0 | 0 | 47.54% |
| [cmpatino/qwen-grpo-r2](https://huggingface.co/cmpatino/qwen-grpo-r2) | 77 | 1,232 | Not unambiguously mapped in the model card |
| [cmpatino/qwen-grpo-r5](https://huggingface.co/cmpatino/qwen-grpo-r5) | 251 | 4,016 | 70.89% |

The endpoint gap is **23.35 percentage points**, reported on all 1,319 GSM8K test
questions using 10-shot greedy evaluation with thinking disabled. These are
publisher-reported results, not measured accuracies from our entropy run.
Source: [pinned model card](https://huggingface.co/cmpatino/qwen-grpo-r5/blob/06c0dac913ec3ca17e3a6418c136be7bf8fcedb8/README.md).

The r5 lineage is 77 r2 steps, then 100 r4 steps, then 74 r5 steps. Nominal prompt
presentations are inferred as steps times 16 prompts per step; the distinct
training IDs were not independently verified. This is additional training
exposure, not an experiment isolating unique dataset size. Learning rate,
rollout group size, and sampling temperature change across stages. The publisher
also selected r5's starting checkpoint using 200 test examples. The original
Qwen3 checkpoint already underwent pretraining and post-training; step zero
does not mean zero lifetime GSM8K exposure.

Other candidates inspected included `x32/Qwen3-0.6B-GRPO-GSM8K-Think`, whose
reported endpoint range is smaller, and
`Atharva1232/Qwen3-0.6B-GRPO-GSM8K`, whose larger reported 8% to 59% gap uses only
100 evaluation questions. The latter's Hugging Face files contained only final
weights; its card mentions intermediate Kaggle artifacts without a usable link.
The chosen series exposes all three required complete weight snapshots directly.

## Measurement protocol

There are 200 questions, 20 independently sampled rollouts per question, and
three checkpoints: **12,000 rollouts**. The original 50 question/injection pairs
are preserved; 150 additional test questions receive new phrase corrections.
See [dataset selection and review](data/gsm8k200/README.md).

The original question-only prompt, full-softmax temperature-1 sampling, EOS
handling, independent ablation caches, and 20-token A plus 20-token B horizons
are retained. This is an entropy-ablation experiment, not an accuracy benchmark.
All checkpoints use the same non-thinking tokenized contexts. The base template
conditionally adds an empty thinking block; the trained templates always add
it. Actual tokenized prompts are compared for every question and correction.
Generated token IDs are spliced directly into B without decoding and
re-tokenization. The forced empty thinking block is preserved in both the actual
A history and the empty-A baseline, and is not counted toward the 20 sampled
tokens.

Question means weight each question equally. Bootstrap intervals use 50,000
draws and paired tests use Holm correction over nine primary comparisons.
Questions with no complete rollout at any checkpoint for a metric are explicitly
listed and excluded from that metric's paired analysis, never imputed as zero.
Sensitivity analyses cover complete A+B samples, source-issue exclusions, and
each selection cohort separately. The across-cohort and sensitivity results are
exploratory, with no multiplicity adjustment across these separate analyses.

## Run

```sh
python3 -m unittest -v
python3 sampled_rlvr_experiment.py --experiment-config configs/qwen06b_gsm8k200.json \
  --limit 3 --rollouts 2 --output-dir runs/qwen06b_smoke
python3 -u run_qwen06b_pipeline.py
```

The pipeline writes resumable raw data to
`runs/qwen06b_gsm8k200_20rollouts`, then validates it, computes statistics, and
creates PNG/PDF plots. Control status is in `runs/qwen06b_control/status.json`.
A pipeline lock prevents duplicate launches. The runner saves every five
rollouts and after every question. Resume after a normal interruption with
`python3 -u run_qwen06b_pipeline.py --resume`. Following a forced kill, verify the
PID in `pipeline.lock` is no longer alive before removing the stale lock.
Keep the source and configuration unchanged while a run is active.

New execution files under `runs/` are local and gitignored. The completed run
is published under `results/qwen06b_gsm8k200_20rollouts`.
If all pinned model files are cached, use `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
when launching or resuming to avoid remote metadata lookups.
