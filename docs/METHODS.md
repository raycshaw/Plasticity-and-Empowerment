# Measurement and interpretation

Let X be the question, A the first generated span, J the stored correction,
and B the continuation after a new user turn. All contexts retain their native
chat framing. Only the full context generates tokens. Each alternative receives
the exact same generated history and uses an independent KV cache.

At each position, the score is:

```text
H(model prediction in ablated context) - H(model prediction in full context)
```

Entropy is computed over the full vocabulary in bits, before sampling the next
token. Negative differences are allowed. Two distributions can have the same
entropy while assigning their probabilities to entirely different tokens, so
zero difference does not imply no contextual influence.

| Saved metric prefix | Full context | Ablated context |
|---|---|---|
| `input` | X, A before current token | Empty X, same A prefix |
| `injection` | X, A, J, B before current token | X, same A, empty J, same B prefix |
| `empowerment` | X, A, J, B before current token | X, empty A, same J, same B prefix |

`empowerment` is a historical column name retained for compatibility with raw
records. In reporting, call it **earlier-output ablation**. It is not a channel
capacity estimate. Empty-context ablation is not marginalization over a source
random variable, so these differences are not conditional mutual information.

## Generation and stopping

Every question/checkpoint has 20 independently seeded sampled rollouts. Sampling
uses temperature 1 with no top-k/top-p filtering. EOS probability is included in
entropy, and EOS is allowed to stop generation. The code's `content_token_count`
counts non-EOS generated tokens; other special tokens are not separately masked.

A is generated afresh, without reusing the Qwen screening prefix. If A reaches
20 non-EOS tokens, its exact token IDs are spliced into the assistant turn, the
stored correction is inserted as a new user message, and B starts. Reference
answers and screening metadata are never placed in model context. No answer
format instruction is added. If A ends early, B is not attempted. Each
checkpoint generates its own A and B, so comparisons include changes in
trajectory distribution as well as changes in model predictions.

`*_bits_per_token` and `*_total_20_bits` require a complete 20-token target span.
Input requires complete A; correction and earlier-output scores require complete
A and B. Missing scores are null. Observed scores include all predicted positions,
including EOS when sampled, and are variable-length diagnostics. They are not
substituted for the fixed-horizon primary scores.

Question means average available fixed-horizon rollout scores. The `_n`, `_sd`,
and `_sem` fields give valid counts, sample SD, and Monte Carlo SEM. The
`_joint_complete` summaries use the same complete A+B rollouts for all metrics.
Different checkpoints can have different retained trajectories even then.
`aggregation_complete` means all requested rollouts were recorded, not that all
rollouts reached both horizons.

## Checkpoints

[experiment_config.py](../experiment_config.py) pins full model revisions:

| Label | Model repository | Revision |
|---|---|---|
| 0 | Raghav-Singhal/1pp-1.7b-raw-sft | `0282a9acbab0c0f4147874b04be180be78b61fbd` |
| 50 | VityaVitalich/1pp-1.7b-raw-sft-grpo-gsm8k | `8e47e5742ae93c215cd92a6798ccdac4d385596d` |
| 100 | VityaVitalich/1pp-1.7b-raw-sft-grpo-gsm8k | `e7da9caa9e8e724cf326595a98fa4cf3ce8eb807` |

The [base model card](https://huggingface.co/Raghav-Singhal/1pp-1.7b-raw-sft)
identifies a pretrained and supervised-fine-tuned model. The
[GRPO model card](https://huggingface.co/VityaVitalich/1pp-1.7b-raw-sft-grpo-gsm8k)
identifies that base model and GSM8K. The pinned commits are labeled
[global step 50](https://huggingface.co/VityaVitalich/1pp-1.7b-raw-sft-grpo-gsm8k/commit/8e47e5742ae93c215cd92a6798ccdac4d385596d)
and [global step 100](https://huggingface.co/VityaVitalich/1pp-1.7b-raw-sft-grpo-gsm8k/commit/e7da9caa9e8e724cf326595a98fa4cf3ce8eb807).
This repository measures those published checkpoints; it does not train them.
It is neither a Pythia experiment nor a Qwen checkpoint comparison. Qwen was used
for dataset screening and separate earlier diagnostics.

The runner verifies tokenizer vocabularies and chat templates across checkpoints.
The historical run records MPS and float16 weights; logits are converted to
float32 for entropy. Rollout seeds are recorded individually. New runs also
record Python/package versions and both runner/configuration-module hashes.

## Statistical scope

The unit of pairing is the question, not the rollout. The primary analysis uses
50 question means at each checkpoint, equal question weights, 50,000 paired
question-bootstrap resamples (seed 20261030), pointwise percentile 95% intervals,
and two-sided paired t-tests. Holm correction covers all nine primary tests
(three metrics times three checkpoint pairs).

Pointwise bootstrap intervals are not multiplicity-adjusted intervals and do not
have to agree with the Holm-adjusted significance decision. Repeated rollouts
reduce within-question sampling noise; they are not 1,000 independent questions
or independent training runs. These intervals and tests describe variation in
this selected question set under the analysis assumptions. They do not account
for dataset selection, all Monte Carlo uncertainty, or variation across training
seeds. The study did not measure final-answer accuracy or correction adoption.


## JSD complement (November 2026)

For each saved B position t, P is the full-context next-token distribution and
Q is the distribution after emptying the correction content or all A content.
Question, other source content, message framing, and exact B history are held
fixed within that comparison. The baseline empties context; it does not
marginalize over a probability law for possible source blocks.

The original score is H(Q) − H(P). The complementary score is
JSD(P,Q) = [KL(P || M) + KL(Q || M)] / 2, where M = (P + Q) / 2.
Both use base-2 logarithms. JSD lies between 0 and 1 bit per token and detects
probability redistribution even when H(P) = H(Q). It is not a percentage of
plasticity, an answer-change probability, or formal directed information.

The matched OLMo analysis averages over the first 100 available non-EOS B tokens,
then over five trials per question, then equally across 48 complete questions.
Paired question-cluster bootstraps preserve checkpoint pairing. Response paths
can differ across checkpoints, even though A token IDs are fixed. The run stopped
after preliminary results, so inference is exploratory.

JSD for correction removal is a correction-sensitivity proxy. JSD for A removal
measures earlier-output dependence, not established environment-control
empowerment. Validate against correction adoption and final answers. The
preliminary value-mention screen is separate from corrected-answer grading.
See [the latest protocol and evidence](updates/2026-11-10.md).
