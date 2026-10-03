# Release Acceptance

This candidate is separate from the anonymous submission snapshot. Do not
replace an anonymous mirror with identifying release material during review.

## Stages

1. Local ZIP download/extraction, syntax, profile and task-manifest tests.
2. Linux CPU tensor regression tests and real ALFWorld/WebShop reset/step checks.
3. GPU rollout, two Actor/Critic updates, save, and resume to a further update.
4. Same existing latest-step150 Actors evaluated using the official entry point.
5. Fresh 150-update training for each benchmark, latest150 evaluated at
   decoding seeds123/456/789, same assets/model revision as the reference.

Stages3-5 require an idle GPU allocation. They have not yet been completed
for this release candidate. Stage1-2 results are recorded outside this source
in the operator's validation evidence; no old paper result is labeled a new
official-release measurement.

## Scientific Profiles

ALFWorld: shared2 dual-head residual alpha1, token gamma/lambda1/1,
turn0.95/0.95, final whitening, next-token boundary, train128x1, mini256,
history2, horizon50, lengths4096/512, seen140, latest150.

WebShop: same credit method with alpha3, mini64, horizon15, small product
corpus, fixed goal IDs0..499 and environment seed0, latest150.

Actor LR1e-6, Critic LR1e-5, KL loss0.01, format penalty0.1, rollout
temperature1, evaluation temperature0.4. New GPU engineering defaults
(micro8/8, remove-padding enabled) are explicit and must be recorded.

## Changed Behavior

Divisibility-padding rows are excluded from trajectory GAE, whitening,
Actor/Critic loss and valid-sample normalization. Credit recursion and
centering use FP32; model computation can remain BF16. These are correctness
changes, not solely speed changes, so full-training results may legitimately
differ from historical ones. VLM correctness tests are not covered by the
two text-benchmark release profiles.

Checkpoint reevaluation checks inference/protocol parity, not training parity.
A two-update smoke checks lifecycle correctness, not convergence. Three
decoding seeds on one checkpoint are not three independent training seeds.

Report all endpoint seeds and new-vs-reference deltas. A deviation larger
than3 percentage points in the three-seed mean is an investigation trigger,
not a statistical equivalence bound or a reason to hide a run. Keep failed
runs and configurations; never select a best intermediate checkpoint or
different seeds to force agreement.
